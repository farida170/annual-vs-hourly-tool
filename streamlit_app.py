import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog

st.set_page_config(page_title="Annual vs Hourly Carbon Accounting — Spain", layout="wide")

# -----------------------------
# Paths (repo-relative)
# -----------------------------
DATA_DIR = Path(__file__).parent / "data"

FILES = {
    "demand": DATA_DIR / "ES_demand_2024.csv",
    "ci": DATA_DIR / "ES_carbon_intensity_hourly_2024.csv",
    "grid_cfe": DATA_DIR / "ES_grid_cfe_2024.csv",
    "cf_wind": DATA_DIR / "ES_cf_wind_2024.csv",
    "cf_solar": DATA_DIR / "ES_cf_solar_2024.csv",
    "price": DATA_DIR / "Spain.csv",  # Ember multi-year hourly prices
}

missing = [k for k, p in FILES.items() if not p.exists()]
if missing:
    st.error(f"Missing required files in /data: {missing}")
    st.stop()

# -----------------------------
# TU Berlin-style tech assumptions (hardcoded for now)
# Replace values with the exact TU table once you paste it in.
# Units: CAPEX €/kW, FOM €/kW-yr, battery energy €/kWh, etc.
# -----------------------------
TU = {
    "fcr": 0.07,  # fixed charge rate (approx WACC/lifetime combo)
    "wind_capex_eur_per_kw": 1200,
    "wind_fom_eur_per_kw_yr": 35,

    "solar_capex_eur_per_kw": 600,
    "solar_fom_eur_per_kw_yr": 15,

    # Battery: split into power (€/kW) and energy (€/kWh)
    "bat_power_capex_eur_per_kw": 350,
    "bat_energy_capex_eur_per_kwh": 200,
    "bat_fom_eur_per_kw_yr": 8,   # applied to power rating (simplification)

    "eta_ch": 0.95,
    "eta_dis": 0.95,
}

# -----------------------------
# Helpers
# -----------------------------
def parse_ts_utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True)

def convert_ci_to_tco2_per_mwh(ci_series):
    x = pd.to_numeric(ci_series, errors="coerce")
    m = x.dropna().mean()
    # Electricity Maps carbonIntensity often in gCO2/kWh
    if m is not None and m > 5:
        return x * 0.001, "gCO2/kWh → tCO2/MWh (×0.001)"
    return x, "tCO2/MWh (assumed)"

def annualized_capex_eur(cap_mw, capex_eur_per_kw, fcr, fom_eur_per_kw_yr):
    cap_kw = cap_mw * 1000.0
    return cap_kw * (capex_eur_per_kw * fcr + fom_eur_per_kw_yr)

def annualized_battery_cost_eur(p_mw, e_mwh, params):
    p_kw = p_mw * 1000.0
    e_kwh = e_mwh * 1000.0
    capex = (p_kw * params["bat_power_capex_eur_per_kw"] + e_kwh * params["bat_energy_capex_eur_per_kwh"])
    annual = capex * params["fcr"] + p_kw * params["bat_fom_eur_per_kw_yr"]
    return annual

def make_heatmap(ts_utc, values, title, cbar_label):
    tmp = pd.DataFrame({"ts": ts_utc, "v": values}).dropna()
    tmp["date"] = tmp["ts"].dt.date
    tmp["hour"] = tmp["ts"].dt.hour
    pivot = tmp.pivot_table(index="date", columns="hour", values="v", aggfunc="mean").reindex(columns=list(range(24)))
    fig = plt.figure(figsize=(10, 4.8))
    plt.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar(label=cbar_label)
    plt.title(title)
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Day of year")
    plt.xticks(ticks=np.arange(0, 24, 3), labels=[str(h) for h in range(0, 24, 3)])
    plt.tight_layout()
    return fig

@st.cache_data
def load_spain_2024():
    demand = pd.read_csv(FILES["demand"])
    ci = pd.read_csv(FILES["ci"])
    wind = pd.read_csv(FILES["cf_wind"])
    solar = pd.read_csv(FILES["cf_solar"])
    grid = pd.read_csv(FILES["grid_cfe"])

    # Ember price file is multi-year
    price = pd.read_csv(FILES["price"])

    demand["ts_utc"] = parse_ts_utc(demand["datetime"])
    ci["ts_utc"] = parse_ts_utc(ci["datetime"])
    wind["ts_utc"] = parse_ts_utc(wind["timestamp"])
    solar["ts_utc"] = parse_ts_utc(solar["timestamp"])

    # grid cfe timestamp is naive -> assume UTC
    grid["ts_utc"] = pd.to_datetime(grid["timestamp"], errors="coerce").dt.tz_localize("UTC")

    # Ember: "Datetime (UTC)" is naive -> localize UTC
    price["ts_utc"] = pd.to_datetime(price["Datetime (UTC)"], errors="coerce").dt.tz_localize("UTC")
    price_2024 = price[(price["ts_utc"].dt.year == 2024)].copy()

    df = (
        demand[["ts_utc", "load_MW"]]
        .merge(ci[["ts_utc", "carbonIntensity", "emissionFactorType"]], on="ts_utc", how="inner")
        .merge(wind[["ts_utc", "cf"]].rename(columns={"cf": "cf_wind"}), on="ts_utc", how="inner")
        .merge(solar[["ts_utc", "cf"]].rename(columns={"cf": "cf_solar"}), on="ts_utc", how="inner")
        .merge(grid[["ts_utc", "cfe"]].rename(columns={"cfe": "grid_cfe"}), on="ts_utc", how="left")
        .merge(price_2024[["ts_utc", "Price (EUR/MWhe)"]].rename(columns={"Price (EUR/MWhe)": "price_eur_per_mwh"}), on="ts_utc", how="left")
        .sort_values("ts_utc")
        .reset_index(drop=True)
    )

    df["demand_mwh"] = pd.to_numeric(df["load_MW"], errors="coerce")  # MW over 1h -> MWh
    df["ci_tco2_per_mwh"], ci_note = convert_ci_to_tco2_per_mwh(df["carbonIntensity"])

    # fill missing prices with annual mean (keeps LP feasible)
    df["price_eur_per_mwh"] = pd.to_numeric(df["price_eur_per_mwh"], errors="coerce")
    df["price_eur_per_mwh"] = df["price_eur_per_mwh"].fillna(df["price_eur_per_mwh"].mean())

    return df, ci_note

# -----------------------------
# Optimization (Route 1: procurement sizing LP)
# Decision vars:
#   capacities: W, S, P_bat, E_bat
#   per-hour: imp_t, exp_t, ch_t, dis_t, soc_t
#
# Balance: W*cfw + S*cfs + dis + imp = demand + ch + exp
# Storage SOC: soc_{t+1} = soc_t + eta_ch*ch - dis/eta_dis
# Bounds: 0<=ch<=P, 0<=dis<=P, 0<=soc<=E, imp>=0, exp>=0
#
# Annual matching: sum(W*cfw + S*cfs) >= sum(demand)
# Hourly matching target x: imp_t <= (1-x)*demand_t  (=> clean share >= x)
# -----------------------------
def solve_procurement_lp(df, policy, x_target, allow_battery, params):
    T = len(df)
    demand = df["demand_mwh"].to_numpy()
    cfw = df["cf_wind"].to_numpy()
    cfs = df["cf_solar"].to_numpy()
    price = df["price_eur_per_mwh"].to_numpy()
    ci = df["ci_tco2_per_mwh"].to_numpy()

    # variable indexing
    # [W, S, P, E] + per-hour blocks: imp, exp, ch, dis, soc
    n_cap = 4
    idx_W, idx_S, idx_P, idx_E = 0, 1, 2, 3

    base = n_cap
    idx_imp = base
    idx_exp = base + T
    idx_ch  = base + 2*T
    idx_dis = base + 3*T
    idx_soc = base + 4*T
    n = base + 5*T

    # Objective c^T x
    c = np.zeros(n)

    # Annualized capacity costs
    # Wind/Solar (€/year) -> convert to €/objective by keeping in EUR and add energy terms also EUR
    c[idx_W] = annualized_capex_eur(1.0, params["wind_capex_eur_per_kw"], params["fcr"], params["wind_fom_eur_per_kw_yr"])
    c[idx_S] = annualized_capex_eur(1.0, params["solar_capex_eur_per_kw"], params["fcr"], params["solar_fom_eur_per_kw_yr"])

    # Battery costs
    if allow_battery:
        # represent as linear in P and E using cost per MW and per MWh
        # compute annualized cost for 1 MW power, 0 energy:
        one_mw_power_cost = annualized_battery_cost_eur(1.0, 0.0, params)
        one_mwh_energy_cost = annualized_battery_cost_eur(0.0, 1.0, params)
        c[idx_P] = one_mw_power_cost
        c[idx_E] = one_mwh_energy_cost
    else:
        # force them to 0 via bounds later
        c[idx_P] = 0.0
        c[idx_E] = 0.0

    # Net energy cost: imports cost - exports revenue
    c[idx_imp:idx_imp+T] = price
    c[idx_exp:idx_exp+T] = -price

    # No explicit cycling cost in this MVP (can add later)

    # Bounds
    bounds = []
    bounds += [(0, None)]  # W
    bounds += [(0, None)]  # S

    if allow_battery:
        bounds += [(0, None)]  # P
        bounds += [(0, None)]  # E
    else:
        bounds += [(0, 0)]     # P fixed to 0
        bounds += [(0, 0)]     # E fixed to 0

    # Per-hour bounds
    for _ in range(T): bounds.append((0, None))  # imp
    for _ in range(T): bounds.append((0, None))  # exp
    for _ in range(T): bounds.append((0, None))  # ch
    for _ in range(T): bounds.append((0, None))  # dis
    for _ in range(T): bounds.append((0, None))  # soc

    # Equality constraints: balance + SOC dynamics
    A_eq = []
    b_eq = []

    # Power balance each hour:
    # W*cfw_t + S*cfs_t + dis_t + imp_t - ch_t - exp_t = demand_t
    for t in range(T):
        row = np.zeros(n)
        row[idx_W] = cfw[t]
        row[idx_S] = cfs[t]
        row[idx_dis + t] = 1.0
        row[idx_imp + t] = 1.0
        row[idx_ch  + t] = -1.0
        row[idx_exp + t] = -1.0
        A_eq.append(row)
        b_eq.append(demand[t])

    # SOC dynamics:
    # soc_{t+1} - soc_t - eta_ch*ch_t + dis_t/eta_dis = 0
    eta_ch = params["eta_ch"]
    eta_dis = params["eta_dis"]
    for t in range(T-1):
        row = np.zeros(n)
        row[idx_soc + (t+1)] = 1.0
        row[idx_soc + t] = -1.0
        row[idx_ch + t] = -eta_ch
        row[idx_dis + t] = 1.0/eta_dis
        A_eq.append(row)
        b_eq.append(0.0)

    # Cyclic SOC: soc_0 - soc_last = 0
    row = np.zeros(n)
    row[idx_soc + 0] = 1.0
    row[idx_soc + (T-1)] = -1.0
    A_eq.append(row)
    b_eq.append(0.0)

    A_eq = np.vstack(A_eq)
    b_eq = np.array(b_eq)

    # Inequality constraints
    A_ub = []
    b_ub = []

    # Storage power/energy limits (only relevant if battery allowed)
    # ch_t <= P ; dis_t <= P ; soc_t <= E
    for t in range(T):
        # ch_t - P <= 0
        row = np.zeros(n)
        row[idx_ch + t] = 1.0
        row[idx_P] = -1.0
        A_ub.append(row); b_ub.append(0.0)

        # dis_t - P <= 0
        row = np.zeros(n)
        row[idx_dis + t] = 1.0
        row[idx_P] = -1.0
        A_ub.append(row); b_ub.append(0.0)

        # soc_t - E <= 0
        row = np.zeros(n)
        row[idx_soc + t] = 1.0
        row[idx_E] = -1.0
        A_ub.append(row); b_ub.append(0.0)

    # Policy constraints
    if policy == "annual":
        # sum(W*cfw + S*cfs) >= sum(demand)
        # convert to -sum(...) <= -sum(demand)
        row = np.zeros(n)
        row[idx_W] = -cfw.sum()
        row[idx_S] = -cfs.sum()
        A_ub.append(row)
        b_ub.append(-demand.sum())

    elif policy == "hourly":
        # imp_t <= (1-x)*demand_t  for all t
        cap = (1.0 - x_target) * demand
        for t in range(T):
            row = np.zeros(n)
            row[idx_imp + t] = 1.0
            A_ub.append(row); b_ub.append(float(cap[t]))

    A_ub = np.vstack(A_ub)
    b_ub = np.array(b_ub)

    # Solve with HiGHS (fast)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        return None, f"LP failed: {res.message}"

    x = res.x
    W = x[idx_W]
    S = x[idx_S]
    P = x[idx_P]
    E = x[idx_E]
    imp = x[idx_imp:idx_imp+T]
    exp = x[idx_exp:idx_exp+T]
    ch  = x[idx_ch:idx_ch+T]
    dis = x[idx_dis:idx_dis+T]
    soc = x[idx_soc:idx_soc+T]

    # Metrics
    emissions = float(np.sum(imp * ci))
    total_demand = float(demand.sum())
    emis_rate = emissions / total_demand if total_demand > 0 else np.nan

    annual_cost = (
        annualized_capex_eur(W, params["wind_capex_eur_per_kw"], params["fcr"], params["wind_fom_eur_per_kw_yr"])
        + annualized_capex_eur(S, params["solar_capex_eur_per_kw"], params["fcr"], params["solar_fom_eur_per_kw_yr"])
        + (annualized_battery_cost_eur(P, E, params) if allow_battery else 0.0)
    )
    energy_net_cost = float(np.sum(price * imp) - np.sum(price * exp))
    total_cost = annual_cost + energy_net_cost
    cost_per_mwh = total_cost / total_demand if total_demand > 0 else np.nan

    # Hourly clean share actually achieved (via imports)
    achieved_share = 1.0 - np.mean(np.divide(imp, demand, out=np.zeros_like(imp), where=demand > 0))

    out = {
        "W_mw": W, "S_mw": S, "BatP_mw": P, "BatE_mwh": E,
        "imports_mwh": float(imp.sum()),
        "exports_mwh": float(exp.sum()),
        "annualized_capacity_cost_eur": annual_cost,
        "energy_net_cost_eur": energy_net_cost,
        "total_cost_eur": total_cost,
        "cost_eur_per_mwh": cost_per_mwh,
        "emissions_tco2": emissions,
        "emissions_rate_tco2_per_mwh": emis_rate,
        "achieved_clean_share": achieved_share,
    }

    ts = pd.DataFrame({
        "ts_utc": df["ts_utc"],
        "demand_mwh": demand,
        "import_mwh": imp,
        "export_mwh": exp,
        "ch_mwh": ch,
        "dis_mwh": dis,
        "soc_mwh": soc,
        "ci_tco2_per_mwh": ci,
        "price_eur_per_mwh": price
    })
    ts["hourly_clean_share"] = 1.0 - np.divide(ts["import_mwh"], ts["demand_mwh"], out=np.zeros(T), where=ts["demand_mwh"]>0)

    return (out, ts), None

# -----------------------------
# Load data
# -----------------------------
df, ci_note = load_spain_2024()

st.title("Spain — Annual vs Hourly Carbon Accounting (TU-style procurement LP)")
st.caption(f"Data: Spain 2024 (hours={len(df)}). Carbon intensity conversion: {ci_note}")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Scenario")
policy = st.sidebar.radio("Accounting policy", ["annual", "hourly"], index=1)
x_target = st.sidebar.slider("Hourly target (x)", 0.80, 1.00, 0.90, 0.01) if policy == "hourly" else 1.0
allow_battery = st.sidebar.checkbox("Include battery", value=True)

st.sidebar.header("Tech cost assumptions (TU-style, editable)")
TU["fcr"] = st.sidebar.number_input("FCR", value=float(TU["fcr"]), step=0.01)
TU["wind_capex_eur_per_kw"] = st.sidebar.number_input("Wind CAPEX (€/kW)", value=float(TU["wind_capex_eur_per_kw"]), step=50.0)
TU["solar_capex_eur_per_kw"] = st.sidebar.number_input("Solar CAPEX (€/kW)", value=float(TU["solar_capex_eur_per_kw"]), step=50.0)
TU["bat_power_capex_eur_per_kw"] = st.sidebar.number_input("Battery power CAPEX (€/kW)", value=float(TU["bat_power_capex_eur_per_kw"]), step=25.0)
TU["bat_energy_capex_eur_per_kwh"] = st.sidebar.number_input("Battery energy CAPEX (€/kWh)", value=float(TU["bat_energy_capex_eur_per_kwh"]), step=10.0)

# -----------------------------
# Tabs
# -----------------------------
tab_heat, tab_emis, tab_cost, tab_tech = st.tabs(["Heatmaps", "Emissions", "Cost", "Technology"])

with tab_heat:
    st.subheader("Heatmaps (Spain 2024)")
    fig1 = make_heatmap(df["ts_utc"], df["grid_cfe"], "Spain — Grid Carbon-Free Share (CFE)", "CFE (0–1)")
    st.pyplot(fig1)

    fig2 = make_heatmap(df["ts_utc"], df["ci_tco2_per_mwh"], "Spain — Grid Carbon Intensity", "tCO₂/MWh")
    st.pyplot(fig2)

# Solve LP once per control set
@st.cache_data(show_spinner=False)
def cached_solve(policy, x_target, allow_battery, tu_params_hash):
    # tu_params_hash only used to invalidate cache; pass TU dict serialized
    return solve_procurement_lp(df, policy, x_target, allow_battery, TU)

with st.spinner("Solving procurement sizing LP (wind + solar + battery)…"):
    sol, err = solve_procurement_lp(df, policy, x_target, allow_battery, TU)

if err:
    st.error(err)
    st.stop()

metrics, ts = sol

with tab_emis:
    st.subheader("Emissions results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Emissions (tCO₂)", f"{metrics['emissions_tco2']:,.0f}")
    c2.metric("Emissions rate (tCO₂/MWh)", f"{metrics['emissions_rate_tco2_per_mwh']:.3f}")
    c3.metric("Achieved clean share", f"{metrics['achieved_clean_share']*100:.1f}%")

    st.markdown("**Hourly clean share distribution (1 − imports/demand):**")
    fig = plt.figure(figsize=(9,3.5))
    plt.hist(ts["hourly_clean_share"].clip(0,1), bins=40)
    plt.xlabel("Hourly clean share")
    plt.ylabel("Hours")
    plt.tight_layout()
    st.pyplot(fig)

with tab_cost:
    st.subheader("Cost results (TU-style annualized CAPEX + net energy cost)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total cost (€/yr)", f"{metrics['total_cost_eur']:,.0f}")
    c2.metric("Cost (€/MWh)", f"{metrics['cost_eur_per_mwh']:.2f}")
    c3.metric("Net energy cost (imports − exports)", f"{metrics['energy_net_cost_eur']:,.0f}")

    st.markdown("**Cost breakdown:**")
    fig = plt.figure(figsize=(9,3.2))
    parts = ["Annualized capacity cost", "Net energy cost"]
    vals = [metrics["annualized_capacity_cost_eur"], metrics["energy_net_cost_eur"]]
    plt.bar(parts, vals)
    plt.ylabel("€/year")
    plt.tight_layout()
    st.pyplot(fig)

with tab_tech:
    st.subheader("Technology portfolio (optimized)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wind (MW)", f"{metrics['W_mw']:,.0f}")
    c2.metric("Solar (MW)", f"{metrics['S_mw']:,.0f}")
    c3.metric("Battery power (MW)", f"{metrics['BatP_mw']:,.0f}")
    c4.metric("Battery energy (MWh)", f"{metrics['BatE_mwh']:,.0f}")

    st.markdown("**Operational summary:**")
    st.write(f"- Imports: {metrics['imports_mwh']:,.0f} MWh")
    st.write(f"- Exports: {metrics['exports_mwh']:,.0f} MWh")

st.subheader("Download outputs")
st.download_button("Download metrics (CSV)", pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8"),
                   "metrics.csv", "text/csv")
st.download_button("Download dispatch timeseries (CSV)", ts.to_csv(index=False).encode("utf-8"),
                   "dispatch_timeseries.csv", "text/csv")
