# streamlit_app.py
# Spain — Annual vs Hourly Carbon Accounting (TU-style procurement sizing LP)
# Streamlit Cloud friendly: representative-period optimization + full-year option

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.optimize import linprog

st.set_page_config(page_title="Annual vs Hourly Carbon Accounting — Spain", layout="wide")

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path(__file__).parent / "data"
FILES = {
    "demand": DATA_DIR / "ES_demand_2024.csv",
    "ci": DATA_DIR / "ES_carbon_intensity_hourly_2024.csv",
    "grid_cfe": DATA_DIR / "ES_grid_cfe_2024.csv",
    "cf_wind": DATA_DIR / "ES_cf_wind_2024.csv",
    "cf_solar": DATA_DIR / "ES_cf_solar_2024.csv",
    "price": DATA_DIR / "Spain.csv",  # Ember multi-year
}
missing = [k for k, p in FILES.items() if not p.exists()]
if missing:
    st.error(f"Missing required files in /data: {missing}")
    st.stop()

# -----------------------------
# TU-style assumptions (hardcoded now; editable in sidebar)
# -----------------------------
TU = {
    "fcr": 0.07,  # fixed charge rate (proxy for WACC + lifetime annualization)
    "wind_capex_eur_per_kw": 1200,
    "wind_fom_eur_per_kw_yr": 35,
    "solar_capex_eur_per_kw": 600,
    "solar_fom_eur_per_kw_yr": 15,
    "bat_power_capex_eur_per_kw": 350,
    "bat_energy_capex_eur_per_kwh": 200,
    "bat_fom_eur_per_kw_yr": 8,
    "eta_ch": 0.95,
    "eta_dis": 0.95,
}

# -----------------------------
# Colormaps to match your figures
# CFE: LOW=BLACK, MID=WHITE, HIGH=DEEP GREEN
# CI:  LOW (good)=DEEP GREEN, MID=WHITE, HIGH (bad)=BLACK
# Price: diverging similar to your red/yellow/green plot
# -----------------------------
BLACK_PLATEAU = 0.22
MID_POS = 0.60

CFE_CMAP = LinearSegmentedColormap.from_list(
    "cfe_black_white_green",
    [
        (0.00, "#000000"),
        (BLACK_PLATEAU, "#000000"),
        (MID_POS, "#FFFFFF"),
        (1.00, "#02590F"),
    ],
)
CI_CMAP = LinearSegmentedColormap.from_list(
    "ci_green_white_black",
    [
        (0.00, "#02590F"),
        (MID_POS, "#FFFFFF"),
        (1.00, "#000000"),
    ],
)
PRICE_CMAP = plt.get_cmap("RdYlGn_r")

# -----------------------------
# Helpers
# -----------------------------
def parse_ts_utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True)

def convert_ci_to_tco2_per_mwh(ci_series):
    x = pd.to_numeric(ci_series, errors="coerce")
    m = x.dropna().mean()
    # Electricity Maps carbonIntensity is often gCO2/kWh
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

def pivot_day_hour(ts_utc, values):
    tmp = pd.DataFrame({"ts": ts_utc, "v": values}).dropna()
    tmp["date"] = tmp["ts"].dt.date
    tmp["hour"] = tmp["ts"].dt.hour
    return tmp.pivot_table(index="date", columns="hour", values="v", aggfunc="mean").reindex(columns=list(range(24)))

def plot_heatmap(pivot, title, cbar_label, cmap, vmin=None, vmax=None, norm=None):
    fig = plt.figure(figsize=(10, 4.8))
    plt.imshow(
        pivot.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    plt.colorbar(label=cbar_label)
    plt.title(title)
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Day of year")
    plt.xticks(ticks=np.arange(0, 24, 3), labels=[str(h) for h in range(0, 24, 3)])
    plt.tight_layout()
    return fig

@st.cache_data
def load_all():
    demand = pd.read_csv(FILES["demand"])
    ci = pd.read_csv(FILES["ci"])
    wind = pd.read_csv(FILES["cf_wind"])
    solar = pd.read_csv(FILES["cf_solar"])
    grid = pd.read_csv(FILES["grid_cfe"])
    price = pd.read_csv(FILES["price"])

    demand["ts_utc"] = parse_ts_utc(demand["datetime"])
    ci["ts_utc"] = parse_ts_utc(ci["datetime"])
    wind["ts_utc"] = parse_ts_utc(wind["timestamp"])
    solar["ts_utc"] = parse_ts_utc(solar["timestamp"])
    # grid timestamp is naive -> assume UTC
    grid["ts_utc"] = pd.to_datetime(grid["timestamp"], errors="coerce").dt.tz_localize("UTC")

    # Ember: robust column detection + tz handling
    dt_col = None
    for c in price.columns:
        if "datetime" in c.lower() and "utc" in c.lower():
            dt_col = c
            break
    if dt_col is None:
        dt_col = price.columns[0]

    price_col = None
    for c in price.columns:
        if "price" in c.lower() and "eur" in c.lower():
            price_col = c
            break
    if price_col is None:
        price_col = price.columns[1]

    ts = pd.to_datetime(price[dt_col], errors="coerce")
    if ts.dt.tz is None:
        price["ts_utc"] = ts.dt.tz_localize("UTC")
    else:
        price["ts_utc"] = ts.dt.tz_convert("UTC")

    price = price.rename(columns={price_col: "price_eur_per_mwh"})
    price["price_eur_per_mwh"] = pd.to_numeric(price["price_eur_per_mwh"], errors="coerce")

    return demand, ci, wind, solar, grid, price

def available_years(demand, ci, wind, solar, grid, price):
    sets = []
    for d, col in [(demand, "ts_utc"), (ci, "ts_utc"), (wind, "ts_utc"), (solar, "ts_utc"), (grid, "ts_utc"), (price, "ts_utc")]:
        years = set(pd.to_datetime(d[col], utc=True).dt.year.dropna().unique().tolist())
        sets.append(years)
    return sorted(set.intersection(*sets)) if sets else []

def build_df_for_year(year, demand, ci, wind, solar, grid, price):
    demand_y = demand[demand["ts_utc"].dt.year == year].copy()
    ci_y = ci[ci["ts_utc"].dt.year == year].copy()
    wind_y = wind[wind["ts_utc"].dt.year == year].copy()
    solar_y = solar[solar["ts_utc"].dt.year == year].copy()
    grid_y = grid[grid["ts_utc"].dt.year == year].copy()
    price_y = price[price["ts_utc"].dt.year == year].copy()

    if len(demand_y) == 0 or len(ci_y) == 0 or len(wind_y) == 0 or len(solar_y) == 0 or len(grid_y) == 0 or len(price_y) == 0:
        return None, None

    df = (
        demand_y[["ts_utc", "load_MW"]]
        .merge(ci_y[["ts_utc", "carbonIntensity"]], on="ts_utc", how="inner")
        .merge(wind_y[["ts_utc", "cf"]].rename(columns={"cf": "cf_wind"}), on="ts_utc", how="inner")
        .merge(solar_y[["ts_utc", "cf"]].rename(columns={"cf": "cf_solar"}), on="ts_utc", how="inner")
        .merge(grid_y[["ts_utc", "cfe"]].rename(columns={"cfe": "grid_cfe"}), on="ts_utc", how="left")
        .merge(price_y[["ts_utc", "price_eur_per_mwh"]], on="ts_utc", how="left")
        .sort_values("ts_utc")
        .reset_index(drop=True)
    )

    df["demand_mwh"] = pd.to_numeric(df["load_MW"], errors="coerce")  # MW over 1h -> MWh
    df["ci_tco2_per_mwh"], ci_note = convert_ci_to_tco2_per_mwh(df["carbonIntensity"])

    if df["price_eur_per_mwh"].isna().any():
        df["price_eur_per_mwh"] = df["price_eur_per_mwh"].fillna(df["price_eur_per_mwh"].mean())

    # drop any rows that are missing critical series
    df = df.dropna(subset=["demand_mwh", "cf_wind", "cf_solar", "ci_tco2_per_mwh", "price_eur_per_mwh"]).reset_index(drop=True)
    return df, ci_note

def slice_df(df, mode):
    df = df.sort_values("ts_utc").reset_index(drop=True)
    if mode == "Full year (slow)":
        return df
    if mode == "First 30 days":
        return df.iloc[:24*30].copy()

    # 4 representative weeks: Jan, Apr, Jul, Oct (7 days each)
    year = int(df["ts_utc"].dt.year.iloc[0])
    chunks = []
    for month in [1, 4, 7, 10]:
        start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        end = start + pd.Timedelta(days=7)
        chunk = df[(df["ts_utc"] >= start) & (df["ts_utc"] < end)]
        if len(chunk) > 0:
            chunks.append(chunk)
    return pd.concat(chunks).reset_index(drop=True) if chunks else df.iloc[:24*14].copy()

# -----------------------------
# Optimization (Route 1: procurement sizing LP)
# -----------------------------
def solve_procurement_lp(df, policy, x_target, allow_battery, params):
    T = len(df)
    demand = df["demand_mwh"].to_numpy()
    cfw = df["cf_wind"].to_numpy()
    cfs = df["cf_solar"].to_numpy()
    price = df["price_eur_per_mwh"].to_numpy()
    ci = df["ci_tco2_per_mwh"].to_numpy()

    # variable indexing:
    # [W,S,P,E] + per-hour blocks: imp, exp, ch, dis, soc
    n_cap = 4
    idx_W, idx_S, idx_P, idx_E = 0, 1, 2, 3
    base = n_cap
    idx_imp = base
    idx_exp = base + T
    idx_ch = base + 2 * T
    idx_dis = base + 3 * T
    idx_soc = base + 4 * T
    n = base + 5 * T

    # objective
    c = np.zeros(n)

    # capacity costs €/year per MW (or per MWh for E)
    c[idx_W] = annualized_capex_eur(1.0, params["wind_capex_eur_per_kw"], params["fcr"], params["wind_fom_eur_per_kw_yr"])
    c[idx_S] = annualized_capex_eur(1.0, params["solar_capex_eur_per_kw"], params["fcr"], params["solar_fom_eur_per_kw_yr"])

    if allow_battery:
        c[idx_P] = annualized_battery_cost_eur(1.0, 0.0, params)  # €/yr per MW
        c[idx_E] = annualized_battery_cost_eur(0.0, 1.0, params)  # €/yr per MWh
    else:
        c[idx_P] = 0.0
        c[idx_E] = 0.0

    # net energy cost
    c[idx_imp : idx_imp + T] = price
    c[idx_exp : idx_exp + T] = -price

    # bounds
    bounds = [(0, None), (0, None)]
    bounds += [(0, None), (0, None)] if allow_battery else [(0, 0), (0, 0)]
    bounds += [(0, None)] * (5 * T)

    # equalities
    A_eq = []
    b_eq = []

    # balance: W*cfw + S*cfs + dis + imp - ch - exp = demand
    for t in range(T):
        row = np.zeros(n)
        row[idx_W] = cfw[t]
        row[idx_S] = cfs[t]
        row[idx_dis + t] = 1.0
        row[idx_imp + t] = 1.0
        row[idx_ch + t] = -1.0
        row[idx_exp + t] = -1.0
        A_eq.append(row)
        b_eq.append(demand[t])

    # SOC dynamics
    eta_ch = params["eta_ch"]
    eta_dis = params["eta_dis"]
    for t in range(T - 1):
        row = np.zeros(n)
        row[idx_soc + (t + 1)] = 1.0
        row[idx_soc + t] = -1.0
        row[idx_ch + t] = -eta_ch
        row[idx_dis + t] = 1.0 / eta_dis
        A_eq.append(row)
        b_eq.append(0.0)

    # cyclic SOC: soc_0 = soc_last
    row = np.zeros(n)
    row[idx_soc + 0] = 1.0
    row[idx_soc + (T - 1)] = -1.0
    A_eq.append(row)
    b_eq.append(0.0)

    A_eq = np.vstack(A_eq)
    b_eq = np.array(b_eq)

    # inequalities
    A_ub = []
    b_ub = []

    # storage constraints: ch<=P, dis<=P, soc<=E
    for t in range(T):
        row = np.zeros(n); row[idx_ch + t] = 1.0; row[idx_P] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(n); row[idx_dis + t] = 1.0; row[idx_P] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(n); row[idx_soc + t] = 1.0; row[idx_E] = -1.0
        A_ub.append(row); b_ub.append(0.0)

    if policy == "annual":
        # sum(W*cfw + S*cfs) >= sum(demand)  ->  -sum(...) <= -sum(demand)
        row = np.zeros(n)
        row[idx_W] = -cfw.sum()
        row[idx_S] = -cfs.sum()
        A_ub.append(row); b_ub.append(-demand.sum())

    elif policy == "hourly":
        # imp_t <= (1-x)*demand_t
        cap = (1.0 - x_target) * demand
        for t in range(T):
            row = np.zeros(n)
            row[idx_imp + t] = 1.0
            A_ub.append(row); b_ub.append(float(cap[t]))

    A_ub = np.vstack(A_ub)
    b_ub = np.array(b_ub)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return None, f"LP failed: {res.message}"

    x = res.x
    W, S, P, E = x[idx_W], x[idx_S], x[idx_P], x[idx_E]
    imp = x[idx_imp:idx_imp + T]
    exp = x[idx_exp:idx_exp + T]
    ch = x[idx_ch:idx_ch + T]
    dis = x[idx_dis:idx_dis + T]
    soc = x[idx_soc:idx_soc + T]

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

    achieved_share = 1.0 - np.mean(np.divide(imp, demand, out=np.zeros_like(imp), where=demand > 0))

    metrics = {
        "W_mw": W, "S_mw": S, "BatP_mw": P, "BatE_mwh": E,
        "imports_mwh": float(imp.sum()), "exports_mwh": float(exp.sum()),
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
        "grid_cfe": df["grid_cfe"].to_numpy(),
        "price_eur_per_mwh": price,
    })
    ts["hourly_clean_share"] = 1.0 - np.divide(ts["import_mwh"], ts["demand_mwh"], out=np.zeros(T), where=ts["demand_mwh"] > 0)

    return (metrics, ts), None

# -----------------------------
# UI
# -----------------------------
st.title("Spain — Annual vs Hourly Carbon Accounting (TU-style procurement LP)")

demand, ci, wind, solar, grid, price = load_all()
years = available_years(demand, ci, wind, solar, grid, price)
if not years:
    st.error("No common years found across all datasets. Check timestamps or add matching-year files.")
    st.stop()

st.sidebar.header("Year")
year = st.sidebar.selectbox("Select year", years, index=len(years) - 1)

df, ci_note = build_df_for_year(year, demand, ci, wind, solar, grid, price)
if df is None or len(df) < 24 * 7:
    st.error(f"Year {year} not available (or too few rows) across all required series.")
    st.stop()

st.caption(f"Year={year} | Loaded {len(df)} hourly rows | Carbon intensity conversion: {ci_note}")

# Solver speed controls (prevents Streamlit Cloud crashes)
st.sidebar.header("Solver speed (Streamlit Cloud)")
fast_mode = st.sidebar.checkbox("Fast mode (recommended)", value=True)
sample_mode = st.sidebar.selectbox("Time slice", ["4 representative weeks", "First 30 days", "Full year (slow)"], index=0)
df_opt = slice_df(df, sample_mode) if fast_mode else df
st.caption(f"Optimization horizon: {len(df_opt)} hours (of {len(df)})")

# Scenario controls
st.sidebar.header("Scenario")
x_target = st.sidebar.slider("Hourly target (x)", 0.80, 1.00, 0.90, 0.01)
allow_battery = st.sidebar.checkbox("Include battery", value=True)

# Tech assumptions
st.sidebar.header("Tech assumptions (TU-style, editable)")
TU["fcr"] = st.sidebar.number_input("FCR", value=float(TU["fcr"]), step=0.01)
TU["wind_capex_eur_per_kw"] = st.sidebar.number_input("Wind CAPEX (€/kW)", value=float(TU["wind_capex_eur_per_kw"]), step=50.0)
TU["solar_capex_eur_per_kw"] = st.sidebar.number_input("Solar CAPEX (€/kW)", value=float(TU["solar_capex_eur_per_kw"]), step=50.0)
TU["bat_power_capex_eur_per_kw"] = st.sidebar.number_input("Battery power CAPEX (€/kW)", value=float(TU["bat_power_capex_eur_per_kw"]), step=25.0)
TU["bat_energy_capex_eur_per_kwh"] = st.sidebar.number_input("Battery energy CAPEX (€/kWh)", value=float(TU["bat_energy_capex_eur_per_kwh"]), step=10.0)

# Solve both (annual + hourly)
try:
    with st.spinner("Solving LPs (Annual and Hourly)…"):
        solA, errA = solve_procurement_lp(df_opt, "annual", 1.0, allow_battery, TU)
        solH, errH = solve_procurement_lp(df_opt, "hourly", x_target, allow_battery, TU)
except Exception as e:
    st.exception(e)
    st.stop()

if errA:
    st.error(errA); st.stop()
if errH:
    st.error(errH); st.stop()

mA, tsA = solA
mH, tsH = solH

# -----------------------------
# Tabs
# -----------------------------
tab_heat, tab_emis, tab_cost, tab_tech = st.tabs(["Heatmaps", "Emissions", "Cost", "Technology"])

with tab_heat:
    st.subheader("Heatmaps (Spain only)")

    piv_cfe = pivot_day_hour(df["ts_utc"], df["grid_cfe"])
    fig_cfe = plot_heatmap(
        piv_cfe,
        f"Spain — Grid Carbon-Free Share (CFE) • {year}",
        "CFE (0–1)",
        cmap=CFE_CMAP,
        vmin=0, vmax=1,
        norm=PowerNorm(gamma=1.25)
    )
    st.pyplot(fig_cfe)

    ci_vals = df["ci_tco2_per_mwh"].to_numpy()
    vmin_ci = float(np.nanpercentile(ci_vals, 5))
    vmax_ci = float(np.nanpercentile(ci_vals, 95))
    piv_ci = pivot_day_hour(df["ts_utc"], df["ci_tco2_per_mwh"])
    fig_ci = plot_heatmap(
        piv_ci,
        f"Spain — Grid Carbon Intensity • {year}",
        "tCO₂/MWh",
        cmap=CI_CMAP,
        vmin=vmin_ci, vmax=vmax_ci,
        norm=PowerNorm(gamma=1.10)
    )
    st.pyplot(fig_ci)

with tab_emis:
    st.subheader("Emissions — Annual vs Hourly (side by side)")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Annual matching")
        st.metric("Emissions (tCO₂)", f"{mA['emissions_tco2']:,.0f}")
        st.metric("Emissions rate (tCO₂/MWh)", f"{mA['emissions_rate_tco2_per_mwh']:.3f}")
        st.metric("Achieved clean share", f"{mA['achieved_clean_share']*100:.1f}%")

    with c2:
        st.markdown(f"### Hourly matching (target {int(x_target*100)}%)")
        st.metric("Emissions (tCO₂)", f"{mH['emissions_tco2']:,.0f}")
        st.metric("Emissions rate (tCO₂/MWh)", f"{mH['emissions_rate_tco2_per_mwh']:.3f}")
        st.metric("Achieved clean share", f"{mH['achieved_clean_share']*100:.1f}%")

    st.markdown("**Hourly clean share distribution (Annual vs Hourly):**")
    fig = plt.figure(figsize=(10, 3.5))
    plt.hist(tsA["hourly_clean_share"].clip(0, 1), bins=40, alpha=0.6, label="Annual")
    plt.hist(tsH["hourly_clean_share"].clip(0, 1), bins=40, alpha=0.6, label="Hourly")
    plt.xlabel("Hourly clean share (1 − imports/demand)")
    plt.ylabel("Hours")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

with tab_cost:
    st.subheader("Cost — Annual vs Hourly (side by side)")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Annual matching")
        st.metric("Total cost (€/yr)", f"{mA['total_cost_eur']:,.0f}")
        st.metric("Cost (€/MWh)", f"{mA['cost_eur_per_mwh']:.2f}")
        st.metric("Net energy cost (imports − exports)", f"{mA['energy_net_cost_eur']:,.0f}")

    with c2:
        st.markdown(f"### Hourly matching (target {int(x_target*100)}%)")
        st.metric("Total cost (€/yr)", f"{mH['total_cost_eur']:,.0f}")
        st.metric("Cost (€/MWh)", f"{mH['cost_eur_per_mwh']:.2f}")
        st.metric("Net energy cost (imports − exports)", f"{mH['energy_net_cost_eur']:,.0f}")

    st.markdown("**Cost breakdown comparison:**")
    fig = plt.figure(figsize=(10, 3.2))
    parts = ["Capex+FOM", "Net energy"]
    annual_vals = [mA["annualized_capacity_cost_eur"], mA["energy_net_cost_eur"]]
    hourly_vals = [mH["annualized_capacity_cost_eur"], mH["energy_net_cost_eur"]]
    x = np.arange(len(parts))
    w = 0.35
    plt.bar(x - w/2, annual_vals, width=w, label="Annual")
    plt.bar(x + w/2, hourly_vals, width=w, label="Hourly")
    plt.xticks(x, parts)
    plt.ylabel("€/year")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Spain — Day-ahead price heatmap")
    piv_p = pivot_day_hour(df["ts_utc"], df["price_eur_per_mwh"])
    fig_p = plot_heatmap(
        piv_p,
        f"Spain — Day-ahead Price (€/MWh) • {year}",
        "Price (€/MWh)",
        cmap=PRICE_CMAP,
        vmin=-200, vmax=200
    )
    st.pyplot(fig_p)

with tab_tech:
    st.subheader("Technology — Annual vs Hourly (side by side)")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Annual matching")
        st.metric("Wind (MW)", f"{mA['W_mw']:,.0f}")
        st.metric("Solar (MW)", f"{mA['S_mw']:,.0f}")
        st.metric("Battery power (MW)", f"{mA['BatP_mw']:,.0f}")
        st.metric("Battery energy (MWh)", f"{mA['BatE_mwh']:,.0f}")

    with c2:
        st.markdown(f"### Hourly matching (target {int(x_target*100)}%)")
        st.metric("Wind (MW)", f"{mH['W_mw']:,.0f}")
        st.metric("Solar (MW)", f"{mH['S_mw']:,.0f}")
        st.metric("Battery power (MW)", f"{mH['BatP_mw']:,.0f}")
        st.metric("Battery energy (MWh)", f"{mH['BatE_mwh']:,.0f}")

# Downloads
st.subheader("Downloads")
metrics_df = pd.DataFrame([
    {"scenario": "annual", "target": 1.0, **mA},
    {"scenario": "hourly", "target": x_target, **mH},
])
st.download_button(
    "Download metrics (CSV)",
    metrics_df.to_csv(index=False).encode("utf-8"),
    "metrics_annual_vs_hourly.csv",
    "text/csv"
)
st.download_button(
    "Download dispatch (annual) CSV",
    tsA.to_csv(index=False).encode("utf-8"),
    "dispatch_annual.csv",
    "text/csv"
)
st.download_button(
    "Download dispatch (hourly) CSV",
    tsH.to_csv(index=False).encode("utf-8"),
    "dispatch_hourly.csv",
    "text/csv"
)
