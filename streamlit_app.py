# streamlit_app.py
# Spain 2024 — Annual vs Hourly Carbon Accounting (TU-style procurement sizing LP)
# Tabs: Emissions | Cost | Technology
# Fixed year: 2024
# Notes:
# - No debug prints (files list / ember columns) in UI
# - Matplotlib norm + vmin/vmax conflict fixed
# - SciPy imported only when user clicks run

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

st.set_page_config(page_title="Annual vs Hourly — Spain (2024)", layout="wide")

YEAR = 2024

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path(__file__).parent / "data"

REQUIRED = [
    "ES_demand_2024.csv",
    "ES_carbon_intensity_hourly_2024.csv",
    "ES_grid_cfe_2024.csv",
    "ES_cf_wind_2024.csv",
    "ES_cf_solar_2024.csv",
    "Spain.csv",
]

if not DATA_DIR.exists():
    st.error("data/ folder not found in the deployed repo. Create a data/ folder and commit it.")
    st.stop()

missing = [f for f in REQUIRED if not (DATA_DIR / f).exists()]
if missing:
    st.error(f"Missing required files in /data: {missing}")
    st.stop()

FILES = {
    "demand": DATA_DIR / "ES_demand_2024.csv",
    "ci": DATA_DIR / "ES_carbon_intensity_hourly_2024.csv",
    "grid_cfe": DATA_DIR / "ES_grid_cfe_2024.csv",
    "cf_wind": DATA_DIR / "ES_cf_wind_2024.csv",
    "cf_solar": DATA_DIR / "ES_cf_solar_2024.csv",
    "price": DATA_DIR / "Spain.csv",
}

# -----------------------------
# TU-style assumptions (editable)
# -----------------------------
# These are "annuitized cost" parameters used in the LP objective.
# Annualized cost = CAPEX * FCR + fixed O&M
TU = {
    "fcr": 0.07,  # fixed charge rate (annuity factor)
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
# TU-style "portfolios / palettes" (simple + thesis-ready)
# -----------------------------
# Palette 1: Wind + Solar only
# Palette 2: Wind + Solar + Batteries
# Palette 3: Wind + Solar + Batteries + (optional) clean firming proxy (not solved yet)
# For now palette 3 behaves like 2, but we show it in UI and reserve it for later upgrades.
PALETTES = {
    "Palette 1 — Wind + Solar": {"wind": True, "solar": True, "battery": False, "firm_clean_proxy": False},
    "Palette 2 — Wind + Solar + Battery": {"wind": True, "solar": True, "battery": True, "firm_clean_proxy": False},
    "Palette 3 — Wind + Solar + Battery + Firm clean (placeholder)": {"wind": True, "solar": True, "battery": True, "firm_clean_proxy": True},
}

# -----------------------------
# Colormaps (match your style)
# -----------------------------
BLACK_PLATEAU = 0.22
MID_POS = 0.60

CFE_CMAP = LinearSegmentedColormap.from_list(
    "cfe_black_white_green",
    [(0.00, "#000000"), (BLACK_PLATEAU, "#000000"), (MID_POS, "#FFFFFF"), (1.00, "#02590F")],
)
CI_CMAP = LinearSegmentedColormap.from_list(
    "ci_green_white_black",
    [(0.00, "#02590F"), (MID_POS, "#FFFFFF"), (1.00, "#000000")],
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
    capex = p_kw * params["bat_power_capex_eur_per_kw"] + e_kwh * params["bat_energy_capex_eur_per_kwh"]
    return capex * params["fcr"] + p_kw * params["bat_fom_eur_per_kw_yr"]

def pivot_day_hour(ts_utc, values):
    tmp = pd.DataFrame({"ts": ts_utc, "v": values}).dropna()
    tmp["date"] = tmp["ts"].dt.date
    tmp["hour"] = tmp["ts"].dt.hour
    return tmp.pivot_table(index="date", columns="hour", values="v", aggfunc="mean").reindex(columns=list(range(24)))

# FIXED: do not pass vmin/vmax when norm is provided
def plot_heatmap(pivot, title, cbar_label, cmap, vmin=None, vmax=None, norm=None):
    fig = plt.figure(figsize=(10, 4.6))
    data = pivot.to_numpy()

    if norm is not None:
        plt.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    else:
        plt.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    plt.colorbar(label=cbar_label)
    plt.title(title)
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Day of year")
    plt.xticks(ticks=np.arange(0, 24, 3), labels=[str(h) for h in range(0, 24, 3)])
    plt.tight_layout()
    return fig

def detect_ember_columns(price_df: pd.DataFrame):
    dt_col = next((c for c in price_df.columns if "datetime" in c.lower() and "utc" in c.lower()), None)
    if dt_col is None:
        dt_col = price_df.columns[0]

    price_col = next((c for c in price_df.columns if "price" in c.lower() and "eur" in c.lower()), None)
    if price_col is None:
        price_col = price_df.columns[1] if len(price_df.columns) > 1 else price_df.columns[0]

    return dt_col, price_col

def slice_df(df, mode):
    df = df.sort_values("ts_utc").reset_index(drop=True)
    if mode == "First 7 days":
        return df.iloc[:24*7].copy()
    if mode == "First 30 days":
        return df.iloc[:24*30].copy()
    if mode == "4 representative weeks":
        year = int(df["ts_utc"].dt.year.iloc[0])
        chunks = []
        for month in [1, 4, 7, 10]:
            start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
            end = start + pd.Timedelta(days=7)
            chunk = df[(df["ts_utc"] >= start) & (df["ts_utc"] < end)]
            if len(chunk) > 0:
                chunks.append(chunk)
        return pd.concat(chunks).reset_index(drop=True) if chunks else df.iloc[:24*14].copy()
    return df  # Full year

# -----------------------------
# Load + build 2024 dataset
# -----------------------------
@st.cache_data
def load_and_build_2024():
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
    grid["ts_utc"] = pd.to_datetime(grid["timestamp"], errors="coerce").dt.tz_localize("UTC")

    dt_col, price_col = detect_ember_columns(price)
    ts = pd.to_datetime(price[dt_col], errors="coerce")
    if ts.dt.tz is None:
        price["ts_utc"] = ts.dt.tz_localize("UTC")
    else:
        price["ts_utc"] = ts.dt.tz_convert("UTC")
    price = price.rename(columns={price_col: "price_eur_per_mwh"})
    price["price_eur_per_mwh"] = pd.to_numeric(price["price_eur_per_mwh"], errors="coerce")

    demand_y = demand[demand["ts_utc"].dt.year == YEAR]
    ci_y = ci[ci["ts_utc"].dt.year == YEAR]
    wind_y = wind[wind["ts_utc"].dt.year == YEAR]
    solar_y = solar[solar["ts_utc"].dt.year == YEAR]
    grid_y = grid[grid["ts_utc"].dt.year == YEAR]
    price_y = price[price["ts_utc"].dt.year == YEAR]

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

    df["demand_mwh"] = pd.to_numeric(df["load_MW"], errors="coerce")
    df["ci_tco2_per_mwh"], ci_note = convert_ci_to_tco2_per_mwh(df["carbonIntensity"])
    df["price_eur_per_mwh"] = df["price_eur_per_mwh"].fillna(df["price_eur_per_mwh"].mean())

    df = df.dropna(subset=["demand_mwh", "cf_wind", "cf_solar", "ci_tco2_per_mwh", "price_eur_per_mwh"]).reset_index(drop=True)

    meta = {"ci_note": ci_note}
    return df, meta

# -----------------------------
# LP model (Route 1)
# -----------------------------
def solve_procurement_lp(df, policy, x_target, palette, params):
    from scipy.optimize import linprog

    tech = PALETTES[palette]
    allow_wind = tech["wind"]
    allow_solar = tech["solar"]
    allow_battery = tech["battery"]

    T = len(df)
    demand = df["demand_mwh"].to_numpy()
    cfw = df["cf_wind"].to_numpy()
    cfs = df["cf_solar"].to_numpy()
    price = df["price_eur_per_mwh"].to_numpy()
    ci = df["ci_tco2_per_mwh"].to_numpy()

    idx_W, idx_S, idx_P, idx_E = 0, 1, 2, 3
    base = 4
    idx_imp = base
    idx_exp = base + T
    idx_ch  = base + 2*T
    idx_dis = base + 3*T
    idx_soc = base + 4*T
    n = base + 5*T

    c = np.zeros(n)

    c[idx_W] = annualized_capex_eur(1.0, params["wind_capex_eur_per_kw"], params["fcr"], params["wind_fom_eur_per_kw_yr"])
    c[idx_S] = annualized_capex_eur(1.0, params["solar_capex_eur_per_kw"], params["fcr"], params["solar_fom_eur_per_kw_yr"])

    if allow_battery:
        c[idx_P] = annualized_battery_cost_eur(1.0, 0.0, params)
        c[idx_E] = annualized_battery_cost_eur(0.0, 1.0, params)
    else:
        c[idx_P] = 0.0
        c[idx_E] = 0.0

    c[idx_imp:idx_imp+T] = price
    c[idx_exp:idx_exp+T] = -price

    # bounds
    bounds = []
    # W, S
    bounds.append((0, None) if allow_wind else (0, 0))
    bounds.append((0, None) if allow_solar else (0, 0))
    # Battery P, E
    if allow_battery:
        bounds += [(0, None), (0, None)]
    else:
        bounds += [(0, 0), (0, 0)]
    # time vars
    bounds += [(0, None)] * (5*T)

    A_eq, b_eq = [], []

    # power balance: W*cfw + S*cfs + dis + imp - ch - exp = demand
    for t in range(T):
        row = np.zeros(n)
        row[idx_W] = cfw[t]
        row[idx_S] = cfs[t]
        row[idx_dis + t] = 1.0
        row[idx_imp + t] = 1.0
        row[idx_ch + t]  = -1.0
        row[idx_exp + t] = -1.0
        A_eq.append(row)
        b_eq.append(demand[t])

    # SOC dynamics
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

    # cyclic SOC
    row = np.zeros(n)
    row[idx_soc + 0] = 1.0
    row[idx_soc + (T-1)] = -1.0
    A_eq.append(row)
    b_eq.append(0.0)

    A_eq = np.vstack(A_eq)
    b_eq = np.array(b_eq)

    A_ub, b_ub = [], []

    # storage constraints
    for t in range(T):
        row = np.zeros(n); row[idx_ch + t] = 1.0; row[idx_P] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(n); row[idx_dis + t] = 1.0; row[idx_P] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(n); row[idx_soc + t] = 1.0; row[idx_E] = -1.0
        A_ub.append(row); b_ub.append(0.0)

    # policy constraints
    if policy == "annual":
        # annual RE generation >= annual demand
        row = np.zeros(n)
        row[idx_W] = -cfw.sum()
        row[idx_S] = -cfs.sum()
        A_ub.append(row); b_ub.append(-demand.sum())

    elif policy == "hourly":
        # limit imports each hour to (1-x)*demand (i.e., clean share >= x)
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
    imp = x[idx_imp:idx_imp+T]
    exp = x[idx_exp:idx_exp+T]
    ch  = x[idx_ch:idx_ch+T]
    dis = x[idx_dis:idx_dis+T]
    soc = x[idx_soc:idx_soc+T]

    emissions = float(np.sum(imp * ci))
    total_demand = float(demand.sum())
    emis_rate = emissions / total_demand if total_demand > 0 else np.nan

    annual_cost = (
        annualized_capex_eur(W, params["wind_capex_eur_per_kw"], params["fcr"], params["wind_fom_eur_per_kw_yr"]) +
        annualized_capex_eur(S, params["solar_capex_eur_per_kw"], params["fcr"], params["solar_fom_eur_per_kw_yr"]) +
        (annualized_battery_cost_eur(P, E, params) if allow_battery else 0.0)
    )
    energy_net_cost = float(np.sum(price * imp) - np.sum(price * exp))
    total_cost = annual_cost + energy_net_cost
    cost_per_mwh = total_cost / total_demand if total_demand > 0 else np.nan

    achieved_clean_share = 1.0 - np.mean(np.divide(imp, demand, out=np.zeros_like(imp), where=demand > 0))

    metrics = {
        "W_mw": W, "S_mw": S, "BatP_mw": P, "BatE_mwh": E,
        "total_cost_eur": total_cost,
        "cost_eur_per_mwh": cost_per_mwh,
        "emissions_tco2": emissions,
        "emissions_rate_tco2_per_mwh": emis_rate,
        "achieved_clean_share": achieved_clean_share,
        "energy_net_cost_eur": energy_net_cost,
        "annualized_tech_cost_eur": annual_cost,
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
    ts["hourly_emissions_tco2"] = ts["import_mwh"] * ts["ci_tco2_per_mwh"]

    return (metrics, ts), None

# -----------------------------
# Load data
# -----------------------------
st.title("Annual vs Hourly Carbon Accounting — Spain (2024)")
st.caption("TU-style procurement sizing LP + visual diagnostics. (Fixed year: 2024)")

try:
    df, meta = load_and_build_2024()
except Exception as e:
    st.exception(e)
    st.stop()

if len(df) < 24 * 7:
    st.error("Too few rows after merge/cleaning. Check timestamp alignment across files.")
    st.stop()

# -----------------------------
# Sidebar (global controls)
# -----------------------------
st.sidebar.header("Scenario controls")

policy_view = st.sidebar.radio("Comparison", ["Annual vs Hourly"], index=0)
x_target = st.sidebar.slider("Hourly target (x)", 0.80, 1.00, 0.90, 0.01)

palette = st.sidebar.selectbox("Technology palette (TU-style)", list(PALETTES.keys()), index=1)

st.sidebar.header("Solver horizon (Cloud-safe)")
horizon = st.sidebar.selectbox(
    "Optimize over",
    ["First 7 days", "First 30 days", "4 representative weeks", "Full year (slow)"],
    index=0
)
df_opt = slice_df(df, horizon)
st.sidebar.caption(f"Optimization hours: {len(df_opt)}")

with st.sidebar.expander("Tech assumptions (TU-style)"):
    TU["fcr"] = st.number_input("FCR", value=float(TU["fcr"]), step=0.01)
    TU["wind_capex_eur_per_kw"] = st.number_input("Wind CAPEX (€/kW)", value=float(TU["wind_capex_eur_per_kw"]), step=50.0)
    TU["solar_capex_eur_per_kw"] = st.number_input("Solar CAPEX (€/kW)", value=float(TU["solar_capex_eur_per_kw"]), step=50.0)
    TU["bat_power_capex_eur_per_kw"] = st.number_input("Battery power CAPEX (€/kW)", value=float(TU["bat_power_capex_eur_per_kw"]), step=25.0)
    TU["bat_energy_capex_eur_per_kwh"] = st.number_input("Battery energy CAPEX (€/kWh)", value=float(TU["bat_energy_capex_eur_per_kwh"]), step=10.0)

# -----------------------------
# Tabs: Emissions | Cost | Technology
# -----------------------------
tab_emis, tab_cost, tab_tech = st.tabs(["Emissions", "Cost", "Technology"])

# -----------------------------
# Emissions tab
# -----------------------------
with tab_emis:
    st.subheader("Emissions diagnostics (Spain 2024)")

    c1, c2 = st.columns(2)
    with c1:
        piv_cfe = pivot_day_hour(df["ts_utc"], df["grid_cfe"])
        st.pyplot(plot_heatmap(
            piv_cfe,
            "Grid carbon-free share (CFE) — Spain 2024",
            "CFE (0–1)",
            CFE_CMAP,
            norm=PowerNorm(gamma=1.25, vmin=0, vmax=1)
        ))
    with c2:
        ci_vals = df["ci_tco2_per_mwh"].to_numpy()
        vmin_ci = float(np.nanpercentile(ci_vals, 5))
        vmax_ci = float(np.nanpercentile(ci_vals, 95))
        piv_ci = pivot_day_hour(df["ts_utc"], df["ci_tco2_per_mwh"])
        st.pyplot(plot_heatmap(
            piv_ci,
            "Grid carbon intensity — Spain 2024",
            "tCO₂/MWh",
            CI_CMAP,
            norm=PowerNorm(gamma=1.10, vmin=vmin_ci, vmax=vmax_ci)
        ))

    st.markdown("### Annual vs Hourly emissions (model-based)")
    st.caption("Click run to compute annual vs hourly procurement outcomes and resulting consumption emissions from grid imports.")

    run = st.button("▶ Run optimization (Emissions)", type="primary", key="run_emis")
    if run:
        with st.spinner("Solving LPs…"):
            solA, errA = solve_procurement_lp(df_opt, "annual", 1.0, palette, TU)
            solH, errH = solve_procurement_lp(df_opt, "hourly", x_target, palette, TU)

        if errA:
            st.error(errA); st.stop()
        if errH:
            st.error(errH); st.stop()

        mA, tsA = solA
        mH, tsH = solH

        colA, colH = st.columns(2, gap="large")
        with colA:
            st.markdown("#### Annual matching")
            st.metric("Emissions (tCO₂)", f"{mA['emissions_tco2']:,.0f}")
            st.metric("Emissions rate (tCO₂/MWh)", f"{mA['emissions_rate_tco2_per_mwh']:.3f}")
            st.metric("Achieved clean share", f"{mA['achieved_clean_share']*100:.1f}%")
        with colH:
            st.markdown(f"#### Hourly matching (target {int(x_target*100)}%)")
            st.metric("Emissions (tCO₂)", f"{mH['emissions_tco2']:,.0f}")
            st.metric("Emissions rate (tCO₂/MWh)", f"{mH['emissions_rate_tco2_per_mwh']:.3f}")
            st.metric("Achieved clean share", f"{mH['achieved_clean_share']*100:.1f}%")

        st.markdown("**Hourly clean share distribution (Annual vs Hourly):**")
        fig = plt.figure(figsize=(10, 3.5))
        plt.hist(tsA["hourly_clean_share"].clip(0, 1), bins=40, alpha=0.6, label="Annual")
        plt.hist(tsH["hourly_clean_share"].clip(0, 1), bins=40, alpha=0.6, label="Hourly")
        plt.xlabel("Hourly clean share (1 − imports/demand)")
        plt理解
::contentReference[oaicite:0]{index=0}
