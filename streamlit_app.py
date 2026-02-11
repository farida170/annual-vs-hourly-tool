import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Annual vs Hourly Carbon Accounting", layout="wide")

# ----------------------------
# Defaults (your uploaded Spain files)
# ----------------------------
DEFAULTS = {
    "demand": "/mnt/data/ES_demand_2024.csv",
    "ci": "/mnt/data/ES_carbon_intensity_hourly_2024.csv",
    "cf_wind": "/mnt/data/ES_cf_wind_2024.csv",
    "cf_solar": "/mnt/data/ES_cf_solar_2024.csv",
    "grid_cfe": "/mnt/data/ES_grid_cfe_2024.csv",
}

# ----------------------------
# Helpers
# ----------------------------
def parse_ts_utc(s):
    return pd.to_datetime(s, errors="coerce", utc=True)

def read_csv(path_or_file):
    if hasattr(path_or_file, "read"):
        return pd.read_csv(path_or_file)
    return pd.read_csv(path_or_file)

def convert_ci_to_tco2_per_mwh(ci_series):
    """
    Many Electricity Maps exports are gCO2/kWh. Convert to tCO2/MWh by *0.001.
    Heuristic: if mean > 5, treat as gCO2/kWh.
    """
    x = pd.to_numeric(ci_series, errors="coerce")
    mean_val = x.dropna().mean()
    if mean_val is not None and mean_val > 5:
        return x * 0.001, "gCO2/kWh → tCO2/MWh (×0.001)"
    return x, "tCO2/MWh (assumed)"

def build_procured_profile_from_cfs(df, wind_share, solar_share):
    """
    Build a procured CFE MWh profile sized so annual procured energy == annual demand (annual 100% match).
    Split by annual energy shares wind_share / solar_share.
    """
    demand_total = df["demand_mwh"].sum()
    wind_cf_sum = df["cf_wind"].sum()
    solar_cf_sum = df["cf_solar"].sum()

    target_wind = wind_share * demand_total
    target_solar = solar_share * demand_total

    cap_wind_mw = target_wind / wind_cf_sum if wind_cf_sum > 0 else 0.0
    cap_solar_mw = target_solar / solar_cf_sum if solar_cf_sum > 0 else 0.0

    procured = cap_wind_mw * df["cf_wind"] + cap_solar_mw * df["cf_solar"]
    return procured, cap_wind_mw, cap_solar_mw

def annual_accounting(df, residual_allocation="proportional_to_demand"):
    """
    Annual matching: annual procured == annual demand by design, but we keep this generic.
    residual_allocation determines how to distribute annual unmatched share back to hours
    for emissions attribution.
    """
    demand = df["demand_mwh"].to_numpy()
    cfe = df["procured_cfe_mwh"].to_numpy()
    ci = df["ci_tco2_per_mwh"].to_numpy()

    total_d = float(np.nansum(demand))
    total_c = float(np.nansum(cfe))
    matched = min(total_c, total_d)
    share = matched / total_d if total_d > 0 else 0.0

    # Residual annual (unmatched) energy over the year:
    annual_unmatched = total_d - matched

    if annual_unmatched <= 1e-9:
        residual = np.zeros_like(demand)
    else:
        if residual_allocation == "proportional_to_demand":
            # residual_t = demand_t * (1 - share)
            residual = demand * (1.0 - share)

        elif residual_allocation == "allocate_to_highest_CI_hours":
            # Put all residual on the highest carbon-intensity hours (stress test)
            residual = np.zeros_like(demand)
            order = np.argsort(ci)[::-1]  # descending CI
            remaining = annual_unmatched
            for idx in order:
                if remaining <= 0:
                    break
                take = min(remaining, demand[idx])
                residual[idx] = take
                remaining -= take

        else:
            residual = demand * (1.0 - share)

    emissions = float(np.nansum(residual * ci))
    return {
        "policy": "annual",
        "target": 1.0,
        "matched_mwh": float(matched),
        "avg_share": float(share),
        "pass_rate_hours": np.nan,
        "emissions_tco2": emissions,
        "emissions_rate_tco2_per_mwh": emissions / total_d if total_d > 0 else np.nan,
    }, residual

def hourly_accounting(df, target):
    demand = df["demand_mwh"].to_numpy()
    cfe = df["procured_cfe_mwh"].to_numpy()
    ci = df["ci_tco2_per_mwh"].to_numpy()

    matched = np.minimum(cfe, demand)
    coverage = np.divide(matched, demand, out=np.zeros_like(matched, dtype=float), where=demand > 0)
    residual = demand - matched
    emissions = float(np.nansum(residual * ci))
    total_d = float(np.nansum(demand))

    return {
        "policy": "hourly",
        "target": float(target),
        "matched_mwh": float(np.nansum(matched)),
        "avg_share": float(np.nanmean(coverage)),
        "pass_rate_hours": float(np.nanmean(coverage >= target)),
        "emissions_tco2": emissions,
        "emissions_rate_tco2_per_mwh": emissions / total_d if total_d > 0 else np.nan,
    }, residual, coverage

def baseline_grid_only(df):
    demand = df["demand_mwh"].to_numpy()
    ci = df["ci_tco2_per_mwh"].to_numpy()
    emissions = float(np.nansum(demand * ci))
    total_d = float(np.nansum(demand))
    return {
        "policy": "grid_only",
        "target": np.nan,
        "matched_mwh": 0.0,
        "avg_share": float(df["grid_cfe_share"].mean()) if df["grid_cfe_share"].notna().any() else np.nan,
        "pass_rate_hours": np.nan,
        "emissions_tco2": emissions,
        "emissions_rate_tco2_per_mwh": emissions / total_d if total_d > 0 else np.nan,
    }

# Simple cost engine (populate with TU Berlin assumptions)
def annualized_cost_eur(cap_mw, capex_eur_per_kw, fcr, fixed_om_eur_per_kw_yr):
    # MW -> kW
    cap_kw = cap_mw * 1000.0
    return cap_kw * (capex_eur_per_kw * fcr + fixed_om_eur_per_kw_yr)

def cost_module(total_demand_mwh, cap_wind_mw, cap_solar_mw, cost_params):
    wind_cost = annualized_cost_eur(
        cap_wind_mw,
        cost_params["wind_capex_eur_per_kw"],
        cost_params["fcr"],
        cost_params["wind_fixed_om_eur_per_kw_yr"],
    )
    solar_cost = annualized_cost_eur(
        cap_solar_mw,
        cost_params["solar_capex_eur_per_kw"],
        cost_params["fcr"],
        cost_params["solar_fixed_om_eur_per_kw_yr"],
    )
    total_cost = wind_cost + solar_cost
    lcoe = total_cost / total_demand_mwh if total_demand_mwh > 0 else np.nan
    return total_cost, lcoe

# ----------------------------
# UI
# ----------------------------
st.title("Annual vs Hourly Carbon Accounting — Interactive Tool (Thesis MVP)")

with st.sidebar:
    st.header("Data")
    mode = st.radio("Use:", ["Spain 2024 (default)", "Upload my own files"], index=0)

    if mode == "Upload my own files":
        up_demand = st.file_uploader("Demand CSV", type=["csv"])
        up_ci = st.file_uploader("Carbon intensity CSV", type=["csv"])
        up_wind = st.file_uploader("Wind CF CSV", type=["csv"])
        up_solar = st.file_uploader("Solar CF CSV", type=["csv"])
        up_gridcfe = st.file_uploader("Grid CFE share CSV (optional)", type=["csv"])
        demand_src = up_demand
        ci_src = up_ci
        wind_src = up_wind
        solar_src = up_solar
        gridcfe_src = up_gridcfe
    else:
        demand_src = DEFAULTS["demand"]
        ci_src = DEFAULTS["ci"]
        wind_src = DEFAULTS["cf_wind"]
        solar_src = DEFAULTS["cf_solar"]
        gridcfe_src = DEFAULTS["grid_cfe"]

    st.header("Scenario")
    hourly_target = st.slider("Hourly target (x)", 0.50, 1.00, 0.90, 0.05)

    wind_share = st.slider("Wind share of annual procured energy", 0.0, 1.0, 0.5, 0.05)
    solar_share = 1.0 - wind_share
    st.caption(f"Solar share = {solar_share:.2f}")

    residual_rule = st.selectbox(
        "Annual residual allocation (for emissions attribution)",
        ["proportional_to_demand", "allocate_to_highest_CI_hours"],
        index=0
    )

    st.header("Costs (TU Berlin-style inputs)")
    use_costs = st.checkbox("Enable cost module", value=True)
    if use_costs:
        fcr = st.number_input("Fixed Charge Rate (FCR)", value=0.07, step=0.01)
        wind_capex = st.number_input("Wind CAPEX (€/kW)", value=1200, step=50)
        solar_capex = st.number_input("Solar CAPEX (€/kW)", value=600, step=50)
        wind_fom = st.number_input("Wind fixed O&M (€/kW-yr)", value=35, step=5)
        solar_fom = st.number_input("Solar fixed O&M (€/kW-yr)", value=15, step=5)

# ----------------------------
# Load + standardize data
# ----------------------------
if demand_src is None or ci_src is None or wind_src is None or solar_src is None:
    st.warning("Please provide demand, carbon intensity, wind CF, and solar CF.")
    st.stop()

demand = read_csv(demand_src)
ci = read_csv(ci_src)
wind = read_csv(wind_src)
solar = read_csv(solar_src)
gridcfe = read_csv(gridcfe_src) if gridcfe_src is not None else None

# Column mapping for your Spain exports (also works for similar structure)
# Demand: datetime, load_MW
# CI: datetime, carbonIntensity
# CFs: timestamp, cf
demand["ts_utc"] = parse_ts_utc(demand.iloc[:, 0]) if "datetime" not in demand.columns else parse_ts_utc(demand["datetime"])
ci["ts_utc"] = parse_ts_utc(ci.iloc[:, 0]) if "datetime" not in ci.columns else parse_ts_utc(ci["datetime"])
wind["ts_utc"] = parse_ts_utc(wind.iloc[:, 0]) if "timestamp" not in wind.columns else parse_ts_utc(wind["timestamp"])
solar["ts_utc"] = parse_ts_utc(solar.iloc[:, 0]) if "timestamp" not in solar.columns else parse_ts_utc(solar["timestamp"])

# grid_cfe can be naive timestamps -> assume UTC if no tz
grid_cfe_share = None
if gridcfe is not None:
    if "timestamp" in gridcfe.columns:
        ts = pd.to_datetime(gridcfe["timestamp"], errors="coerce")
        if ts.dt.tz is None:
            gridcfe["ts_utc"] = ts.dt.tz_localize("UTC")
        else:
            gridcfe["ts_utc"] = ts.dt.tz_convert("UTC")
    else:
        ts = pd.to_datetime(gridcfe.iloc[:, 0], errors="coerce")
        gridcfe["ts_utc"] = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")

# Build canonical df
df = (
    demand[["ts_utc", "load_MW"]].rename(columns={"load_MW": "load"})
    .merge(ci[["ts_utc", "carbonIntensity"]] if "carbonIntensity" in ci.columns else ci[["ts_utc", ci.columns[1]]].rename(columns={ci.columns[1]:"carbonIntensity"}), on="ts_utc", how="inner")
    .merge(wind[["ts_utc", "cf"]].rename(columns={"cf":"cf_wind"}), on="ts_utc", how="inner")
    .merge(solar[["ts_utc", "cf"]].rename(columns={"cf":"cf_solar"}), on="ts_utc", how="inner")
)

if gridcfe is not None and "cfe" in gridcfe.columns:
    df = df.merge(gridcfe[["ts_utc", "cfe"]].rename(columns={"cfe":"grid_cfe_share"}), on="ts_utc", how="left")
else:
    df["grid_cfe_share"] = np.nan

df = df.sort_values("ts_utc").reset_index(drop=True)

# Units
df["demand_mwh"] = pd.to_numeric(df["load"], errors="coerce")
df["ci_tco2_per_mwh"], ci_note = convert_ci_to_tco2_per_mwh(df["carbonIntensity"])

# Basic checks
n_hours = len(df)
st.caption(f"Loaded **{n_hours}** hourly rows. Carbon intensity conversion: **{ci_note}**.")

if df["demand_mwh"].isna().mean() > 0.01:
    st.warning("Demand has many missing values. Consider cleaning/interpolating.")
if df["ci_tco2_per_mwh"].isna().mean() > 0.01:
    st.warning("Carbon intensity has many missing values. Consider cleaning/interpolating.")

# Build procured CFE from CFs (annual 100% match design)
df["procured_cfe_mwh"], cap_wind_mw, cap_solar_mw = build_procured_profile_from_cfs(
    df, wind_share=wind_share, solar_share=solar_share
)

# ----------------------------
# Run accounting
# ----------------------------
base = baseline_grid_only(df)
annual, annual_residual = annual_accounting(df, residual_allocation=residual_rule)
hourly, hourly_residual, hourly_cov = hourly_accounting(df, target=hourly_target)

kpis = pd.DataFrame([base, annual, hourly])

# Avoided vs baseline
baseline_emis = base["emissions_tco2"]
kpis["avoided_emissions_tco2_vs_grid"] = baseline_emis - kpis["emissions_tco2"]

# Costs (simple annualized CAPEX + FOM)
total_demand = float(df["demand_mwh"].sum())
if use_costs:
    cost_params = dict(
        fcr=fcr,
        wind_capex_eur_per_kw=wind_capex,
        solar_capex_eur_per_kw=solar_capex,
        wind_fixed_om_eur_per_kw_yr=wind_fom,
        solar_fixed_om_eur_per_kw_yr=solar_fom,
    )
    total_cost, lcoe = cost_module(total_demand, cap_wind_mw, cap_solar_mw, cost_params)
    kpis.loc[kpis["policy"].isin(["annual","hourly"]), "annualized_cost_eur"] = total_cost
    kpis.loc[kpis["policy"].isin(["annual","hourly"]), "cost_eur_per_mwh"] = lcoe

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns([1.05, 0.95], gap="large")

with col1:
    st.subheader("Key results")
    st.dataframe(
        kpis.rename(columns={
            "avg_share":"Avg matched share",
            "pass_rate_hours":"Pass rate (hours)",
            "emissions_tco2":"Emissions (tCO₂)",
            "emissions_rate_tco2_per_mwh":"Emissions rate (tCO₂/MWh)",
        }),
        use_container_width=True
    )

    st.markdown("**Implied procurement sizing (annual 100% by design):**")
    st.write(f"- Wind capacity: **{cap_wind_mw:,.0f} MW**")
    st.write(f"- Solar capacity: **{cap_solar_mw:,.0f} MW**")

    # Emissions rate bar chart
    st.subheader("Emissions rate comparison")
    fig = plt.figure(figsize=(8,3.6))
    labels = []
    vals = []
    for _, r in kpis.iterrows():
        if r["policy"] == "grid_only":
            labels.append("grid_only")
        elif r["policy"] == "annual":
            labels.append("annual_100")
        else:
            labels.append(f"hourly_{int(hourly_target*100)}")
        vals.append(r["emissions_rate_tco2_per_mwh"])
    plt.bar(labels, vals)
    plt.ylabel("tCO₂/MWh")
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("What hours are driving the gap?")

    # Duration curve of residual load for hourly
    st.markdown("**Residual (unmatched) demand duration curve — hourly accounting**")
    resid_sorted = np.sort(hourly_residual)[::-1]
    fig2 = plt.figure(figsize=(8,3.2))
    plt.plot(resid_sorted)
    plt.ylabel("Unmatched demand (MWh per hour)")
    plt.xlabel("Hour rank (descending)")
    plt.tight_layout()
    st.pyplot(fig2)

    # Heatmap of hourly coverage
    st.markdown("**Hourly coverage heatmap (UTC hour-of-day vs day-of-year)**")
    tmp = pd.DataFrame({"ts_utc": df["ts_utc"], "cov": hourly_cov})
    tmp["date"] = tmp["ts_utc"].dt.date
    tmp["hour"] = tmp["ts_utc"].dt.hour
    pivot = tmp.pivot_table(index="date", columns="hour", values="cov", aggfunc="mean").reindex(columns=list(range(24)))
    fig3 = plt.figure(figsize=(8,4.8))
    plt.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar(label="Coverage (0–1)")
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Day of year")
    plt.xticks(ticks=np.arange(0,24,3), labels=[str(h) for h in range(0,24,3)])
    plt.tight_layout()
    st.pyplot(fig3)

# ----------------------------
# Downloads
# ----------------------------
st.subheader("Download outputs")
out_ts = df[["ts_utc","demand_mwh","procured_cfe_mwh","ci_tco2_per_mwh","grid_cfe_share"]].copy()
out_ts["hourly_coverage"] = hourly_cov
out_ts["hourly_unmatched_mwh"] = hourly_residual
out_ts["hourly_emissions_tco2"] = hourly_residual * df["ci_tco2_per_mwh"].to_numpy()

st.download_button(
    "Download KPI table (CSV)",
    data=kpis.to_csv(index=False).encode("utf-8"),
    file_name="kpis_annual_vs_hourly.csv",
    mime="text/csv"
)
st.download_button(
    "Download hourly timeseries (CSV)",
    data=out_ts.to_csv(index=False).encode("utf-8"),
    file_name="timeseries_hourly_outputs.csv",
    mime="text/csv"
)

st.caption("Next thesis-grade steps: add storage optimization (or PyPSA adapter), TU Berlin tech cost table import, and scenario batch runner.")
