import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Annual vs Hourly Carbon Accounting", layout="wide")
st.title("Annual vs Hourly Carbon Accounting — Interactive Tool")

st.sidebar.header("Upload required files")
up_demand = st.sidebar.file_uploader("Demand CSV (hourly)", type=["csv"])
up_ci = st.sidebar.file_uploader("Carbon Intensity CSV (hourly)", type=["csv"])
up_wind = st.sidebar.file_uploader("Wind CF CSV (hourly)", type=["csv"])
up_solar = st.sidebar.file_uploader("Solar CF CSV (hourly)", type=["csv"])
up_gridcfe = st.sidebar.file_uploader("Grid CFE share CSV (optional)", type=["csv"])

st.sidebar.header("Scenario controls")
hourly_target = st.sidebar.slider("Hourly target (x)", 0.50, 1.00, 0.90, 0.05)
wind_share = st.sidebar.slider("Wind share of annual procured energy", 0.0, 1.0, 0.5, 0.05)
solar_share = 1.0 - wind_share
residual_rule = st.sidebar.selectbox(
    "Annual residual allocation (for emissions attribution)",
    ["proportional_to_demand", "allocate_to_highest_CI_hours"],
    index=0
)

def parse_ts_utc(s):
    return pd.to_datetime(s, errors="coerce", utc=True)

def convert_ci_to_tco2_per_mwh(ci_series):
    x = pd.to_numeric(ci_series, errors="coerce")
    m = x.dropna().mean()
    if m is not None and m > 5:  # heuristic: likely gCO2/kWh
        return x * 0.001, "gCO2/kWh → tCO2/MWh (×0.001)"
    return x, "tCO2/MWh (assumed)"

def build_procured_profile_from_cfs(df, wind_share, solar_share):
    demand_total = df["demand_mwh"].sum()
    wind_cf_sum = df["cf_wind"].sum()
    solar_cf_sum = df["cf_solar"].sum()

    target_wind = wind_share * demand_total
    target_solar = solar_share * demand_total

    cap_wind_mw = target_wind / wind_cf_sum if wind_cf_sum > 0 else 0.0
    cap_solar_mw = target_solar / solar_cf_sum if solar_cf_sum > 0 else 0.0

    procured = cap_wind_mw * df["cf_wind"] + cap_solar_mw * df["cf_solar"]
    return procured, cap_wind_mw, cap_solar_mw

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

def annual_accounting(df, residual_allocation="proportional_to_demand"):
    demand = df["demand_mwh"].to_numpy()
    cfe = df["procured_cfe_mwh"].to_numpy()
    ci = df["ci_tco2_per_mwh"].to_numpy()

    total_d = float(np.nansum(demand))
    total_c = float(np.nansum(cfe))
    matched = min(total_c, total_d)
    share = matched / total_d if total_d > 0 else 0.0
    annual_unmatched = total_d - matched

    if annual_unmatched <= 1e-9:
        residual = np.zeros_like(demand)
    else:
        if residual_allocation == "proportional_to_demand":
            residual = demand * (1.0 - share)
        elif residual_allocation == "allocate_to_highest_CI_hours":
            residual = np.zeros_like(demand)
            order = np.argsort(ci)[::-1]
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

# Require core uploads
if up_demand is None or up_ci is None or up_wind is None or up_solar is None:
    st.info("Upload Demand, Carbon Intensity, Wind CF, and Solar CF CSVs to begin.")
    st.stop()

demand = pd.read_csv(up_demand)
ci = pd.read_csv(up_ci)
wind = pd.read_csv(up_wind)
solar = pd.read_csv(up_solar)
gridcfe = pd.read_csv(up_gridcfe) if up_gridcfe is not None else None

# Robust-ish column detection (first column timestamp)
demand_ts = "datetime" if "datetime" in demand.columns else demand.columns[0]
ci_ts = "datetime" if "datetime" in ci.columns else ci.columns[0]
wind_ts = "timestamp" if "timestamp" in wind.columns else wind.columns[0]
solar_ts = "timestamp" if "timestamp" in solar.columns else solar.columns[0]

demand["ts_utc"] = parse_ts_utc(demand[demand_ts])
ci["ts_utc"] = parse_ts_utc(ci[ci_ts])
wind["ts_utc"] = parse_ts_utc(wind[wind_ts])
solar["ts_utc"] = parse_ts_utc(solar[solar_ts])

# numeric columns (fallback to 2nd column)
demand_val = "load_MW" if "load_MW" in demand.columns else demand.columns[1]
ci_val = "carbonIntensity" if "carbonIntensity" in ci.columns else ci.columns[1]
wind_val = "cf" if "cf" in wind.columns else wind.columns[1]
solar_val = "cf" if "cf" in solar.columns else solar.columns[1]

df = (
    demand[["ts_utc", demand_val]].rename(columns={demand_val: "load"})
    .merge(ci[["ts_utc", ci_val]].rename(columns={ci_val: "carbonIntensity"}), on="ts_utc", how="inner")
    .merge(wind[["ts_utc", wind_val]].rename(columns={wind_val: "cf_wind"}), on="ts_utc", how="inner")
    .merge(solar[["ts_utc", solar_val]].rename(columns={solar_val: "cf_solar"}), on="ts_utc", how="inner")
)

if gridcfe is not None:
    gc_ts = "timestamp" if "timestamp" in gridcfe.columns else gridcfe.columns[0]
    gc_val = "cfe" if "cfe" in gridcfe.columns else gridcfe.columns[1]
    ts = pd.to_datetime(gridcfe[gc_ts], errors="coerce")
    gridcfe["ts_utc"] = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    df = df.merge(gridcfe[["ts_utc", gc_val]].rename(columns={gc_val: "grid_cfe_share"}), on="ts_utc", how="left")
else:
    df["grid_cfe_share"] = np.nan

df = df.sort_values("ts_utc").reset_index(drop=True)
df["demand_mwh"] = pd.to_numeric(df["load"], errors="coerce")
df["ci_tco2_per_mwh"], ci_note = convert_ci_to_tco2_per_mwh(df["carbonIntensity"])

st.caption(f"Loaded **{len(df)}** hourly rows. Carbon intensity conversion: **{ci_note}**.")

df["procured_cfe_mwh"], cap_wind_mw, cap_solar_mw = build_procured_profile_from_cfs(df, wind_share, solar_share)

base = baseline_grid_only(df)
annual, annual_resid = annual_accounting(df, residual_allocation=residual_rule)
hourly, hourly_resid, hourly_cov = hourly_accounting(df, hourly_target)

kpis = pd.DataFrame([base, annual, hourly])
baseline_emis = base["emissions_tco2"]
kpis["avoided_emissions_tco2_vs_grid"] = baseline_emis - kpis["emissions_tco2"]

col1, col2 = st.columns([1.05, 0.95], gap="large")

with col1:
    st.subheader("Key results")
    st.dataframe(kpis, use_container_width=True)

    st.markdown("**Implied procurement sizing (annual 100% by design):**")
    st.write(f"- Wind capacity: **{cap_wind_mw:,.0f} MW**")
    st.write(f"- Solar capacity: **{cap_solar_mw:,.0f} MW**")

    st.subheader("Emissions rate comparison")
    fig = plt.figure(figsize=(8, 3.6))
    labels = ["grid_only", "annual_100", f"hourly_{int(hourly_target*100)}"]
    vals = [base["emissions_rate_tco2_per_mwh"], annual["emissions_rate_tco2_per_mwh"], hourly["emissions_rate_tco2_per_mwh"]]
    plt.bar(labels, vals)
    plt.ylabel("tCO₂/MWh")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("When does hourly matching fail?")
    st.markdown("**Residual demand duration curve (hourly accounting)**")
    resid_sorted = np.sort(hourly_resid)[::-1]
    fig2 = plt.figure(figsize=(8, 3.2))
    plt.plot(resid_sorted)
    plt.ylabel("Unmatched demand (MWh per hour)")
    plt.xlabel("Hour rank (descending)")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("**Hourly coverage heatmap (UTC hour-of-day vs day-of-year)**")
    tmp = pd.DataFrame({"ts_utc": df["ts_utc"], "cov": hourly_cov})
    tmp["date"] = tmp["ts_utc"].dt.date
    tmp["hour"] = tmp["ts_utc"].dt.hour
    pivot = tmp.pivot_table(index="date", columns="hour", values="cov", aggfunc="mean").reindex(columns=list(range(24)))
    fig3 = plt.figure(figsize=(8, 4.8))
    plt.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar(label="Coverage (0–1)")
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Day of year")
    plt.xticks(ticks=np.arange(0, 24, 3), labels=[str(h) for h in range(0, 24, 3)])
    plt.tight_layout()
    st.pyplot(fig3)

st.subheader("Download outputs")
out_ts = df[["ts_utc","demand_mwh","procured_cfe_mwh","ci_tco2_per_mwh","grid_cfe_share"]].copy()
out_ts["hourly_coverage"] = hourly_cov
out_ts["hourly_unmatched_mwh"] = hourly_resid
out_ts["hourly_emissions_tco2"] = hourly_resid * df["ci_tco2_per_mwh"].to_numpy()

st.download_button("Download KPI table (CSV)", kpis.to_csv(index=False).encode("utf-8"), "kpis_annual_vs_hourly.csv", "text/csv")
st.download_button("Download hourly timeseries (CSV)", out_ts.to_csv(index=False).encode("utf-8"), "timeseries_hourly_outputs.csv", "text/csv")
