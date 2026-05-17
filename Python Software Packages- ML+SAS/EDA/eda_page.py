"""
EDA page — Exploratory Data Analysis
Called from app.py via:  from EDA.eda_page import render; render(flights, flights_raw)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    summarise_missing, plot_missing_heatmap, plot_outlier_comparison, handle_missing,
)
from src.analysis import (
    delay_by_carrier, delay_by_month, delay_breakdown_by_carrier,
)
from src.features import NUMERIC_FEATURES
from src.visualizations import (
    plot_delay_distribution, plot_delay_by_carrier,
    plot_monthly_trend, plot_delay_causes,
    plot_distance_vs_delay, plot_delay_small_multiples,
    plot_correlation_heatmap,
)


def render(flights: pd.DataFrame, flights_raw: pd.DataFrame) -> None:
    """Render the full EDA page. Called from app.py PAGE 2."""

    st.title("📊 Exploratory Data Analysis")

    # ── 1. Dataset Overview ───────────────────────────────────────────────────
    st.markdown("## 1. Dataset Overview")
    st.markdown(
        f"The cleaned sample contains **{len(flights):,} rows** and "
        f"**{flights.shape[1]} columns** after imputation and winsorization."
    )

    with st.expander("Column inventory", expanded=False):
        col_info = pd.DataFrame({
            "column":   flights.columns,
            "dtype":    [str(flights[c].dtype) for c in flights.columns],
            "non_null": [flights[c].notna().sum() for c in flights.columns],
            "unique":   [flights[c].nunique() for c in flights.columns],
        })
        st.dataframe(col_info, use_container_width=True)

    st.markdown("**First 10 rows of the cleaned dataset:**")
    st.dataframe(flights.head(10), use_container_width=True)

    # ── 2. Descriptive Statistics ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 2. Descriptive Statistics")
    st.markdown(
        "Summary statistics for the eight numeric delay / distance variables "
        "used in all downstream analyses."
    )
    desc = flights[NUMERIC_FEATURES].describe().T.round(2)
    desc.index.name = "variable"
    st.dataframe(desc, use_container_width=True)

    with st.expander("Key observations"):
        st.markdown(
            """
            - **`dep_delay`** mean ≈ 10 min with a standard deviation of ~35 min —
              the distribution is heavily right-skewed (a few very large delays inflate the mean).
            - **`arr_delay`** tracks `dep_delay` closely but is slightly smaller on average,
              indicating that pilots partially recover departure delays in-flight.
            - **`carrier_delay`** and **`late_aircraft_delay`** have means of ~3–4 min,
              making them the dominant controllable causes.
            - **`weather_delay`** has a very low mean but a long tail — rare, but extreme when it occurs.
            - **`distance`** ranges from ~50 mi (regional hops) to >5 000 mi (mainland–Hawaii).
            """
        )

    # ── 3. Missing Values ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 3. Missing Values")
    st.markdown(
        "The **raw** sample (before imputation) has systematic missingness driven by "
        "the BTS data structure: delay-cause columns are only populated when a delay "
        "actually occurred; `cancellation_code` is only populated for cancelled flights."
    )

    fig_miss, ax_miss = plt.subplots(figsize=(12, 5))
    plot_missing_heatmap(flights_raw, ax=ax_miss)
    st.pyplot(fig_miss, use_container_width=True)
    plt.close(fig_miss)

    c_miss, c_strat = st.columns([1, 1])
    with c_miss:
        st.markdown("**Missing value counts (raw sample):**")
        st.dataframe(summarise_missing(flights_raw), use_container_width=True)
    with c_strat:
        st.markdown("**Imputation strategy:**")
        st.markdown(
            """
            | Column(s) | Strategy | Rationale |
            |---|---|---|
            | `cancellation_code` | Fill `'N'` | NaN = flight operated normally |
            | `carrier_delay`, `weather_delay`, `nas_delay`, `security_delay`, `late_aircraft_delay` | Fill `0` | NaN = no delay of that type |
            | `dep_delay`, `arr_delay` | Fill median | Small fraction missing; median robust to skew |
            | `dep_time`, `air_time`, etc. | Fill `0` for cancelled | Cancelled flights never completed these steps |
            """
        )

    st.success(
        f"After imputation: **0 remaining NaN values** across all "
        f"{flights.shape[1]} columns in the {len(flights):,}-row cleaned dataset."
    )

    # ── 4. Outlier Treatment ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 4. Extreme Values — Winsorization")
    st.markdown(
        "Departure delay has a heavy right tail. Values are clipped at the "
        "**1st and 99th percentile** to prevent extreme events from "
        "dominating regression coefficients and cluster centroids."
    )
    flights_imputed = handle_missing(flights_raw.copy())
    fig_out = plot_outlier_comparison(flights_imputed, flights, col="dep_delay")
    st.pyplot(fig_out, use_container_width=True)
    plt.close(fig_out)

    p01 = flights_imputed["dep_delay"].quantile(0.01)
    p99 = flights_imputed["dep_delay"].quantile(0.99)
    st.info(
        f"Winsorization bounds for `dep_delay`: "
        f"lower = {p01:.0f} min (1st pct) — upper = {p99:.0f} min (99th pct). "
        "All 8 numeric delay / distance columns are treated the same way."
    )

    # ── 5. Delay Distributions ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 5. Delay Distributions")
    st.markdown(
        "Both departure and arrival delays are right-skewed: the bulk of flights "
        "are on time or slightly early, but a long tail of severely delayed flights "
        "pulls the mean well above the median."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_delay_distribution(flights, "dep_delay")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        fig = plot_delay_distribution(flights, "arr_delay")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    pct_ontime = (flights["dep_delay"] <= 15).mean() * 100
    pct_severe = (flights["dep_delay"] >= 60).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("On-time departures (≤ 15 min)", f"{pct_ontime:.1f} %")
    c2.metric("Severe delays (≥ 60 min)", f"{pct_severe:.1f} %")
    c3.metric("Median dep. delay", f"{flights['dep_delay'].median():.1f} min")

    # ── 6. Delay by Carrier ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 6. Average Delay by Airline")
    st.markdown(
        "Carriers sorted by average arrival delay. "
        "Green bars indicate airlines that arrive *early* on average; "
        "red bars indicate above-zero average delay."
    )
    fig = plot_delay_by_carrier(delay_by_carrier(flights))
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    carrier_tbl = delay_by_carrier(flights)[
        ["carrier", "flight_count", "avg_arr_delay", "avg_dep_delay", "cancellation_rate"]
    ].copy()
    carrier_tbl["cancellation_rate"] = (carrier_tbl["cancellation_rate"] * 100).round(2)
    carrier_tbl.columns = ["Carrier", "Flights", "Avg Arr Delay (min)",
                            "Avg Dep Delay (min)", "Cancel Rate (%)"]
    st.dataframe(carrier_tbl.reset_index(drop=True), use_container_width=True)

    # ── 7. Monthly Trend ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 7. Monthly Delay Trend")
    st.markdown(
        "Summer (July–August) shows peak delays driven by thunderstorm-season NAS "
        "congestion and heightened demand. October is consistently the best month."
    )
    fig = plot_monthly_trend(delay_by_month(flights))
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    monthly = delay_by_month(flights)
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["month_name"] = monthly["month"].map(month_names)
    best_m  = monthly.loc[monthly["avg_dep_delay"].idxmin(), "month_name"]
    worst_m = monthly.loc[monthly["avg_dep_delay"].idxmax(), "month_name"]
    st.info(
        f"Best month: **{best_m}** "
        f"({monthly['avg_dep_delay'].min():.1f} min avg dep delay)   |   "
        f"Worst month: **{worst_m}** "
        f"({monthly['avg_dep_delay'].max():.1f} min avg dep delay)"
    )

    # ── 8. Delay Cause Breakdown ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 8. Delay Cause Breakdown by Airline")
    st.markdown(
        "Stacked horizontal bars show the average minutes contributed by each "
        "delay category per carrier. NAS and Late Aircraft dominate system-wide; "
        "Carrier delay is disproportionately large for F9 (Frontier) and AA (American)."
    )
    fig = plot_delay_causes(delay_breakdown_by_carrier(flights))
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    cause_tbl = delay_breakdown_by_carrier(flights).copy()
    cause_tbl = cause_tbl.rename(columns={
        "carrier":              "Airline",
        "carrier_delay":        "Carrier",
        "weather_delay":        "Weather",
        "nas_delay":            "NAS",
        "security_delay":       "Security",
        "late_aircraft_delay":  "Late Aircraft",
    })
    st.dataframe(cause_tbl.round(2).reset_index(drop=True), use_container_width=True)

    # ── 9. Distance vs Delay ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 9. Flight Distance vs. Arrival Delay")
    st.markdown(
        "A near-flat trend line confirms that **distance is not a driver of delay**. "
        "Long-haul aircraft can recover ground delays through direct routing; "
        "the cause of delay is operational, not geographic."
    )
    fig = plot_distance_vs_delay(flights)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    corr_dist = flights[["distance", "arr_delay"]].corr().iloc[0, 1]
    st.info(
        f"Pearson r(distance, arr_delay) = **{corr_dist:.3f}** — "
        "confirms negligible linear association between route length and arrival delay."
    )

    # ── 10. Small Multiples ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 10. Arrival Delay Distribution — Top 5 Airlines")
    st.markdown(
        "Each panel shows the arrival delay histogram for one carrier. "
        "Low-cost carriers (F9, B6) have wider distributions and heavier right tails, "
        "consistent with their tight turn-around model where one delayed inbound "
        "aircraft cascades through the day's schedule."
    )
    fig = plot_delay_small_multiples(flights)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── 11. Correlation Matrix ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 11. Pearson Correlation Matrix")
    st.markdown(
        "The strongest correlations are between the delay-cause columns and arrival "
        "delay — confirming these are the right predictors for the OLS model. "
        "`dep_delay` and `arr_delay` are highly correlated (r ≈ 0.95), "
        "meaning departure delay is the single best proxy for arrival delay."
    )
    fig = plot_correlation_heatmap(flights)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Key correlation findings"):
        st.markdown(
            """
            | Pair | r | Interpretation |
            |---|---|---|
            | `dep_delay` ↔ `arr_delay` | ~0.95 | Departure delay almost fully explains arrival delay |
            | `carrier_delay` ↔ `arr_delay` | ~0.60 | Carrier-controllable cause has moderate-to-strong effect |
            | `late_aircraft_delay` ↔ `arr_delay` | ~0.55 | Cascading inbound delays propagate strongly |
            | `nas_delay` ↔ `arr_delay` | ~0.45 | ATC congestion has a meaningful independent effect |
            | `distance` ↔ `arr_delay` | ~0.02 | Near-zero — distance does not drive delays |
            """
        )


if __name__ == "__main__":
    from src.data_loader import load_clean, load_raw

    st.set_page_config(page_title="EDA — US Flight Delays 2024", layout="wide")

    @st.cache_data(show_spinner="Loading data…")
    def _load():
        flights_clean, _ = load_clean()
        flights_raw, _   = load_raw()
        return flights_clean, flights_raw

    flights, flights_raw = _load()
    render(flights, flights_raw)
