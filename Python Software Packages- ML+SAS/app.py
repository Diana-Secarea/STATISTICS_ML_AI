"""
Multi-page Streamlit application — US Flight Delay Analysis 2024
Organization: US Airline Industry (BTS On-Time Performance Data)

Pages:
  1. Home
  2. Data Overview      (missing values, stats)
  3. Geographic Analysis (GeoPandas + choropleth)
  4. Feature Engineering (encoding + scaling)
  5. Modeling           (clustering, logistic regression, OLS)
  6. Results & Conclusions
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# ── project modules ──────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader  import load_clean, load_raw, summarise_missing, plot_missing_heatmap, plot_outlier_comparison, cap_outliers, NUMERIC_COLS_TO_CAP, handle_missing
from src.analysis     import delay_by_carrier, delay_by_month, delay_breakdown_by_carrier, merge_flights_airports, airport_summary, delay_by_origin
from src.features     import encode, scale, encoding_summary, scaling_summary, NUMERIC_FEATURES
from src.visualizations import (plot_delay_distribution, plot_delay_by_carrier,
                                  plot_monthly_trend, plot_delay_causes,
                                  plot_distance_vs_delay, plot_delay_small_multiples)
from src.models       import (run_clustering, plot_clusters, cluster_summary,
                               run_logistic, plot_roc_curve, plot_confusion_matrix, lr_feature_importance,
                               run_ols, plot_ols_coefficients, plot_ols_residuals)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Flight Delay Analysis 2024",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & cleaning data…")
def get_data():
    flights, airports = load_clean()
    return flights, airports

@st.cache_data(show_spinner="Loading raw sample for EDA…")
def get_raw():
    flights_raw, _ = load_raw()
    return flights_raw

# ── sidebar navigation ────────────────────────────────────────────────────────

PAGES = [
    "Home",
    "Data Overview",
    "Geographic Analysis",
    "Feature Engineering",
    "Modeling",
    "Results & Conclusions",
]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Data source:** US Bureau of Transportation Statistics  \n"
    "**Dataset:** On-Time Performance 2024  \n"
    "**Rows:** 7 M total · 100 k sample"
)


# =============================================================================
# PAGE 1 — HOME
# =============================================================================
if page == PAGES[0]:
    st.title("✈ US Airline On-Time Performance — 2024")
    st.markdown(
        """
        ### Business Question
        > *Which airlines and routes suffer the worst delays in 2024,
        > and can we predict flight cancellations before they happen?*

        ---
        ### About the Organisation
        The **US Airline Industry** transports over 900 million passengers annually.
        Understanding delay patterns is critical for:
        - Airlines optimising crew and gate scheduling
        - Airports allocating ground resources
        - Regulators setting performance benchmarks

        ### Dataset
        | | |
        |---|---|
        | Source | US Bureau of Transportation Statistics (BTS) |
        | Period | Full year 2024 |
        | Raw rows | ~7 million flights |
        | Sample used | 100 000 flights (stratified random) |
        | Airport reference | OpenFlights — 7 000+ airports with lat/lon |

        ### Python Facilities Demonstrated
        | # | Facility | Module |
        |---|---|---|
        | 1 | Streamlit — display & graphics | `app.py` |
        | 2 | GeoPandas — geospatial analysis | Page 3 |
        | 3 | Missing & extreme values | `data_loader.py` |
        | 4 | Encoding (OHE + Label) | `features.py` |
        | 5 | Scaling (StandardScaler) | `features.py` |
        | 6 | pandas groupby & aggregation | `analysis.py` |
        | 7 | merge / join | `analysis.py` |
        | 8 | matplotlib charts | `visualizations.py` |
        | 9 | scikit-learn — KMeans clustering | `models.py` |
        | 10 | scikit-learn — Logistic Regression | `models.py` |
        | 11 | statsmodels — OLS regression | `models.py` |
        """
    )

    flights, _ = get_data()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Flights in sample", f"{len(flights):,}")
    col2.metric("Airlines", flights["op_unique_carrier"].nunique())
    col3.metric("Origin airports", flights["origin"].nunique())
    col4.metric("Avg departure delay", f"{flights['dep_delay'].mean():.1f} min")


# =============================================================================
# PAGE 2 — DATA OVERVIEW
# =============================================================================
elif page == PAGES[1]:
    st.title("Data Overview")
    flights, _ = get_data()
    flights_raw = get_raw()

    st.subheader("Dataset snapshot")
    st.dataframe(flights.head(10), use_container_width=True)

    st.subheader("Descriptive statistics")
    st.dataframe(flights[NUMERIC_FEATURES].describe().round(2), use_container_width=True)

    # --- Missing values ---
    st.subheader("Facility 3a — Missing values")
    st.markdown(
        "The raw dataset has several columns with systematic missingness. "
        "Below is the heatmap of the **raw** sample (before imputation)."
    )
    fig_miss, ax_miss = plt.subplots(figsize=(12, 5))
    plot_missing_heatmap(flights_raw, ax=ax_miss)
    st.pyplot(fig_miss, use_container_width=True)
    plt.close(fig_miss)

    miss_df = summarise_missing(flights_raw)
    st.dataframe(miss_df, use_container_width=True)

    with st.expander("Imputation strategy"):
        st.markdown(
            """
            | Column(s) | Strategy | Rationale |
            |---|---|---|
            | `cancellation_code` | Fill `'N'` | NaN = flight operated normally |
            | `carrier_delay`, `weather_delay`, `nas_delay`, `security_delay`, `late_aircraft_delay` | Fill `0` | NaN = no delay of that type |
            | `dep_delay`, `arr_delay` | Fill median | Small fraction missing; median robust to skew |
            | `dep_time`, `air_time`, `actual_elapsed_time`, … | Fill `0` for cancelled | Cancelled flights never completed these metrics |
            """
        )

    # --- Outlier treatment ---
    st.subheader("Facility 3b — Extreme values (winsorization)")
    st.markdown(
        "Departure delay has a heavy right tail (large positive outliers). "
        "Values are capped at the **1st and 99th percentile**."
    )
    flights_imputed = handle_missing(flights_raw.copy())
    fig_out = plot_outlier_comparison(flights_imputed, flights, col="dep_delay")
    st.pyplot(fig_out, use_container_width=True)
    plt.close(fig_out)

    # --- Charts ---
    st.subheader("Delay distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_delay_distribution(flights, "dep_delay")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        fig = plot_delay_distribution(flights, "arr_delay")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Monthly delay trend")
    fig = plot_monthly_trend(delay_by_month(flights))
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =============================================================================
# PAGE 3 — GEOGRAPHIC ANALYSIS  (GeoPandas — Facility 2)
# =============================================================================
elif page == PAGES[2]:
    st.title("Geographic Analysis")
    st.markdown("**Facility 2 — GeoPandas**: airports are loaded as a GeoDataFrame and plotted on a US map coloured by average arrival delay.")

    flights, airports = get_data()
    merged = merge_flights_airports(flights, airports)
    ap_stats = airport_summary(merged)

    # --- Build GeoDataFrame ---
    gdf = gpd.GeoDataFrame(
        ap_stats,
        geometry=gpd.points_from_xy(ap_stats["origin_lon"], ap_stats["origin_lat"]),
        crs="EPSG:4326",
    )
    # US airports only (rough bounding box)
    gdf_us = gdf[
        (gdf["origin_lon"].between(-170, -60)) &
        (gdf["origin_lat"].between(15, 72))
    ].copy()

    st.metric("Airports on map", len(gdf_us))

    # colour scale for delay
    delay_min, delay_max = gdf_us["avg_arr_delay"].min(), gdf_us["avg_arr_delay"].max()
    norm = plt.Normalize(vmin=delay_min, vmax=delay_max)
    cmap = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("#D0E8F2")

    # plot points sized by flight count, coloured by avg delay
    sc = ax.scatter(
        gdf_us["origin_lon"],
        gdf_us["origin_lat"],
        c=gdf_us["avg_arr_delay"],
        s=gdf_us["flight_count"] / gdf_us["flight_count"].max() * 300 + 10,
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        zorder=5,
    )
    plt.colorbar(sc, ax=ax, label="Avg Arrival Delay (min)", shrink=0.6)
    ax.set_xlim(-170, -60)
    ax.set_ylim(15, 72)
    ax.set_title("US Airport Average Arrival Delay 2024\n(circle size = flight volume)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # annotate top-5 busiest airports
    top5 = gdf_us.nlargest(5, "flight_count")
    for _, row in top5.iterrows():
        ax.annotate(
            row["origin"],
            xy=(row["origin_lon"], row["origin_lat"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="black",
        )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Top 20 airports by flight volume")
    top20 = gdf_us.nlargest(20, "flight_count")[
        ["origin", "origin_airport_name", "origin_city",
         "flight_count", "avg_arr_delay", "cancellation_rate"]
    ].reset_index(drop=True)
    st.dataframe(top20, use_container_width=True)

    st.subheader("Delay by origin airport (top 20)")
    fig = plot_delay_by_carrier(
        delay_by_origin(flights, 20)
        .rename(columns={"origin": "carrier", "avg_arr_delay": "avg_arr_delay"})
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =============================================================================
# PAGE 4 — FEATURE ENGINEERING
# =============================================================================
elif page == PAGES[3]:
    st.title("⚙️ Feature Engineering")
    flights, _ = get_data()

    # --- Encoding ---
    st.subheader("Facility 4 — Encoding")
    st.markdown(
        """
        | Column | Method | Reason |
        |---|---|---|
        | `op_unique_carrier` | One-hot encoding (OHE) | 15 airlines — low cardinality, no ordinal meaning |
        | `origin`, `dest` | Label encoding | 344 airports — OHE would create 344 columns |
        """
    )
    df_enc, encoders = encode(flights)
    enc_summary = encoding_summary(flights, df_enc)
    col1, col2 = st.columns([1, 2])
    col1.metric("Columns before encoding", flights.shape[1])
    col2.metric("Columns after encoding", df_enc.shape[1])
    st.dataframe(enc_summary, use_container_width=True)

    st.markdown("**Sample of one-hot encoded carrier columns:**")
    ohe_cols = [c for c in df_enc.columns if c.startswith("op_unique_carrier_")]
    st.dataframe(df_enc[["origin", "dest"] + ohe_cols].head(8), use_container_width=True)

    # --- Scaling ---
    st.subheader("Facility 5 — Scaling (StandardScaler)")
    st.markdown(
        "StandardScaler transforms each numeric feature to **zero mean and unit variance**: "
        r"$z = (x - \mu) / \sigma$"
    )
    df_scaled, scaler = scale(df_enc)
    scale_df = scaling_summary(df_enc, df_scaled)
    st.dataframe(scale_df, use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df_enc["dep_delay"].hist(bins=50, ax=axes[0], color="#1976D2", edgecolor="white")
    axes[0].set_title("dep_delay — before scaling")
    axes[0].set_xlabel("Minutes")
    df_scaled["dep_delay"].hist(bins=50, ax=axes[1], color="#43A047", edgecolor="white")
    axes[1].set_title("dep_delay — after StandardScaler")
    axes[1].set_xlabel("Standard deviations")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =============================================================================
# PAGE 5 — MODELING
# =============================================================================
elif page == PAGES[4]:
    st.title("Modeling")
    flights, _ = get_data()

    tab1, tab2, tab3 = st.tabs(["KMeans Clustering", "Logistic Regression", "OLS Regression"])

    # ── KMeans ───────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Facility 9 — KMeans Clustering")
        st.markdown(
            """
            Flights are grouped into **4 delay tiers** based on:
            `dep_delay`, `arr_delay`, `distance`, `air_time`.

            Features are standardised internally before clustering so all dimensions contribute equally.
            """
        )
        with st.spinner("Running KMeans…"):
            c_df, km = run_clustering(flights)

        st.subheader("Cluster centroids (original scale)")
        st.dataframe(cluster_summary(c_df), use_container_width=True)

        fig = plot_clusters(c_df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(
            """
            **Economic interpretation:**
            - **Tier 0** (~early arrivals, short haul): efficient regional operations.
            - **Tier 1–2** (minimal delay): the bulk of long-haul domestic flights.
            - **Tier 3** (>130 min delay): systemic disruptions — weather, cascading delays.
              These flights are prime candidates for proactive passenger rebooking algorithms.
            """
        )

    # ── Logistic Regression ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Facility 10 — Logistic Regression: Predicting Cancellations")
        st.markdown(
            """
            **Target:** `cancelled` (1 = cancelled, 0 = operated)
            **Imbalance:** ~1.35 % cancelled — handled with `class_weight='balanced'`
            **Train/test split:** 80/20, stratified
            """
        )
        with st.spinner("Training logistic regression…"):
            lr = run_logistic(flights)

        col1, col2 = st.columns(2)
        col1.metric("AUC-ROC", f"{lr['auc']:.3f}")
        col2.metric("Recall (cancelled)", f"{lr['report']['1']['recall']:.3f}")

        c1, c2 = st.columns(2)
        with c1:
            fig = plot_roc_curve(lr)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            fig = plot_confusion_matrix(lr)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.subheader("Feature coefficients")
        st.dataframe(lr_feature_importance(lr), use_container_width=True)

        st.markdown(
            """
            **Economic interpretation:**
            The model achieves AUC 0.999 — departure delay is an overwhelming predictor of cancellation.
            Airlines can deploy this model in real time: once a departure delay exceeds a threshold,
            automatically trigger proactive rebooking workflows before the flight is officially cancelled.
            """
        )

    # ── OLS ───────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Facility 11 — OLS Multiple Regression: Explaining Arrival Delay")
        st.markdown(
            r"""
            **Target:** $\log(1 + \text{arr\_delay} + \text{offset})$ — log-transform to reduce skew
            **Features:** dep_delay, distance, air_time, delay causes, month, day_of_week, carrier dummies
            """
        )
        with st.spinner("Fitting OLS…"):
            ols = run_ols(flights)

        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{ols['results'].rsquared:.3f}")
        col2.metric("Adj. R²", f"{ols['results'].rsquared_adj:.3f}")
        col3.metric("Observations", f"{int(ols['results'].nobs):,}")

        st.subheader("Coefficient plot (top 15 by magnitude)")
        fig = plot_ols_coefficients(ols, top_n=15)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Residual plot")
        fig = plot_ols_residuals(ols)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Full OLS summary table"):
            st.text(str(ols["results"].summary()))

        st.markdown(
            """
            **Economic interpretation:**
            NAS (National Airspace System) and weather delays carry the largest positive coefficients —
            meaning systemic, uncontrollable disruptions have the greatest impact on arrival delay.
            Carrier-specific dummies (e.g. Hawaiian Airlines `carrier_HA` = +0.45) reveal structural
            delay tendencies beyond what flight-level features explain.
            """
        )


# =============================================================================
# PAGE 6 — RESULTS & CONCLUSIONS
# =============================================================================
elif page == PAGES[5]:
    st.title("Results & Conclusions")
    flights, _ = get_data()

    st.subheader("Key findings")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Delay performance by airline**
            - Frontier (F9) and American (AA) have the highest average arrival delays (>11 min).
            - Alaska (AS) and Hawaiian (HA) are the most punctual carriers.
            - Summer months (July–August) see average delays >18 minutes; October is the best month.
            """
        )
        fig = plot_delay_by_carrier(delay_by_carrier(flights))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown(
            """
            **Root cause breakdown**
            - NAS (Air Traffic Control) and late arriving aircraft together account for the
              majority of delay minutes across all carriers.
            - Weather delays are rare but extreme — highest variance.
            - Carrier-controllable delays (turnaround, maintenance) dominate at F9 and AA.
            """
        )
        fig = plot_delay_causes(delay_breakdown_by_carrier(flights))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Small-multiples: arrival delay per top-5 airline")
    fig = plot_delay_small_multiples(flights)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Recommendations for the industry")
    st.markdown(
        """
        | # | Recommendation | Supporting evidence |
        |---|---|---|
        | 1 | **Deploy real-time cancellation prediction** at gate level | LR model AUC 0.999 — dep_delay is a dominant signal |
        | 2 | **Prioritise NAS/ATC coordination for high-volume hubs** (ATL, ORD, DFW) | GeoPandas map: busiest airports = highest delay |
        | 3 | **Target summer schedule de-peaking** (reduce July/August departures) | Monthly trend: July avg dep delay 18.9 min vs Oct 4.3 min |
        | 4 | **Audit carrier-controllable delays at F9 and AA** | OLS carrier dummies + delay cause breakdown |
        | 5 | **Use Tier-3 cluster as early warning trigger** | 130+ min delays follow a distinct delay-cause profile |
        """
    )

    st.info(
        "All analysis is based on a 100 000-row stratified sample of 2024 BTS On-Time Performance data. "
        "Conclusions are directionally valid; production models should retrain on the full 7 M-row dataset."
    )
