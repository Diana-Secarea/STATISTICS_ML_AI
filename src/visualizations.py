"""
Facility 8: matplotlib visualizations.

All functions return a matplotlib Figure so they can be embedded in
Streamlit via st.pyplot(fig) or saved to disk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# 1. Distribution of departure delay
# ---------------------------------------------------------------------------

def plot_delay_distribution(df: pd.DataFrame,
                             col: str = "dep_delay",
                             bins: int = 60) -> plt.Figure:
    """Histogram of departure (or arrival) delay with a zero-delay marker."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[col].dropna(), bins=bins, color="#F321D4", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="On-time threshold")
    ax.set_xlabel("Delay (minutes)")
    ax.set_ylabel("Number of flights")
    ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Average arrival delay by carrier (horizontal bar chart)
# ---------------------------------------------------------------------------

def plot_delay_by_carrier(carrier_df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of avg arrival delay per carrier.
    carrier_df must have columns: carrier, avg_arr_delay.
    """
    df = carrier_df.sort_values("avg_arr_delay")
    colors = ["#F44336" if v >= 0 else "#4CAF50" for v in df["avg_arr_delay"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df["carrier"], df["avg_arr_delay"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Average Arrival Delay (minutes)")
    ax.set_title("Average Arrival Delay by Airline")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Monthly delay trend (line chart)
# ---------------------------------------------------------------------------

def plot_monthly_trend(monthly_df: pd.DataFrame) -> plt.Figure:
    """
    Dual-line chart: avg dep delay and avg arr delay by month.
    monthly_df must have: month, avg_dep_delay, avg_arr_delay.
    """
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_df["month"], monthly_df["avg_dep_delay"],
            marker="o", label="Departure delay", color="#1976D2")
    ax.plot(monthly_df["month"], monthly_df["avg_arr_delay"],
            marker="s", label="Arrival delay", color="#E53935", linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xticks(monthly_df["month"])
    ax.set_xticklabels(month_labels[:len(monthly_df)])
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Delay (minutes)")
    ax.set_title("Monthly Delay Trend (2024)")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Stacked bar — delay causes by carrier
# ---------------------------------------------------------------------------

def plot_delay_causes(cause_df: pd.DataFrame) -> plt.Figure:
    """
    Stacked horizontal bar showing breakdown of delay causes per carrier.
    cause_df: carrier | carrier_delay | weather_delay | nas_delay |
              security_delay | late_aircraft_delay
    """
    cause_cols = ["carrier_delay", "weather_delay", "nas_delay",
                  "security_delay", "late_aircraft_delay"]
    labels = ["Carrier", "Weather", "NAS", "Security", "Late Aircraft"]
    colors = ["#1565C0", "#EF6C00", "#2E7D32", "#6A1B9A", "#C62828"]

    df = cause_df.set_index("carrier")[cause_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="barh", stacked=True, ax=ax, color=colors)
    ax.set_xlabel("Average Delay (minutes)")
    ax.set_title("Delay Cause Breakdown by Airline")
    ax.legend(labels, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Scatter — distance vs arrival delay
# ---------------------------------------------------------------------------

def plot_distance_vs_delay(df: pd.DataFrame,
                            sample: int = 5000) -> plt.Figure:
    """Scatter plot of flight distance vs arrival delay (sampled)."""
    sub = df.dropna(subset=["distance", "arr_delay"]).sample(
        min(sample, len(df)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(sub["distance"], sub["arr_delay"],
               alpha=0.25, s=8, color="#455A64")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="On-time")
    # trend line
    z = np.polyfit(sub["distance"], sub["arr_delay"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sub["distance"].min(), sub["distance"].max(), 200)
    ax.plot(x_line, p(x_line), color="#F57F17", linewidth=1.5, label="Trend")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Arrival Delay (minutes)")
    ax.set_title("Flight Distance vs. Arrival Delay")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Small-multiples — delay distribution per top-5 carriers
# ---------------------------------------------------------------------------

def plot_delay_small_multiples(df: pd.DataFrame, top_n: int = 5) -> plt.Figure:
    """One histogram per top-N carrier (by flight volume)."""
    top_carriers = df["op_unique_carrier"].value_counts().head(top_n).index.tolist()
    fig, axes = plt.subplots(1, top_n, figsize=(14, 4), sharey=True)
    for ax, carrier in zip(axes, top_carriers):
        vals = df.loc[df["op_unique_carrier"] == carrier, "arr_delay"].dropna()
        ax.hist(vals, bins=40, color="#5C6BC0", edgecolor="white", linewidth=0.3)
        ax.set_title(carrier, fontsize=11)
        ax.set_xlabel("Arr Delay (min)")
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Flights")
    fig.suptitle("Arrival Delay Distribution — Top 5 Airlines", fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Pearson correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame,
                              cols: list[str] | None = None) -> plt.Figure:
    """Lower-triangle Pearson correlation heatmap for numeric delay variables."""
    import seaborn as sns

    if cols is None:
        cols = [
            "dep_delay", "arr_delay", "distance", "air_time",
            "carrier_delay", "weather_delay", "nas_delay", "late_aircraft_delay",
        ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, annot_kws={"size": 9},
    )
    ax.set_title("Pearson Correlation Matrix — Flight Delay Variables")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")          # headless — avoids needing a display

    from src.data_loader import load_clean
    from src.analysis import delay_by_carrier, delay_by_month, delay_breakdown_by_carrier

    flights, _ = load_clean()

    figs = [
        ("delay_distribution", plot_delay_distribution(flights)),
        ("delay_by_carrier",   plot_delay_by_carrier(delay_by_carrier(flights))),
        ("monthly_trend",      plot_monthly_trend(delay_by_month(flights))),
        ("delay_causes",       plot_delay_causes(delay_breakdown_by_carrier(flights))),
        ("distance_vs_delay",  plot_distance_vs_delay(flights)),
        ("small_multiples",    plot_delay_small_multiples(flights)),
    ]

    import os, pathlib
    out = pathlib.Path("outputs")
    out.mkdir(exist_ok=True)
    for name, fig in figs:
        path = out / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")
