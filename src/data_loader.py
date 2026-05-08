"""
Facility 3 (partial): Missing & extreme values handling.
Loads flight + airport data, draws a reproducible 100k-row sample,
imputes / drops missing values, and caps outliers via winsorization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLIGHTS_PATH = os.path.join(BASE_DIR, "data", "flight_data_2024.csv")
AIRPORTS_PATH = os.path.join(BASE_DIR, "data", "Airports-Only.csv")

SAMPLE_SIZE = 100_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (flights_sample, airports) DataFrames."""
    flights_full = pd.read_csv(FLIGHTS_PATH, low_memory=False)
    flights = flights_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    airports = pd.read_csv(AIRPORTS_PATH, encoding="latin-1")
    return flights, airports


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def summarise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy summary of missing values per column."""
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    return (
        pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
    )


def plot_missing_heatmap(df: pd.DataFrame, ax=None) -> plt.Axes:
    """Plot a heatmap of missing values (columns with any NaN only)."""
    cols_with_na = df.columns[df.isna().any()].tolist()
    sample = df[cols_with_na].isna().astype(int).head(500)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(sample.T, cbar=False, yticklabels=True, xticklabels=False,
                cmap="YlOrRd", ax=ax)
    ax.set_title("Missing Values Heatmap (first 500 rows, columns with NaN)")
    ax.set_xlabel("Rows (sample)")
    ax.set_ylabel("Column")
    plt.tight_layout()
    return ax


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation strategy:
    - cancellation_code: fill with 'N' (not cancelled — vast majority).
    - delay columns (carrier_delay etc.): fill 0 (no delay recorded = 0 min).
    - dep_delay / arr_delay: fill with column median (small fraction missing).
    - rows still missing after imputation: drop.
    """
    df = df.copy()

    # cancellation_code is NaN for non-cancelled flights
    df["cancellation_code"] = df["cancellation_code"].fillna("N")

    # delay breakdown columns — NaN means the flight was not delayed by that cause
    delay_cause_cols = ["carrier_delay", "weather_delay", "nas_delay",
                        "security_delay", "late_aircraft_delay"]
    df[delay_cause_cols] = df[delay_cause_cols].fillna(0)

    # primary delay measures — fill with median
    for col in ["dep_delay", "arr_delay"]:
        df[col] = df[col].fillna(df[col].median())

    # cancelled flights never completed: fill flight-completion columns with 0
    # so they are not dropped by dropna() and remain available for cancellation models
    completion_cols = [
        "dep_time", "taxi_out", "wheels_off", "wheels_on", "taxi_in",
        "arr_time", "actual_elapsed_time", "air_time",
    ]
    for col in completion_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # drop any remaining rows with NaN (tiny fraction)
    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# Extreme values (winsorization)
# ---------------------------------------------------------------------------

def cap_outliers(df: pd.DataFrame,
                 cols: list[str],
                 lower_pct: float = 0.01,
                 upper_pct: float = 0.99) -> pd.DataFrame:
    """Cap values at [lower_pct, upper_pct] percentiles (winsorization)."""
    df = df.copy()
    for col in cols:
        lo = df[col].quantile(lower_pct)
        hi = df[col].quantile(upper_pct)
        df[col] = df[col].clip(lo, hi)
    return df


def plot_outlier_comparison(df_raw: pd.DataFrame,
                            df_capped: pd.DataFrame,
                            col: str = "dep_delay") -> plt.Figure:
    """Side-by-side boxplots before and after capping."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df_raw[col].plot(kind="box", ax=axes[0], title=f"{col} — before capping")
    df_capped[col].plot(kind="box", ax=axes[1], title=f"{col} — after capping")
    plt.suptitle("Outlier Treatment: Winsorization", fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

NUMERIC_COLS_TO_CAP = [
    "dep_delay", "arr_delay", "distance",
    "carrier_delay", "weather_delay", "nas_delay",
    "late_aircraft_delay", "air_time",
]


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Public entry point used by all other modules.
    Returns (flights_clean, airports).
    """
    flights_raw, airports = load_raw()
    flights_imputed = handle_missing(flights_raw)
    flights_clean = cap_outliers(flights_imputed, NUMERIC_COLS_TO_CAP)
    return flights_clean, airports


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flights_raw, airports = load_raw()

    print("=== Raw sample ===")
    print(f"Shape: {flights_raw.shape}")
    print(summarise_missing(flights_raw).to_string())

    flights_clean, _ = load_clean()
    print("\n=== After cleaning ===")
    print(f"Shape: {flights_clean.shape}")
    print(f"Remaining NaN: {flights_clean.isna().sum().sum()}")

    fig = plot_outlier_comparison(flights_raw, flights_clean)
    plt.show()
