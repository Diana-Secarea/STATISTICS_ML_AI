"""
Facility 6: pandas grouping & aggregation.
Facility 7: merge / join — enriches flights with airport geo data.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Facility 6 — groupby aggregation
# ---------------------------------------------------------------------------

def delay_by_carrier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average arrival delay, cancellation rate, and flight count per carrier.
    """
    agg = (
        df.groupby("op_unique_carrier")
        .agg(
            flight_count=("arr_delay", "count"),
            avg_arr_delay=("arr_delay", "mean"),
            avg_dep_delay=("dep_delay", "mean"),
            cancellation_rate=("cancelled", "mean"),
            avg_distance=("distance", "mean"),
        )
        .round(2)
        .reset_index()
        .rename(columns={"op_unique_carrier": "carrier"})
        .sort_values("avg_arr_delay", ascending=False)
    )
    return agg


def delay_by_origin(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Average delays aggregated by origin airport (top N busiest)."""
    agg = (
        df.groupby("origin")
        .agg(
            flight_count=("arr_delay", "count"),
            avg_arr_delay=("arr_delay", "mean"),
            avg_dep_delay=("dep_delay", "mean"),
            cancellation_rate=("cancelled", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("flight_count", ascending=False)
        .head(top_n)
    )
    return agg


def delay_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly trend of average departure delay."""
    agg = (
        df.groupby("month")
        .agg(
            flight_count=("dep_delay", "count"),
            avg_dep_delay=("dep_delay", "mean"),
            avg_arr_delay=("arr_delay", "mean"),
            cancellation_rate=("cancelled", "mean"),
        )
        .round(2)
        .reset_index()
    )
    return agg


def delay_breakdown_by_carrier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Breakdown of delay causes (carrier, weather, NAS, late aircraft)
    per airline — useful for pie/stacked bar charts.
    """
    cause_cols = ["carrier_delay", "weather_delay", "nas_delay",
                  "security_delay", "late_aircraft_delay"]
    agg = (
        df.groupby("op_unique_carrier")[cause_cols]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"op_unique_carrier": "carrier"})
    )
    return agg


# ---------------------------------------------------------------------------
# Facility 7 — merge / join
# ---------------------------------------------------------------------------

def merge_flights_airports(flights: pd.DataFrame,
                            airports: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join flights with airport data on origin IATA code.
    Adds Name, City, Country, Latitude, Longitude of origin airport.
    """
    airports_clean = (
        airports[["IATA", "Name", "City", "Country", "Latitude", "Longitude"]]
        .rename(columns={
            "Name": "origin_airport_name",
            "City": "origin_city",
            "Country": "origin_country",
            "Latitude": "origin_lat",
            "Longitude": "origin_lon",
        })
        .dropna(subset=["IATA"])
        .drop_duplicates(subset=["IATA"])
    )

    merged = pd.merge(
        flights,
        airports_clean,
        left_on="origin",
        right_on="IATA",
        how="inner",
    ).drop(columns=["IATA"])

    return merged


def route_stats(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Per-route (origin → dest) statistics after the merge.
    Includes geo coordinates of the origin for map plotting.
    """
    agg = (
        merged.groupby(["origin", "dest", "origin_lat", "origin_lon",
                        "origin_airport_name", "origin_city"])
        .agg(
            flight_count=("arr_delay", "count"),
            avg_arr_delay=("arr_delay", "mean"),
            cancellation_rate=("cancelled", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("flight_count", ascending=False)
    )
    return agg


def airport_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Per-airport summary with lat/lon — fed directly into GeoPandas.
    """
    agg = (
        merged.groupby(["origin", "origin_airport_name",
                        "origin_city", "origin_lat", "origin_lon"])
        .agg(
            flight_count=("arr_delay", "count"),
            avg_arr_delay=("arr_delay", "mean"),
            avg_dep_delay=("dep_delay", "mean"),
            cancellation_rate=("cancelled", "mean"),
        )
        .round(3)
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_loader import load_clean

    flights, airports = load_clean()

    print("=== Delay by carrier ===")
    print(delay_by_carrier(flights).to_string(index=False))

    print("\n=== Delay by month ===")
    print(delay_by_month(flights).to_string(index=False))

    print("\n=== Merge with airports ===")
    merged = merge_flights_airports(flights, airports)
    print(f"Merged shape: {merged.shape}")
    print(merged[["origin", "origin_airport_name", "origin_lat", "origin_lon"]].head())

    print("\n=== Airport summary (top 5) ===")
    print(airport_summary(merged).head().to_string(index=False))
