"""
Facility 4: Encoding methods — one-hot encoding + label encoding.
Facility 5: Scaling methods — StandardScaler on numeric features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

# Columns to one-hot encode (low-medium cardinality categoricals)
OHE_COLS = ["op_unique_carrier"]

# Columns to label-encode (higher cardinality, used for tree/distance models)
LABEL_ENC_COLS = ["origin", "dest"]

# Numeric features used downstream for scaling / modeling
NUMERIC_FEATURES = [
    "dep_delay", "arr_delay", "distance", "air_time",
    "carrier_delay", "weather_delay", "nas_delay", "late_aircraft_delay",
]


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    One-hot encode carrier column; label-encode origin/dest airports.
    Returns (encoded_df, encoders_dict) so encoders can be reused on test data.
    """
    df = df.copy()
    encoders: dict = {}

    # One-hot encoding — creates dummy columns, drops the first to avoid multicollinearity
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True, dtype=int)

    # Label encoding — replaces string with integer code
    for col in LABEL_ENC_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def encoding_summary(df_original: pd.DataFrame, df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Show which columns were added / transformed."""
    new_cols = set(df_encoded.columns) - set(df_original.columns)
    changed_cols = set(LABEL_ENC_COLS)
    rows = []
    for col in sorted(new_cols):
        rows.append({"column": col, "type": "one-hot (new)", "unique_values": int(df_encoded[col].nunique())})
    for col in sorted(changed_cols):
        rows.append({"column": col, "type": "label-encoded", "unique_values": int(df_encoded[col].nunique())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale(df: pd.DataFrame,
          cols: list[str] | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler (zero mean, unit variance) to numeric columns.
    Returns (scaled_df, fitted_scaler).
    """
    if cols is None:
        cols = NUMERIC_FEATURES

    # keep only cols that exist after encoding
    cols = [c for c in cols if c in df.columns]

    df = df.copy()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


def scaling_summary(df_before: pd.DataFrame,
                    df_after: pd.DataFrame,
                    cols: list[str] | None = None) -> pd.DataFrame:
    """Compare mean and std before/after scaling."""
    if cols is None:
        cols = [c for c in NUMERIC_FEATURES if c in df_before.columns]
    rows = []
    for col in cols:
        rows.append({
            "column": col,
            "mean_before": round(df_before[col].mean(), 3),
            "std_before":  round(df_before[col].std(), 3),
            "mean_after":  round(df_after[col].mean(), 6),
            "std_after":   round(df_after[col].std(), 6),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, StandardScaler]:
    """
    Run full feature-engineering pipeline.
    Returns (feature_df, encoders, scaler).
    """
    df_enc, encoders = encode(df)
    df_scaled, scaler = scale(df_enc)
    return df_scaled, encoders, scaler


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_loader import load_clean

    flights, _ = load_clean()
    print("Before encoding:", flights.shape)

    df_enc, encoders = encode(flights)
    print("After encoding:", df_enc.shape)
    print(encoding_summary(flights, df_enc).to_string(index=False))

    df_scaled, scaler = scale(df_enc)
    print("\nScaling summary:")
    print(scaling_summary(df_enc, df_scaled).to_string(index=False))
