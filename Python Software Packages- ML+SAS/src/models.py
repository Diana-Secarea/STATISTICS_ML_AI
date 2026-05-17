"""
Facility 9:  scikit-learn KMeans clustering — segment flights into delay tiers.
Facility 10: scikit-learn LogisticRegression — predict cancellation.
Facility 11: statsmodels OLS — explain arrival delay (multiple regression).
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLUSTER_FEATURES = ["dep_delay", "arr_delay", "distance", "air_time"]

_LR_FEATURES = [
    "dep_delay", "distance", "air_time",
    "carrier_delay", "weather_delay", "nas_delay", "late_aircraft_delay",
    "month", "day_of_week",
]

_OLS_FEATURES = [
    "dep_delay", "distance", "air_time",
    "carrier_delay", "weather_delay", "nas_delay", "late_aircraft_delay",
    "month", "day_of_week",
]


def _select_and_drop_na(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    avail = [c for c in cols if c in df.columns]
    return df[avail].dropna()


# ---------------------------------------------------------------------------
# Facility 9 — KMeans clustering
# ---------------------------------------------------------------------------

def run_clustering(df: pd.DataFrame,
                   k: int = 4,
                   features: list[str] | None = None) -> tuple[pd.DataFrame, KMeans]:
    """
    Segment flights into k delay tiers using KMeans.
    Features are scaled internally so raw values can stay in the returned df.
    Returns (df_with_cluster_label, fitted_kmeans).
    """
    if features is None:
        features = _CLUSTER_FEATURES

    sub = _select_and_drop_na(df, features).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(sub[features])

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    sub["cluster"] = km.fit_predict(X)

    # label clusters by mean arrival delay (0 = best, k-1 = worst)
    order = (
        sub.groupby("cluster")["arr_delay"].mean()
        .sort_values()
        .reset_index()
        .reset_index()  # gives us the rank
        .rename(columns={"index": "tier", "cluster": "old_label"})
    )
    remap = dict(zip(order["old_label"], order["tier"]))
    sub["cluster"] = sub["cluster"].map(remap)
    return sub, km


def plot_clusters(cluster_df: pd.DataFrame) -> plt.Figure:
    """Scatter of dep_delay vs arr_delay coloured by cluster tier."""
    k = cluster_df["cluster"].nunique()
    palette = plt.colormaps.get_cmap("tab10").resampled(k)

    fig, ax = plt.subplots(figsize=(9, 6))
    for tier in range(k):
        mask = cluster_df["cluster"] == tier
        ax.scatter(
            cluster_df.loc[mask, "dep_delay"],
            cluster_df.loc[mask, "arr_delay"],
            s=6, alpha=0.35,
            color=palette(tier),
            label=f"Tier {tier}",
        )
    ax.set_xlabel("Departure Delay (min)")
    ax.set_ylabel("Arrival Delay (min)")
    ax.set_title(f"KMeans Flight Clusters (k={k}) — Delay Tiers")
    ax.legend(markerscale=3)
    plt.tight_layout()
    return fig


def cluster_summary(cluster_df: pd.DataFrame) -> pd.DataFrame:
    return (
        cluster_df.groupby("cluster")[_CLUSTER_FEATURES]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"cluster": "tier"})
    )


# ---------------------------------------------------------------------------
# Facility 10 — Logistic Regression (predict cancellation)
# ---------------------------------------------------------------------------

def run_logistic(df: pd.DataFrame,
                 features: list[str] | None = None,
                 test_size: float = 0.2) -> dict:
    """
    Predict flight cancellation (binary).
    Returns dict with: model, X_test, y_test, y_pred, y_prob, report, auc.
    """
    if features is None:
        features = _LR_FEATURES

    sub = _select_and_drop_na(df, features + ["cancelled"]).copy()

    # Re-balance: cancellations are rare (~1.4 %) — stratify + class_weight
    X = sub[features].values
    y = sub["cancelled"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    report = classification_report(y_te, y_pred, output_dict=True)
    auc = roc_auc_score(y_te, y_prob)

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "X_test": X_te,
        "y_test": y_te,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": report,
        "auc": auc,
    }


def plot_roc_curve(lr_result: dict) -> plt.Figure:
    """ROC curve for the logistic regression model."""
    fpr, tpr, _ = roc_curve(lr_result["y_test"], lr_result["y_prob"])
    auc = lr_result["auc"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1565C0", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Flight Cancellation Prediction")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(lr_result: dict) -> plt.Figure:
    cm = confusion_matrix(lr_result["y_test"], lr_result["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["On-time/Operated", "Cancelled"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Cancellation Prediction")
    plt.tight_layout()
    return fig


def lr_feature_importance(lr_result: dict) -> pd.DataFrame:
    model = lr_result["model"]
    coefs = model.coef_[0]
    return (
        pd.DataFrame({"feature": lr_result["features"], "coefficient": coefs})
        .reindex(pd.Index(range(len(coefs))))
        .assign(feature=lr_result["features"], coefficient=coefs)
        .sort_values("coefficient", key=abs, ascending=False)
    )


# ---------------------------------------------------------------------------
# Facility 11 — statsmodels OLS (explain arrival delay)
# ---------------------------------------------------------------------------

def run_ols(df: pd.DataFrame,
            features: list[str] | None = None) -> dict:
    """
    OLS regression: log(1 + max(arr_delay+offset,0)) ~ flight characteristics.
    Returns dict with: results (RegressionResultsWrapper), X, y, features.
    """
    if features is None:
        features = _OLS_FEATURES

    # also include carrier dummies so we can compare airlines
    carrier_dummies = pd.get_dummies(
        df["op_unique_carrier"], prefix="carrier", drop_first=True, dtype=float
    )
    dummy_cols = carrier_dummies.columns.tolist()

    sub = _select_and_drop_na(df, features + ["arr_delay"]).copy()
    sub = sub.join(carrier_dummies, how="left").dropna()

    # log-transform the target — arr_delay can be negative (early arrivals)
    # shift so minimum is 1 before taking log
    offset = max(0, -sub["arr_delay"].min()) + 1
    sub["log_arr_delay"] = np.log(sub["arr_delay"] + offset)

    X = sm.add_constant(sub[features + dummy_cols])
    y = sub["log_arr_delay"]

    results = sm.OLS(y, X).fit()

    return {
        "results": results,
        "X": X,
        "y": y,
        "features": features + dummy_cols,
        "offset": offset,
    }


def plot_ols_coefficients(ols_result: dict, top_n: int = 15) -> plt.Figure:
    """Bar chart of the top-N OLS coefficients by absolute value."""
    params = ols_result["results"].params.drop("const", errors="ignore")
    pvalues = ols_result["results"].pvalues.drop("const", errors="ignore")

    coef_df = (
        pd.DataFrame({"feature": params.index, "coef": params.values,
                      "pvalue": pvalues.values})
        .sort_values("coef", key=abs, ascending=False)
        .head(top_n)
    )

    colors = ["#1976D2" if p < 0.05 else "#B0BEC5" for p in coef_df["pvalue"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(coef_df["feature"], coef_df["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("OLS Coefficient")
    ax.set_title(f"OLS Coefficients — Top {top_n} (blue = p < 0.05)")
    plt.tight_layout()
    return fig


def plot_ols_residuals(ols_result: dict, sample: int = 3000) -> plt.Figure:
    """Residual plot: fitted values vs residuals."""
    results = ols_result["results"]
    fitted = results.fittedvalues.sample(min(sample, len(results.fittedvalues)),
                                         random_state=42)
    resid = results.resid.loc[fitted.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, resid, alpha=0.2, s=6, color="#37474F")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("OLS Residual Plot")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from src.data_loader import load_clean
    import pathlib

    flights, _ = load_clean()
    out = pathlib.Path("outputs")
    out.mkdir(exist_ok=True)

    # --- clustering ---
    print("Running KMeans...")
    c_df, km = run_clustering(flights)
    print(cluster_summary(c_df).to_string(index=False))
    fig = plot_clusters(c_df)
    fig.savefig(out / "clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- logistic regression ---
    print("\nRunning Logistic Regression...")
    lr = run_logistic(flights)
    print(f"AUC: {lr['auc']:.3f}")
    print(pd.DataFrame(lr["report"]).T.round(3).to_string())
    for name, fig in [("roc_curve", plot_roc_curve(lr)),
                      ("confusion_matrix", plot_confusion_matrix(lr))]:
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- OLS ---
    print("\nRunning OLS...")
    ols = run_ols(flights)
    print(f"R²: {ols['results'].rsquared:.3f}")
    print(ols["results"].summary().tables[1])
    for name, fig in [("ols_coefficients", plot_ols_coefficients(ols)),
                      ("ols_residuals", plot_ols_residuals(ols))]:
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\nAll model outputs saved to outputs/")
