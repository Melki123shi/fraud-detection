import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.metrics import precision_recall_curve

from src.data_processing import DATA_PROCESSED
from src.modeling import get_best_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "notebooks" / "images"

SHAP_SAMPLE_SIZE = 1000


def _ensure_images_dir(dataset: str):
    path = IMAGES_DIR / f"explainability-{dataset}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_data(dataset: str):
    prefix = "fraud" if dataset == "fraud" else "credit"
    X_train = pd.read_csv(DATA_PROCESSED / f"X_train_{prefix}.csv")
    X_test = pd.read_csv(DATA_PROCESSED / f"X_test_{prefix}.csv")
    y_train = pd.read_csv(DATA_PROCESSED / f"y_train_{prefix}.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_PROCESSED / f"y_test_{prefix}.csv").iloc[:, 0]
    feature_cols = list(X_train.columns)
    return X_train, X_test, y_train, y_test, feature_cols


def _get_explainer(model, X_train):
    model_name = type(model).__name__
    if (
        "XGB" in model_name
        or "LGBM" in model_name
        or "RandomForest" in model_name
    ):
        return shap.TreeExplainer(model)
    elif "Logistic" in model_name:
        return shap.LinearExplainer(model, X_train)
    else:
        return shap.KernelExplainer(
            model.predict_proba,
            X_train[:100],
        )


def _compute_shap(model, X, explainer):
    shap_values = explainer.shap_values(np.array(X))
    # LightGBM TreeExplainer returns a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values, dtype=np.float64)
    # 3-D output: (n_samples, n_features, n_classes) — take positive class
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Built-in feature importance
# ---------------------------------------------------------------------------
def get_builtin_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not expose feature_importances_ or coef_")

    total = importances.sum()
    if total > 0:
        importances = importances / total

    df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return df


def plot_builtin_importance(
    model,
    feature_names: list,
    dataset: str,
    top_n: int = 15,
):
    save_dir = _ensure_images_dir(dataset)
    df = get_builtin_importance(model, feature_names).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(
        f"Top {top_n} Features -- Built-in Importance "
        f"({type(model).__name__}, {dataset})"
    )
    plt.tight_layout()
    path = save_dir / f"builtin_importance_top{top_n}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return df


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------
def generate_shap_summary(
    model,
    X_train,
    feature_names: list,
    dataset: str,
    explainer=None,
):
    save_dir = _ensure_images_dir(dataset)
    model_name = type(model).__name__

    if explainer is None:
        explainer = _get_explainer(model, X_train)

    n = min(SHAP_SAMPLE_SIZE, len(X_train))
    X_sample = X_train.sample(n=n, random_state=42)
    shap_values = _compute_shap(model, X_sample, explainer)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        np.array(X_sample),
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Summary Plot -- {model_name} ({dataset})")
    plt.tight_layout()
    path = save_dir / "shap_summary.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return shap_values, explainer, X_sample


def generate_shap_force_plots(
    model,
    X_test,
    y_test,
    feature_names: list,
    dataset: str,
    explainer=None,
):
    save_dir = _ensure_images_dir(dataset)
    model_name = type(model).__name__

    if explainer is None:
        X_dummy = pd.DataFrame(
            np.zeros((1, len(feature_names))),
            columns=feature_names,
        )
        explainer = _get_explainer(model, X_dummy)

    y_prob = model.predict_proba(X_test)[:, 1]
    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(
        y_test,
        y_prob,
    )
    f1_scores = (
        2
        * (precision_arr[:-1] * recall_arr[:-1])
        / (precision_arr[:-1] + recall_arr[:-1] + 1e-12)
    )
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds_pr[best_idx])
    y_pred = (y_prob >= optimal_threshold).astype(int)

    tp_idx = np.where((y_pred == 1) & (y_test.values == 1))[0]
    fp_idx = np.where((y_pred == 1) & (y_test.values == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_test.values == 1))[0]

    # If no FP at optimal threshold, lower threshold to find one
    if len(fp_idx) == 0:
        for fallback in np.arange(optimal_threshold - 0.05, 0.0, -0.05):
            fp_idx = np.where(
                ((y_prob >= fallback) & (y_test.values == 0))
            )[0]
            if len(fp_idx) > 0:
                break

    # Deduplicated list of sample positions needed
    seen = set()
    needed_indices = []
    for idx_list in [tp_idx, fp_idx, fn_idx]:
        if len(idx_list) > 0:
            i = idx_list[0]
            if i not in seen:
                seen.add(i)
                needed_indices.append(i)

    if not needed_indices:
        print("No TP, FP, or FN samples found.")
        return {}

    if hasattr(X_test, "iloc"):
        X_subset = X_test.iloc[needed_indices]
    else:
        X_subset = X_test[needed_indices]
    shap_values = _compute_shap(model, X_subset, explainer)

    plots = {}
    for label, idx_list in [("TP", tp_idx), ("FP", fp_idx), ("FN", fn_idx)]:
        if len(idx_list) == 0:
            print(f"No {label} found in test set at threshold {optimal_threshold:.4f}.")
            continue
        idx = idx_list[0]
        local_pos = needed_indices.index(idx)
        x = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]
        shap_val = np.array(shap_values[local_pos, :])

        plt.figure()
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = expected[1] if len(expected) == 2 else expected[0]
        shap.force_plot(
            expected,
            shap_val,
            x,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        plt.title(
            f"SHAP Force Plot -- {label} ({model_name}, {dataset})\n"
            f"Threshold: {optimal_threshold:.4f}"
        )
        path = save_dir / f"shap_force_{label}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
        if hasattr(y_test, "iloc"):
            true_val = int(y_test.iloc[idx])
        else:
            true_val = int(y_test[idx])
        plots[label] = {
            "index": int(idx),
            "true": true_val,
            "pred": int(y_pred[idx]),
            "threshold": round(optimal_threshold, 4),
            "probability": round(float(y_prob[idx]), 4),
            "path": str(path),
        }

    return plots


def compare_importance_sources(
    model,
    shap_values,
    X_sample,
    feature_names: list,
    dataset: str,
    top_n: int = 10,
):
    save_dir = _ensure_images_dir(dataset)

    builtin = get_builtin_importance(model, feature_names).head(top_n)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "shap_importance": mean_abs_shap,
            }
        )
        .sort_values("shap_importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    comparison = builtin.merge(
        shap_df,
        on="feature",
        how="outer",
    ).fillna(0)
    comparison["builtin_rank"] = (
        comparison["importance"].rank(ascending=False).astype(int)
    )
    comparison["shap_rank"] = (
        comparison["shap_importance"].rank(ascending=False).astype(int)
    )
    comparison = comparison.sort_values("importance", ascending=False).reset_index(
        drop=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison))
    width = 0.35
    ax.bar(
        x - width / 2,
        comparison["importance"],
        width,
        label="Built-in",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        comparison["shap_importance"],
        width,
        label="SHAP mean |SHAP|",
        color="coral",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["feature"], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(
        "Feature Importance: Built-in vs SHAP"
    )
    ax.legend()
    plt.tight_layout()
    path = save_dir / "importance_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return comparison


def run_full_explainability(dataset="fraud"):
    _ensure_images_dir(dataset)
    model = get_best_model(dataset)
    X_train, X_test, y_train, y_test, feature_cols = _get_data(dataset)

    print(f"\n{'=' * 60}")
    print(f"EXPLAINABILITY -- {dataset.upper()} DATASET")
    print(f"{'=' * 60}")

    explainer = _get_explainer(model, X_train)

    builtin_df = plot_builtin_importance(
        model,
        feature_cols,
        dataset,
    )

    shap_values, explainer, X_sample = generate_shap_summary(
        model,
        X_train,
        feature_cols,
        dataset,
        explainer=explainer,
    )

    force_plots = generate_shap_force_plots(
        model,
        X_test,
        y_test,
        feature_cols,
        dataset,
        explainer=explainer,
    )

    comparison = compare_importance_sources(
        model,
        shap_values,
        X_sample,
        feature_cols,
        dataset,
    )

    return {
        "builtin_importance": builtin_df,
        "shap_summary_path": str(
            IMAGES_DIR / f"explainability-{dataset}" / "shap_summary.png"
        ),
        "force_plots": force_plots,
        "comparison": comparison,
    }
