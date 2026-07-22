import numpy as np
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score,
    confusion_matrix, precision_score, recall_score,
    classification_report, average_precision_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_processing import (
    process_fraud_data, process_creditcard_data,
    prepare_modeling_data_fraud, prepare_modeling_data_credit,
    DATA_PROCESSED
)

COST_FP = 1.0
COST_FN = 10.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "notebooks" / "images"


def _ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (IMAGES_DIR / "modeling").mkdir(parents=True, exist_ok=True)


def _get_fraud_data():
    '''
    This function loads and processes the fraud dataset, 
    returning the training and test sets, as well as 
    the SMOTE-resampled training set 
    and feature columns.
    '''
    fraud_df, _, _ = process_fraud_data()
    return prepare_modeling_data_fraud(fraud_df)


def _get_credit_data():
    '''
    This function loads and processes the credit card dataset, 
    returning the training and test sets, as well as the 
    SMOTE-resampled training set and feature columns.
    '''
    credit_df = process_creditcard_data()
    return prepare_modeling_data_credit(credit_df)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def get_logistic_regression():
    '''
    This function returns a logistic regression model 
    with predefined parameters.
    '''
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def get_random_forest():
    '''
    This function returns a random forest model 
    with predefined parameters.
    '''
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def get_xgboost():
    '''
    This function returns an XGBoost model
    with predefined parameters.'''
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )


def get_lightgbm():
    '''
    This function returns a LightGBM model
    with predefined parameters.
    '''
    return LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
def get_param_grid(model_name: str, fast: bool = False) -> Dict:
    '''
    This function returns a parameter grid for hyperparameter tuning
    based on the model name. Use fast=True for quicker tuning runs.

    Parameters:
    - model_name: The name of the model for which to get the parameter grid.
    - fast: If True, return a smaller grid for faster tuning.

    Returns:
    A dictionary representing the parameter grid for the specified model.
    '''
    if fast:
        return {
            "LogisticRegression": {
                "C": [0.01, 1.0, 100.0],
                "solver": ["lbfgs", "saga"],
            },
            "RandomForest": {
                "n_estimators": [100, 200],
                "max_depth": [8, 12],
                "min_samples_split": [10],
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
            },
            "LightGBM": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
            },
        }.get(model_name, {})

    param_grids = {
        "LogisticRegression": {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["saga"],
            "l1_ratio": [0.3, 0.5, 0.7],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [8, 12, 16, None],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", "log2", None],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "gamma": [0, 0.1, 0.2],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
        },
    }
    return param_grids.get(model_name, {})


def tune_hyperparameters(
    model_name: str, X_train, y_train,
    X_test=None, y_test=None,
    n_iter=20, cv=5, random_state=42,
    sample_size=None,
) -> Dict:
    '''
    This function performs hyperparameter tuning using RandomizedSearchCV
    with StratifiedKFold cross-validation, optimizing for AUC-PR.
    When X_test and y_test are provided, the best estimator is evaluated
    on the untouched test set with a full post-tuning report.

    Parameters:
    - model_name: The name of the model to tune.
    - X_train: The feature set for training.
    - y_train: The true labels for training.
    - X_test: The held-out test feature set (optional).
    - y_test: The held-out test labels (optional).
    - n_iter: The number of parameter settings sampled (default is 20).
    - cv: The number of cross-validation folds (default is 5).
    - random_state: The random seed for reproducibility (default is 42).
    - sample_size: Optional stratified sample size for faster tuning.

    Returns:
    A dictionary containing the best parameters, best score, the 
    best estimator, and (when test data is supplied) the test-set
    evaluation results.
    '''
    model_factories = {
        "LogisticRegression": get_logistic_regression,
        "RandomForest": get_random_forest,
        "XGBoost": get_xgboost,
        "LightGBM": get_lightgbm,
    }

    if model_name not in model_factories:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_factories[model_name]()
    param_grid = get_param_grid(model_name)

    X_tune, y_tune = X_train, y_train
    if sample_size is not None and len(X_train) > sample_size:
        from sklearn.utils import resample
        X_tune, y_tune = resample(
            X_train, y_train,
            n_samples=sample_size,
            stratify=y_train,
            random_state=random_state,
        )
        print(f"  Using stratified sample: {len(X_tune):,} rows")

    result = {
        "best_params": model.get_params(),
        "best_score": None,
        "best_estimator": model,
        "tuned": False,
    }

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring="average_precision",
        n_jobs=-1,
        verbose=0,
        random_state=random_state,
        return_train_score=False,
    )

    print(f"Fitting {model_name} with hyperparameter grid...")
    search.fit(X_tune, y_tune)

    result = {
        "best_params": search.best_params_,
        "best_score": round(float(search.best_score_), 4),
        "best_estimator": search.best_estimator_,
        "tuned": True,
        "cv_results": search.cv_results_,
    }

    if X_test is not None and y_test is not None:
        result["test_evaluation"] = _post_tuning_evaluation(
            search.best_estimator_, X_test, y_test, model_name,
        )

    return result


# ---------------------------------------------------------------------------
# Post-tuning test-set evaluation
# ---------------------------------------------------------------------------
def _post_tuning_evaluation(model, X_test, y_test, model_name: str) -> Dict:
    '''
    Evaluate the best estimator on the untouched test set.

    Produces:
    1. Precision-Recall curve + AUC-PR
    2. F1 at the optimal (max-F1) threshold
    3. Confusion matrix + cost analysis

    The PR curve plot is saved to the images directory.

    Parameters:
    - model: The fitted best estimator.
    - X_test: Held-out test features.
    - y_test: Held-out test labels.
    - model_name: Name of the model (used for the saved plot).

    Returns:
    A dictionary with all evaluation artefacts.
    '''
    _ensure_dirs()

    y_prob = model.predict_proba(X_test)[:, 1]

    # --- 1. Precision-Recall curve + AUC-PR ---
    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(y_test, y_prob)
    auc_pr = round(float(auc(recall_arr, precision_arr)), 4)
    avg_precision = round(float(average_precision_score(y_test, y_prob)), 4)

    _save_pr_curve(precision_arr, recall_arr, auc_pr, model_name)

    # --- 2. F1 at optimal threshold ---
    f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (
        precision_arr[:-1] + recall_arr[:-1] + 1e-12
    )
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = round(float(thresholds_pr[best_idx]), 4)
    f1_at_optimal = round(float(f1_scores[best_idx]), 4)
    precision_at_optimal = round(float(precision_arr[best_idx]), 4)
    recall_at_optimal = round(float(recall_arr[best_idx]), 4)

    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

    # --- 3. Confusion matrix + cost analysis ---
    cm = confusion_matrix(y_test, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()

    total_cost = fp * COST_FP + fn * COST_FN
    cost_per_sample = round(total_cost / len(y_test), 4)

    cm_dict = {
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }
    cost_dict = {
        "cost_fp": COST_FP,
        "cost_fn": COST_FN,
        "total_cost": round(float(total_cost), 4),
        "cost_per_sample": cost_per_sample,
        "false_positive_cost": round(float(fp * COST_FP), 4),
        "false_negative_cost": round(float(fn * COST_FN), 4),
    }

    # --- Console output ---
    print(f"\n{'='*60}")
    print(f"POST-TUNING TEST-SET EVALUATION — {model_name}")
    print(f"{'='*60}")
    print(f"  AUC-PR:                  {auc_pr}")
    print(f"  Average Precision:       {avg_precision}")
    print(f"  Optimal Threshold:       {optimal_threshold}")
    print(f"  F1 @ Optimal Threshold:  {f1_at_optimal}")
    print(f"  Precision @ Optimal:     {precision_at_optimal}")
    print(f"  Recall @ Optimal:        {recall_at_optimal}")
    print(f"\n  Confusion Matrix (optimal threshold):")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"\n  Cost Analysis:")
    print(f"    FP cost (unit): {COST_FP}   |  total FP cost: {cost_dict['false_positive_cost']}")
    print(f"    FN cost (unit): {COST_FN}   |  total FN cost: {cost_dict['false_negative_cost']}")
    print(f"    Total cost: {cost_dict['total_cost']}  |  cost / sample: {cost_per_sample}")
    print(f"{'='*60}\n")

    return {
        "auc_pr": auc_pr,
        "average_precision": avg_precision,
        "optimal_threshold": optimal_threshold,
        "f1_at_optimal_threshold": f1_at_optimal,
        "precision_at_optimal": precision_at_optimal,
        "recall_at_optimal": recall_at_optimal,
        "confusion_matrix": cm_dict,
        "cost_analysis": cost_dict,
        "classification_report": classification_report(
            y_test, y_pred_optimal, output_dict=True,
        ),
    }


def _save_pr_curve(precision_arr, recall_arr, auc_pr: float, model_name: str):
    '''
    Plot and save the Precision-Recall curve for a single model.
    '''
    _ensure_dirs()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_arr, precision_arr, linewidth=2, label=f"PR curve (AUC-PR = {auc_pr})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name} (post-tuning)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = IMAGES_DIR / "modeling" / f"{model_name.lower()}_pr_curve_post_tuning.png"
    fig.savefig(path, dpi=150)
    print(f"PR curve saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> Dict:
    '''
    This function evaluates a trained model on the test set,
    returning various performance metrics.

    Parameters:
    - model: The trained model to evaluate.
    - X_test: The feature set for testing.
    - y_test: The true labels for the test set.

    Returns:
    A dictionary containing evaluation metrics such as AUC-PR,
    average precision, F1 score, precision, recall, accuracy,
    confusion matrix, and classification report.
    '''
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    avg_precision = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model": type(model).__name__,
        "auc_pr": round(float(auc_pr), 4),
        "average_precision": round(float(avg_precision), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "accuracy": round(float((tp + tn) / (tp + tn + fp + fn)), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def cross_validate_model(model, X, y, cv=5) -> Dict:
    '''
    This function performs cross-validation on a given model,
    returning the mean and standard deviation of various metrics.
    
    Parameters:
    - model: The model to cross-validate.
    - X: The feature set for cross-validation.
    - y: The true labels for the feature set.
    - cv: The number of cross-validation folds (default is 5).

    Returns:
    A dictionary containing the mean and standard deviation of
    AUC-PR, F1 score, precision, and recall 
    across the cross-validation folds.
    '''
    scoring = {
        "auc_pr": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=cv_splitter, scoring=scoring,
        return_train_score=False, n_jobs=-1
    )
    return {
        metric: {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
        }
        for metric, scores in cv_results.items()
        if metric in scoring
    }


# ---------------------------------------------------------------------------
# Training pipelines
# ---------------------------------------------------------------------------
def train_all_models(dataset="fraud", use_smote=True, do_tune=False, n_iter=20):
    '''
    This function trains multiple models on the specified dataset,
    optionally using SMOTE for resampling. It evaluates each model,
    performs cross-validation, and saves the results and best model.

    Parameters:
    - dataset: The dataset to use for training ("fraud" or "credit").
    - use_smote: Whether to use SMOTE for resampling
      the training data (default is True).
    - do_tune: Whether to perform hyperparameter tuning
      using RandomizedSearchCV (default is False).
    - n_iter: Number of parameter settings sampled for RandomizedSearchCV
      (default is 20, only used if do_tune is True).

    Returns:
    A dictionary containing the results of model evaluations,
    the trained models, the best model's name and instance,
    feature columns, and the training and test sets.
    '''
    _ensure_dirs()

    if dataset == "fraud":
        X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, feature_cols = _get_fraud_data()
        train_X, train_y = (X_train_smote, y_train_smote) if use_smote else (X_train, y_train)
        dataset_name = "fraud"
    else:
        X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, feature_cols = _get_credit_data()
        train_X, train_y = (X_train_smote, y_train_smote) if use_smote else (X_train, y_train)
        dataset_name = "credit"

    models = {
        "LogisticRegression": get_logistic_regression(),
        "RandomForest": get_random_forest(),
        "XGBoost": get_xgboost(),
        "LightGBM": get_lightgbm(),
    }

    results = []
    trained_models = {}

    print(f"\n{'='*60}")
    print(f"TRAINING MODELS — {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    print(f"Training on {'SMOTE-resampled' if use_smote else 'original'} data: {train_X.shape}")
    print(f"Test set (untouched): {X_test.shape}")
    if do_tune:
        print(f"Hyperparameter tuning: ENABLED (n_iter={n_iter}, cv=3, sample_size=50000)")
    print()

    for name, model in models.items():
        print(f"Training {name}...")

        if do_tune:
            effective_n_iter = min(n_iter, 10) if dataset == "credit" else n_iter
            sample_size = 50000 if dataset == "credit" else None
            tune_result = tune_hyperparameters(
                name, train_X, train_y,
                n_iter=effective_n_iter, cv=3,
                random_state=42,
                sample_size=sample_size,
            )
            if tune_result["tuned"]:
                model = tune_result["best_estimator"]
                print(f"  Tuned best params: {tune_result['best_params']}")
                print(f"  Tuned CV AUC-PR: {tune_result['best_score']}")
            else:
                model.fit(train_X, train_y)
        else:
            model.fit(train_X, train_y)

        # Evaluate on original (non-SMOTE) test set
        metrics = evaluate_model(model, X_test, y_test)
        cv_metrics = cross_validate_model(model, X_train, y_train, cv=5)

        result = {
            "name": name,
            "dataset": dataset_name,
            "use_smote": use_smote,
            "tuned": do_tune,
            "model": model,
            "evaluation": metrics,
            "cross_validation": cv_metrics,
        }
        results.append(result)
        trained_models[name] = model

        print(f"  AUC-PR:  {metrics['auc_pr']}")
        print(f"  F1:      {metrics['f1_score']}")
        print(f"  Prec:    {metrics['precision']}")
        print(f"  Recall:  {metrics['recall']}")
        print(f"  ConfMat: TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")
        print()

    # Save results (exclude model objects from JSON)
    results_path = DATA_PROCESSED / f"model_results_{dataset_name}.json"
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != "model"}
        json_results.append(jr)
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Save best model by F1
    best = max(results, key=lambda r: r["evaluation"]["f1_score"])
    best_name = best["name"]
    best_model = trained_models[best_name]
    model_path = MODELS_DIR / f"best_model_{dataset_name}.pkl"
    joblib.dump(best_model, model_path)
    print(f"Best model ({best_name}) saved to {model_path}")

    return {
        "results": results,
        "trained_models": trained_models,
        "best_model_name": best_name,
        "best_model": best_model,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
    }


def compare_models(results: List[Dict]) -> pd.DataFrame:
    '''
    This function takes a list of model evaluation results and returns a
    pandas DataFrame summarizing the performance metrics of each model.

    Parameters:
    - results: A list of dictionaries containing model evaluation results.

    Returns:
    A pandas DataFrame with columns for model name, dataset, AUC-PR,
    F1 score, precision, recall, accuracy, and cross-validation metrics.
    '''
    rows = []
    for r in results:
        eval_m = r["evaluation"]
        cv_m = r["cross_validation"]
        rows.append({
            "Model": r["name"],
            "Dataset": r["dataset"],
            "AUC-PR": eval_m["auc_pr"],
            "F1-Score": eval_m["f1_score"],
            "Precision": eval_m["precision"],
            "Recall": eval_m["recall"],
            "Accuracy": eval_m["accuracy"],
            "CV AUC-PR (mean)": cv_m.get("auc_pr", {}).get("mean", "-"),
            "CV F1 (mean)": cv_m.get("f1", {}).get("mean", "-"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("F1-Score", ascending=False).reset_index(drop=True)


def get_best_model(dataset="fraud"):
    '''
    This function loads the best saved model for the specified dataset.

    Parameters:
    - dataset: The dataset for which to load the best model 
    ("fraud" or "credit").

    Returns:
    The best trained model for the specified dataset.
    '''
    model_path = MODELS_DIR / f"best_model_{dataset}.pkl"
    if not model_path.exists():
        msg = (
            f"No saved model found at {model_path}. "
            "Run train_all_models first."
        )
        raise FileNotFoundError(msg)
    return joblib.load(model_path)


def get_feature_importance(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    '''
    This function extracts feature importance from a trained model
    and returns a DataFrame sorted by importance.

    Parameters:
    - model: The trained model (must have feature_importances_ or coef_).
    - feature_names: List of feature names.
    - top_n: Number of top features to return.

    Returns:
    A pandas DataFrame with columns 'feature' and 'importance',
    sorted by importance in descending order.
    '''
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
    else:
        raise ValueError(
            "Model does not expose feature_importances_ or coef_."
        )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def plot_feature_importance(
    model, feature_names: List[str], dataset: str,
    top_n: int = 20, save: bool = True
) -> pd.DataFrame:
    '''
    This function plots feature importance for a trained model
    and optionally saves the plot.

    Parameters:
    - model: The trained model.
    - feature_names: List of feature names.
    - dataset: Dataset name for the plot title.
    - top_n: Number of top features to plot.
    - save: Whether to save the plot to file.

    Returns:
    A pandas DataFrame with the top features and their importance.
    '''
    _ensure_dirs()
    df = get_feature_importance(model, feature_names, top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(
        f"Top {top_n} Feature Importances — "
        f"{type(model).__name__} ({dataset})"
    )
    plt.tight_layout()

    if save:
        path = IMAGES_DIR / f"modeling/{dataset}-feature-importance.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    plt.show()
    return df
