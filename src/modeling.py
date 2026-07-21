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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "notebooks" / "images"


def _ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


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
def get_param_grid(model_name: str) -> Dict:
    '''
    This function returns a parameter grid for hyperparameter tuning
    based on the model name.

    Parameters:
    - model_name: The name of the model for which to get the parameter grid.

    Returns:
    A dictionary representing the parameter grid for the specified model.
    '''
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


def tune_hyperparameters(model_name: str, X_train, y_train, n_iter=20, cv=5, random_state=42) -> Dict:
    '''
    This function performs hyperparameter tuning using RandomizedSearchCV
    with StratifiedKFold cross-validation, optimizing for AUC-PR.

    Parameters:
    - model_name: The name of the model to tune.
    - X_train: The feature set for training.
    - y_train: The true labels for training.
    - n_iter: The number of parameter settings sampled (default is 20).
    - cv: The number of cross-validation folds (default is 5).
    - random_state: The random seed for reproducibility (default is 42).

    Returns:
    A dictionary containing the best parameters, best score, and the 
    best estimator.
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

    if not param_grid:
        return {
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
        verbose=1,
        random_state=random_state,
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    return {
        "best_params": search.best_params_,
        "best_score": round(float(search.best_score_), 4),
        "best_estimator": search.best_estimator_,
        "tuned": True,
        "cv_results": search.cv_results_,
    }


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
        print(f"Hyperparameter tuning: ENABLED (n_iter={n_iter})")
    print()

    for name, model in models.items():
        print(f"Training {name}...")

        if do_tune:
            tune_result = tune_hyperparameters(name, train_X, train_y, n_iter=n_iter)
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
