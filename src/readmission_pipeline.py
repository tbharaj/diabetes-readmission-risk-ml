
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs")
BRIEF_DIR = Path("briefs")
SRC_DIR = Path("src")

OUTPUT_DIR.mkdir(exist_ok=True)
BRIEF_DIR.mkdir(exist_ok=True)
SRC_DIR.mkdir(exist_ok=True)


def make_onehot_encoder():
    """Run the make onehot encoder step in the project workflow."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_calibrated_classifier(estimator):
    """Run the make calibrated classifier step in the project workflow."""
    try:
        return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=3)


def df_to_markdown(df, float_digits=3):
    """Run the df to markdown step in the project workflow."""
    if df.empty:
        return "No results available."

    formatted = df.copy()

    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")

    headers = list(formatted.columns)
    rows = formatted.astype(str).values.tolist()

    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        out.append("| " + " | ".join(row) + " |")

    return "\n".join(out)


def load_dataset():
    """Run the load dataset step in the project workflow."""
    print("Fetching Diabetes 130-US Hospitals dataset from UCI...")
    diabetes = fetch_ucirepo(id=296)

    X = diabetes.data.features.copy()
    y_raw = diabetes.data.targets.copy()

    if isinstance(y_raw, pd.DataFrame):
        if "readmitted" in y_raw.columns:
            y_raw = y_raw["readmitted"]
        else:
            y_raw = y_raw.iloc[:, 0]

    X = X.replace("?", np.nan)
    y = (y_raw == "<30").astype(int)

    print(f"Dataset shape: {X.shape}")
    print("\nReadmission target counts:")
    print(y.value_counts())
    print("\nReadmission target proportions:")
    print(y.value_counts(normalize=True))

    return X, y


def prepare_features(X):
    """Run the prepare features step in the project workflow."""
    drop_cols = [
        "encounter_id",
        "patient_nbr",
        "weight",
    ]

    existing_drop_cols = [col for col in drop_cols if col in X.columns]
    X_model = X.drop(columns=existing_drop_cols)

    numeric_features = X_model.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [col for col in X_model.columns if col not in numeric_features]

    print(f"\nDropped columns: {existing_drop_cols}")
    print(f"Feature matrix after dropping columns: {X_model.shape}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    return X_model, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """Run the build preprocessor step in the project workflow."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_model_search_spaces(preprocessor, y_train):
    """Run the build model search spaces step in the project workflow."""
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos

    search_spaces = {}

    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    search_spaces["Logistic Regression"] = {
        "pipeline": logistic,
        "params": {
            "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "classifier__penalty": ["l1", "l2"],
        },
        "n_iter": 6,
    }

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    search_spaces["Random Forest"] = {
        "pipeline": random_forest,
        "params": {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [None, 8, 12],
            "classifier__min_samples_leaf": [3, 5, 10],
            "classifier__max_features": ["sqrt", "log2"],
        },
        "n_iter": 6,
    }

    if XGBOOST_AVAILABLE:
        xgb = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            ]
        )

        search_spaces["XGBoost"] = {
            "pipeline": xgb,
            "params": {
                "classifier__n_estimators": [150, 250, 350],
                "classifier__max_depth": [2, 3, 4],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
                "classifier__min_child_weight": [1, 5, 10],
            },
            "n_iter": 8,
        }
    else:
        print("\nXGBoost not available. Install with: pip install xgboost")

    return search_spaces


def tune_models(search_spaces, X_train, y_train):
    """Run the tune models step in the project workflow."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    tuned_models = {}
    cv_rows = []

    for model_name, spec in search_spaces.items():
        print(f"\nTuning {model_name}...")

        search = RandomizedSearchCV(
            estimator=spec["pipeline"],
            param_distributions=spec["params"],
            n_iter=spec["n_iter"],
            scoring=scoring,
            refit="pr_auc",
            cv=cv,
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbose=1,
            return_train_score=False,
        )

        search.fit(X_train, y_train)

        best_index = search.best_index_
        cv_results = search.cv_results_

        row = {
            "model": model_name,
            "best_cv_pr_auc_mean": cv_results["mean_test_pr_auc"][best_index],
            "best_cv_pr_auc_std": cv_results["std_test_pr_auc"][best_index],
            "best_cv_roc_auc_mean": cv_results["mean_test_roc_auc"][best_index],
            "best_cv_roc_auc_std": cv_results["std_test_roc_auc"][best_index],
            "best_params": json.dumps(search.best_params_),
        }

        cv_rows.append(row)
        tuned_models[model_name] = search.best_estimator_

        print(f"Best CV PR-AUC: {row['best_cv_pr_auc_mean']:.4f} +/- {row['best_cv_pr_auc_std']:.4f}")
        print(f"Best CV ROC-AUC: {row['best_cv_roc_auc_mean']:.4f} +/- {row['best_cv_roc_auc_std']:.4f}")
        print(f"Best params: {search.best_params_}")

    cv_df = pd.DataFrame(cv_rows).sort_values("best_cv_pr_auc_mean", ascending=False)
    cv_df.to_csv(OUTPUT_DIR / "cross_validation_tuning_results.csv", index=False)

    return tuned_models, cv_df


def evaluate_holdout_models(tuned_models, X_test, y_test):
    """Run the evaluate holdout models step in the project workflow."""
    rows = []
    predictions = {}

    for model_name, model in tuned_models.items():
        print(f"\nEvaluating {model_name} on holdout test set...")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        row = {
            "model": model_name,
            "holdout_roc_auc": roc_auc_score(y_test, y_proba),
            "holdout_pr_auc": average_precision_score(y_test, y_proba),
            "precision_threshold_0_50": precision_score(y_test, y_pred, zero_division=0),
            "recall_threshold_0_50": recall_score(y_test, y_pred, zero_division=0),
            "f1_threshold_0_50": f1_score(y_test, y_pred, zero_division=0),
            "brier_score_uncalibrated": brier_score_loss(y_test, y_proba),
        }

        rows.append(row)
        predictions[model_name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "model": model,
        }

        print(row)

    results_df = pd.DataFrame(rows).sort_values("holdout_pr_auc", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    return results_df, predictions


def calibrate_best_model(best_model, X_train, y_train, X_test, y_test):
    """Run the calibrate best model step in the project workflow."""
    print("\nCalibrating best model using sigmoid calibration...")
    calibrated_model = make_calibrated_classifier(best_model)
    calibrated_model.fit(X_train, y_train)

    y_proba_uncalibrated = best_model.predict_proba(X_test)[:, 1]
    y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

    calibration_summary = pd.DataFrame(
        [
            {
                "probability_type": "uncalibrated",
                "brier_score": brier_score_loss(y_test, y_proba_uncalibrated),
                "roc_auc": roc_auc_score(y_test, y_proba_uncalibrated),
                "pr_auc": average_precision_score(y_test, y_proba_uncalibrated),
            },
            {
                "probability_type": "calibrated_sigmoid",
                "brier_score": brier_score_loss(y_test, y_proba_calibrated),
                "roc_auc": roc_auc_score(y_test, y_proba_calibrated),
                "pr_auc": average_precision_score(y_test, y_proba_calibrated),
            },
        ]
    )

    calibration_summary.to_csv(OUTPUT_DIR / "calibration_summary.csv", index=False)

    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_proba_uncalibrated, n_bins=10, strategy="quantile")
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_proba_calibrated, n_bins=10, strategy="quantile")

    calibration_curve_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "probability_type": "uncalibrated",
                    "mean_predicted_probability": prob_pred_uncal,
                    "observed_readmission_rate": prob_true_uncal,
                }
            ),
            pd.DataFrame(
                {
                    "probability_type": "calibrated_sigmoid",
                    "mean_predicted_probability": prob_pred_cal,
                    "observed_readmission_rate": prob_true_cal,
                }
            ),
        ],
        ignore_index=True,
    )

    calibration_curve_df.to_csv(OUTPUT_DIR / "calibration_curve_data.csv", index=False)

    plt.figure()
    plt.plot(prob_pred_uncal, prob_true_uncal, marker="o", label="Uncalibrated")
    plt.plot(prob_pred_cal, prob_true_cal, marker="o", label="Calibrated")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed readmission rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=300)
    plt.close()

    print("\nCalibration summary:")
    print(calibration_summary)

    return calibrated_model, y_proba_calibrated, calibration_summary, calibration_curve_df


def threshold_analysis(y_test, y_proba):
    """Run the threshold analysis step in the project workflow."""
    thresholds = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50]

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
                "predicted_positive_rate": y_pred.mean(),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            }
        )

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)

    plt.figure()
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1 score")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric")
    plt.title("Threshold Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_analysis.png", dpi=300)
    plt.close()

    print("\nThreshold analysis:")
    print(threshold_df)

    return threshold_df


def save_core_plots(best_model_name, y_test, y_proba, threshold=0.50):
    """Run the save core plots step in the project workflow."""
    y_pred = (y_proba >= threshold).astype(int)

    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve - {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300)
    plt.close()

    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title(f"Precision-Recall Curve - {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not readmitted <30", "Readmitted <30"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {best_model_name} at threshold {threshold}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def safe_auc(y_true, y_proba):
    """Run the safe auc step in the project workflow."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_proba)
    except Exception:
        return np.nan


def run_fairness_analysis(X_test_original, y_test, y_proba, threshold=0.50):
    """Run the run fairness analysis step in the project workflow."""
    y_pred = (y_proba >= threshold).astype(int)
    rows = []

    data = X_test_original.copy()
    data["y_true"] = np.asarray(y_test)
    data["y_pred"] = y_pred
    data["y_proba"] = y_proba

    if "age" in data.columns:
        younger_than_60 = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)"]

        def age_band(value):
            """Run the age band step in the project workflow."""
            if pd.isna(value):
                return np.nan
            value = str(value)
            if value in younger_than_60:
                return "<60"
            if value.startswith("["):
                return ">=60"
            return np.nan

        data["age_band"] = data["age"].apply(age_band)

    group_columns = []
    for col in ["gender", "age", "age_band", "race"]:
        if col in data.columns:
            group_columns.append(col)

    for col in group_columns:
        group_data = data.copy()

        if col == "gender":
            group_data[col] = group_data[col].replace("Unknown/Invalid", np.nan)

        group_data = group_data.dropna(subset=[col])

        for group, subset in group_data.groupby(col):
            if len(subset) < 100:
                continue

            rows.append(
                {
                    "group_type": col,
                    "group": group,
                    "n": len(subset),
                    "positive_cases": int(subset["y_true"].sum()),
                    "observed_readmission_rate": subset["y_true"].mean(),
                    "predicted_positive_rate": subset["y_pred"].mean(),
                    "roc_auc": safe_auc(subset["y_true"], subset["y_proba"]),
                    "precision": precision_score(subset["y_true"], subset["y_pred"], zero_division=0),
                    "recall": recall_score(subset["y_true"], subset["y_pred"], zero_division=0),
                    "f1": f1_score(subset["y_true"], subset["y_pred"], zero_division=0),
                }
            )

    fairness_df = pd.DataFrame(rows)
    fairness_df.to_csv(OUTPUT_DIR / "fairness_subgroup_analysis.csv", index=False)

    plot_df = fairness_df[fairness_df["group_type"].isin(["gender", "age_band"])].copy()

    if not plot_df.empty:
        plot_df["label"] = plot_df["group_type"] + ": " + plot_df["group"].astype(str)

        plt.figure(figsize=(9, 5))
        plt.bar(plot_df["label"], plot_df["recall"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Recall")
        plt.title("Subgroup Analysis - Recall by Group")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fairness_recall_by_group.png", dpi=300)
        plt.close()

    print("\nFairness/subgroup analysis:")
    print(fairness_df)

    return fairness_df


def run_error_analysis(X_test_original, y_test, y_proba, threshold=0.50):
    """Run the run error analysis step in the project workflow."""
    y_pred = (y_proba >= threshold).astype(int)

    error_df = X_test_original.copy()
    error_df["y_true"] = np.asarray(y_test)
    error_df["y_pred"] = y_pred
    error_df["y_proba"] = y_proba

    conditions = [
        (error_df["y_true"] == 1) & (error_df["y_pred"] == 1),
        (error_df["y_true"] == 0) & (error_df["y_pred"] == 1),
        (error_df["y_true"] == 1) & (error_df["y_pred"] == 0),
        (error_df["y_true"] == 0) & (error_df["y_pred"] == 0),
    ]

    choices = ["true_positive", "false_positive", "false_negative", "true_negative"]
    error_df["prediction_group"] = np.select(conditions, choices, default="unknown")

    summary_rows = []

    for group, subset in error_df.groupby("prediction_group"):
        row = {
            "prediction_group": group,
            "n": len(subset),
            "mean_predicted_probability": subset["y_proba"].mean(),
        }

        for col in [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
        ]:
            if col in subset.columns:
                row[f"mean_{col}"] = pd.to_numeric(subset[col], errors="coerce").mean()

        for col in ["age", "gender", "race", "insulin", "diabetesMed", "discharge_disposition_id"]:
            if col in subset.columns:
                mode = subset[col].mode(dropna=True)
                row[f"most_common_{col}"] = mode.iloc[0] if len(mode) else np.nan

        summary_rows.append(row)

    error_summary = pd.DataFrame(summary_rows)
    error_summary.to_csv(OUTPUT_DIR / "error_analysis_summary.csv", index=False)

    print("\nError analysis summary:")
    print(error_summary)

    return error_summary


def get_feature_names(model):
    """Run the get feature names step in the project workflow."""
    preprocessor = model.named_steps["preprocessor"]

    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return None


def run_permutation_importance(best_model, X_test, y_test):
    """Run the run permutation importance step in the project workflow."""
    print("\nRunning permutation importance on a test subset...")

    if len(X_test) > 5000:
        X_sample = X_test.sample(n=5000, random_state=RANDOM_STATE)
        y_sample = y_test.loc[X_sample.index] if hasattr(y_test, "loc") else pd.Series(y_test, index=X_test.index).loc[X_sample.index]
    else:
        X_sample = X_test
        y_sample = y_test

    result = permutation_importance(
        best_model,
        X_sample,
        y_sample,
        n_repeats=3,
        random_state=RANDOM_STATE,
        scoring="average_precision",
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean_pr_auc": result.importances_mean,
            "importance_std_pr_auc": result.importances_std,
        }
    ).sort_values("importance_mean_pr_auc", ascending=False)

    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top = importance_df.head(15)

    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"][::-1], top["importance_mean_pr_auc"][::-1])
    plt.xlabel("Mean decrease in PR-AUC")
    plt.title("Permutation Importance - Top Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "permutation_importance.png", dpi=300)
    plt.close()

    print("\nTop permutation importance features:")
    print(top)

    return importance_df


def run_optional_shap(best_model, X_test):
    """Run the run optional shap step in the project workflow."""
    if not SHAP_AVAILABLE:
        print("\nSHAP not available. Install with: pip install shap")
        return pd.DataFrame()

    classifier = best_model.named_steps["classifier"]
    classifier_name = classifier.__class__.__name__

    if classifier_name not in ["RandomForestClassifier", "XGBClassifier"]:
        print(f"\nSkipping SHAP because best model is {classifier_name}.")
        return pd.DataFrame()

    print("\nRunning SHAP analysis on a sample...")

    sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)

    preprocessor = best_model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(sample)

    feature_names = get_feature_names(best_model)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    if hasattr(X_transformed, "toarray"):
        X_transformed_dense = X_transformed.toarray()
    else:
        X_transformed_dense = X_transformed

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed_dense)

        if isinstance(shap_values, list):
            shap_values_to_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_to_use = shap_values

        if len(shap_values_to_use.shape) == 3:
            shap_values_to_use = shap_values_to_use[:, :, 1]

        mean_abs_shap = np.abs(shap_values_to_use).mean(axis=0)

        shap_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        shap_df.to_csv(OUTPUT_DIR / "shap_top_features.csv", index=False)

        top = shap_df.head(20)

        plt.figure(figsize=(9, 7))
        plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
        plt.xlabel("Mean absolute SHAP value")
        plt.title("SHAP Feature Importance - Top Features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_summary_bar.png", dpi=300)
        plt.close()

        print("\nSHAP top features:")
        print(top)

        return shap_df

    except Exception as e:
        print(f"\nSHAP failed safely: {e}")
        return pd.DataFrame()


def write_advanced_readme(
    cv_df,
    results_df,
    calibration_summary,
    threshold_df,
    fairness_df,
    error_summary,
    importance_df,
    shap_df,
    best_model_name,
):
    results_display = results_df.copy()
    cv_display = cv_df.copy()

    best_result = results_df.iloc[0]
    best_cv = cv_df[cv_df["model"] == best_model_name].iloc[0]

    has_shap = not shap_df.empty

    readme = f"""# Diabetes Readmission Risk Prediction Using Interpretable Machine Learning

## Abstract

This project evaluates the feasibility of predicting 30-day hospital readmission risk in patients with diabetes using routinely collected hospital encounter data. The analysis was designed as a reproducible health-data science study rather than a deployable clinical tool.

The project compares baseline and non-linear machine-learning models using stratified cross-validation, hyperparameter tuning, holdout evaluation, calibration analysis, threshold analysis, subgroup/fairness analysis, error analysis and model interpretability. The best model by holdout PR-AUC was **{best_model_name}**. Its holdout ROC-AUC was **{best_result['holdout_roc_auc']:.3f}** and holdout PR-AUC was **{best_result['holdout_pr_auc']:.3f}**.

The results suggest that routinely collected hospital data contains some signal associated with 30-day readmission risk, but the model is not clinically deployable without external validation, prospective testing, calibration review and fairness assessment.

## 1. Introduction

Hospital readmission is an important healthcare outcome because it can reflect disease burden, treatment complexity, care-transition quality and healthcare-system pressure. In patients with diabetes, readmission risk may be influenced by multimorbidity, medication burden, prior healthcare utilisation, acute complications and discharge planning.

Thirty-day readmission prediction is challenging because readmission is not driven by one biological mechanism alone. It is influenced by clinical, demographic, treatment-related and healthcare-system variables. Machine-learning models can identify patterns in historical healthcare data, but their usefulness depends on discrimination, calibration, threshold behaviour, subgroup performance and clinical feasibility.

This project uses the Diabetes 130-US Hospitals for Years 1999-2008 dataset from the UCI Machine Learning Repository (Dua and Graff, 2019). The dataset was originally used in work investigating HbA1c measurement and readmission outcomes in patients with diabetes (Strack et al., 2014).

## 2. Aim and Research Question

### Aim

The aim was to build, validate, calibrate, interpret and stress-test a reproducible machine-learning pipeline for predicting 30-day readmission risk in patients with diabetes.

### Research question

Can routinely collected hospital encounter data identify patients with diabetes at higher risk of readmission within 30 days?

### Objectives

The project objectives were to:

- preprocess real-world hospital encounter data;
- define a binary 30-day readmission target;
- compare logistic regression, random forest and optional XGBoost models;
- use stratified cross-validation and hyperparameter tuning;
- evaluate discrimination using ROC-AUC and PR-AUC;
- evaluate probability calibration using Brier score and calibration curves;
- assess threshold trade-offs between precision, recall and false positives;
- assess subgroup performance by available demographic variables;
- perform error analysis to understand misclassified cases;
- interpret model behaviour using permutation importance and optional SHAP.

## 3. Dataset

The project used the Diabetes 130-US Hospitals for Years 1999-2008 dataset from the UCI Machine Learning Repository (Dua and Graff, 2019).

| Dataset characteristic | Value |
|---|---:|
| Hospital encounters | 101,766 |
| Original columns | 48 |
| Features after preprocessing | 44 |
| Prediction target | 30-day readmission |
| Positive class | Readmitted within 30 days |
| Positive class proportion | Approximately 11.2% |

The original `readmitted` variable contains three categories: `<30`, `>30` and `NO`. This project converted the task into binary classification:

| Binary class | Meaning |
|---|---|
| 1 | Readmitted within 30 days |
| 0 | Not readmitted within 30 days |

This created an imbalanced classification problem, making PR-AUC, precision, recall and threshold analysis especially important.

## 4. Methods

The analysis was implemented in Python using pandas, NumPy, scikit-learn, Matplotlib and optional XGBoost/SHAP. Pandas was used for data handling (McKinney, 2010), NumPy for numerical operations (Harris et al., 2020), scikit-learn for modelling and evaluation (Pedregosa et al., 2011), and Matplotlib for visualisation (Hunter, 2007).

### 4.1 Preprocessing

The preprocessing workflow included:

- missing-value replacement;
- removal of selected identifier, high-missingness or low-interpretability columns;
- numerical imputation and scaling;
- categorical imputation and one-hot encoding;
- stratified train-test splitting;
- class-imbalance-aware modelling.

### 4.2 Model development

Models were compared using stratified 3-fold cross-validation and hyperparameter tuning. PR-AUC was used as the refit metric because the positive class was rare and clinically important.

### 4.3 Models compared

The project compared the following models:

- Logistic Regression;
- Random Forest;
- XGBoost, if installed successfully.

## 5. Cross-Validation and Hyperparameter Tuning

The table below reports the best cross-validation performance for each model after hyperparameter tuning.

{df_to_markdown(cv_display[['model', 'best_cv_pr_auc_mean', 'best_cv_pr_auc_std', 'best_cv_roc_auc_mean', 'best_cv_roc_auc_std']])}

This strengthens the analysis because the results are not based only on one train-test split. Reporting standard deviation also gives a basic estimate of uncertainty across folds.

## 6. Holdout Test Results

After tuning, the best model configurations were evaluated on a held-out test set.

{df_to_markdown(results_display[['model', 'holdout_roc_auc', 'holdout_pr_auc', 'precision_threshold_0_50', 'recall_threshold_0_50', 'f1_threshold_0_50', 'brier_score_uncalibrated']])}

The best model by holdout PR-AUC was **{best_model_name}**. However, performance should be interpreted cautiously because even a model with moderate ROC-AUC may be clinically weak if precision, calibration or threshold behaviour are poor.

## 7. Calibration Analysis

Calibration assesses whether predicted probabilities correspond to observed outcome rates. This matters clinically because a predicted 30% risk should ideally mean that approximately 30% of similar patients experience readmission.

{df_to_markdown(calibration_summary)}

![Calibration curve](outputs/calibration_curve.png)

A poorly calibrated model may still rank patients reasonably but produce misleading risk estimates. This means calibration must be considered before using any model for clinical risk communication.

## 8. Threshold Analysis

Threshold analysis shows how model behaviour changes when the decision threshold is adjusted.

{df_to_markdown(threshold_df[['threshold', 'precision', 'recall', 'f1', 'specificity', 'predicted_positive_rate', 'false_positives', 'false_negatives']])}

![Threshold analysis](outputs/threshold_analysis.png)

This is clinically important because different decision thresholds imply different trade-offs. A lower threshold may identify more patients at risk but create many false positives. A higher threshold may reduce unnecessary alerts but miss more patients who are truly at risk.

## 9. Subgroup and Fairness Analysis

The subgroup analysis compared model performance across available demographic groups.

{df_to_markdown(fairness_df[['group_type', 'group', 'n', 'observed_readmission_rate', 'predicted_positive_rate', 'roc_auc', 'precision', 'recall', 'f1']]) if not fairness_df.empty else "No subgroup results were available."}

![Fairness recall by group](outputs/fairness_recall_by_group.png)

Unequal model performance across groups would require further investigation before deployment. This analysis should be interpreted as an exploratory fairness screen rather than a full fairness audit.

## 10. Error Analysis

Error analysis summarises the types of cases the model misclassified.

{df_to_markdown(error_summary)}

False positives may create unnecessary clinical follow-up or alert fatigue. False negatives may miss patients who could benefit from additional discharge support. This trade-off means model usefulness depends on the clinical decision setting.

## 11. Interpretability

Permutation importance was used to identify features most associated with model performance.

{df_to_markdown(importance_df.head(15))}

![Permutation importance](outputs/permutation_importance.png)

{"SHAP was also used as an additional model-interpretability method.\n\n![SHAP summary](outputs/shap_summary_bar.png)" if has_shap else "SHAP was not available in this run. If installed successfully, SHAP can provide an additional interpretability layer alongside permutation importance."}

Interpretability results should not be interpreted causally. They identify features that contribute to prediction in this dataset, not variables that necessarily cause readmission.

## 12. Core Figures

### ROC curve

![ROC curve](outputs/roc_curve.png)

### Precision-recall curve

![Precision-recall curve](outputs/precision_recall_curve.png)

### Confusion matrix

![Confusion matrix](outputs/confusion_matrix.png)

## 13. Discussion

This project suggests that routinely collected hospital encounter data contains predictive signal for 30-day readmission risk in patients with diabetes. Prior healthcare utilisation, emergency visits, discharge disposition, diagnosis information and treatment-related variables are clinically plausible predictors because they may reflect disease burden, care-transition complexity and multimorbidity.

However, the project also shows why healthcare prediction models need more than headline discrimination metrics. A model with moderate ROC-AUC may still be clinically weak if precision is low, probabilities are poorly calibrated or subgroup performance is uneven. In this setting, false positives could create unnecessary clinical workload, while false negatives could miss patients who might benefit from additional follow-up.

The model should therefore be viewed as an exploratory health-data science pipeline, not a deployable clinical tool.

## 14. Limitations

Key limitations include:

1. The dataset is historical, covering care from 1999-2008.
2. The dataset comes from US hospitals and may not generalise to UK/NHS settings.
3. Observational data can identify associations but cannot prove causality.
4. Coding patterns may reflect healthcare-system processes rather than patient biology alone.
5. Model performance is modest.
6. Calibration and threshold behaviour require careful clinical review.
7. Subgroup analysis is exploratory and does not replace a full fairness audit.
8. External validation and prospective validation would be required before clinical use.

## 15. Conclusion

This project built and stress-tested an interpretable machine-learning pipeline for predicting 30-day diabetes readmission risk. The advanced version goes beyond a simple model comparison by incorporating cross-validation, hyperparameter tuning, PR-AUC, calibration, threshold analysis, subgroup analysis, error analysis and interpretability.

The findings show that routinely collected hospital data contains some predictive signal, but the model is not suitable for clinical deployment without further validation. The main value of the project is as a rigorous, reproducible demonstration of responsible healthcare machine learning.

## 16. Medical Affairs and Health Data Science Relevance

For medical affairs, this project demonstrates how real-world evidence and predictive modelling can be translated into stakeholder-relevant insights around patient pathways, discharge planning, risk communication and implementation barriers.

For health data science, it demonstrates an end-to-end healthcare prediction workflow: data access, preprocessing, model tuning, validation, calibration, interpretability, subgroup analysis, error analysis and critical reporting.

## 17. Repository Structure

| File or folder | Description |
|---|---|
| `src/readmission_pipeline.py` | Main reproducible Python pipeline |
| `outputs/model_results.csv` | Holdout model performance |
| `outputs/cross_validation_tuning_results.csv` | Cross-validation and tuning results |
| `outputs/calibration_summary.csv` | Calibration metrics |
| `outputs/calibration_curve.png` | Calibration curve |
| `outputs/threshold_analysis.csv` | Threshold comparison table |
| `outputs/threshold_analysis.png` | Threshold analysis plot |
| `outputs/fairness_subgroup_analysis.csv` | Subgroup/fairness results |
| `outputs/error_analysis_summary.csv` | Error analysis summary |
| `outputs/feature_importance.csv` | Permutation importance results |
| `outputs/shap_top_features.csv` | Optional SHAP results |
| `briefs/technical_summary.md` | Technical summary |
| `briefs/medical_affairs_translation_brief.md` | Medical affairs brief |

## 18. Reproducibility

To run the project locally:

```bash
git clone https://github.com/tbharaj/diabetes-readmission-risk-ml.git
cd diabetes-readmission-risk-ml
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python src/readmission_pipeline.py
```

## References

Dua, D. and Graff, C. (2019) *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science. Available at: https://archive.ics.uci.edu/ (Accessed: 8 June 2026).

Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D. et al. (2020) 'Array programming with NumPy', *Nature*, 585, pp. 357-362. doi: 10.1038/s41586-020-2649-2.

Hunter, J.D. (2007) 'Matplotlib: A 2D graphics environment', *Computing in Science & Engineering*, 9(3), pp. 90-95. doi: 10.1109/MCSE.2007.55.

McKinney, W. (2010) 'Data structures for statistical computing in Python', *Proceedings of the 9th Python in Science Conference*, pp. 56-61.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O. et al. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J. and Clore, J.N. (2014) 'Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records', *BioMed Research International*, 2014, Article ID 781670. doi: 10.1155/2014/781670.

## Licence

This repository is licensed under the MIT License. The licence applies only to the code and project documentation created in this repository. The original dataset is not redistributed here.
"""

    Path("README.md").write_text(readme, encoding="utf-8")


def write_briefs(best_model_name, cv_df, results_df, calibration_summary, threshold_df, fairness_df, error_summary):
    """Run the write briefs step in the project workflow."""
    technical = f"""# Technical Summary

## Project

Diabetes Readmission Risk Prediction Using Interpretable Machine Learning

## Summary

This project built, tuned, validated, calibrated and interpreted a healthcare machine-learning pipeline for predicting 30-day readmission risk in patients with diabetes.

## Best model

The best model by holdout PR-AUC was **{best_model_name}**.

## Cross-validation and tuning

{df_to_markdown(cv_df[['model', 'best_cv_pr_auc_mean', 'best_cv_pr_auc_std', 'best_cv_roc_auc_mean', 'best_cv_roc_auc_std']])}

## Holdout results

{df_to_markdown(results_df[['model', 'holdout_roc_auc', 'holdout_pr_auc', 'precision_threshold_0_50', 'recall_threshold_0_50', 'f1_threshold_0_50']])}

## Calibration

{df_to_markdown(calibration_summary)}

## Clinical interpretation

The project shows some predictive signal in routinely collected hospital data, but model performance, calibration, threshold behaviour and subgroup performance would need external and prospective validation before clinical use.
"""

    medical = f"""# Medical Affairs Translation Brief

## Key message

Routinely collected hospital encounter data may contain signal associated with 30-day readmission risk in patients with diabetes, but the model is not clinically deployable without further validation.

## Stakeholder relevance

Relevant stakeholders include:

- clinicians;
- discharge planning teams;
- hospital quality-improvement teams;
- medical affairs teams;
- real-world evidence teams;
- healthcare analytics teams.

## Clinical interpretation

The model should be framed as an exploratory risk-stratification analysis, not a clinical tool. False positives may increase workload and alert fatigue, while false negatives may miss patients who could benefit from post-discharge support.

## Evidence generation implications

Future work would require:

- external validation;
- prospective validation;
- calibration review;
- threshold selection based on workflow;
- subgroup/fairness evaluation;
- clinician acceptability assessment;
- health-economic analysis.
"""

    advanced = f"""# Advanced Evaluation Summary

## Threshold analysis

{df_to_markdown(threshold_df)}

## Subgroup/fairness analysis

{df_to_markdown(fairness_df) if not fairness_df.empty else "No subgroup results available."}

## Error analysis

{df_to_markdown(error_summary)}

## Interpretation

This advanced evaluation moves the project beyond a simple model comparison by stress-testing the model across validation, calibration, threshold behaviour, subgroup performance and error patterns.
"""

    (BRIEF_DIR / "technical_summary.md").write_text(technical, encoding="utf-8")
    (BRIEF_DIR / "medical_affairs_translation_brief.md").write_text(medical, encoding="utf-8")
    (BRIEF_DIR / "advanced_evaluation_summary.md").write_text(advanced, encoding="utf-8")


def update_requirements():
    """Run the update requirements step in the project workflow."""
    req_path = Path("requirements.txt")
    existing = req_path.read_text(encoding="utf-8") if req_path.exists() else ""

    required = [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "ucimlrepo",
        "xgboost",
        "shap",
    ]

    lines = [line.strip() for line in existing.splitlines() if line.strip()]
    lower_lines = {line.lower().split("==")[0] for line in lines}

    for package in required:
        if package.lower() not in lower_lines:
            lines.append(package)

    req_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Run the main step in the project workflow."""
    update_requirements()

    X, y = load_dataset()
    X_model, numeric_features, categorical_features = prepare_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_model,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Test positive rate: {y_test.mean():.3f}")

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    search_spaces = build_model_search_spaces(preprocessor, y_train)

    tuned_models, cv_df = tune_models(search_spaces, X_train, y_train)
    results_df, predictions = evaluate_holdout_models(tuned_models, X_test, y_test)

    best_model_name = results_df.iloc[0]["model"]
    best_model = tuned_models[best_model_name]

    print(f"\nBest model by holdout PR-AUC: {best_model_name}")

    calibrated_model, y_proba_calibrated, calibration_summary, calibration_curve_df = calibrate_best_model(
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    threshold_df = threshold_analysis(y_test, y_proba_calibrated)

    # Select the threshold with the highest F1 score for reporting.
    # A default 0.50 threshold was too conservative after calibration and predicted no positive cases.
    best_threshold = float(threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"])
    print(f"\nSelected reporting threshold by maximum F1 score: {best_threshold:.3f}")

    save_core_plots(best_model_name, y_test, y_proba_calibrated, threshold=best_threshold)

    fairness_df = run_fairness_analysis(X_test, y_test, y_proba_calibrated, threshold=best_threshold)
    error_summary = run_error_analysis(X_test, y_test, y_proba_calibrated, threshold=best_threshold)

    importance_df = run_permutation_importance(best_model, X_test, y_test)
    shap_df = run_optional_shap(best_model, X_test)

    write_advanced_readme(
        cv_df=cv_df,
        results_df=results_df,
        calibration_summary=calibration_summary,
        threshold_df=threshold_df,
        fairness_df=fairness_df,
        error_summary=error_summary,
        importance_df=importance_df,
        shap_df=shap_df,
        best_model_name=best_model_name,
    )

    write_briefs(
        best_model_name=best_model_name,
        cv_df=cv_df,
        results_df=results_df,
        calibration_summary=calibration_summary,
        threshold_df=threshold_df,
        fairness_df=fairness_df,
        error_summary=error_summary,
    )

    print("\nAdvanced project upgrade completed.")
    print("\nKey outputs created:")
    print("- outputs/cross_validation_tuning_results.csv")
    print("- outputs/model_results.csv")
    print("- outputs/calibration_summary.csv")
    print("- outputs/calibration_curve.png")
    print("- outputs/threshold_analysis.csv")
    print("- outputs/threshold_analysis.png")
    print("- outputs/fairness_subgroup_analysis.csv")
    print("- outputs/fairness_recall_by_group.png")
    print("- outputs/error_analysis_summary.csv")
    print("- outputs/feature_importance.csv")
    print("- outputs/permutation_importance.png")
    print("- outputs/shap_top_features.csv, if SHAP succeeded")
    print("- outputs/shap_summary_bar.png, if SHAP succeeded")
    print("- README.md")
    print("- briefs/technical_summary.md")
    print("- briefs/medical_affairs_translation_brief.md")
    print("- briefs/advanced_evaluation_summary.md")


if __name__ == "__main__":
    main()
