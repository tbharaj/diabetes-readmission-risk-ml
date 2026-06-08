# ============================================================
# Diabetes Readmission Risk ML Project
# Predicting 30-day hospital readmission risk using interpretable ML
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)

from sklearn.inspection import permutation_importance


RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("Loading UCI Diabetes 130-US Hospitals dataset...")

diabetes = fetch_ucirepo(id=296)

X_raw = diabetes.data.features
y_raw = diabetes.data.targets

df = pd.concat([X_raw, y_raw], axis=1)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())


# ------------------------------------------------------------
# 2. Inspect target variable
# ------------------------------------------------------------

print("\nReadmission categories:")
print(df["readmitted"].value_counts(dropna=False))


# ------------------------------------------------------------
# 3. Create binary target
# ------------------------------------------------------------

df = df.replace("?", np.nan)

# Positive class: readmitted within 30 days
y = (df["readmitted"] == "<30").astype(int)

# Features
X = df.drop(columns=["readmitted"])

print("\nTarget counts:")
print(y.value_counts())

print("\nTarget proportions:")
print(y.value_counts(normalize=True))


# ------------------------------------------------------------
# 4. Drop identifier / weak columns
# ------------------------------------------------------------

drop_cols = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "medical_specialty",
]

X = X.drop(columns=[col for col in drop_cols if col in X.columns])

print("\nFeature matrix shape after dropping selected columns:")
print(X.shape)


# ------------------------------------------------------------
# 5. Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining set:")
print(X_train.shape)

print("\nTest set:")
print(X_test.shape)

print("\nTrain positive rate:")
print(round(y_train.mean(), 3))

print("\nTest positive rate:")
print(round(y_test.mean(), 3))


# ------------------------------------------------------------
# 6. Preprocessing
# ------------------------------------------------------------

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ------------------------------------------------------------
# 7. Build models
# ------------------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

random_forest_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        ),
    ]
)

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
}


# ------------------------------------------------------------
# 8. Train and evaluate models
# ------------------------------------------------------------

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n=== {name} ===")
    print("ROC-AUC:", round(auc, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 score:", round(f1, 3))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append(
        {
            "model": name,
            "roc_auc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    )

results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

print("\nModel comparison:")
print(results_df)

results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)


# ------------------------------------------------------------
# 9. Select best model by ROC-AUC
# ------------------------------------------------------------

best_model_name = results_df.iloc[0]["model"]
best_model = models[best_model_name]

print("\nBest model by ROC-AUC:")
print(best_model_name)


# ------------------------------------------------------------
# 10. Plot ROC curve
# ------------------------------------------------------------

RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"ROC Curve — {best_model_name}")
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 11. Plot precision-recall curve
# ------------------------------------------------------------

PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"Precision-Recall Curve — {best_model_name}")
plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 12. Confusion matrix
# ------------------------------------------------------------

y_pred_best = best_model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best)
plt.title(f"Confusion Matrix — {best_model_name}")
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 13. Permutation importance
# ------------------------------------------------------------

sample_n = min(3000, len(X_test))

X_sample = X_test.sample(sample_n, random_state=RANDOM_STATE)
y_sample = y_test.loc[X_sample.index]

print("\nCalculating permutation importance...")

perm = permutation_importance(
    best_model,
    X_sample,
    y_sample,
    n_repeats=5,
    random_state=RANDOM_STATE,
    scoring="roc_auc",
)

importance_df = pd.DataFrame(
    {
        "feature": X_sample.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
).sort_values("importance_mean", ascending=False)

print("\nTop 15 important features:")
print(importance_df.head(15))

importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)


# ------------------------------------------------------------
# 14. Plot feature importance
# ------------------------------------------------------------

top_features = importance_df.head(10).sort_values("importance_mean")

plt.figure(figsize=(8, 5))
plt.barh(top_features["feature"], top_features["importance_mean"])
plt.xlabel("Permutation importance")
plt.title("Top Features by Permutation Importance")
plt.savefig(OUTPUT_DIR / "permutation_importance.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 15. Save technical summary
# ------------------------------------------------------------

summary_text = f"""
# Technical Summary

## Project title

Predicting 30-Day Diabetes Readmission Risk Using Interpretable Machine Learning

## Research question

Can routinely collected hospital encounter data identify patients with diabetes at higher risk of readmission within 30 days?

## Dataset

This project uses the UCI Diabetes 130-US Hospitals dataset. The target variable was converted into a binary outcome:

- 1 = readmitted within 30 days
- 0 = not readmitted within 30 days

## Methods

The project used a reproducible machine-learning pipeline including:

- Missing-value handling
- Categorical encoding
- Stratified train-test split
- Logistic regression baseline
- Random forest comparison model
- ROC-AUC, precision, recall, F1 score and confusion matrix evaluation
- Permutation importance for interpretation

## Best model

Best model by ROC-AUC: {best_model_name}

## Results

{results_df.to_string(index=False)}

## Interpretation

This project is a prototype health-data analysis. Model outputs may help identify patterns associated with readmission risk, but they should not be interpreted as causal or clinically deployable.

## Limitations

1. The dataset is historical, covering care from 1999–2008.
2. The data comes from US hospitals and may not generalise to UK/NHS settings.
3. Observational data can identify associations but cannot prove causality.
4. Missing values and coding patterns may reflect healthcare-system processes.
5. The target is imbalanced, so accuracy alone would be misleading.
6. Sensitive variables such as age, race and gender require fairness assessment.
7. External and prospective validation would be required before clinical use.
"""

Path("briefs").mkdir(exist_ok=True)
Path("briefs/technical_summary.md").write_text(summary_text, encoding="utf-8")


# ------------------------------------------------------------
# 16. Save medical affairs translation brief
# ------------------------------------------------------------

medical_affairs_text = """
# Medical Affairs Translation Brief

## Project title

Reducing 30-Day Diabetes Readmission Risk Through Real-World Evidence and Interpretable Machine Learning

## Clinical problem

Patients with diabetes may be at risk of hospital readmission due to disease complexity, comorbidities, medication changes, discharge planning challenges and follow-up care gaps. Thirty-day readmission is clinically important because it can reflect patient deterioration, poor care transitions or avoidable pressure on healthcare systems.

## Evidence source

This project uses a public real-world hospital encounter dataset involving patients with diabetes.

## Analytical question

Can routinely collected hospital encounter variables be used to flag patients who may be at higher risk of readmission within 30 days?

## Medical affairs relevance

This project is relevant to medical affairs because it connects real-world evidence, patient pathway optimisation, clinical education and stakeholder communication.

A medical affairs team could use similar evidence-generation thinking to support clinician education around:

- readmission risk factors
- discharge planning
- medication review
- follow-up prioritisation
- patient pathway improvement

## Stakeholders

- Diabetologists and endocrinologists
- Diabetes specialist nurses
- Hospital pharmacists
- Primary care teams
- Patients and carers
- NHS or payer decision-makers
- Medical affairs and real-world evidence teams

## Risks and limitations

- The dataset is historical and may not reflect current practice.
- The data comes from US hospitals and may not generalise to the NHS.
- Observational data can identify associations but cannot prove causality.
- Machine-learning models can reinforce bias if underlying healthcare access or coding patterns are unequal.
- Clinical deployment would require prospective validation, governance review and clinician oversight.

## Personal reflection

This project connects biomedical science, machine learning, real-world evidence and healthcare communication. It strengthened my understanding of how health data can be analysed technically, interpreted clinically and translated into stakeholder-facing evidence for healthcare decision-making.
"""

Path("briefs/medical_affairs_translation_brief.md").write_text(
    medical_affairs_text,
    encoding="utf-8",
)


# ------------------------------------------------------------
# 17. Final printout
# ------------------------------------------------------------

print("\nProject complete.")
print("\nSaved outputs:")
print("- outputs/model_results.csv")
print("- outputs/feature_importance.csv")
print("- outputs/roc_curve.png")
print("- outputs/precision_recall_curve.png")
print("- outputs/confusion_matrix.png")
print("- outputs/permutation_importance.png")
print("- briefs/technical_summary.md")
print("- briefs/medical_affairs_translation_brief.md")

print(
    """
Final interpretation note:

This project is a prototype machine-learning analysis and should not be interpreted
as a clinically deployable model. Its value is in demonstrating health-data preprocessing,
model comparison, evaluation, interpretation and responsible discussion of limitations.
"""
)
