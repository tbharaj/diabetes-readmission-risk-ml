# Technical Summary

## Project

Diabetes Readmission Risk Prediction Using Interpretable Machine Learning

## Summary

This project built, tuned, validated, calibrated and interpreted a healthcare machine-learning pipeline for predicting 30-day readmission risk in patients with diabetes.

## Best model

The best model by holdout PR-AUC was **XGBoost**.

## Cross-validation and tuning

| model | best_cv_pr_auc_mean | best_cv_pr_auc_std | best_cv_roc_auc_mean | best_cv_roc_auc_std |
| --- | --- | --- | --- | --- |
| XGBoost | 0.215 | 0.004 | 0.670 | 0.002 |
| Logistic Regression | 0.200 | 0.006 | 0.647 | 0.004 |
| Random Forest | 0.199 | 0.004 | 0.658 | 0.004 |

## Holdout results

| model | holdout_roc_auc | holdout_pr_auc | precision_threshold_0_50 | recall_threshold_0_50 | f1_threshold_0_50 |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.680 | 0.231 | 0.179 | 0.620 | 0.277 |
| Random Forest | 0.673 | 0.220 | 0.187 | 0.523 | 0.276 |
| Logistic Regression | 0.659 | 0.212 | 0.172 | 0.554 | 0.263 |

## Calibration

| probability_type | brier_score | roc_auc | pr_auc |
| --- | --- | --- | --- |
| uncalibrated | 0.224 | 0.680 | 0.231 |
| calibrated_sigmoid | 0.094 | 0.681 | 0.232 |

## Clinical interpretation

The project shows some predictive signal in routinely collected hospital data, but model performance, calibration, threshold behaviour and subgroup performance would need external and prospective validation before clinical use.
