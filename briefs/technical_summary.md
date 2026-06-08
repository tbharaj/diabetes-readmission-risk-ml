
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

Best model by ROC-AUC: Random Forest

## Results

              model  roc_auc  precision   recall  f1_score
      Random Forest 0.658310   0.172108 0.538529  0.260851
Logistic Regression 0.648482   0.168866 0.557904  0.259259

## Interpretation

This project is a prototype health-data analysis. Model outputs may help identify patterns associated with readmission risk, but they should not be interpreted as causal or clinically deployable.

## Limitations

1. The dataset is historical, covering care from 1999-2008.
2. The data comes from US hospitals and may not generalise to UK/NHS settings.
3. Observational data can identify associations but cannot prove causality.
4. Missing values and coding patterns may reflect healthcare-system processes.
5. The target is imbalanced, so accuracy alone would be misleading.
6. Sensitive variables such as age, race and gender require fairness assessment.
7. External and prospective validation would be required before clinical use.
