# Diabetes Readmission Risk ML

## Project aim

This project builds an interpretable machine-learning pipeline to predict 30-day hospital readmission risk in patients with diabetes using real-world hospital encounter data.

The aim is not to create a clinically deployable tool, but to demonstrate responsible health-data preprocessing, model evaluation, interpretation and limitations.

## Research question

Can routinely collected hospital encounter data identify patients with diabetes at higher risk of readmission within 30 days?

## Dataset

This project uses the Diabetes 130-US Hospitals for Years 1999–2008 dataset from the UCI Machine Learning Repository.

The raw dataset is not redistributed in this repository. The project fetches the data using `ucimlrepo`.

## Methods

- Binary classification: readmitted within 30 days vs not within 30 days
- Missing-value handling
- Categorical encoding
- Stratified train-test split
- Logistic regression baseline
- Random forest comparison model
- Evaluation using ROC-AUC, precision, recall, F1 score and confusion matrix
- Permutation importance for interpretation
- Clinical limitations and bias discussion

## Important limitation

This is a prototype educational project and is not suitable for clinical decision-making without external and prospective validation.
