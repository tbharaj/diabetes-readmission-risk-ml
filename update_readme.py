from pathlib import Path

readme = """# Diabetes Readmission Risk Prediction Using Interpretable Machine Learning

## Abstract

Hospital readmission is an important healthcare outcome because it can reflect disease burden, care-transition quality, treatment complexity and wider pressure on healthcare systems. This project developed an interpretable machine-learning pipeline to predict 30-day hospital readmission risk in patients with diabetes using routinely collected hospital encounter data.

The aim was not to create a clinically deployable model. Instead, the project aimed to demonstrate responsible health-data preprocessing, binary classification, model evaluation, class-imbalance awareness, feature interpretation and critical discussion of clinical limitations. Two models were compared: logistic regression and random forest. The Random Forest model achieved the highest ROC-AUC of 0.658, but precision remained low, showing that the model would require substantial validation and refinement before any clinical use.

## 1. Introduction

Diabetes is a long-term metabolic condition associated with substantial clinical complexity, multimorbidity and healthcare utilisation. Patients with diabetes may experience repeated hospital admissions due to complications, comorbid disease, medication burden or challenges in care transitions after discharge.

Thirty-day readmission is commonly used as a healthcare quality and risk indicator. However, predicting readmission is difficult because it is influenced by clinical, demographic, treatment-related and healthcare-system factors. Machine-learning methods can identify patterns in routinely collected healthcare data, but their outputs must be interpreted cautiously, particularly when models are trained on historical observational datasets.

This project uses the Diabetes 130-US Hospitals for Years 1999-2008 dataset from the UCI Machine Learning Repository (Dua and Graff, 2019). The dataset was introduced in a study examining the relationship between HbA1c measurement and hospital readmission outcomes in patients with diabetes (Strack et al., 2014).

## 2. Aim and Research Question

### Aim

The aim of this project was to build a reproducible and interpretable machine-learning pipeline for predicting 30-day hospital readmission risk in patients with diabetes.

### Research question

Can routinely collected hospital encounter data identify patients with diabetes at higher risk of readmission within 30 days?

### Project objectives

The project objectives were to:

- preprocess real-world hospital encounter data;
- convert readmission status into a binary classification target;
- compare a logistic regression baseline model with a random forest model;
- evaluate models using clinically relevant classification metrics;
- interpret model behaviour using permutation importance;
- discuss limitations, bias risk and clinical implementation barriers.

## 3. Dataset

This project used the Diabetes 130-US Hospitals for Years 1999-2008 dataset from the UCI Machine Learning Repository (Dua and Graff, 2019). The dataset contains hospital encounter records for patients with diabetes across multiple US hospitals.

The raw dataset is not redistributed in this repository. Instead, the data is accessed programmatically using the `ucimlrepo` Python package.

### Dataset summary

| Dataset characteristic | Value |
|---|---:|
| Hospital encounters | 101,766 |
| Original columns | 48 |
| Features used after preprocessing | 44 |
| Prediction target | 30-day readmission |
| Positive class | Readmitted within 30 days |
| Positive class proportion | Approximately 11.2% |

### Target variable

The original `readmitted` variable contains three categories:

| Original category | Meaning |
|---|---|
| `<30` | Patient readmitted within 30 days |
| `>30` | Patient readmitted after 30 days |
| `NO` | Patient not readmitted |

For this project, the target was converted into a binary classification problem:

| Binary class | Meaning |
|---|---|
| `1` | Readmitted within 30 days |
| `0` | Not readmitted within 30 days |

This created an imbalanced classification task, because only approximately 11.2% of encounters belonged to the positive class.

## 4. Methods

The analysis was implemented in Python using pandas, NumPy, scikit-learn and Matplotlib. Pandas was used for data handling (McKinney, 2010), NumPy supported numerical operations (Harris et al., 2020), scikit-learn was used for model development and evaluation (Pedregosa et al., 2011), and Matplotlib was used for visualisation (Hunter, 2007).

### 4.1 Preprocessing

The preprocessing workflow included:

- loading the dataset using `ucimlrepo`;
- replacing missing-value markers with appropriate missing values;
- removing selected identifier, high-missingness or low-interpretability columns;
- separating numerical and categorical variables;
- imputing missing values;
- encoding categorical variables;
- scaling numerical variables where appropriate;
- creating a stratified train-test split to preserve class proportions.

### 4.2 Models

Two machine-learning models were trained and compared.

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline interpretable classification model |
| Random Forest | Non-linear comparison model able to capture feature interactions |

Logistic regression was included as a baseline because it is widely used, relatively interpretable and suitable for comparison in healthcare prediction tasks. Random forest was included because it can capture non-linear relationships and interactions between variables.

### 4.3 Evaluation metrics

The models were evaluated using:

- ROC-AUC;
- precision;
- recall;
- F1 score;
- confusion matrix;
- precision-recall curve;
- permutation importance.

Accuracy was not used as the main metric because the dataset was imbalanced. In this setting, a model could achieve high accuracy by mostly predicting the majority class while failing to identify patients at risk of 30-day readmission.

## 5. Results

### 5.1 Model performance

| Model | ROC-AUC | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Random Forest | 0.658 | 0.172 | 0.539 | 0.261 |
| Logistic Regression | 0.648 | 0.169 | 0.558 | 0.259 |

The Random Forest model achieved the highest ROC-AUC at 0.658. However, both models showed low precision. This means that many patients predicted as high risk would be false positives. In a clinical context, this matters because false positives could lead to unnecessary follow-up, inefficient resource allocation or alert fatigue.

The Logistic Regression model achieved slightly higher recall than the Random Forest model, but the difference was small. Overall, the results suggest that routinely collected hospital encounter data contains some predictive signal for 30-day readmission risk, but model performance is not strong enough for clinical deployment.

### 5.2 Figures

#### ROC curve

![ROC curve](outputs/roc_curve.png)

#### Precision-recall curve

![Precision-recall curve](outputs/precision_recall_curve.png)

#### Confusion matrix

![Confusion matrix](outputs/confusion_matrix.png)

#### Permutation importance

![Permutation importance](outputs/permutation_importance.png)

### 5.3 Key model drivers

Permutation importance suggested that the most influential features included:

| Feature group | Clinical interpretation |
|---|---|
| Prior inpatient visits | May reflect previous healthcare utilisation and disease burden |
| Discharge disposition | May reflect care-transition complexity |
| Emergency visits | May indicate unstable disease or urgent care needs |
| Diagnosis codes | May capture comorbidity and clinical complexity |
| Insulin use | May reflect treatment intensity or disease severity |
| Age | May relate to frailty, comorbidity and readmission risk |
| Metformin use | May reflect diabetes treatment profile |
| Diabetes medication status | May reflect active pharmacological management |
| Number of lab procedures | May indicate clinical complexity or monitoring burden |
| Number of diagnoses | May reflect multimorbidity |

These features are clinically plausible. However, they should not be interpreted as causal drivers. The model identifies associations in observational data, not proof that changing these variables would directly reduce readmission risk.

## 6. Discussion

This project demonstrates both the potential and the limitations of machine learning in healthcare prediction tasks. The model identified clinically plausible signals associated with readmission risk, including previous inpatient use, emergency visits, discharge disposition and treatment-related variables. These factors are consistent with the idea that readmission risk is influenced by disease complexity, prior healthcare utilisation and care-transition challenges.

However, the modest ROC-AUC and low precision show that the model is not clinically deployable. In healthcare, a model must be evaluated not only by statistical performance but also by clinical consequences. A low-precision readmission model could incorrectly flag many patients as high risk, increasing workload and potentially reducing clinician trust.

The class imbalance is also important. Since only approximately 11.2% of encounters involved readmission within 30 days, the positive class was much smaller than the negative class. This makes precision, recall, F1 score and precision-recall analysis more informative than accuracy alone.

The project also highlights the importance of interpretability. Permutation importance helped identify which features contributed most to model performance. This makes the analysis more transparent than reporting performance metrics alone. However, interpretability methods also have limits. Feature importance can be affected by correlated variables, data coding patterns and historical healthcare processes.

## 7. Limitations

Several limitations should be considered.

First, the dataset is historical and covers hospital encounters from 1999-2008. Clinical practice, diabetes management and hospital discharge processes may have changed since then.

Second, the dataset comes from US hospitals, meaning the results may not generalise to UK/NHS settings.

Third, the analysis uses observational data. The model can identify associations but cannot prove causality.

Fourth, some variables may reflect healthcare-system behaviour rather than patient biology alone. For example, number of lab procedures may reflect clinical severity, hospital policy, documentation practice or care intensity.

Fifth, the model showed low precision. This limits its usefulness as a clinical decision-support tool.

Sixth, sensitive variables such as age, race and gender require careful fairness assessment before any real-world implementation.

Finally, external validation and prospective validation would be required before the model could be considered for clinical use.

## 8. Conclusion

This project built a reproducible machine-learning pipeline to predict 30-day hospital readmission risk in patients with diabetes using real-world hospital encounter data. The Random Forest model achieved the best ROC-AUC, but both models had low precision, showing that the analysis is best viewed as an educational and exploratory health-data science project rather than a deployable clinical tool.

The project demonstrates key skills relevant to health data science, healthcare analytics and medical affairs: data preprocessing, binary classification, model comparison, evaluation of imbalanced outcomes, interpretation of model drivers and responsible discussion of clinical limitations.

## 9. Medical Affairs and Health Data Science Relevance

This project is relevant to medical affairs because it translates a technical machine-learning analysis into clinically meaningful insights. The findings can be framed around patient pathway improvement, discharge planning, clinician education, real-world evidence generation and risk communication.

The project is also relevant to health data science because it demonstrates the full workflow of a healthcare prediction task: dataset access, preprocessing, model training, evaluation, interpretation, reporting and limitation analysis.

## 10. Repository Structure

| File or folder | Description |
|---|---|
| `src/readmission_pipeline.py` | Main reproducible Python pipeline |
| `outputs/model_results.csv` | Model performance results |
| `outputs/feature_importance.csv` | Permutation importance results |
| `outputs/roc_curve.png` | ROC curve |
| `outputs/precision_recall_curve.png` | Precision-recall curve |
| `outputs/confusion_matrix.png` | Confusion matrix |
| `outputs/permutation_importance.png` | Feature importance plot |
| `briefs/technical_summary.md` | Technical project summary |
| `briefs/medical_affairs_translation_brief.md` | Medical affairs translation brief |
| `requirements.txt` | Python package requirements |

## 11. Reproducibility

To run the project locally:

1. Clone the repository.
2. Create a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Run `src/readmission_pipeline.py`.

The project fetches the dataset programmatically and saves model results, plots and interpretation outputs into the `outputs` folder.

## References

Dua, D. and Graff, C. (2019) *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science. Available at: https://archive.ics.uci.edu/ (Accessed: 8 June 2026).

Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D. et al. (2020) 'Array programming with NumPy', *Nature*, 585, pp. 357-362. doi: 10.1038/s41586-020-2649-2.

Hunter, J.D. (2007) 'Matplotlib: A 2D graphics environment', *Computing in Science & Engineering*, 9(3), pp. 90-95. doi: 10.1109/MCSE.2007.55.

McKinney, W. (2010) 'Data structures for statistical computing in Python', *Proceedings of the 9th Python in Science Conference*, pp. 56-61.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O. et al. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J. and Clore, J.N. (2014) 'Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records', *BioMed Research International*, 2014, Article ID 781670. doi: 10.1155/2014/781670.

## Licence

This repository is licensed under the MIT License.

The licence applies only to the code and project documentation created in this repository. The original dataset is not redistributed here.
"""

Path("README.md").write_text(readme, encoding="utf-8")

print("Report-style README written successfully.")
print("Line count:", len(readme.splitlines()))
