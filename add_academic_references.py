from pathlib import Path
import re

readme_path = Path("README.md")
text = readme_path.read_text(encoding="utf-8")

# ---------- Add / strengthen in-text citations ----------

replacements = {
    "Machine-learning models can identify patterns in historical healthcare data, but their usefulness depends on discrimination, calibration, threshold behaviour, subgroup performance and clinical feasibility.":
    "Machine-learning models can identify patterns in historical healthcare data, but their usefulness depends on discrimination, calibration, threshold behaviour, subgroup performance, implementation feasibility and clinical safety (Rajkomar, Dean and Kohane, 2019; Kelly et al., 2019).",

    "The analysis was implemented in Python using pandas, NumPy, scikit-learn, Matplotlib and optional XGBoost/SHAP. Pandas was used for data handling (McKinney, 2010), NumPy for numerical operations (Harris et al., 2020), scikit-learn for modelling and evaluation (Pedregosa et al., 2011), and Matplotlib for visualisation (Hunter, 2007).":
    "The analysis was implemented in Python using pandas, NumPy, scikit-learn, Matplotlib and optional XGBoost/SHAP. Pandas was used for data handling (McKinney, 2010), NumPy for numerical operations (Harris et al., 2020), scikit-learn for modelling and evaluation (Pedregosa et al., 2011), and Matplotlib for visualisation (Hunter, 2007). Random forest was included as an ensemble baseline based on Breiman's original method (Breiman, 2001), XGBoost was included as a stronger gradient-boosting model (Chen and Guestrin, 2016), and SHAP was used for model interpretation where available (Lundberg and Lee, 2017).",

    "Models were compared using stratified 3-fold cross-validation and hyperparameter tuning. PR-AUC was used as the refit metric because the positive class was rare and clinically important.":
    "Models were compared using stratified 3-fold cross-validation and hyperparameter tuning. PR-AUC was used as the refit metric because the positive class was rare and clinically important; precision-recall analysis is often more informative than ROC analysis when evaluating binary classifiers on imbalanced datasets (Saito and Rehmsmeier, 2015).",

    "Calibration assesses whether predicted probabilities correspond to observed outcome rates. This matters clinically because a predicted 30% risk should ideally mean that approximately 30% of similar patients experience readmission.":
    "Calibration assesses whether predicted probabilities correspond to observed outcome rates. This matters clinically because a predicted 30% risk should ideally mean that approximately 30% of similar patients experience readmission; poor calibration is a major limitation of predictive analytics in clinical settings (Van Calster et al., 2019).",

    "Unequal model performance across groups would require further investigation before deployment. This analysis should be interpreted as an exploratory fairness screen rather than a full fairness audit.":
    "Unequal model performance across groups would require further investigation before deployment. This analysis should be interpreted as an exploratory fairness screen rather than a full fairness audit, particularly because healthcare algorithms can reproduce or amplify structural inequities if target definitions, proxies or deployment contexts are poorly chosen (Obermeyer et al., 2019).",

    "The model should therefore be viewed as an exploratory health-data science pipeline, not a deployable clinical tool.":
    "The model should therefore be viewed as an exploratory health-data science pipeline, not a deployable clinical tool. This cautious interpretation is consistent with prediction-model reporting and risk-of-bias guidance, which emphasises transparent reporting, validation, calibration, applicability and bias assessment before clinical use (Collins et al., 2015; Wolff et al., 2019).",

    "External validation and prospective validation would be required before clinical use.":
    "External validation and prospective validation would be required before clinical use, because performance in a retrospective development dataset does not guarantee safe or useful performance in another clinical setting (Collins et al., 2015; Wolff et al., 2019)."
}

for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new)
    else:
        print("Warning: could not find exact paragraph to replace:")
        print(old[:120] + "...")

# Add a short reporting standards paragraph before Methods if not already present
standards_paragraph = """The write-up is also informed by established prediction-model reporting and appraisal principles. TRIPOD emphasises transparent reporting of prediction-model development and validation studies (Collins et al., 2015), while PROBAST provides a framework for considering risk of bias and applicability in prediction-model studies (Wolff et al., 2019)."""

if standards_paragraph not in text:
    text = text.replace(
        "## 4. Methods\n\n",
        "## 4. Methods\n\n" + standards_paragraph + "\n\n"
    )

# ---------- Replace references with expanded 15-source Harvard list ----------

references = """## References

Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5-32. doi: 10.1023/A:1010933404324.

Chen, T. and Guestrin, C. (2016) 'XGBoost: A scalable tree boosting system', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785-794. doi: 10.1145/2939672.2939785.

Collins, G.S., Reitsma, J.B., Altman, D.G. and Moons, K.G.M. (2015) 'Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): the TRIPOD statement', *Annals of Internal Medicine*, 162(1), pp. 55-63. doi: 10.7326/M14-0697.

Dua, D. and Graff, C. (2019) *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science. Available at: https://archive.ics.uci.edu/ (Accessed: 9 June 2026).

Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D. et al. (2020) 'Array programming with NumPy', *Nature*, 585, pp. 357-362. doi: 10.1038/s41586-020-2649-2.

Hunter, J.D. (2007) 'Matplotlib: A 2D graphics environment', *Computing in Science & Engineering*, 9(3), pp. 90-95. doi: 10.1109/MCSE.2007.55.

Kelly, C.J., Karthikesalingam, A., Suleyman, M., Corrado, G. and King, D. (2019) 'Key challenges for delivering clinical impact with artificial intelligence', *BMC Medicine*, 17, Article 195. doi: 10.1186/s12916-019-1426-2.

Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774.

McKinney, W. (2010) 'Data structures for statistical computing in Python', *Proceedings of the 9th Python in Science Conference*, pp. 56-61. doi: 10.25080/Majora-92bf1922-00a.

Obermeyer, Z., Powers, B., Vogeli, C. and Mullainathan, S. (2019) 'Dissecting racial bias in an algorithm used to manage the health of populations', *Science*, 366(6464), pp. 447-453. doi: 10.1126/science.aax2342.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O. et al. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.

Rajkomar, A., Dean, J. and Kohane, I. (2019) 'Machine learning in medicine', *New England Journal of Medicine*, 380(14), pp. 1347-1358. doi: 10.1056/NEJMra1814259.

Saito, T. and Rehmsmeier, M. (2015) 'The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets', *PLOS ONE*, 10(3), e0118432. doi: 10.1371/journal.pone.0118432.

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J. and Clore, J.N. (2014) 'Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records', *BioMed Research International*, 2014, Article ID 781670. doi: 10.1155/2014/781670.

Van Calster, B., McLernon, D.J., van Smeden, M., Wynants, L. and Steyerberg, E.W. (2019) 'Calibration: the Achilles heel of predictive analytics', *BMC Medicine*, 17, Article 230. doi: 10.1186/s12916-019-1466-7.

Wolff, R.F., Moons, K.G.M., Riley, R.D., Whiting, P.F., Westwood, M., Collins, G.S. et al. (2019) 'PROBAST: A tool to assess the risk of bias and applicability of prediction model studies', *Annals of Internal Medicine*, 170(1), pp. 51-58. doi: 10.7326/M18-1376.
"""

if "## References" in text and "## Licence" in text:
    text = re.sub(
        r"## References\n.*?## Licence",
        references.strip() + "\n\n## Licence",
        text,
        flags=re.S
    )
elif "## References" in text:
    text = text.split("## References")[0].rstrip() + "\n\n" + references.strip() + "\n"
else:
    text = text.rstrip() + "\n\n" + references.strip() + "\n"

readme_path.write_text(text, encoding="utf-8")

print("README updated with expanded in-text citations and 15-reference Harvard list.")
print("Reference count:", text.count("\\n") - text.split("## References")[0].count("\\n"))
