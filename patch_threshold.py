from pathlib import Path

path = Path("src/readmission_pipeline.py")
text = path.read_text(encoding="utf-8")

old = '''    threshold_df = threshold_analysis(y_test, y_proba_calibrated)

    # Use 0.50 as a conservative default reporting threshold.
    save_core_plots(best_model_name, y_test, y_proba_calibrated, threshold=0.50)

    fairness_df = run_fairness_analysis(X_test, y_test, y_proba_calibrated, threshold=0.50)
    error_summary = run_error_analysis(X_test, y_test, y_proba_calibrated, threshold=0.50)
'''

new = '''    threshold_df = threshold_analysis(y_test, y_proba_calibrated)

    # Select the threshold with the highest F1 score for reporting.
    # A default 0.50 threshold was too conservative after calibration and predicted no positive cases.
    best_threshold = float(threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"])
    print(f"\\nSelected reporting threshold by maximum F1 score: {best_threshold:.3f}")

    save_core_plots(best_model_name, y_test, y_proba_calibrated, threshold=best_threshold)

    fairness_df = run_fairness_analysis(X_test, y_test, y_proba_calibrated, threshold=best_threshold)
    error_summary = run_error_analysis(X_test, y_test, y_proba_calibrated, threshold=best_threshold)
'''

if old not in text:
    raise SystemExit("Could not find the block to replace.")

text = text.replace(old, new)
path.write_text(text, encoding="utf-8")

print("Threshold patch applied successfully.")
