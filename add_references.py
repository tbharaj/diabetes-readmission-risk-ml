from pathlib import Path

readme_path = Path("README.md")
readme = readme_path.read_text(encoding="utf-8")

references = """
## References

Dua, D. and Graff, C. (2019) *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science. Available at: https://archive.ics.uci.edu/ (Accessed: 8 June 2026).

Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D. et al. (2020) 'Array programming with NumPy', *Nature*, 585, pp. 357-362. doi: 10.1038/s41586-020-2649-2.

Hunter, J.D. (2007) 'Matplotlib: A 2D graphics environment', *Computing in Science & Engineering*, 9(3), pp. 90-95. doi: 10.1109/MCSE.2007.55.

McKinney, W. (2010) 'Data structures for statistical computing in Python', *Proceedings of the 9th Python in Science Conference*, pp. 56-61. doi: 10.25080/Majora-92bf1922-00a.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O. et al. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J. and Clore, J.N. (2014) 'Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records', *BioMed Research International*, 2014, Article ID 781670. doi: 10.1155/2014/781670.
"""

# Remove any old References section if it already exists
if "## References" in readme:
    readme = readme.split("## References")[0].rstrip()

readme = readme.rstrip() + "\n\n" + references.strip() + "\n"

readme_path.write_text(readme, encoding="utf-8")

print("Harvard-style references added to README.md")
