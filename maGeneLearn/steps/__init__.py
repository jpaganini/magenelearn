# maGeneLearn/steps/__init__.py

"""
Subpackage containing the individual pipeline step scripts:
00_split_dataset.py, 01_chisq_selection.py, …
"""

__all__ = [
    "00_split_dataset",
    "01_chisq_selection",
    "02_muvr_feature_selection",
    "03_extract_features",
    "04_train_model",
    "05_evaluate_model",
]
