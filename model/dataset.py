import numpy as np
from pathlib import Path


# ============================================================
# FEATURE INFORMATION
# ============================================================

FEATURES = [
    {
        "name": "Red Blood Cells (RBC)",
        "unit": "millions/µL",
        "description": "Cells that primarily transport oxygen through the bloodstream.",
        "healthy": "Approximately 4.0 to 6.0 million/µL"
    },
    {
        "name": "White Blood Cells (WBC)",
        "unit": "thousand/µL",
        "description": "Immune system cells that help defend the body.",
        "healthy": "Approximately 4 to 10 thousand/µL"
    },
    {
        "name": "Platelets (PLT)",
        "unit": "thousand/µL",
        "description": "Cells that play an essential role in blood clotting.",
        "healthy": "Approximately 150 to 400 thousand/µL"
    },
    {
        "name": "Blood Glucose (GLY)",
        "unit": "g/L",
        "description": "Concentration of glucose in the blood.",
        "healthy": "Approximately 0.70 to 1.10 g/L when fasting"
    },
    {
        "name": "CRP",
        "unit": "mg/L",
        "description": "A protein whose level typically rises during inflammation.",
        "healthy": "Generally below 5 mg/L"
    }
]


LABELS = [
    "Healthy",
    "Diabetes",
    "Infection",
    "Anemia",
    "Leukemia"
]


CONDITION_PROFILES = [
    {
        "name": "Diabetes",
        "details": (
            "Synthetic teaching profile used by this model:\n"
            "fasting blood glucose around 2.5 g/L.\n\n"
            "In clinical care, a fasting venous plasma glucose of 1.26 g/L "
            "or more is one diagnostic criterion and normally requires "
            "confirmation. This app cannot diagnose diabetes."
        )
    },
    {
        "name": "Infection",
        "details": (
            "Synthetic teaching profile used by this model:\n"
            "CRP strictly above 5 mg/L.\n\n"
            "Demo rule: this application always classifies a CRP above "
            "5 mg/L as Infection. This is an intentional simplification "
            "for learning purposes, not a clinical diagnostic rule."
        )
    },
    {
        "name": "Anemia",
        "details": (
            "Synthetic teaching profile used by this model:\n"
            "RBC around 3.5 million/µL.\n\n"
            "Clinical assessment of anemia relies mainly on hemoglobin and "
            "other complete blood count results; an RBC count alone is not "
            "diagnostic."
        )
    },
    {
        "name": "Leukemia",
        "details": (
            "Synthetic teaching profile used by this model:\n"
            "RBC around 3.5 million/µL, WBC around 30 thousand/µL, "
            "platelets around 100 thousand/µL, and CRP at or below "
            "5 mg/L.\n\n"
            "Blood-count abnormalities can require further evaluation, but "
            "they cannot diagnose leukemia without specialist testing."
        )
    }
]


# ============================================================
# DATASET LOADING
# ============================================================

DATASET_PATH = Path(__file__).with_name("blood_test_dataset.csv")
TRAINING_ROWS = 200


def generate_dataset():

    data = np.loadtxt(
        DATASET_PATH,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3, 4, 5, 6)
    )

    X = data[:, :5]
    y = data[:, 5].astype(int)

    X_train = X[:TRAINING_ROWS]
    y_train = y[:TRAINING_ROWS]

    X_test = X[TRAINING_ROWS:]
    y_test = y[TRAINING_ROWS:]

    # Normalize using only the training portion of the saved dataset.

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std

    return (
        X_train_normalized,
        y_train,
        X_test_normalized,
        y_test,
        mean,
        std
    )


def normalize_input(values, mean, std):

    values = np.array(values, dtype=float)

    return (values - mean) / std
