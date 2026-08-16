import numpy as np


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


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(samples_per_class=50, seed=42):

    rng = np.random.default_rng(seed)

    X = []
    y = []

    # --------------------------------------------------------
    # Healthy
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        values = [
            rng.normal(5.0, 0.25),      # RBC
            rng.normal(7.0, 0.8),       # WBC
            rng.normal(250, 25),        # PLT
            rng.normal(0.90, 0.08),     # GLY
            rng.normal(2.0, 0.8)        # CRP
        ]

        X.append(values)
        y.append(0)

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        values = [
            rng.normal(5.0, 0.25),
            rng.normal(7.0, 0.8),
            rng.normal(250, 25),
            rng.normal(2.5, 0.30),     # high blood glucose
            rng.normal(2.0, 0.8)
        ]

        X.append(values)
        y.append(1)

    # --------------------------------------------------------
    # Infection
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        values = [
            rng.normal(5.0, 0.25),
            rng.normal(18.0, 2.0),      # high WBC count
            rng.normal(250, 25),
            rng.normal(0.90, 0.08),
            rng.normal(100, 15)         # high CRP
        ]

        X.append(values)
        y.append(2)

    # --------------------------------------------------------
    # Anemia
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        values = [
            rng.normal(3.5, 0.20),      # low RBC count
            rng.normal(7.0, 0.8),
            rng.normal(220, 25),
            rng.normal(0.90, 0.08),
            rng.normal(2.0, 0.8)
        ]

        X.append(values)
        y.append(3)

    # --------------------------------------------------------
    # Leukemia
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        values = [
            rng.normal(3.5, 0.25),      # low RBC count
            rng.normal(30.0, 3.0),      # very high WBC count
            rng.normal(100, 15),        # low platelet count
            rng.normal(0.90, 0.08),
            rng.normal(80, 15)           # high CRP
        ]

        X.append(values)
        y.append(4)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # Shuffle samples
    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    split = int(len(X) * 0.8)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

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
