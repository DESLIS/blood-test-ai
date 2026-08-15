import numpy as np


# ============================================================
# INFORMATIONS SUR LES INDICATEURS
# ============================================================

FEATURES = [
    {
        "name": "Globules rouges (GR)",
        "unit": "millions/µL",
        "description": "Cellules qui transportent principalement l'oxygène dans le sang.",
        "healthy": "Environ 4,0 à 6,0 millions/µL"
    },
    {
        "name": "Globules blancs (GB)",
        "unit": "milliers/µL",
        "description": "Cellules du système immunitaire qui participent à la défense de l'organisme.",
        "healthy": "Environ 4 à 10 milliers/µL"
    },
    {
        "name": "Plaquettes (PLT)",
        "unit": "milliers/µL",
        "description": "Cellules qui jouent un rôle essentiel dans la coagulation du sang.",
        "healthy": "Environ 150 à 400 milliers/µL"
    },
    {
        "name": "Glycémie (GLY)",
        "unit": "g/L",
        "description": "Concentration de glucose dans le sang.",
        "healthy": "Environ 0,70 à 1,10 g/L à jeun"
    },
    {
        "name": "CRP",
        "unit": "mg/L",
        "description": "Protéine dont le taux augmente notamment lors d'une inflammation.",
        "healthy": "Généralement inférieure à 5 mg/L"
    }
]


LABELS = [
    "Sain",
    "Diabète",
    "Infection",
    "Anémie",
    "Leucémie"
]


# ============================================================
# GÉNÉRATION DU DATASET
# ============================================================

def generate_dataset(samples_per_class=50, seed=42):

    rng = np.random.default_rng(seed)

    X = []
    y = []

    # --------------------------------------------------------
    # Sain
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        values = [
            rng.normal(5.0, 0.25),      # GR
            rng.normal(7.0, 0.8),       # GB
            rng.normal(250, 25),        # PLT
            rng.normal(0.90, 0.08),     # GLY
            rng.normal(2.0, 0.8)        # CRP
        ]

        X.append(values)
        y.append(0)

    # --------------------------------------------------------
    # Diabète
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        values = [
            rng.normal(5.0, 0.25),
            rng.normal(7.0, 0.8),
            rng.normal(250, 25),
            rng.normal(2.5, 0.30),     # glycémie élevée
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
            rng.normal(18.0, 2.0),      # GB élevés
            rng.normal(250, 25),
            rng.normal(0.90, 0.08),
            rng.normal(100, 15)         # CRP élevée
        ]

        X.append(values)
        y.append(2)

    # --------------------------------------------------------
    # Anémie
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        values = [
            rng.normal(3.5, 0.20),      # GR faibles
            rng.normal(7.0, 0.8),
            rng.normal(220, 25),
            rng.normal(0.90, 0.08),
            rng.normal(2.0, 0.8)
        ]

        X.append(values)
        y.append(3)

    # --------------------------------------------------------
    # Leucémie
    # --------------------------------------------------------

    for _ in range(samples_per_class):

        values = [
            rng.normal(3.5, 0.25),      # GR faibles
            rng.normal(30.0, 3.0),      # GB très élevés
            rng.normal(100, 15),        # PLT faibles
            rng.normal(0.90, 0.08),
            rng.normal(80, 15)           # CRP élevée
        ]

        X.append(values)
        y.append(4)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # Mélange des exemples
    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    # --------------------------------------------------------
    # Séparation entraînement / test
    # --------------------------------------------------------

    split = int(len(X) * 0.8)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    # --------------------------------------------------------
    # Normalisation
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