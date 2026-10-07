"""Divide el dataset en conjuntos de entrenamiento y prueba.

Versión 2.0 — Maneja codificación One-Hot de variables categóricas
(grado, grupo, especialidad) y devuelve los encoders para reutilizarlos.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CATEGORICAL_COLS,
    DATASET_FILE,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def load_and_encode_data():
    """Carga el dataset, codifica las categóricas y divide los datos.

    Retorna:
        X_train, X_test, y_train, y_test, encoder, feature_names
    """
    datos = pd.read_csv(DATASET_FILE)

    # Separamos features y target
    X_raw = datos[CATEGORICAL_COLS + NUMERIC_COLS]
    y = datos[TARGET]

    # One-Hot Encoding para las columnas categóricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_encoded = encoder.fit_transform(X_raw[CATEGORICAL_COLS])
    cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()

    # Combinamos categóricas codificadas + numéricas
    X_numeric = X_raw[NUMERIC_COLS].values
    import numpy as np
    X_combined = np.hstack([X_cat_encoded, X_numeric])

    feature_names = cat_feature_names + NUMERIC_COLS

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, encoder, feature_names


def main() -> None:
    X_train, X_test, y_train, y_test, encoder, feature_names = load_and_encode_data()

    print("===== DIVISIÓN DEL DATASET =====")
    print(f"Total de registros: {len(X_train) + len(X_test)}")
    print(f"Registros de entrenamiento: {len(X_train)}")
    print(f"Registros de prueba: {len(X_test)}")

    print(f"\n===== FEATURES DESPUÉS DE ENCODING ({len(feature_names)}) =====")
    for i, name in enumerate(feature_names):
        print(f"  [{i:2d}] {name}")

    print("\n===== ENTRENAMIENTO =====")
    print("Distribución de clases:")
    print(y_train.value_counts().rename({0: "BAJO", 1: "ALTO"}))

    print("\n===== PRUEBA =====")
    print("Distribución de clases:")
    print(y_test.value_counts().rename({0: "BAJO", 1: "ALTO"}))


if __name__ == "__main__":
    main()
