"""Divide el dataset en conjuntos de entrenamiento y prueba."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATASET_FILE, FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def load_and_split_data():
    """Carga el dataset y realiza una división estratificada."""
    datos = pd.read_csv(DATASET_FILE)
    X = datos[FEATURES]
    y = datos[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def main() -> None:
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = load_and_split_data()

    print("===== DIVISIÓN DEL DATASET =====")
    print(f"Total de registros: {len(X_entrenamiento) + len(X_prueba)}")
    print(f"Registros de entrenamiento: {len(X_entrenamiento)}")
    print(f"Registros de prueba: {len(X_prueba)}")

    print("\n===== ENTRENAMIENTO =====")
    print("Distribución de clases:")
    print(y_entrenamiento.value_counts())

    print("\n===== PRUEBA =====")
    print("Distribución de clases:")
    print(y_prueba.value_counts())

    print("\n===== PRIMEROS 5 DE ENTRENAMIENTO =====")
    print(X_entrenamiento.head())

    print("\n===== PRIMEROS 5 DE PRUEBA =====")
    print(X_prueba.head())


if __name__ == "__main__":
    main()
