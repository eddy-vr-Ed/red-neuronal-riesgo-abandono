"""Muestra el proceso de estandarización sin modificar el dataset original."""

from sklearn.preprocessing import StandardScaler

from src.config import FEATURES
from src.data.prepare import load_and_split_data


def main() -> None:
    X_entrenamiento, X_prueba, _, _ = load_and_split_data()

    escalador = StandardScaler()
    X_entrenamiento_escalado = escalador.fit_transform(X_entrenamiento)
    X_prueba_escalado = escalador.transform(X_prueba)

    print("===== DATOS ORIGINALES =====")
    print(X_entrenamiento.head())

    print("\n===== DATOS ESTANDARIZADOS =====")
    print(X_entrenamiento_escalado[:5])

    print("\n===== TAMAÑOS =====")
    print(f"Características procesadas: {FEATURES}")
    print(f"Entrenamiento: {X_entrenamiento_escalado.shape}")
    print(f"Prueba: {X_prueba_escalado.shape}")


if __name__ == "__main__":
    main()
