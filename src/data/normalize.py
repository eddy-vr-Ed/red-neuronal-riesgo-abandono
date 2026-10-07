"""Muestra el proceso de estandarización de los datos nuevos.

Versión 2.0 — Demuestra cómo se transforman las variables categóricas
y numéricas antes de alimentar la red neuronal.
"""

from sklearn.preprocessing import StandardScaler

from src.config import NUMERIC_COLS
from src.data.prepare import load_and_encode_data


def main() -> None:
    X_train, X_test, _, _, _, feature_names = load_and_encode_data()

    escalador = StandardScaler()
    X_train_scaled = escalador.fit_transform(X_train)
    X_test_scaled = escalador.transform(X_test)

    print("===== DATOS ORIGINALES (primeros 5) =====")
    print(X_train[:5])

    print("\n===== DATOS ESTANDARIZADOS (primeros 5) =====")
    print(X_train_scaled[:5])

    print(f"\n===== TAMAÑOS =====")
    print(f"Features: {feature_names}")
    print(f"Entrenamiento: {X_train_scaled.shape}")
    print(f"Prueba: {X_test_scaled.shape}")


if __name__ == "__main__":
    main()
