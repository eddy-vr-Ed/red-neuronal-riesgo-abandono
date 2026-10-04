"""Entrena, evalúa y guarda la red neuronal."""

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.config import DATASET_FILE, FEATURES, MODELS_DIR, MODEL_FILE, RANDOM_STATE, SCALER_FILE, TARGET
from src.data.prepare import load_and_split_data
from src.model.architecture import build_model


def main() -> None:
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    print("===== DATASET =====")
    print(f"Archivo: {DATASET_FILE}")

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = load_and_split_data()

    print("\n===== DIVISIÓN DE DATOS =====")
    print(f"Entrenamiento: {len(X_entrenamiento)}")
    print(f"Prueba: {len(X_prueba)}")
    print(f"Características: {FEATURES}")
    print(f"Variable objetivo: {TARGET}")

    escalador = StandardScaler()
    X_entrenamiento = escalador.fit_transform(X_entrenamiento)
    X_prueba = escalador.transform(X_prueba)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(escalador, SCALER_FILE)

    print("\n===== DATOS NORMALIZADOS =====")
    print(X_entrenamiento[:3])

    modelo = build_model()
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n===== ARQUITECTURA DE LA RED =====")
    modelo.summary()

    print("\n===== ENTRENANDO MODELO =====")
    modelo.fit(
        X_entrenamiento,
        y_entrenamiento,
        epochs=100,
        batch_size=16,
        verbose=1,
    )

    modelo.save(MODEL_FILE)

    perdida, precision = modelo.evaluate(X_prueba, y_prueba, verbose=0)
    print("\n===== RESULTADOS =====")
    print(f"Pérdida: {perdida:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Precisión porcentual: {precision * 100:.2f}%")

    probabilidades = modelo.predict(X_prueba, verbose=0)
    predicciones = np.argmax(probabilidades, axis=1)

    print("\n===== PRIMERAS PREDICCIONES =====")
    for i in range(min(10, len(y_prueba))):
        print(
            f"Real: {y_prueba.iloc[i]} | "
            f"Predicción: {predicciones[i]} | "
            f"Probabilidades: {probabilidades[i]}"
        )

    print("\n===== MATRIZ DE CONFUSIÓN =====")
    print(confusion_matrix(y_prueba, predicciones))

    print("\n===== REPORTE DE CLASIFICACIÓN =====")
    print(
        classification_report(
            y_prueba,
            predicciones,
            target_names=["BAJO", "ALTO"],
        )
    )

    print("\nArtefactos guardados correctamente:")
    print(f"- Modelo: {MODEL_FILE}")
    print(f"- Escalador: {SCALER_FILE}")


if __name__ == "__main__":
    main()
