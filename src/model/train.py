"""Entrena, evalúa y guarda la red neuronal.

Versión 2.0 — Usa el dataset por grado, codifica variables categóricas,
guarda los encoders para la interfaz web y genera métricas por grado.
"""

import json
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATASET_FILE,
    ENCODERS_FILE,
    MODELS_DIR,
    MODEL_FILE,
    RANDOM_STATE,
    SCALER_FILE,
    TARGET,
)
from src.data.prepare import load_and_encode_data
from src.model.architecture import build_model


def main() -> None:
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    print("=" * 60)
    print("  RED NEURONAL — PREDICCIÓN DE RIESGO DE ABANDONO v2.0")
    print("  Modelo específico por grado y grupo")
    print("=" * 60)

    print(f"\n===== DATASET =====")
    print(f"Archivo: {DATASET_FILE}")

    X_train, X_test, y_train, y_test, encoder, feature_names = load_and_encode_data()

    print(f"\n===== DIVISIÓN DE DATOS =====")
    print(f"Entrenamiento: {len(X_train)} registros")
    print(f"Prueba: {len(X_test)} registros")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Variable objetivo: {TARGET}")

    # Escalado (estandarización)
    escalador = StandardScaler()
    X_train = escalador.fit_transform(X_train)
    X_test = escalador.transform(X_test)

    # Guardar artefactos de preprocesamiento
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(escalador, SCALER_FILE)
    joblib.dump(encoder, ENCODERS_FILE)

    print(f"\n===== DATOS NORMALIZADOS (primeras 3 filas) =====")
    print(X_train[:3])

    # Construir modelo con el número dinámico de features
    input_dim = X_train.shape[1]
    modelo = build_model(input_dim)
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n===== ARQUITECTURA DE LA RED ({input_dim} entradas) =====")
    modelo.summary()

    print(f"\n===== ALIMENTANDO LA RED NEURONAL =====")
    print("Proceso de alimentación paso a paso:")
    print(f"  1. Datos crudos -> {len(feature_names)} columnas originales")
    print(f"  2. One-Hot Encoding -> Variables categóricas convertidas a numéricas")
    print(f"  3. Estandarización -> Media=0, Desv.Estándar=1")
    print(f"  4. Entrada a la red -> {input_dim} neuronas de entrada")
    print(f"  5. Forward Propagation -> Capas ocultas procesan los datos")
    print(f"  6. Salida Softmax -> Probabilidad BAJO vs ALTO")

    print(f"\n===== ENTRENANDO MODELO =====")
    history = modelo.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        verbose=1,
    )

    modelo.save(MODEL_FILE)

    # Guardar el historial de entrenamiento como JSON para la web
    history_path = MODELS_DIR / "training_history.json"
    history_data = {
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }
    with open(history_path, "w") as f:
        json.dump(history_data, f)

    print(f"\n===== GENERANDO GRÁFICA DE APRENDIZAJE =====")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Gráfica de Precisión
        axes[0].plot(history.history["accuracy"], label="Entrenamiento", linewidth=2)
        axes[0].plot(history.history["val_accuracy"], label="Validación", linewidth=2)
        axes[0].set_title("Precisión (Accuracy) de la Red Neuronal")
        axes[0].set_xlabel("Época (Iteración)")
        axes[0].set_ylabel("Precisión")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Gráfica de Pérdida
        axes[1].plot(history.history["loss"], label="Entrenamiento", linewidth=2)
        axes[1].plot(history.history["val_loss"], label="Validación", linewidth=2)
        axes[1].set_title("Errores (Loss) de la Red Neuronal")
        axes[1].set_xlabel("Época (Iteración)")
        axes[1].set_ylabel("Pérdida")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = MODELS_DIR / "curvas_aprendizaje.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Gráfica guardada exitosamente en: {plot_path}")
    except ImportError:
        print("La librería matplotlib no está instalada, se omite la gráfica.")

    print(f"\n===== DEMOSTRACIÓN DIDÁCTICA: PESOS SINÁPTICOS =====")
    pesos, sesgos = modelo.layers[0].get_weights()
    print("La red ajustó internamente estos pesos sin que nosotros los programáramos:")
    print(f"Pesos de la primera neurona: {pesos[:, 0].flatten().round(4)}")

    perdida, precision = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"\n===== RESULTADOS GLOBALES =====")
    print(f"Pérdida: {perdida:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Precisión porcentual: {precision * 100:.2f}%")

    probabilidades = modelo.predict(X_test, verbose=0)
    predicciones = np.argmax(probabilidades, axis=1)

    print(f"\n===== PRIMERAS PREDICCIONES =====")
    for i in range(min(10, len(y_test))):
        print(
            f"Real: {y_test.iloc[i]} | "
            f"Predicción: {predicciones[i]} | "
            f"Probabilidades: {probabilidades[i]}"
        )

    print(f"\n===== MATRIZ DE CONFUSIÓN =====")
    print(confusion_matrix(y_test, predicciones))

    print(f"\n===== REPORTE DE CLASIFICACIÓN =====")
    print(
        classification_report(
            y_test,
            predicciones,
            target_names=["BAJO", "ALTO"],
        )
    )

    print(f"\nArtefactos guardados correctamente:")
    print(f"- Modelo: {MODEL_FILE}")
    print(f"- Escalador: {SCALER_FILE}")
    print(f"- Encoders: {ENCODERS_FILE}")
    print(f"- Historial: {history_path}")


if __name__ == "__main__":
    main()
