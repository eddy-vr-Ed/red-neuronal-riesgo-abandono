"""Realiza una predicción interactiva con el modelo entrenado."""

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import FEATURES, MODEL_FILE, SCALER_FILE


def solicitar_float(mensaje: str, minimo: float, maximo: float) -> float:
    """Solicita un número dentro de un rango válido."""
    while True:
        try:
            valor = float(input(mensaje))
        except ValueError:
            print("Valor inválido. Ingresa un número.")
            continue

        if minimo <= valor <= maximo:
            return valor

        print(f"El valor debe estar entre {minimo} y {maximo}.")


def solicitar_int(mensaje: str, minimo: int, maximo: int) -> int:
    """Solicita un entero dentro de un rango válido."""
    while True:
        try:
            valor = int(input(mensaje))
        except ValueError:
            print("Valor inválido. Ingresa un número entero.")
            continue

        if minimo <= valor <= maximo:
            return valor

        print(f"El valor debe estar entre {minimo} y {maximo}.")


def main() -> None:
    modelo = keras.models.load_model(MODEL_FILE)
    escalador = joblib.load(SCALER_FILE)

    print("===== PREDICCIÓN DE RIESGO DE ABANDONO =====")

    horas_estudio = solicitar_float("Horas de estudio por semana: ", 1, 30)
    asistencia = solicitar_float("Porcentaje de asistencia: ", 50, 100)
    promedio = solicitar_float("Promedio académico: ", 5, 10)
    materias_reprobadas = solicitar_int("Número de materias reprobadas: ", 0, 6)

    datos_estudiante = pd.DataFrame(
        [
            {
                "horas_estudio": horas_estudio,
                "asistencia": asistencia,
                "promedio": promedio,
                "materias_reprobadas": materias_reprobadas,
            }
        ],
        columns=FEATURES,
    )

    datos_estudiante_escalados = escalador.transform(datos_estudiante)
    probabilidades = modelo.predict(datos_estudiante_escalados, verbose=0)

    prediccion = int(np.argmax(probabilidades, axis=1)[0])
    resultado = "BAJO" if prediccion == 0 else "ALTO"

    probabilidad_bajo = probabilidades[0][0] * 100
    probabilidad_alto = probabilidades[0][1] * 100

    print("\n===== RESULTADO =====")
    print(f"Horas de estudio: {horas_estudio}")
    print(f"Asistencia: {asistencia}%")
    print(f"Promedio: {promedio}")
    print(f"Materias reprobadas: {materias_reprobadas}")
    print(f"\nRiesgo estimado: {resultado}")
    print(f"Probabilidad BAJO: {probabilidad_bajo:.2f}%")
    print(f"Probabilidad ALTO: {probabilidad_alto:.2f}%")


if __name__ == "__main__":
    main()
