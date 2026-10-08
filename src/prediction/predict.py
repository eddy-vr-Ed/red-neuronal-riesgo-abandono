"""Realiza una predicción interactiva con el modelo entrenado."""

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import ENCODERS_FILE, MODEL_FILE, SCALER_FILE
from src.prediction.preprocessing import preparar_matriz_modelo


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


def predecir_estudiante(modelo, escalador, encoder, datos_estudiante: pd.DataFrame) -> np.ndarray:
    """Aplica el mismo preprocesamiento de Flask y devuelve probabilidades."""
    datos_escalados = preparar_matriz_modelo(datos_estudiante, encoder, escalador)
    return modelo.predict(datos_escalados, verbose=0)


def main() -> None:
    modelo = keras.models.load_model(MODEL_FILE)
    escalador = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODERS_FILE)

    print("===== PREDICCIÓN DE RIESGO DE ABANDONO =====")

    grado = solicitar_int("Cuatrimestre (1, 4, 7 o 10): ", 1, 10)
    if grado not in {1, 4, 7, 10}:
        raise ValueError("El cuatrimestre debe ser 1, 4, 7 o 10.")

    grupo = input("Grupo (A, B, C o D): ").strip().upper()
    especialidad = "general"
    if grado == 10:
        especialidad = input("Especialidad (software/redes): ").strip().lower()

    horas_semana_totales = 25 if grado == 10 else 35
    asistencia = solicitar_float("Asistencia semanal (%): ", 0, 100)
    promedio = solicitar_float("Promedio académico (0-10): ", 0, 10)
    materias_reprobadas = solicitar_int("Número de materias reprobadas: ", 0, 7)

    datos_estudiante = pd.DataFrame(
        [
            {
                "grado": grado,
                "grupo": grupo,
                "especialidad": especialidad,
                "horas_semana_totales": horas_semana_totales,
                "asistencia_semanal": asistencia,
                "promedio": promedio,
                "materias_reprobadas": materias_reprobadas,
            }
        ]
    )

    probabilidades = predecir_estudiante(modelo, escalador, encoder, datos_estudiante)

    prediccion = int(np.argmax(probabilidades, axis=1)[0])
    resultado = "BAJO" if prediccion == 0 else "ALTO"

    probabilidad_bajo = probabilidades[0][0] * 100
    probabilidad_alto = probabilidades[0][1] * 100

    print("\n===== RESULTADO =====")
    print(f"Cuatrimestre: {grado}")
    print(f"Grupo: {grupo}")
    print(f"Especialidad: {especialidad}")
    print(f"Horas semanales: {horas_semana_totales}")
    print(f"Asistencia semanal: {asistencia}%")
    print(f"Promedio: {promedio}")
    print(f"Materias reprobadas: {materias_reprobadas}")
    print(f"\nRiesgo estimado: {resultado}")
    print(f"Probabilidad BAJO: {probabilidad_bajo:.2f}%")
    print(f"Probabilidad ALTO: {probabilidad_alto:.2f}%")


if __name__ == "__main__":
    main()
