"""Inspecciona y valida rápidamente el dataset por grado.

Versión 2.0 — Muestra estadísticas por grado y grupo.
"""

import pandas as pd

from src.config import DATASET_FILE, FEATURES_RAW, NUMERIC_COLS, TARGET


def main() -> None:
    datos = pd.read_csv(DATASET_FILE)

    print("===== INFORMACIÓN DEL DATASET =====")
    print(f"Total de registros: {len(datos)}")

    print("\n===== COLUMNAS =====")
    print(datos.columns.tolist())

    print("\n===== PRIMEROS 10 REGISTROS =====")
    print(datos.head(10).to_string(index=False))

    print("\n===== DISTRIBUCIÓN POR GRADO Y GRUPO =====")
    tabla = datos.groupby(["grado", "grupo", "especialidad"]).agg(
        total=("riesgo_abandono", "count"),
        riesgo_alto=("riesgo_abandono", "sum"),
        promedio_mean=("promedio", "mean"),
        asistencia_mean=("asistencia_semanal", "mean"),
    ).reset_index()
    tabla["riesgo_pct"] = (tabla["riesgo_alto"] / tabla["total"] * 100).round(1)
    print(tabla.to_string(index=False))

    print("\n===== ESTADÍSTICAS NUMÉRICAS =====")
    print(datos[NUMERIC_COLS].describe().round(2))

    print("\n===== DISTRIBUCIÓN GLOBAL DEL RIESGO =====")
    print(datos[TARGET].value_counts().rename({0: "BAJO", 1: "ALTO"}))

    print("\n===== VALORES NULOS =====")
    print(datos.isnull().sum())

    missing_columns = [c for c in FEATURES_RAW + [TARGET] if c not in datos.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")


if __name__ == "__main__":
    main()
