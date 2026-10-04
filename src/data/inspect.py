"""Inspecciona y valida rápidamente el dataset."""

import pandas as pd

from src.config import DATASET_FILE, FEATURES, TARGET


def main() -> None:
    datos = pd.read_csv(DATASET_FILE)

    print("===== INFORMACIÓN DEL DATASET =====")
    print(f"Total de registros: {len(datos)}")

    print("\n===== COLUMNAS =====")
    print(datos.columns.tolist())

    print("\n===== PRIMEROS 10 REGISTROS =====")
    print(datos.head(10))

    print("\n===== ÚLTIMOS 10 REGISTROS =====")
    print(datos.tail(10))

    print("\n===== VALORES MÍNIMOS =====")
    print(datos.min(numeric_only=True))

    print("\n===== VALORES MÁXIMOS =====")
    print(datos.max(numeric_only=True))

    print("\n===== DISTRIBUCIÓN DEL RIESGO =====")
    print(datos[TARGET].value_counts())

    print("\n===== VALORES NULOS =====")
    print(datos.isnull().sum())

    missing_columns = [column for column in FEATURES + [TARGET] if column not in datos.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")


if __name__ == "__main__":
    main()
