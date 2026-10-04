"""Genera el dataset sintético utilizado por el proyecto."""

import numpy as np
import pandas as pd

from src.config import DATASET_FILE, RANDOM_STATE


def generate_dataset(num_estudiantes: int = 200) -> pd.DataFrame:
    """Genera datos sintéticos y calcula la etiqueta de riesgo."""
    np.random.seed(RANDOM_STATE)

    horas_estudio = np.random.uniform(1, 30, num_estudiantes)
    asistencia = np.random.uniform(50, 100, num_estudiantes)
    promedio = np.random.uniform(5.0, 10.0, num_estudiantes)
    materias_reprobadas = np.random.randint(0, 7, num_estudiantes)

    horas_norm = (horas_estudio - 1) / (30 - 1)
    asistencia_norm = (asistencia - 50) / (100 - 50)
    promedio_norm = (promedio - 5) / (10 - 5)
    reprobadas_norm = materias_reprobadas / 6

    riesgo = (
        0.25 * (1 - horas_norm)
        + 0.30 * (1 - asistencia_norm)
        + 0.25 * (1 - promedio_norm)
        + 0.20 * reprobadas_norm
    )

    ruido = np.random.normal(0, 0.08, num_estudiantes)
    riesgo += ruido

    riesgo_abandono = (riesgo >= np.median(riesgo)).astype(int)

    datos = pd.DataFrame(
        {
            "horas_estudio": np.round(horas_estudio, 1),
            "asistencia": np.round(asistencia, 1),
            "promedio": np.round(promedio, 1),
            "materias_reprobadas": materias_reprobadas,
            "riesgo_abandono": riesgo_abandono,
        }
    )

    return datos.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    """Genera y guarda el dataset."""
    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    datos = generate_dataset()
    datos.to_csv(DATASET_FILE, index=False)

    print("Dataset creado correctamente.")
    print(f"Archivo: {DATASET_FILE.relative_to(DATASET_FILE.parents[2])}")
    print(f"Total de estudiantes: {len(datos)}")
    print("\nDistribución de clases:")
    print(datos["riesgo_abandono"].value_counts())
    print("\nPrimeros 10 registros:")
    print(datos.head(10))


if __name__ == "__main__":
    main()
