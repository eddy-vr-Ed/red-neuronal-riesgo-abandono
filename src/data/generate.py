"""Genera el dataset sintético utilizado por el proyecto."""

import numpy as np
import pandas as pd

from src.config import DATASET_FILE, RANDOM_STATE


def generate_dataset(num_estudiantes: int = 400) -> pd.DataFrame:
    """Genera datos estocásticos y calcula el riesgo con reglas NO lineales."""
    np.random.seed(RANDOM_STATE)

    # Aumentamos la muestra a 400 para que la red neuronal tenga más ejemplos de las reglas raras
    horas_estudio = np.random.uniform(1, 30, num_estudiantes)
    asistencia = np.random.uniform(50, 100, num_estudiantes)
    promedio = np.random.uniform(5.0, 10.0, num_estudiantes)
    materias_reprobadas = np.random.randint(0, 7, num_estudiantes)

    riesgo_abandono = np.zeros(num_estudiantes, dtype=int)

    for i in range(num_estudiantes):
        h = horas_estudio[i]
        a = asistencia[i]
        p = promedio[i]
        r = materias_reprobadas[i]
        
        riesgo = 0.0
        
        # === REGLAS NO LINEALES (Por qué usamos Redes Neuronales) ===
        
        # 1. "La Bola de Nieve": 3 o más reprobadas = Abandono casi inminente
        if r >= 3:
            riesgo = 0.85
            
        # 2. "El Genio Flojo": Excelente promedio, pero dejó de asistir = Abandono sorpresa
        elif p >= 8.5 and a < 65.0:
            riesgo = 0.90
            
        # 3. "Esfuerzo sin resultados": Estudia más de 20 horas pero apenas aprueba = Desgaste
        elif h >= 20.0 and p <= 6.5:
            riesgo = 0.85
            
        # 4. Estudiante regular (cálculo lineal normal ponderado)
        else:
            h_norm = (h - 1) / 29
            a_norm = (a - 50) / 50
            p_norm = (p - 5) / 5
            
            # Base lineal normal si todo va bien
            riesgo = 0.3 * (1 - h_norm) + 0.4 * (1 - a_norm) + 0.3 * (1 - p_norm)

        # Agregamos aleatoriedad humana
        riesgo += np.random.normal(0, 0.05)
        
        # Más de 0.55 de índice de riesgo se clasifica como abandono
        riesgo_abandono[i] = 1 if riesgo >= 0.55 else 0

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
