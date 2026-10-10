"""Genera el dataset sintético a partir de los horarios reales de los 13 grupos.

Lee 'horarios_resumen.csv' para obtener las horas reales por grupo y genera
estudiantes ficticios con distribuciones de asistencia y promedio realistas.
Las reglas de riesgo varían por grado para simular patrones no lineales.
"""

import numpy as np
import pandas as pd

from src.config import (
    DATASET_FILE,
    HORARIOS_RESUMEN,
    RANDOM_STATE,
)


def load_group_metadata() -> pd.DataFrame:
    """Carga los metadatos reales de los 13 grupos desde horarios_resumen.csv."""
    resumen = pd.read_csv(HORARIOS_RESUMEN)
    return resumen[["grado", "grupo", "especialidad", "horas_semana_totales"]].copy()


def generate_dataset(
    estudiantes_por_grupo: int = 50,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Genera datos estocásticos usando los horarios reales como base.

    Para cada grupo:
      - Asigna horas_semana_totales del CSV real (dato fijo por grupo).
      - Genera asistencia_semanal y promedio con distribuciones realistas.
      - Calcula riesgo_abandono con reglas NO lineales que varían por grado.
    """
    np.random.seed(seed)

    grupos = load_group_metadata()
    registros = []

    for _, fila in grupos.iterrows():
        grado = int(fila["grado"])
        grupo = fila["grupo"]
        especialidad = fila["especialidad"]
        horas = int(fila["horas_semana_totales"])

        for _ in range(estudiantes_por_grupo):
            # Arquetipos de estudiantes para cubrir todo el espectro real:
            # 60% regulares/buenos, 20% en riesgo medio, 20% en riesgo crítico
            arquetipo = np.random.choice(["regular", "medio", "critico"], p=[0.60, 0.20, 0.20])

            if arquetipo == "regular":
                asistencia = np.clip(np.random.normal(88, 6), 75, 100)
                promedio = np.clip(np.random.normal(8.5, 0.8), 7.0, 10.0)
                reprobadas = 0 if np.random.rand() > 0.08 else 1
            elif arquetipo == "medio":
                asistencia = np.clip(np.random.normal(68, 7), 55, 82)
                promedio = np.clip(np.random.normal(6.8, 0.7), 5.8, 7.8)
                reprobadas = int(np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2]))
            else:  # "critico"
                # Cubre asistencias bajas (15% a 60%), promedios reprobatorios y múltiples materias reprobadas
                asistencia = np.clip(np.random.uniform(15, 60), 10, 60)
                promedio = np.clip(np.random.normal(5.0, 1.2), 1.0, 6.5)
                reprobadas = int(np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.3, 0.3, 0.2, 0.1]))

            # === CÁLCULO DE RIESGO (varía por grado y factores académicos) ===
            riesgo = _calcular_riesgo(grado, horas, asistencia, promedio, reprobadas)

            registros.append({
                "grado": grado,
                "grupo": grupo,
                "especialidad": especialidad,
                "horas_semana_totales": horas,
                "asistencia_semanal": round(float(asistencia), 1),
                "promedio": round(float(promedio), 1),
                "materias_reprobadas": reprobadas,
                "riesgo_abandono": riesgo,
            })

    df = pd.DataFrame(registros)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _calcular_riesgo(
    grado: int,
    horas: int,
    asistencia: float,
    promedio: float,
    reprobadas: int,
) -> int:
    """Calcula el riesgo de abandono con reglas no lineales por grado.

    EXPLICACIÓN PARA EXPOSICIÓN:
    Estas reglas son las que la Red Neuronal tiene que APRENDER a descubrir.
    Nosotros las usamos para generar datos de entrenamiento, pero la red
    nunca ve estas reglas directamente — solo ve los datos resultantes.
    """
    riesgo_score = 0.0

    # === FACTOR 1: Materias reprobadas (el indicador más fuerte) ===
    if reprobadas >= 3:
        riesgo_score = 0.90
    elif reprobadas == 2:
        riesgo_score = 0.75
    elif reprobadas == 1:
        riesgo_score += 0.25

    # === FACTOR 2: Asistencia crítica (<60%) ===
    if asistencia < 60.0:
        riesgo_score = max(riesgo_score, 0.85)

    # === FACTOR 3: Promedio reprobatorio (<6.0) ===
    if promedio < 6.0:
        riesgo_score = max(riesgo_score, 0.80)

    # === REGLAS ESPECÍFICAS POR GRADO (para casos intermedios) ===
    if riesgo_score < 0.70:
        a_norm = (asistencia - 30) / 70.0
        p_norm = promedio / 10.0

        if grado == 1:
            # 1ro: Choque de primer ingreso, alta exigencia de asistencia
            base = 0.5 * (1 - a_norm) + 0.3 * (1 - p_norm)
            if asistencia < 75.0:
                base += 0.15
            riesgo_score = max(riesgo_score, base)

        elif grado == 4:
            # 4to: Fase intermedia de adaptación
            base = 0.4 * (1 - a_norm) + 0.35 * (1 - p_norm)
            riesgo_score = max(riesgo_score, base)

        elif grado == 7:
            # 7mo: Materias de especialidad difíciles
            base = 0.45 * (1 - a_norm) + 0.35 * (1 - p_norm)
            if promedio < 7.0 and asistencia < 80.0:
                base += 0.10
            riesgo_score = max(riesgo_score, base)

        elif grado == 10:
            # 10mo: Poca carga horaria (25 hrs), riesgo si se desconectan
            factor_horas = horas / 35.0
            base = 0.40 * (1 - a_norm) + 0.35 * (1 - p_norm) + 0.15 * (1 - factor_horas)
            riesgo_score = max(riesgo_score, base)

    # Ruido estocástico suave
    riesgo_score += np.random.normal(0, 0.03)

    return 1 if riesgo_score >= 0.50 else 0


def main() -> None:
    """Genera y guarda el dataset basado en horarios reales."""
    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("===== CARGANDO HORARIOS REALES =====")
    grupos = load_group_metadata()
    print(f"Grupos encontrados: {len(grupos)}")
    print(grupos.to_string(index=False))

    print("\n===== GENERANDO ESTUDIANTES =====")
    datos = generate_dataset()
    datos.to_csv(DATASET_FILE, index=False)

    print(f"Archivo: {DATASET_FILE.name}")
    print(f"Total de estudiantes generados: {len(datos)}")

    print("\n===== DISTRIBUCIÓN POR GRADO =====")
    for grado in sorted(datos["grado"].unique()):
        subset = datos[datos["grado"] == grado]
        riesgo_alto = subset["riesgo_abandono"].sum()
        print(
            f"  Grado {grado:2d} -> {len(subset):3d} estudiantes | "
            f"Riesgo ALTO: {riesgo_alto:3d} ({riesgo_alto / len(subset) * 100:.1f}%) | "
            f"Riesgo BAJO: {len(subset) - riesgo_alto:3d} ({(len(subset) - riesgo_alto) / len(subset) * 100:.1f}%)"
        )

    print("\n===== DISTRIBUCIÓN GLOBAL =====")
    print(datos["riesgo_abandono"].value_counts().rename({0: "BAJO", 1: "ALTO"}))

    print("\n===== PRIMEROS 10 REGISTROS =====")
    print(datos.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
