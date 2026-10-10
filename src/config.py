"""Configuración central del proyecto de predicción de riesgo de abandono.

Versión 2.0 — Modelo específico por grado y grupo.
"""

from pathlib import Path

# Raíz del repositorio.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directorios principales.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# Archivos de entrada (datos reales de horarios).
HORARIOS_RESUMEN = RAW_DATA_DIR / "horarios_resumen.csv"
HORARIOS_DETALLE = RAW_DATA_DIR / "horarios_detalle.csv"

# Dataset generado a partir de los horarios reales.
DATASET_FILE = RAW_DATA_DIR / "estudiantes_por_grado.csv"

# Artefactos del modelo.
MODEL_FILE = MODELS_DIR / "modelo_riesgo.keras"
SCALER_FILE = MODELS_DIR / "escalador.pkl"
ENCODERS_FILE = MODELS_DIR / "encoders.pkl"

# Columnas categóricas que se codifican con One-Hot Encoding.
CATEGORICAL_COLS = ["grado", "grupo", "especialidad"]

# Características numéricas que se escalan con StandardScaler.
NUMERIC_COLS = [
    "horas_semana_totales",
    "asistencia_semanal",
    "promedio",
    "materias_reprobadas",
]

# Todas las columnas de entrada (antes de codificar).
FEATURES_RAW = CATEGORICAL_COLS + NUMERIC_COLS

# Variable objetivo.
TARGET = "riesgo_abandono"

# Grados válidos del programa.
GRADOS_VALIDOS = [1, 4, 7, 10]

# Estructura de grupos por grado (datos reales de la institución).
GRUPOS_POR_GRADO = {
    1:  {"grupos": ["A", "B", "C", "D"], "especialidad": "general"},
    4:  {"grupos": ["A", "B", "C"],       "especialidad": "general"},
    7:  {"grupos": ["A", "B", "C"],       "especialidad": "general"},
    10: {"grupos": [
            {"grupo": "A", "especialidad": "software"},
            {"grupo": "B", "especialidad": "software"},
            {"grupo": "A", "especialidad": "redes"},
        ]},
}

# Reproducibilidad.
RANDOM_STATE = 42
TEST_SIZE = 0.25
