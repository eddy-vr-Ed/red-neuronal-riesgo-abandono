"""Configuración central del proyecto de predicción de riesgo de abandono."""

from pathlib import Path

# Raíz del repositorio.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directorios principales.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# Archivos de entrada y artefactos del modelo.
DATASET_FILE = RAW_DATA_DIR / "estudiantes_200.csv"
MODEL_FILE = MODELS_DIR / "modelo_riesgo.keras"
SCALER_FILE = MODELS_DIR / "escalador.pkl"

# Características y variable objetivo.
FEATURES = [
    "horas_estudio",
    "asistencia",
    "promedio",
    "materias_reprobadas",
]
TARGET = "riesgo_abandono"

# Reproducibilidad.
RANDOM_STATE = 42
TEST_SIZE = 0.25
