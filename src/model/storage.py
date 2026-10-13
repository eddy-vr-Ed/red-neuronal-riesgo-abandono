"""Gestión de artefactos activos y almacenamiento persistente."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from tensorflow import keras

from src.config import (
    ENCODERS_FILE,
    INITIAL_ENCODERS_FILE,
    INITIAL_MODEL_FILE,
    INITIAL_SCALER_FILE,
    MODEL_FILE,
    MODEL_STORAGE_DIR,
    SCALER_FILE,
)


ARTIFACTS = {
    INITIAL_MODEL_FILE: MODEL_FILE,
    INITIAL_SCALER_FILE: SCALER_FILE,
    INITIAL_ENCODERS_FILE: ENCODERS_FILE,
}


def initialize_model_storage() -> None:
    """Copia artefactos iniciales al almacenamiento activo si faltan.

    No sobrescribe archivos existentes: si el modelo activo ya fue actualizado
    por retroalimentación, se conserva tal como está.
    """
    MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    missing_initial = [source for source in ARTIFACTS if not source.exists()]
    if missing_initial:
        names = ", ".join(str(path) for path in missing_initial)
        raise FileNotFoundError(f"Faltan artefactos iniciales del modelo: {names}")

    for source, destination in ARTIFACTS.items():
        if not destination.exists():
            shutil.copy2(source, destination)


def active_artifacts_available() -> bool:
    """Indica si el modelo activo y sus preprocesadores están disponibles."""
    return MODEL_FILE.exists() and SCALER_FILE.exists() and ENCODERS_FILE.exists()


def save_model_atomically(model: keras.Model, destination: Path = MODEL_FILE) -> None:
    """Guarda un modelo Keras evitando dejar un archivo parcial ante fallas."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp.keras"
    )
    try:
        model.save(temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            if temp_path.is_dir():
                shutil.rmtree(temp_path, ignore_errors=True)
            else:
                temp_path.unlink(missing_ok=True)
