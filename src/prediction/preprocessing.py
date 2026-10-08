"""Utilidades compartidas para preparar datos antes de la inferencia."""

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_COLS, FEATURES_RAW, NUMERIC_COLS


def preparar_matriz_modelo(
    datos: pd.DataFrame,
    encoder,
    escalador,
) -> np.ndarray:
    """Convierte datos crudos al vector numerico esperado por la red neuronal.

    Mantiene en un solo lugar el orden de columnas, el One-Hot Encoding de
    variables categoricas y el escalado usado tanto por Flask como por el CLI.
    """
    datos_ordenados = datos[FEATURES_RAW]
    cat_encoded = encoder.transform(datos_ordenados[CATEGORICAL_COLS])
    num_raw = datos_ordenados[NUMERIC_COLS].values
    datos_combinados = np.hstack([cat_encoded, num_raw])
    return escalador.transform(datos_combinados)

