"""Define la arquitectura de la red neuronal."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model() -> keras.Model:
    """Construye la red neuronal de clasificación binaria."""
    return keras.Sequential(
        [
            keras.Input(shape=(4,)),
            layers.Dense(8, activation="relu"),
            layers.Dense(4, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ],
        name="modelo_riesgo_abandono",
    )
