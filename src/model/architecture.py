"""Arquitectura de la Red Neuronal (Perceptrón Multicapa / MLP).

Nivel: Técnico Superior Universitario (TSU) en Desarrollo de Software.
Estructura:
- 1 Capa de Entrada: Recibe el vector numérico estandarizado.
- 2 Capas Ocultas: 16 y 8 neuronas con función de activación ReLU.
- 1 Capa de Salida: 2 neuronas con activación Softmax (Probabilidad: BAJO vs ALTO).
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim: int) -> keras.Model:
    """Construye un modelo secuencial simple para clasificación binaria.
    
    Parámetros:
        input_dim (int): Cantidad de características de entrada tras One-Hot Encoding.
    """
    modelo = keras.Sequential([
        # 1. Capa de Entrada: Especifica cuántas características entran a la red
        keras.Input(shape=(input_dim,)),

        # 2. Primera Capa Oculta: 16 neuronas para aprender patrones iniciales
        layers.Dense(16, activation="relu", name="capa_oculta_1"),

        # 3. Segunda Capa Oculta: 8 neuronas para refinar las combinaciones
        layers.Dense(8, activation="relu", name="capa_oculta_2"),

        # 4. Capa de Salida: 2 neuronas para calcular la probabilidad (Softmax: suma 100%)
        # Índice 0: Riesgo BAJO | Índice 1: Riesgo ALTO
        layers.Dense(2, activation="softmax", name="capa_salida")
    ], name="mlp_riesgo_escolar")

    return modelo

