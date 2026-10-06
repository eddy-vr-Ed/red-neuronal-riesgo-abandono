"""Define la arquitectura de la red neuronal (Perceptrón Multicapa)."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model() -> keras.Model:
    """Construye la red neuronal de clasificación binaria.
    
    EXPLICACIÓN PARA EXPOSICIÓN:
    Utilizamos un esquema "Sequential" que apila capas de neuronas artificiales.
    Es un Perceptrón Multicapa (MLP), el modelo clásico de una Red Neuronal.
    """
    return keras.Sequential(
        [
            # === CAPA DE ENTRADA ===
            # Recibe los 4 datos del estudiante (Horas, Asistencia, Promedio, Reprobadas)
            keras.Input(shape=(4,)),
            
            # === PRIMERA CAPA OCULTA ===
            # 8 neuronas artificiales que extraen patrones matemáticos de los datos.
            # Función de activación 'relu': Imita cómo una neurona biológica se activa 
            # solo cuando recibe suficiente estímulo (si valor > 0, pasa; si no, 0).
            layers.Dense(8, activation="relu"),
            
            # === SEGUNDA CAPA OCULTA ===
            # 4 neuronas adicionales para procesar niveles lógicos más profundos.
            layers.Dense(4, activation="relu"),
            
            # === CAPA DE SALIDA ===
            # 2 neuronas porque solo hay 2 resultados posibles (Riesgo BAJO o ALTO).
            # Función de activación 'softmax': Convierte los resultados numéricos de las neuronas 
            # en porcentajes de probabilidad (ej. 80% Bajo, 20% Alto).
            layers.Dense(2, activation="softmax"),
        ],
        name="modelo_riesgo_abandono",
    )
