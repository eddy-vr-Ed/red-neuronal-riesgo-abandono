"""Servidor Web con Flask para exponer la Red Neuronal."""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow import keras

app = Flask(__name__)

# Definimos las rutas a los modelos entrenados
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "modelo_riesgo.keras"
SCALER_FILE = MODELS_DIR / "escalador.pkl"

# Cargamos el escalador (Solo una vez al iniciar el servidor)
try:
    escalador = joblib.load(SCALER_FILE)
    modelo = keras.models.load_model(MODEL_FILE)
    MODELO_CARGADO = True
except Exception as e:
    print(f"Error cargando el modelo o escalador: {e}")
    print("Asegúrate de ejecutar primero: python -m src.model.train")
    MODELO_CARGADO = False

FEATURES = ["horas_estudio", "asistencia", "promedio", "materias_reprobadas"]


@app.route("/")
def index():
    """Sirve la página web principal."""
    return render_template("index.html", modelo_listo=MODELO_CARGADO)


@app.route("/grafica")
def grafica():
    """Devuelve la imagen de las curvas de aprendizaje."""
    return send_from_directory(MODELS_DIR, "curvas_aprendizaje.png")


@app.route("/predecir", methods=["POST"])
def predecir():
    """Endpoint de la API que recibe datos de la web y usa la Red Neuronal."""
    if not MODELO_CARGADO:
        return jsonify({"error": "El modelo no ha sido entrenado aún."}), 500

    try:
        # Obtenemos los datos del formulario web
        data = request.json
        horas_estudio = float(data.get("horas_estudio", 0))
        asistencia = float(data.get("asistencia", 0))
        promedio = float(data.get("promedio", 0))
        materias_reprobadas = int(data.get("materias_reprobadas", 0))

        # Construimos el DataFrame para el escalador
        datos_estudiante = pd.DataFrame(
            [
                {
                    "horas_estudio": horas_estudio,
                    "asistencia": asistencia,
                    "promedio": promedio,
                    "materias_reprobadas": materias_reprobadas,
                }
            ],
            columns=FEATURES,
        )

        # 1. Normalización idéntica al entrenamiento
        datos_escalados = escalador.transform(datos_estudiante)

        # 2. Predicción con la Red Neuronal (Forward Propagation)
        probabilidades = modelo.predict(datos_escalados, verbose=0)
        
        # 3. Interpretación de la capa Softmax
        prob_bajo = float(probabilidades[0][0] * 100)
        prob_alto = float(probabilidades[0][1] * 100)
        
        prediccion_idx = int(np.argmax(probabilidades, axis=1)[0])
        riesgo_str = "ALTO" if prediccion_idx == 1 else "BAJO"

        return jsonify({
            "riesgo": riesgo_str,
            "probabilidad_bajo": f"{prob_bajo:.1f}%",
            "probabilidad_alto": f"{prob_alto:.1f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
