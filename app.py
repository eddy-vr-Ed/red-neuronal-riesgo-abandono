"""Servidor Web con Flask para exponer la Red Neuronal v2.0."""

import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow import keras

app = Flask(__name__)

# Definimos las rutas a los modelos entrenados
from src.config import (
    CATEGORICAL_COLS,
    ENCODERS_FILE,
    MODEL_FILE,
    MODELS_DIR,
    NUMERIC_COLS,
    SCALER_FILE,
)

HISTORY_FILE = MODELS_DIR / "training_history.json"

# Cargamos el modelo y los preprocesadores
try:
    escalador = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODERS_FILE)
    modelo = keras.models.load_model(MODEL_FILE)
    MODELO_CARGADO = True
except Exception as e:
    print(f"Error cargando el modelo o preprocesadores: {e}")
    print("Asegúrate de ejecutar primero: python -m src.model.train")
    MODELO_CARGADO = False


@app.route("/")
def index():
    """Sirve la página web principal."""
    # Intentamos cargar el historial de entrenamiento para mostrarlo en el frontend
    history = {}
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            pass
            
    return render_template(
        "index.html", 
        modelo_listo=MODELO_CARGADO,
        history=json.dumps(history)
    )


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
        grado = int(data.get("grado", 1))
        grupo = str(data.get("grupo", "A"))
        especialidad = str(data.get("especialidad", "general"))
        horas = float(data.get("horas_semana_totales") or 30)
        asistencia = float(data.get("asistencia_semanal") or 0)
        promedio = float(data.get("promedio") or 0)
        materias_reprobadas = int(data.get("materias_reprobadas") or 0)

        # Construimos el DataFrame para los encoders (variables crudas)
        datos_estudiante = pd.DataFrame(
            [{
                "grado": grado,
                "grupo": grupo,
                "especialidad": especialidad,
                "horas_semana_totales": horas,
                "asistencia_semanal": asistencia,
                "promedio": promedio,
                "materias_reprobadas": materias_reprobadas,
            }],
        )

        # 1. Codificación One-Hot de categóricas
        cat_encoded = encoder.transform(datos_estudiante[CATEGORICAL_COLS])
        
        # 2. Combinación con numéricas
        num_raw = datos_estudiante[NUMERIC_COLS].values
        X_combined = np.hstack([cat_encoded, num_raw])

        # Si el escalador en memoria no coincide con las dimensiones de entrada, recargar desde disco
        global escalador, modelo
        if getattr(escalador, "n_features_in_", None) != X_combined.shape[1]:
            escalador = joblib.load(SCALER_FILE)
            modelo = keras.models.load_model(MODEL_FILE)

        # 3. Normalización (Estandarización)
        X_scaled = escalador.transform(X_combined)

        # 4. Predicción con la Red Neuronal (Forward Propagation)
        probabilidades = modelo.predict(X_scaled, verbose=0)
        
        # 5. Interpretación de la capa Softmax
        prob_bajo = float(probabilidades[0][0] * 100)
        prob_alto = float(probabilidades[0][1] * 100)
        
        prediccion_idx = int(np.argmax(probabilidades, axis=1)[0])
        riesgo_str = "ALTO" if prediccion_idx == 1 else "BAJO"

        # Obtenemos pesos sinápticos de la primera capa para demostración de aprendizaje
        primeros_pesos = modelo.layers[0].get_weights()[0][:5, 0].round(4).tolist()

        return jsonify({
            "riesgo": riesgo_str,
            "probabilidad_bajo": f"{prob_bajo:.1f}%",
            "probabilidad_alto": f"{prob_alto:.1f}%",
            "pesos_ejemplo": primeros_pesos,
            "debug": {
                "features_entrada": int(X_scaled.shape[1]),
                "cat_encoded_shape": int(cat_encoded.shape[1])
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/retroalimentar", methods=["POST"])
def retroalimentar():
    """Permite a la red neuronal aprender de una consulta real mediante retroalimentación (Fine-Tuning online)."""
    global modelo
    if not MODELO_CARGADO:
        return jsonify({"error": "Modelo no cargado"}), 500

    try:
        data = request.json
        grado = int(data.get("grado", 1))
        grupo = str(data.get("grupo", "A"))
        especialidad = str(data.get("especialidad", "general"))
        horas = float(data.get("horas_semana_totales") or 30)
        asistencia = float(data.get("asistencia_semanal") or 0)
        promedio = float(data.get("promedio") or 0)
        materias_reprobadas = int(data.get("materias_reprobadas") or 0)
        etiqueta_real = int(data.get("etiqueta_real", 1))  # 0: BAJO, 1: ALTO

        datos_estudiante = pd.DataFrame([{
            "grado": grado,
            "grupo": grupo,
            "especialidad": especialidad,
            "horas_semana_totales": horas,
            "asistencia_semanal": asistencia,
            "promedio": promedio,
            "materias_reprobadas": materias_reprobadas,
        }])

        cat_encoded = encoder.transform(datos_estudiante[CATEGORICAL_COLS])
        num_raw = datos_estudiante[NUMERIC_COLS].values
        X_combined = np.hstack([cat_encoded, num_raw])
        X_scaled = escalador.transform(X_combined)
        y_real = np.array([etiqueta_real])

        pesos_antes = modelo.layers[0].get_weights()[0][:4, 0].round(4).tolist()

        # Entrenamiento en 1 paso (Backpropagation y actualización de pesos sinápticos)
        hist = modelo.fit(X_scaled, y_real, epochs=1, verbose=0)
        loss_obtenido = float(hist.history["loss"][0])

        pesos_despues = modelo.layers[0].get_weights()[0][:4, 0].round(4).tolist()

        # Guardar modelo actualizado
        modelo.save(MODEL_FILE)

        return jsonify({
            "status": "ok",
            "mensaje": "¡La red neuronal ajustó sus pesos sinápticos con esta consulta!",
            "loss": round(loss_obtenido, 4),
            "pesos_antes": pesos_antes,
            "pesos_despues": pesos_despues,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predecir_lote", methods=["POST"])
def predecir_lote():
    """Procesa un archivo Excel o CSV con múltiples estudiantes y devuelve el reporte completo."""
    if not MODELO_CARGADO:
        return jsonify({"error": "El modelo no ha sido entrenado aún."}), 500

    if "archivo" not in request.files:
        return jsonify({"error": "No se subió ningún archivo."}), 400

    file = request.files["archivo"]
    if not file.filename:
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    try:
        if file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            return jsonify({"error": "Formato no válido. Sube un archivo .xlsx o .csv"}), 400

        # Normalizamos nombres de columnas (minúsculas y sin acentos)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Validamos columnas requeridas
        cols_requeridas = ["grado", "grupo", "asistencia_semanal", "promedio"]
        faltantes = [c for c in cols_requeridas if c not in df.columns]
        if faltantes:
            return jsonify({"error": f"Faltan columnas obligatorias: {faltantes}"}), 400

        # Rellenamos columnas opcionales o faltantes
        if "especialidad" not in df.columns:
            df["especialidad"] = "general"
        else:
            df["especialidad"] = df["especialidad"].fillna("general").astype(str).str.lower()
            df["especialidad"] = df["especialidad"].replace({
                "ric": "redes",
                "dgs": "software",
                "dsm": "general"
            })

        if "horas_semana_totales" not in df.columns:
            # Asignamos según grado si no vienen especificadas
            df["horas_semana_totales"] = df["grado"].apply(lambda g: 25 if int(g) == 10 else 35)

        if "materias_reprobadas" not in df.columns:
            df["materias_reprobadas"] = 0

        # Aseguramos tipos
        df["grado"] = df["grado"].astype(int)
        df["grupo"] = df["grupo"].astype(str).str.upper()
        df["horas_semana_totales"] = df["horas_semana_totales"].astype(float)
        df["asistencia_semanal"] = df["asistencia_semanal"].astype(float)
        df["promedio"] = df["promedio"].astype(float)
        df["materias_reprobadas"] = df["materias_reprobadas"].astype(int)

        # Preprocesamiento por lotes
        cat_encoded = encoder.transform(df[CATEGORICAL_COLS])
        num_raw = df[NUMERIC_COLS].values
        X_combined = np.hstack([cat_encoded, num_raw])
        X_scaled = escalador.transform(X_combined)

        # Predicción masiva
        probabilidades = modelo.predict(X_scaled, verbose=0)
        predicciones_idx = np.argmax(probabilidades, axis=1)

        # Añadimos resultados al DataFrame
        df["riesgo_predicho"] = ["ALTO" if p == 1 else "BAJO" for p in predicciones_idx]
        df["prob_bajo_pct"] = (probabilidades[:, 0] * 100).round(1)
        df["prob_alto_pct"] = (probabilidades[:, 1] * 100).round(1)

        total_alumnos = len(df)
        total_alto = int((predicciones_idx == 1).sum())
        total_bajo = int((predicciones_idx == 0).sum())

        # Agrupación por grado y grupo
        resumen_grupos = df.groupby(["grado", "grupo", "especialidad"]).agg(
            total=("riesgo_predicho", "count"),
            en_riesgo=("riesgo_predicho", lambda x: (x == "ALTO").sum())
        ).reset_index()
        resumen_grupos["pct_riesgo"] = (resumen_grupos["en_riesgo"] / resumen_grupos["total"] * 100).round(1)

        # Guardamos el archivo procesado en memoria / temporal para descarga
        resultado_json = {
            "total_alumnos": total_alumnos,
            "total_alto": total_alto,
            "total_bajo": total_bajo,
            "pct_alto": round((total_alto / total_alumnos * 100), 1) if total_alumnos > 0 else 0,
            "grupos": resumen_grupos.to_dict(orient="records"),
            "detalle": df[[
                "grado", "grupo", "especialidad", "horas_semana_totales", 
                "asistencia_semanal", "promedio", "materias_reprobadas", 
                "riesgo_predicho", "prob_alto_pct"
            ]].to_dict(orient="records")
        }

        # Guardamos en CSV temporal para exportar
        reporte_path = MODELS_DIR / "ultimo_reporte_lote.csv"
        df.to_csv(reporte_path, index=False)

        return jsonify(resultado_json)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error procesando lote: {str(e)}"}), 500


@app.route("/descargar_plantilla")
def descargar_plantilla():
    """Genera y descarga una plantilla Excel de ejemplo con la estructura correcta."""
    from io import BytesIO
    from flask import send_file

    datos_ejemplo = [
        {"grado": 1, "grupo": "A", "especialidad": "general", "horas_semana_totales": 35, "asistencia_semanal": 88.5, "promedio": 8.5, "materias_reprobadas": 0},
        {"grado": 1, "grupo": "B", "especialidad": "general", "horas_semana_totales": 35, "asistencia_semanal": 55.0, "promedio": 5.8, "materias_reprobadas": 2},
        {"grado": 4, "grupo": "A", "especialidad": "general", "horas_semana_totales": 35, "asistencia_semanal": 92.0, "promedio": 9.0, "materias_reprobadas": 0},
        {"grado": 7, "grupo": "C", "especialidad": "general", "horas_semana_totales": 35, "asistencia_semanal": 60.0, "promedio": 6.2, "materias_reprobadas": 1},
        {"grado": 10, "grupo": "A", "especialidad": "redes", "horas_semana_totales": 25, "asistencia_semanal": 30.0, "promedio": 5.1, "materias_reprobadas": 2},
        {"grado": 10, "grupo": "A", "especialidad": "software", "horas_semana_totales": 25, "asistencia_semanal": 94.0, "promedio": 9.5, "materias_reprobadas": 0},
        {"grado": 10, "grupo": "B", "especialidad": "software", "horas_semana_totales": 25, "asistencia_semanal": 70.0, "promedio": 7.2, "materias_reprobadas": 1},
    ]
    df_ejemplo = pd.DataFrame(datos_ejemplo)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_ejemplo.to_excel(writer, index=False, sheet_name="Estudiantes")
    output.seek(0)

    return send_file(
        output,
        download_name="plantilla_estudiantes_riesgo.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/descargar_reporte")
def descargar_reporte():
    """Descarga el último reporte generado en CSV."""
    reporte_path = MODELS_DIR / "ultimo_reporte_lote.csv"
    if not reporte_path.exists():
        return "No hay reporte generado aún.", 404
    return send_from_directory(MODELS_DIR, "ultimo_reporte_lote.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)


