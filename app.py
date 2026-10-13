"""Servidor Flask para exponer la red neuronal de riesgo de abandono."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from zipfile import BadZipFile
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from tensorflow import keras

from src.config import (
    ENCODERS_FILE,
    INITIAL_ENCODERS_FILE,
    INITIAL_MODEL_FILE,
    INITIAL_SCALER_FILE,
    MAX_BATCH_ROWS,
    MAX_UPLOAD_MB,
    MODEL_FILE,
    MODEL_STORAGE_DIR,
    MODELS_DIR,
    SCALER_FILE,
)
from src.model.storage import (
    active_artifacts_available,
    initialize_model_storage,
    save_model_atomically,
)
from src.prediction.preprocessing import preparar_matriz_modelo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

HISTORY_FILE = MODELS_DIR / "training_history.json"
REPORTS_DIR = Path(
    os.environ.get(
        "REPORTS_DIR",
        Path(tempfile.gettempdir()) / "red_neuronal_riesgo_reportes",
    )
).expanduser().resolve()
PREDICTION_REQUIRED_FIELDS = [
    "grado",
    "grupo",
    "horas_semana_totales",
    "asistencia_semanal",
    "promedio",
    "materias_reprobadas",
]
BATCH_REQUIRED_COLUMNS = [
    "grado",
    "grupo",
    "especialidad",
    "horas_semana_totales",
    "asistencia_semanal",
    "promedio",
    "materias_reprobadas",
]
VALID_GROUPS_BY_GRADE = {
    1: {"A", "B", "C", "D"},
    4: {"A", "B", "C"},
    7: {"A", "B", "C"},
    10: {"A", "B"},
}

modelo = None
escalador = None
encoder = None
MODELO_CARGADO = False
MODEL_LOCK = threading.RLock()


def _load_runtime_artifacts() -> bool:
    """Inicializa y carga modelo activo, escalador y codificadores."""
    global modelo, escalador, encoder, MODELO_CARGADO
    try:
        initialize_model_storage()
        escalador = joblib.load(SCALER_FILE)
        encoder = joblib.load(ENCODERS_FILE)
        modelo = keras.models.load_model(MODEL_FILE)
        MODELO_CARGADO = True
        logger.info("Modelo activo cargado desde %s", MODEL_FILE)
    except Exception:
        MODELO_CARGADO = False
        logger.exception("No se pudieron cargar el modelo o sus preprocesadores")
    return MODELO_CARGADO


_load_runtime_artifacts()


def _datos_estudiante_desde_json(data: dict) -> pd.DataFrame:
    """Valida y convierte el JSON de prediccion individual a un DataFrame."""
    if not isinstance(data, dict):
        raise ValueError("La solicitud debe enviarse en formato JSON.")

    faltantes = [campo for campo in PREDICTION_REQUIRED_FIELDS if data.get(campo) in (None, "")]
    if faltantes:
        raise ValueError(f"Faltan campos obligatorios: {', '.join(faltantes)}.")

    grado = _valor_entero(data["grado"], "grado")
    grupo = str(data["grupo"]).strip().upper()
    especialidad = str(data.get("especialidad") or "general").strip().lower()
    horas = _valor_float(data["horas_semana_totales"], "horas_semana_totales")
    asistencia = _valor_float(data["asistencia_semanal"], "asistencia_semanal")
    promedio = _valor_float(data["promedio"], "promedio")
    materias_reprobadas = _valor_entero(data["materias_reprobadas"], "materias_reprobadas")

    if grado not in {1, 4, 7, 10}:
        raise ValueError("El grado debe ser 1, 4, 7 o 10.")
    if not grupo:
        raise ValueError("El grupo es obligatorio.")
    if especialidad not in {"general", "redes", "software"}:
        raise ValueError("La especialidad debe ser general, redes o software.")
    _validar_grupo_especialidad(grado, grupo, especialidad)
    if horas <= 0:
        raise ValueError("Las horas por semana deben ser mayores a 0.")
    if not 0 <= asistencia <= 100:
        raise ValueError("La asistencia semanal debe estar entre 0 y 100.")
    if not 0 <= promedio <= 10:
        raise ValueError("El promedio debe estar entre 0 y 10.")
    if materias_reprobadas < 0:
        raise ValueError("Las materias reprobadas no pueden ser negativas.")

    return pd.DataFrame([{
        "grado": grado,
        "grupo": grupo,
        "especialidad": especialidad,
        "horas_semana_totales": horas,
        "asistencia_semanal": asistencia,
        "promedio": promedio,
        "materias_reprobadas": materias_reprobadas,
    }])


def _validar_grupo_especialidad(grado: int, grupo: str, especialidad: str) -> None:
    grupos_validos = VALID_GROUPS_BY_GRADE.get(grado, set())
    if grupo not in grupos_validos:
        raise ValueError(
            f"El grupo {grupo} no es valido para el grado {grado}. "
            f"Grupos validos: {', '.join(sorted(grupos_validos))}."
        )
    if grado == 10:
        if especialidad not in {"redes", "software"}:
            raise ValueError("Para grado 10 la especialidad debe ser redes o software.")
        if especialidad == "redes" and grupo != "A":
            raise ValueError("La especialidad redes solo esta disponible en grado 10 grupo A.")
    elif especialidad != "general":
        raise ValueError("Para grados 1, 4 y 7 la especialidad debe ser general.")


def _es_booleano(valor) -> bool:
    return isinstance(valor, (bool, np.bool_))


def _valor_float(valor, campo: str) -> float:
    if _es_booleano(valor):
        raise ValueError(f"El campo {campo} debe ser numerico, no booleano.")
    try:
        numero = float(valor)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"El campo {campo} debe tener un numero valido.") from exc
    if not np.isfinite(numero):
        raise ValueError(f"El campo {campo} debe ser un numero finito.")
    return numero


def _valor_entero(valor, campo: str) -> int:
    if _es_booleano(valor):
        raise ValueError(f"El campo {campo} debe ser entero, no booleano.")
    numero = _valor_float(valor, campo)
    if not numero.is_integer():
        raise ValueError(f"El campo {campo} debe ser un entero sin decimales.")
    return int(numero)


def _serie_contiene_booleanos(serie: pd.Series) -> bool:
    valores = serie.dropna()
    if valores.map(_es_booleano).any():
        return True
    return valores.astype(str).str.strip().str.lower().isin({
        "true",
        "false",
        "verdadero",
        "falso",
    }).any()


def _validar_y_preparar_lote(df: pd.DataFrame) -> pd.DataFrame:
    """Valida un archivo por lote y conserva solo las columnas usadas por el modelo."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    duplicadas = sorted(df.columns[df.columns.duplicated()].unique())
    if duplicadas:
        raise ValueError(f"El archivo contiene columnas duplicadas: {', '.join(duplicadas)}.")

    faltantes = [c for c in BATCH_REQUIRED_COLUMNS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(faltantes)}.")

    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(f"El archivo supera el limite de {MAX_BATCH_ROWS} filas.")

    df = df[BATCH_REQUIRED_COLUMNS].copy()
    columnas_con_vacios = [
        col for col in BATCH_REQUIRED_COLUMNS
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any()
    ]
    if columnas_con_vacios:
        raise ValueError(
            "El archivo contiene valores vacios en las columnas: "
            f"{', '.join(columnas_con_vacios)}. "
            "Complete esos datos antes de ejecutar el diagnostico."
        )

    numeric_cols = ["grado", "horas_semana_totales", "asistencia_semanal", "promedio", "materias_reprobadas"]
    columnas_booleanas = [col for col in numeric_cols if _serie_contiene_booleanos(df[col])]
    if columnas_booleanas:
        raise ValueError(
            "El archivo contiene valores booleanos en columnas numericas: "
            f"{', '.join(columnas_booleanas)}."
        )

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    columnas_invalidas = [
        col for col in numeric_cols
        if df[col].isna().any() or not np.isfinite(df[col]).all()
    ]
    if columnas_invalidas:
        raise ValueError(
            "El archivo contiene datos numericos invalidos en las columnas: "
            f"{', '.join(columnas_invalidas)}."
        )

    columnas_enteras = ["grado", "materias_reprobadas"]
    columnas_con_decimales = [
        col for col in columnas_enteras
        if (df[col] % 1 != 0).any()
    ]
    if columnas_con_decimales:
        raise ValueError(
            "El archivo contiene decimales en columnas que deben ser enteras: "
            f"{', '.join(columnas_con_decimales)}."
        )

    df["grado"] = df["grado"].astype(int)
    df["grupo"] = df["grupo"].astype(str).str.strip().str.upper()
    df["especialidad"] = df["especialidad"].astype(str).str.strip().str.lower().replace({
        "ric": "redes",
        "dgs": "software",
        "dsm": "general",
    })
    df["horas_semana_totales"] = df["horas_semana_totales"].astype(float)
    df["asistencia_semanal"] = df["asistencia_semanal"].astype(float)
    df["promedio"] = df["promedio"].astype(float)
    df["materias_reprobadas"] = df["materias_reprobadas"].astype(int)

    grados_invalidos = sorted(set(df.loc[~df["grado"].isin([1, 4, 7, 10]), "grado"]))
    if grados_invalidos:
        raise ValueError(f"El archivo contiene grados no validos: {grados_invalidos}.")
    if (df["horas_semana_totales"] <= 0).any():
        raise ValueError("Las horas por semana deben ser mayores a 0.")
    if ((df["asistencia_semanal"] < 0) | (df["asistencia_semanal"] > 100)).any():
        raise ValueError("La asistencia semanal debe estar entre 0 y 100.")
    if ((df["promedio"] < 0) | (df["promedio"] > 10)).any():
        raise ValueError("El promedio debe estar entre 0 y 10.")
    if (df["materias_reprobadas"] < 0).any():
        raise ValueError("Las materias reprobadas no pueden ser negativas.")
    if (~df["especialidad"].isin(["general", "redes", "software"])).any():
        raise ValueError("La especialidad debe ser general, redes o software.")
    for row in df[["grado", "grupo", "especialidad"]].itertuples(index=False):
        _validar_grupo_especialidad(int(row.grado), str(row.grupo), str(row.especialidad))

    return df


def _validar_etiqueta_real(data: dict) -> int:
    if not isinstance(data, dict) or data.get("etiqueta_real") in (None, ""):
        raise ValueError("La retroalimentacion requiere etiqueta_real con valor 0 o 1.")
    try:
        etiqueta_real = _valor_entero(data["etiqueta_real"], "etiqueta_real")
    except ValueError as exc:
        raise ValueError("La etiqueta_real debe ser 0 (BAJO) o 1 (ALTO), sin decimales.") from exc
    if etiqueta_real not in {0, 1}:
        raise ValueError("La etiqueta_real debe ser 0 (BAJO) o 1 (ALTO).")
    return etiqueta_real


def _feedback_autorizado() -> bool:
    token = os.environ.get("FEEDBACK_TOKEN")
    if not token:
        return False
    auth_header = request.headers.get("Authorization", "")
    header_token = request.headers.get("X-Feedback-Token")
    bearer_token = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else None
    return token in {header_token, bearer_token}


def _report_path(report_id: str) -> Path:
    try:
        safe_id = str(uuid.UUID(report_id))
    except (TypeError, ValueError) as exc:
        raise ValueError("Identificador de reporte invalido.") from exc
    return REPORTS_DIR / f"{safe_id}.csv"


@app.errorhandler(413)
def archivo_demasiado_grande(_error):
    return jsonify({"error": f"El archivo supera el limite de {MAX_UPLOAD_MB} MB."}), 413


@app.route("/")
def index():
    """Sirve la página web principal."""
    history = {}
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                history = json.load(file)
        except Exception:
            logger.exception("No se pudo leer el historial de entrenamiento")

    return render_template(
        "index.html",
        modelo_listo=MODELO_CARGADO,
        history=json.dumps(history),
    )


@app.route("/health")
def health():
    """Comprueba que el servicio y los artefactos ML estén disponibles."""
    model_loaded = bool(MODELO_CARGADO and modelo is not None and escalador is not None and encoder is not None)
    status_code = 200 if model_loaded and active_artifacts_available() else 503
    return jsonify({
        "status": "ok" if status_code == 200 else "unavailable",
        "model_loaded": model_loaded,
        "storage_dir": str(MODEL_STORAGE_DIR),
        "artifacts": {
            "model": MODEL_FILE.exists(),
            "scaler": SCALER_FILE.exists(),
            "encoders": ENCODERS_FILE.exists(),
            "initial_model": INITIAL_MODEL_FILE.exists(),
            "initial_scaler": INITIAL_SCALER_FILE.exists(),
            "initial_encoders": INITIAL_ENCODERS_FILE.exists(),
        },
    }), status_code


@app.route("/grafica")
def grafica():
    """Devuelve la imagen de las curvas de aprendizaje."""
    return send_from_directory(MODELS_DIR, "curvas_aprendizaje.png")


@app.route("/predecir", methods=["POST"])
def predecir():
    """Recibe datos individuales y devuelve una inferencia sin entrenar el modelo."""
    if not MODELO_CARGADO:
        return jsonify({"error": "El modelo no esta disponible."}), 500

    try:
        datos_estudiante = _datos_estudiante_desde_json(request.get_json(silent=True))
        X_scaled = preparar_matriz_modelo(datos_estudiante, encoder, escalador)

        with MODEL_LOCK:
            probabilidades = modelo.predict(X_scaled, verbose=0)
            primeros_pesos = modelo.layers[0].get_weights()[0][:5, 0].round(4).tolist()

        prob_bajo = float(probabilidades[0][0] * 100)
        prob_alto = float(probabilidades[0][1] * 100)
        prediccion_idx = int(np.argmax(probabilidades, axis=1)[0])
        riesgo_str = "ALTO" if prediccion_idx == 1 else "BAJO"

        return jsonify({
            "riesgo": riesgo_str,
            "probabilidad_bajo": f"{prob_bajo:.1f}%",
            "probabilidad_alto": f"{prob_alto:.1f}%",
            "pesos_ejemplo": primeros_pesos,
            "aprendizaje_realizado": False,
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("No se pudo procesar la prediccion individual")
        return jsonify({"error": "No se pudo procesar la prediccion."}), 400


@app.route("/retroalimentar", methods=["POST"])
def retroalimentar():
    """Ajusta pesos con una etiqueta real proporcionada por el usuario autorizado."""
    if not MODELO_CARGADO:
        return jsonify({"error": "El modelo no esta disponible."}), 500
    if not os.environ.get("FEEDBACK_TOKEN"):
        return jsonify({"error": "La retroalimentacion no esta habilitada en este entorno."}), 503
    if not _feedback_autorizado():
        return jsonify({"error": "No autorizado para retroalimentar el modelo."}), 401

    try:
        data = request.get_json(silent=True)
        etiqueta_real = _validar_etiqueta_real(data)
        datos_estudiante = _datos_estudiante_desde_json(data)
        X_scaled = preparar_matriz_modelo(datos_estudiante, encoder, escalador)
        y_real = np.array([etiqueta_real])

        with MODEL_LOCK:
            pesos_modelo_antes = modelo.get_weights()
            pesos_antes = modelo.layers[0].get_weights()[0][:4, 0].round(4).tolist()
            hist = modelo.fit(X_scaled, y_real, epochs=1, verbose=0)
            loss_obtenido = float(hist.history["loss"][0])
            pesos_despues = modelo.layers[0].get_weights()[0][:4, 0].round(4).tolist()
            try:
                save_model_atomically(modelo, MODEL_FILE)
            except Exception:
                modelo.set_weights(pesos_modelo_antes)
                logger.exception("Fallo el guardado del modelo; pesos en memoria restaurados")
                raise

        return jsonify({
            "status": "ok",
            "mensaje": "La red neuronal actualizo sus pesos con la etiqueta real proporcionada.",
            "loss": round(loss_obtenido, 4),
            "pesos_antes": pesos_antes,
            "pesos_despues": pesos_despues,
            "aprendizaje_realizado": True,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("No se pudo aplicar la retroalimentacion")
        return jsonify({"error": "No se pudo aplicar la retroalimentacion."}), 500


@app.route("/predecir_lote", methods=["POST"])
def predecir_lote():
    """Procesa un archivo XLSX o CSV y devuelve el reporte asociado a un id."""
    if not MODELO_CARGADO:
        return jsonify({"error": "El modelo no esta disponible."}), 500

    if "archivo" not in request.files:
        return jsonify({"error": "No se subio ningun archivo."}), 400

    file = request.files["archivo"]
    if not file.filename:
        return jsonify({"error": "Nombre de archivo vacio."}), 400

    try:
        filename = file.filename.lower()
        try:
            if filename.endswith(".xlsx"):
                df = pd.read_excel(file)
            elif filename.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                return jsonify({"error": "Formato no valido. Sube un archivo .xlsx o .csv"}), 400
        except (BadZipFile, UnicodeDecodeError, pd.errors.ParserError, ValueError, OSError):
            logger.exception("Archivo de lote vacio, corrupto o ilegible")
            return jsonify({"error": "El archivo esta vacio, corrupto o no se puede leer."}), 400

        if df.empty:
            return jsonify({"error": "El archivo no contiene registros para procesar."}), 400

        df = _validar_y_preparar_lote(df)
        X_scaled = preparar_matriz_modelo(df, encoder, escalador)

        with MODEL_LOCK:
            probabilidades = modelo.predict(X_scaled, verbose=0)

        predicciones_idx = np.argmax(probabilidades, axis=1)
        df["riesgo_predicho"] = ["ALTO" if p == 1 else "BAJO" for p in predicciones_idx]
        df["prob_bajo_pct"] = (probabilidades[:, 0] * 100).round(1)
        df["prob_alto_pct"] = (probabilidades[:, 1] * 100).round(1)

        total_alumnos = len(df)
        total_alto = int((predicciones_idx == 1).sum())
        total_bajo = int((predicciones_idx == 0).sum())
        resumen_grupos = df.groupby(["grado", "grupo", "especialidad"]).agg(
            total=("riesgo_predicho", "count"),
            en_riesgo=("riesgo_predicho", lambda x: (x == "ALTO").sum()),
        ).reset_index()
        resumen_grupos["pct_riesgo"] = (resumen_grupos["en_riesgo"] / resumen_grupos["total"] * 100).round(1)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_id = str(uuid.uuid4())
        df.to_csv(_report_path(report_id), index=False)

        return jsonify({
            "total_alumnos": total_alumnos,
            "total_alto": total_alto,
            "total_bajo": total_bajo,
            "pct_alto": round((total_alto / total_alumnos * 100), 1) if total_alumnos > 0 else 0,
            "grupos": resumen_grupos.to_dict(orient="records"),
            "detalle": df[[
                "grado", "grupo", "especialidad", "horas_semana_totales",
                "asistencia_semanal", "promedio", "materias_reprobadas",
                "riesgo_predicho", "prob_alto_pct",
            ]].to_dict(orient="records"),
            "report_id": report_id,
            "download_url": f"/descargar_reporte?id={report_id}",
            "aprendizaje_realizado": False,
        })

    except pd.errors.EmptyDataError:
        return jsonify({"error": "El archivo esta vacio o no tiene columnas."}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Error procesando prediccion por lote")
        return jsonify({"error": "Error procesando el archivo por lote."}), 500


@app.route("/descargar_plantilla")
def descargar_plantilla():
    """Genera y descarga una plantilla Excel de ejemplo con la estructura correcta."""
    from io import BytesIO

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
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/descargar_reporte")
def descargar_reporte():
    """Descarga un reporte generado por lote usando su identificador."""
    report_id = request.args.get("id")
    if not report_id:
        return "Falta el identificador del reporte.", 400
    try:
        reporte_path = _report_path(report_id)
    except ValueError as exc:
        return str(exc), 400
    if not reporte_path.exists():
        return "No hay reporte generado para ese identificador.", 404
    return send_file(
        reporte_path,
        as_attachment=True,
        download_name=f"reporte_riesgo_{report_id}.csv",
        mimetype="text/csv",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
