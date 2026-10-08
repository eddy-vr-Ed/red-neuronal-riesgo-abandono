# Arquitectura del proyecto

## 1. Arquitectura general

El proyecto separa responsabilidades en cuatro bloques:

- `src/data/`: generacion, inspeccion, preparacion y normalizacion de datos.
- `src/model/`: definicion, entrenamiento, evaluacion y guardado de la red neuronal.
- `src/prediction/`: inferencia individual por linea de comandos.
- `app.py`, `templates/` y `static/`: aplicacion Flask e interfaz web.

`src/config.py` concentra rutas, columnas de entrada, variable objetivo y parametros compartidos para evitar valores duplicados en varios archivos.

## 2. Flujo de entrenamiento

```text
data/raw/estudiantes_por_grado.csv
    ↓
src.data.prepare.load_and_encode_data()
    ↓
train_test_split sobre datos crudos
    ↓
OneHotEncoder ajustado con entrenamiento
    ↓
StandardScaler ajustado con entrenamiento
    ↓
src.model.architecture.build_model()
    ↓
Entrenamiento Keras
    ↓
models/modelo_riesgo.keras
models/escalador.pkl
models/encoders.pkl
models/training_history.json
models/curvas_aprendizaje.png
```

El `StandardScaler` se ajusta solo con datos de entrenamiento y transforma entrenamiento/prueba por separado. El `OneHotEncoder` tambien se ajusta solo con entrenamiento para reducir riesgo de data leakage.

## 3. Flujo de prediccion individual

La interfaz web envia un JSON a `/predecir` con:

- `grado`
- `grupo`
- `especialidad`
- `horas_semana_totales`
- `asistencia_semanal`
- `promedio`
- `materias_reprobadas`

Flask construye un `DataFrame`, aplica `encoders.pkl`, concatena las variables numericas, aplica `escalador.pkl` y ejecuta `modelo_riesgo.keras`. La respuesta contiene la clase estimada BAJO/ALTO y las probabilidades de la capa Softmax.

## 4. Flujo de prediccion por lote

`/predecir_lote` acepta archivos `.xlsx` o `.csv`. El backend:

1. Lee el archivo con Pandas.
2. Normaliza nombres de columnas.
3. Valida columnas obligatorias.
4. Completa columnas opcionales cuando faltan.
5. Aplica el mismo encoder, escalador y modelo que la prediccion individual.
6. Devuelve resumen por grupo y detalle por estudiante.
7. Genera `models/ultimo_reporte_lote.csv` para descarga.

Ese reporte es una salida temporal de ejecucion, no un artefacto fuente.

## 5. Rol de Flask

`app.py` carga los artefactos del modelo al iniciar y expone:

- `/`: renderiza `templates/index.html`.
- `/grafica`: sirve la imagen de curvas de aprendizaje.
- `/predecir`: inferencia individual.
- `/predecir_lote`: inferencia masiva.
- `/descargar_plantilla`: plantilla `.xlsx`.
- `/descargar_reporte`: ultimo reporte generado.
- `/retroalimentar`: endpoint experimental de ajuste online.

## 6. Rol de templates y archivos estaticos

- `templates/index.html` define la estructura visual y los formularios.
- `static/js/main.js` maneja seleccion de grado/grupo, validaciones, llamadas `fetch` a Flask y renderizado de resultados.
- `static/css/style.css` contiene el estilo visual de la interfaz.

El endpoint `/retroalimentar` existe en backend, pero no forma parte del flujo principal visible del frontend.

## 7. Rol de los artefactos del modelo

- `modelo_riesgo.keras`: red neuronal entrenada.
- `escalador.pkl`: `StandardScaler` usado para estandarizar el vector de entrada.
- `encoders.pkl`: `OneHotEncoder` usado para variables categoricas.
- `training_history.json`: historial de entrenamiento para evidencia academica.
- `curvas_aprendizaje.png`: grafica de entrenamiento.

## 8. Limitaciones

- El dataset es sintetico y no representa comportamiento real de estudiantes.
- La precision del modelo depende de reglas artificiales usadas para generar datos.
- Las predicciones son demostrativas y no deben usarse para tomar decisiones academicas.
- Los artefactos serializados pueden requerir versiones compatibles de Scikit-learn, TensorFlow y Keras.
- La retroalimentacion online es experimental y puede modificar el modelo guardado si se integra al frontend.
