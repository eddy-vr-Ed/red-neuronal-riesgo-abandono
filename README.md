# Red Neuronal - Prediccion de Riesgo de Abandono Escolar

Proyecto universitario de Inteligencia Artificial que utiliza una red neuronal para clasificar el riesgo de abandono academico de estudiantes. La version actual funciona como aplicacion web Flask, permite prediccion individual y procesamiento por lote, y usa un dataset sintetico ampliado con variables academicas e institucionales.

> Nota academica: el dataset es sintetico y fue construido para fines educativos. El sistema no debe usarse como evaluacion real de estudiantes ni como herramienta institucional de decision.

## Objetivo

Clasificar cada registro en una de dos clases:

- BAJO: menor riesgo estimado de abandono.
- ALTO: mayor riesgo estimado de abandono.

El modelo trabaja con estas variables actuales:

- grado
- grupo
- especialidad
- horas_semana_totales
- asistencia_semanal
- promedio
- materias_reprobadas

## Arquitectura real del proyecto

```text
red-neuronal-riesgo-abandono/
├── app.py                    # Aplicacion Flask y endpoints de prediccion
├── data/raw/                 # CSV fuente y dataset sintetico actual
├── docs/                     # Documentacion tecnica
├── models/                   # Modelo, scaler, encoder y evidencias de entrenamiento
├── src/config.py             # Rutas, columnas y parametros centrales
├── src/data/                 # Generacion, inspeccion, preparacion y normalizacion
├── src/model/                # Arquitectura y entrenamiento de la red neuronal
├── src/prediction/           # Predictor CLI individual
├── static/                   # CSS y JavaScript del frontend
├── templates/                # Plantillas HTML Flask
└── tests/                    # Pruebas basicas del proyecto
```

## Flujo del sistema

```text
Dataset sintetico
    ↓
Preprocesamiento
    ↓
One-Hot Encoding
    ↓
Escalado
    ↓
Red neuronal
    ↓
Prediccion individual o por lote
    ↓
Resultado BAJO / ALTO
```

## Tecnologias

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Joblib
- Matplotlib
- HTML, CSS y JavaScript
- OpenPyXL para lectura y escritura de archivos `.xlsx`

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecucion de scripts

```bash
python -m src.data.inspect
python -m src.data.prepare
python -m src.data.normalize
python -m src.model.train
python -m unittest discover -s tests -v
```

## Ejecucion de la app Flask

```bash
python app.py
```

La aplicacion queda disponible normalmente en:

```text
http://127.0.0.1:5000
```

## Prediccion por linea de comandos

```bash
python -m src.prediction.predict
```

El predictor CLI carga los mismos artefactos que la aplicacion Flask:

- `models/modelo_riesgo.keras`
- `models/escalador.pkl`
- `models/encoders.pkl`

## Prediccion por lote

La interfaz permite cargar archivos con multiples estudiantes en estos formatos:

- `.xlsx`
- `.csv`

Columnas requeridas:

- `grado`
- `grupo`
- `asistencia_semanal`
- `promedio`

Columnas opcionales:

- `especialidad`
- `horas_semana_totales`
- `materias_reprobadas`

Si faltan columnas opcionales, la aplicacion asigna valores por defecto coherentes con el flujo actual.

## Artefactos generados

El entrenamiento genera o actualiza:

- `models/modelo_riesgo.keras`
- `models/escalador.pkl`
- `models/encoders.pkl`
- `models/training_history.json`
- `models/curvas_aprendizaje.png`

La prediccion por lote genera temporalmente:

- `models/ultimo_reporte_lote.csv`

Ese ultimo archivo es una salida de ejecucion y no debe tratarse como fuente del proyecto.

## Endpoints principales

- `/`: interfaz web.
- `/predecir`: prediccion individual desde JSON.
- `/predecir_lote`: prediccion masiva desde archivo `.xlsx` o `.csv`.
- `/descargar_plantilla`: plantilla Excel con columnas esperadas.
- `/descargar_reporte`: descarga del ultimo reporte por lote generado.
- `/retroalimentar`: endpoint experimental de ajuste online; no esta integrado como flujo principal del frontend.

## Flujo recomendado con GitHub

La rama `main` debe conservar una version estable. Para preparar cambios:

```bash
git checkout dev
git pull origin dev
git checkout -b fix/segunda-entrega
```

Antes de integrar:

```bash
python -m unittest discover -s tests -v
git status
git diff --stat
```

No se recomienda hacer merge directo a `main`; usa Pull Request y revisa que no haya cambios masivos por saltos de linea.

## Integrantes

- Maria Fernanda Osorio Landa: Lider tecnico / apoyo a desarrollador principal
- Aracely Hernandez Cedillo: Desarrollador principal
- Miguel Angel Carrillo Hernandez: Investigador / QA
- Jesus Eduardo Vazquez Rodriguez: Analista / apoyo de QA
- Maria Guadalupe Herrera Rafael: Oradora principal

## Limitaciones

- El dataset es sintetico.
- La precision depende de reglas artificiales de generacion y del conjunto de entrenamiento disponible.
- La salida BAJO / ALTO es demostrativa y no representa un diagnostico academico real.
- La compatibilidad de artefactos serializados puede depender de versiones de TensorFlow, Keras y Scikit-learn.
