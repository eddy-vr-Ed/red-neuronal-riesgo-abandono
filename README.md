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
├── src/prediction/           # Prediccion individual y preprocesamiento compartido
│   ├── predict.py            # Predictor CLI individual
│   └── preprocessing.py      # One-Hot Encoding y escalado para Flask y CLI
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

Tambien usa `src/prediction/preprocessing.py`, la misma utilidad de One-Hot Encoding y escalado que utiliza Flask antes de enviar datos al modelo.

## Prediccion por lote

La interfaz permite cargar archivos con multiples estudiantes en estos formatos:

- `.xlsx`
- `.csv`

Columnas requeridas:

- `grado`
- `grupo`
- `especialidad`
- `horas_semana_totales`
- `asistencia_semanal`
- `promedio`
- `materias_reprobadas`

Si el archivo contiene columnas extra, como `riesgo_predicho` o `prob_alto_pct`, el sistema las ignora. Si existen valores vacios, NaN o infinitos en columnas requeridas, se muestra un error claro al usuario.

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

## Creditos y licencias de uso

El codigo fuente fue desarrollado por los integrantes del equipo con fines academicos y educativos. El dataset `data/raw/estudiantes_por_grado.csv` contiene datos sinteticos generados para este proyecto y no incluye informacion personal ni datos reales de estudiantes.

Las librerias utilizadas pertenecen a sus respectivos autores y organizaciones:

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Joblib
- Matplotlib
- OpenPyXL
- HTML, CSS y JavaScript

Las fuentes consultadas se encuentran en la seccion "Fuentes de consulta".

## Uso de herramientas de inteligencia artificial

Durante el desarrollo de este proyecto se utilizaron herramientas de inteligencia artificial como apoyo academico y tecnico.

La inteligencia artificial se utilizo principalmente para:

- Resolver dudas relacionadas con Python, Flask, TensorFlow/Keras y las librerias utilizadas.
- Apoyar en la explicacion de conceptos sobre redes neuronales.
- Proponer y revisar fragmentos de codigo.
- Apoyar en la identificacion y solucion de errores durante las pruebas.
- Orientar sobre la organizacion, documentacion y control de calidad del proyecto.

Las decisiones sobre la estructura del proyecto, los datos utilizados, la preparacion y division del conjunto de datos, la arquitectura del modelo, las pruebas y la integracion final fueron revisadas y realizadas por los integrantes del equipo.

El conjunto de datos utilizado es sintetico y fue generado especificamente con fines educativos. El modelo y sus resultados fueron ejecutados y comprobados por el equipo antes de integrarlos al proyecto.

La herramienta de inteligencia artificial se utilizo como apoyo durante el proceso de aprendizaje y desarrollo, y no como sustituto de la revision y participacion de los integrantes del equipo.

## Fuentes de consulta

### Redes neuronales

- Video de consulta sobre redes neuronales: https://www.youtube.com/watch?v=M6oDiCQCins
- Video de consulta sobre redes neuronales: https://www.youtube.com/watch?v=6vwfT3-mBBw
- Video de consulta sobre redes neuronales: https://www.youtube.com/watch?v=xSjlvulOiQY

### Aplicacion de redes neuronales en la industria

- Amazon Web Services (AWS). ¿Que es el OCR? - Explicacion del reconocimiento optico de caracteres. https://aws.amazon.com/es/what-is/ocr/
- Fuente de consulta proporcionada durante la investigacion: https://share.google/AFVKok3mHEx0YQNBS

### Tecnologias y librerias utilizadas

- BBVA. TensorFlow: la biblioteca de codigo abierto de Google para acelerar la adopcion de la IA. https://www.bbva.com/es/innovacion/tensorflow-la-biblioteca-de-codigo-abierto-de-google-para-acelerar-la-adopcion-de-la-ia/
- ENAE. NumPy. https://www.enae.es/blog/numpy
- NVIDIA. Pandas Python. https://www.nvidia.com/en-us/glossary/pandas-python/
- Scikit-learn. Sitio oficial. https://scikit-learn.org/stable/
- SciPy. Sitio oficial. https://scipy.org/es/faq/
- Liora. Joblib: What is this Python library and how do I use it? https://liora.io/en/joblib-what-is-this-python-library-and-how-do-i-use-it
- Flask. Documentacion oficial. https://flask.palletsprojects.com/
- OpenPyXL. Documentacion oficial. https://openpyxl.readthedocs.io/
- Matplotlib. Documentacion oficial. https://matplotlib.org/
