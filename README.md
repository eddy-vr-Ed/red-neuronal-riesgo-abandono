# Red Neuronal - Predicción de Riesgo de Abandono

Proyecto universitario de Inteligencia Artificial que utiliza una red neuronal para clasificar el riesgo de abandono académico de un estudiante en dos categorías: **BAJO** y **ALTO**.

## Objetivo

A partir de cuatro características académicas:

- Horas de estudio por semana.
- Porcentaje de asistencia.
- Promedio académico.
- Número de materias reprobadas.

la red neuronal genera una clasificación de riesgo y muestra la probabilidad estimada de cada clase.

> **Nota académica:** el dataset es sintético y fue creado para fines educativos. Las predicciones no deben interpretarse como una evaluación real de un estudiante.

## Arquitectura del proyecto

```text
RedNeuronal/
├── data/
│   └── raw/
│       └── estudiantes_200.csv
├── docs/
│   └── arquitectura.md
├── models/
│   ├── escalador.pkl
│   └── modelo_riesgo.keras
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── generate.py
│   │   ├── inspect.py
│   │   ├── normalize.py
│   │   └── prepare.py
│   ├── model/
│   │   ├── architecture.py
│   │   └── train.py
│   └── prediction/
│       └── predict.py
├── tests/
│   └── test_project.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Flujo del sistema

```text
Dataset CSV
    ↓
Inspección
    ↓
Preparación y división
    ↓
Estandarización
    ↓
Red neuronal
    ↓
Evaluación
    ↓
Modelo + escalador
    ↓
Predicción de un estudiante
```

## Requisitos

Se recomienda que los integrantes del equipo utilicen la misma versión de Python y trabajen dentro de un entorno virtual.

### 1. Crear entorno virtual

```bash
python -m venv .venv
```

### 2. Activar entorno en Windows

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecución

### Revisar el dataset

```bash
python -m src.data.inspect
```

### Preparar los datos

```bash
python -m src.data.prepare
```

### Ver la normalización

```bash
python -m src.data.normalize
```

### Entrenar el modelo

```bash
python -m src.model.train
```

Esto actualiza los artefactos:

- `models/modelo_riesgo.keras`
- `models/escalador.pkl`

### Realizar una predicción

```bash
python -m src.prediction.predict
```

### Ejecutar pruebas básicas

```bash
python -m unittest discover -s tests -v
```

## Generar nuevamente el dataset

El dataset sintético puede regenerarse con:

```bash
python -m src.data.generate
```

**Importante:** al regenerar el dataset cambia la información utilizada para entrenar el modelo; después de hacerlo se recomienda volver a ejecutar el entrenamiento.

## Trabajo colaborativo con GitHub

La rama `main` debe conservar una versión estable del proyecto. Cada integrante debe trabajar en una rama propia y posteriormente integrar sus cambios mediante Pull Request.

Ejemplo:

```text
main
├── feature/datos
├── feature/modelo
├── feature/prediccion
└── feature/documentacion
```

Antes de comenzar una tarea:

```bash
git checkout main
git pull origin main
git checkout -b feature/nombre-de-la-tarea
```

Al terminar:

```bash
git add .
git commit -m "feat: descripcion del cambio"
git push -u origin feature/nombre-de-la-tarea
```

## Integrantes

* Maria Fernanda Osorio Landa: Líder Técnico/ Apoyo a desarrollador principal
* Aracely Hernández Cedillo: Desarrollador principal
* Miguel Ángel Carrillo Hernández: Investigador / QA
* Jesús Eduardo Vazquez Rodriguez: Analista / apoyo de QA
* Maria Guadalupe Herrera Rafael: Oradora principal

## Fuentes de consulta

Durante la investigación y desarrollo del proyecto se consultaron las siguientes fuentes:

### Redes neuronales

* Video de consulta sobre redes neuronales: https://www.youtube.com/watch?v=M6oDiCQCins
* Video de consulta sobre redes neuronales: https://www.youtube.com/watch?v=6vwfT3-mBBw
* Video de consulta sobre redes neuronales: https://www.youtube.com/watch?v=xSjlvulOiQY

### Aplicación de redes neuronales en la industria

* Amazon Web Services (AWS). *¿Qué es el OCR? - Explicación del reconocimiento óptico de caracteres*. https://aws.amazon.com/es/what-is/ocr/
* Fuente de consulta proporcionada durante la investigación: https://share.google/AFVKok3mHEx0YQNBS

### Tecnologías y librerías utilizadas

* BBVA. *TensorFlow: la biblioteca de código abierto de Google para acelerar la adopción de la IA*. https://www.bbva.com/es/innovacion/tensorflow-la-biblioteca-de-codigo-abierto-de-google-para-acelerar-la-adopcion-de-la-ia/
* ENAE. *NumPy*. https://www.enae.es/blog/numpy
* NVIDIA. *Pandas Python*. https://www.nvidia.com/en-us/glossary/pandas-python/
* Scikit-learn. Sitio oficial. https://scikit-learn.org/stable/
* SciPy. Sitio oficial. https://scipy.org/es/faq/
* Liora. *Joblib: What is this Python library and how do I use it?* https://liora.io/en/joblib-what-is-this-python-library-how-do-i-use-it

## Licencias y uso de recursos

* El código fuente de este proyecto fue desarrollado por los integrantes del equipo con fines académicos y educativos.
* El archivo `data/raw/estudiantes_200.csv` contiene datos sintéticos generados para este proyecto. No contiene información personal ni datos reales de estudiantes.
* Las librerías y herramientas utilizadas en el proyecto pertenecen a sus respectivos autores y organizaciones, y se utilizan de acuerdo con las licencias correspondientes:

  * TensorFlow / Keras
  * NumPy
  * Pandas
  * Scikit-learn
  * SciPy
  * Joblib
* Las fuentes de consulta utilizadas para la investigación se encuentran indicadas en la sección **Fuentes de consulta** de este README.

## Uso de herramientas de inteligencia artificial

Durante el desarrollo de este proyecto se utilizaron herramientas de inteligencia artificial como apoyo académico y técnico.

La inteligencia artificial se utilizó principalmente para:

* Resolver dudas relacionadas con Python, TensorFlow y las librerías utilizadas.
* Apoyar en la explicación de conceptos sobre redes neuronales.
* Proponer y revisar fragmentos de código.
* Apoyar en la identificación y solución de errores durante las pruebas.
* Orientar sobre la organización y documentación del proyecto.

Las decisiones sobre la estructura del proyecto, los datos utilizados, la preparación y división del conjunto de datos, la arquitectura del modelo, las pruebas y la integración final fueron revisadas y realizadas por los integrantes del equipo.

El conjunto de datos utilizado en el prototipo es sintético y fue generado específicamente con fines educativos. El modelo y sus resultados fueron ejecutados y comprobados por el equipo antes de integrarlos al proyecto.

La herramienta de inteligencia artificial se utilizó como apoyo durante el proceso de aprendizaje y desarrollo, y no como sustituto de la revisión y participación de los integrantes del equipo.