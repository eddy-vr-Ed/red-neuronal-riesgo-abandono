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

## Rama Ary
* Esta es la rama en la que trabajare