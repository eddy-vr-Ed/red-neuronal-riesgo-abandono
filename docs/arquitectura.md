# Arquitectura del proyecto

## 1. Principio de organización

El proyecto se reorganizó por **responsabilidad** en lugar de mantener todos los scripts, datos y artefactos en la raíz del repositorio.

## 2. Capas principales

### `data/`
Contiene los datos utilizados por el proyecto. La carpeta `raw/` conserva el dataset original que alimenta el proceso.

### `models/`
Contiene los artefactos generados por el entrenamiento:

- `modelo_riesgo.keras`: red neuronal entrenada.
- `escalador.pkl`: `StandardScaler` utilizado para transformar las características.

### `src/data/`
Agrupa las tareas relacionadas con los datos:

- `generate.py`: genera el dataset sintético.
- `inspect.py`: revisa la calidad y estructura del dataset.
- `prepare.py`: divide los datos en entrenamiento y prueba.
- `normalize.py`: demuestra la estandarización de las características.

### `src/model/`
Agrupa la lógica de aprendizaje automático:

- `architecture.py`: define las capas de la red neuronal.
- `train.py`: prepara los datos, entrena, evalúa y guarda los artefactos.

### `src/prediction/`
Contiene la lógica de inferencia para que el usuario introduzca los datos de un estudiante y obtenga el riesgo estimado.

### `tests/`
Contiene pruebas automáticas básicas sobre la estructura del dataset y la presencia de los artefactos necesarios.

### `docs/`
Conserva documentación técnica que no pertenece al código de ejecución.

## 3. Configuración centralizada

`src/config.py` concentra rutas, nombres de archivos, características, variable objetivo y parámetros de división para evitar rutas escritas directamente en varios scripts.

## 4. Ventajas para el equipo

Esta organización facilita:

- localizar rápidamente cada parte del proyecto;
- trabajar en paralelo sin modificar los mismos archivos innecesariamente;
- incorporar pruebas y documentación;
- subir el repositorio a GitHub con una estructura clara;
- explicar la arquitectura durante la presentación del proyecto.

## 5. Regla de modificación

Cuando se agregue funcionalidad nueva, primero debe identificarse la responsabilidad a la que pertenece. La lógica de datos no debe mezclarse con la lógica de predicción y la definición de la red debe mantenerse separada del proceso de entrenamiento.
