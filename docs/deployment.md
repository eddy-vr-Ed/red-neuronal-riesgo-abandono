# Despliegue en servidor de compañero

Esta guia prepara una demo academica desplegable sin asumir Render, Railway ni otro proveedor. No se debe aplicar ninguna configuracion en el servidor hasta confirmar sus caracteristicas reales.

## Dictamen actual

Estado: listo con condiciones.

Condiciones pendientes:

- Confirmar sistema operativo, arquitectura y recursos del servidor.
- Ejecutar instalacion limpia en Linux con Python 3.12 y `pip check`.
- Medir memoria en el servidor real bajo carga de modelo, prediccion, lote y retroalimentacion.
- Configurar un directorio persistente para `MODEL_STORAGE_DIR` con permisos de escritura.
- Configurar HTTPS/proxy o confirmar que la demo sera solo por red privada.

## Datos necesarios del servidor

Antes de desplegar, pedir al compañero:

- Sistema operativo y arquitectura: por ejemplo Ubuntu 22.04/24.04 x86_64.
- Python disponible: version exacta de `python3 --version`, o confirmar soporte Docker.
- Recursos: RAM, CPU y espacio libre.
- Acceso: SSH, panel web, SFTP, CI/CD u otro mecanismo.
- URL: dominio, subdominio, IP, HTTPS y proxy existente.
- Directorio persistente: ruta y permisos para conservar el modelo actualizado.
- Servicio activo: systemd, supervisor, Docker Compose, panel del hosting u otra forma de reinicio.
- Politicas de archivos temporales: ubicacion y limpieza de reportes.

## Variables de entorno

Configurar variables en el sistema, panel del hosting o archivo de entorno cargado explicitamente por el servicio. No asumir que `.env` se carga solo.

| Variable | Uso | Valor recomendado para demo |
| --- | --- | --- |
| `MODEL_STORAGE_DIR` | Artefactos activos y persistentes del modelo | ruta persistente, fuera del repo si es posible |
| `REPORTS_DIR` | Reportes temporales por lote | ruta temporal o persistente con limpieza |
| `FEEDBACK_TOKEN` | Autoriza `/retroalimentar` | secreto generado en el servidor |
| `PORT` | Puerto HTTP | `5000` o el puerto del proxy |
| `WEB_CONCURRENCY` | Procesos Gunicorn | `1` |
| `GUNICORN_THREADS` | Hilos Gunicorn | `1` |
| `GUNICORN_TIMEOUT` | Timeout de requests | `120` |
| `MAX_UPLOAD_MB` | Tamano maximo de archivo | `2` |
| `MAX_BATCH_ROWS` | Filas maximas por lote | `100` |
| `TF_NUM_INTRAOP_THREADS` | Hilos internos TensorFlow | `1` |
| `TF_NUM_INTEROP_THREADS` | Hilos entre operaciones TensorFlow | `1` |
| `OMP_NUM_THREADS` | Hilos OpenMP | `1` |

El diseno actual usa un modelo mutable en memoria. Mantener `WEB_CONCURRENCY=1` y `GUNICORN_THREADS=1` evita que varios procesos intenten modificar y guardar el mismo modelo al mismo tiempo.

Generar token:

```bash
python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
```

## Instalacion Linux con Gunicorn

Ejemplo para Linux con Python 3.12:

```bash
cd /ruta/al/proyecto
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Variables para una prueba manual:

```bash
export MODEL_STORAGE_DIR=/ruta/persistente/red-neuronal-modelo
export REPORTS_DIR=/ruta/persistente/red-neuronal-reportes
export FEEDBACK_TOKEN="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)"
export PORT=5000
export WEB_CONCURRENCY=1
export GUNICORN_THREADS=1
export GUNICORN_TIMEOUT=120
export MAX_UPLOAD_MB=2
export MAX_BATCH_ROWS=100
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1
```

Arranque:

```bash
gunicorn app:app --config gunicorn.conf.py
```

Comprobacion:

```bash
curl -f "http://127.0.0.1:$PORT/health"
curl -X POST "http://127.0.0.1:$PORT/predecir" \
  -H "Content-Type: application/json" \
  -d '{"grado":1,"grupo":"A","especialidad":"general","horas_semana_totales":35,"asistencia_semanal":88,"promedio":8.5,"materias_reprobadas":0}'
```

## Plantilla systemd

No aplicarla hasta conocer rutas y usuario reales. Guardar como `/etc/systemd/system/red-neuronal.service` solo si el servidor usa systemd:

```ini
[Unit]
Description=Red Neuronal Riesgo Escolar
After=network.target

[Service]
User=USUARIO_DEL_SERVIDOR
Group=USUARIO_DEL_SERVIDOR
WorkingDirectory=/ruta/al/proyecto
Environment="MODEL_STORAGE_DIR=/ruta/persistente/red-neuronal-modelo"
Environment="REPORTS_DIR=/ruta/persistente/red-neuronal-reportes"
Environment="FEEDBACK_TOKEN=CAMBIAR_EN_EL_SERVIDOR"
Environment="PORT=5000"
Environment="WEB_CONCURRENCY=1"
Environment="GUNICORN_THREADS=1"
Environment="GUNICORN_TIMEOUT=120"
Environment="MAX_UPLOAD_MB=2"
Environment="MAX_BATCH_ROWS=100"
Environment="TF_NUM_INTRAOP_THREADS=1"
Environment="TF_NUM_INTEROP_THREADS=1"
Environment="OMP_NUM_THREADS=1"
ExecStart=/ruta/al/proyecto/.venv/bin/gunicorn app:app --config gunicorn.conf.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Activacion, si ya fue revisada:

```bash
sudo systemctl daemon-reload
sudo systemctl enable red-neuronal
sudo systemctl start red-neuronal
sudo systemctl status red-neuronal
```

## Plantilla Nginx

Usar solo si el servidor ya maneja Nginx como proxy. Ajustar dominio, certificados y puerto:

```nginx
server {
    listen 80;
    server_name ejemplo.com;

    client_max_body_size 2m;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

Si se expone a internet, habilitar HTTPS antes de compartir la URL.

## Artefactos y compatibilidad

Metadatos inspeccionados localmente:

- `models/modelo_riesgo.keras`: `keras_version` = `3.16.0.dev2026100304`, `date_saved` = `2026-10-08@00:02:06`.
- `models/escalador.pkl`: `_sklearn_version` = `1.9.1`.
- `models/encoders.pkl`: `_sklearn_version` = `1.9.1`.
- Entrada del modelo cargado: `(None, 15)`.
- Salida del modelo cargado: `(None, 2)`.
- Encoder: 11 columnas one-hot (`grado`, `grupo`, `especialidad`) mas 4 numericas, total 15 features.

El entorno local disponible no coincide con el objetivo de despliegue: se ejecuto en Windows con Python `3.14.5`, TensorFlow `2.22.0-rc0`, `tf.keras` `3.16.0.dev2026100304`, scikit-learn `1.9.1`, pandas `3.0.6`, numpy `2.5.3`. En ese entorno los artefactos cargaron y las pruebas pasaron, pero la instalacion limpia Linux/Python 3.12 queda pendiente en el servidor real.

`requirements.txt` fija Python de proyecto mediante `.python-version` `3.12.11` y usa `tensorflow-cpu==2.21.0`, `keras==3.15.1`, `scikit-learn==1.9.1`. En Linux se debe ejecutar `pip check` y cargar realmente modelo, escalador y encoder antes de compartir la URL.

## Verificaciones ejecutadas localmente

Ejecutado en Windows local, usando almacenamiento temporal para no sobrescribir artefactos originales:

- `python -m py_compile app.py src\config.py src\model\storage.py src\prediction\preprocessing.py src\prediction\predict.py tests\test_project.py`: OK.
- `node --check static\js\main.js`: OK.
- `python -m pip check`: OK en el Python local activo.
- `python -m unittest discover -s tests -v`: 28 pruebas OK.
- Carga real de modelo, scaler y encoder: OK.
- Coherencia de features/categorias/dimension de entrada/salida: OK.
- `/`, `/grafica`, `/health`, prediccion individual, lote CSV/XLSX, plantilla, reporte y retroalimentacion autorizada: OK por test client Flask.
- Rechazo de retroalimentacion sin token o con token incorrecto: OK.
- Prediccion no modifica pesos: OK.
- Retroalimentacion modifica pesos: OK.
- Persistencia tras proceso independiente con el mismo `MODEL_STORAGE_DIR`: OK.
- Guardado atomico con rollback de pesos en memoria ante fallo simulado de guardado: OK.

No ejecutado:

- Instalacion limpia en Linux con Python 3.12.
- Gunicorn real en Linux.
- Nginx/systemd en servidor real.
- Medicion real de memoria en el servidor final.

## Medicion de memoria

Ejecutar en Linux para basar la RAM recomendada en datos reales:

```bash
python - <<'PY'
from pathlib import Path
from io import BytesIO
import os
import pandas as pd

def rss_mb():
    status = Path("/proc/self/status").read_text()
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return round(int(line.split()[1]) / 1024, 1)
    return None

print("mem_start_hwm_mb", rss_mb())
import app
print("mem_after_load_hwm_mb", rss_mb())

payload = {
    "grado": 1,
    "grupo": "A",
    "especialidad": "general",
    "horas_semana_totales": 35,
    "asistencia_semanal": 88,
    "promedio": 8.5,
    "materias_reprobadas": 0,
}
client = app.app.test_client()
print("predict", client.post("/predecir", json=payload).status_code)
print("mem_after_predict_hwm_mb", rss_mb())

df = pd.DataFrame([payload] * 20)
csv_bytes = df.to_csv(index=False).encode("utf-8")
print("batch_csv", client.post(
    "/predecir_lote",
    data={"archivo": (BytesIO(csv_bytes), "lote.csv")},
    content_type="multipart/form-data",
).status_code)
print("mem_after_batch_hwm_mb", rss_mb())

print("feedback", client.post(
    "/retroalimentar",
    json={**payload, "etiqueta_real": 0},
    headers={"X-Feedback-Token": os.environ["FEEDBACK_TOKEN"]},
).status_code)
print("mem_after_feedback_hwm_mb", rss_mb())
PY
```

Hasta medir en el servidor real, no afirmar que una cantidad de RAM es suficiente. Como punto de partida prudente para TensorFlow CPU, pedir al menos 1 GB de RAM disponible y validar con la medicion anterior.

## Respaldo y restauracion

Respaldo:

```bash
tar -czf red-neuronal-model-backup.tar.gz -C "$MODEL_STORAGE_DIR" modelo_riesgo.keras escalador.pkl encoders.pkl
```

Restauracion:

```bash
mkdir -p "$MODEL_STORAGE_DIR"
tar -xzf red-neuronal-model-backup.tar.gz -C "$MODEL_STORAGE_DIR"
```

Los reportes de `REPORTS_DIR` son temporales. Definir una politica simple: borrar al terminar la demo o programar limpieza diaria si la URL permanece activa.

## Checklist antes de compartir URL

- `/health` responde `200` y muestra `model_loaded: true`.
- La pagina principal carga sin exponer `FEEDBACK_TOKEN`.
- `/grafica` devuelve PNG.
- Prediccion individual devuelve BAJO/ALTO y `aprendizaje_realizado: false`.
- CSV y XLSX generan `report_id` y descarga de reporte.
- `/retroalimentar` devuelve `401` sin token y con token incorrecto.
- `/retroalimentar` con token correcto y etiqueta real devuelve `aprendizaje_realizado: true`.
- Reiniciar el servicio conserva el modelo actualizado en `MODEL_STORAGE_DIR`.
- Hay respaldo del modelo antes de retirar o reiniciar la demo.
- Se monitoreo memoria durante la prueba final.
