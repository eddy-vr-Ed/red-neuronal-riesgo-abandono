"""Pruebas del proyecto y de los flujos críticos de Flask."""

from io import BytesIO
import json
import numpy as np
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import pandas as pd

from src.config import (
    DATASET_FILE,
    FEATURES_RAW,
    INITIAL_ENCODERS_FILE,
    INITIAL_MODEL_FILE,
    INITIAL_SCALER_FILE,
    TARGET,
)


VALID_PAYLOAD = {
    "grado": 1,
    "grupo": "A",
    "especialidad": "general",
    "horas_semana_totales": 35,
    "asistencia_semanal": 88.0,
    "promedio": 8.5,
    "materias_reprobadas": 0,
}


def _batch_dataframe() -> pd.DataFrame:
    return pd.DataFrame([
        VALID_PAYLOAD,
        {
            "grado": 10,
            "grupo": "A",
            "especialidad": "redes",
            "horas_semana_totales": 25,
            "asistencia_semanal": 40.0,
            "promedio": 6.0,
            "materias_reprobadas": 2,
        },
    ])


class TestProjectFiles(unittest.TestCase):
    def test_dataset_exists(self):
        self.assertTrue(DATASET_FILE.exists(), "No existe el dataset de entrada.")

    def test_dataset_schema(self):
        datos = pd.read_csv(DATASET_FILE)
        self.assertEqual(list(datos.columns), FEATURES_RAW + [TARGET])
        self.assertGreater(len(datos), 0)
        self.assertEqual(datos.isnull().sum().sum(), 0)

    def test_initial_model_artifacts_exist(self):
        self.assertTrue(INITIAL_MODEL_FILE.exists(), "No existe el modelo inicial.")
        self.assertTrue(INITIAL_SCALER_FILE.exists(), "No existe el escalador inicial.")
        self.assertTrue(INITIAL_ENCODERS_FILE.exists(), "No existe el encoder inicial.")


class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._model_dir = tempfile.TemporaryDirectory()
        cls._reports_dir = tempfile.TemporaryDirectory()
        os.environ["FEEDBACK_TOKEN"] = "test-token"
        os.environ["REPORTS_DIR"] = cls._reports_dir.name

        import src.config as config

        config.MODEL_STORAGE_DIR = Path(cls._model_dir.name)
        config.MODEL_FILE = config.MODEL_STORAGE_DIR / "modelo_riesgo.keras"
        config.SCALER_FILE = config.MODEL_STORAGE_DIR / "escalador.pkl"
        config.ENCODERS_FILE = config.MODEL_STORAGE_DIR / "encoders.pkl"

        import app

        cls.app_module = app
        cls.client = app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls._model_dir.cleanup()
        cls._reports_dir.cleanup()

    def test_index_responds(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('id="feedback-form"', html)
        self.assertIn('id="feedback-token"', html)
        self.assertNotIn("test-token", html)

    def test_template_download_responds(self):
        response = self.client.get("/descargar_plantilla")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.mimetype,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    def test_graph_responds(self):
        response = self.client.get("/grafica")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")

    def test_health_reports_model_artifacts(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["artifacts"]["model"])
        self.assertTrue(data["artifacts"]["scaler"])
        self.assertTrue(data["artifacts"]["encoders"])

    def test_prediction_rejects_incomplete_payload(self):
        response = self.client.post("/predecir", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_prediction_rejects_invalid_values(self):
        payload = {**VALID_PAYLOAD, "asistencia_semanal": 120}
        response = self.client.post("/predecir", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("asistencia semanal", response.get_json()["error"].lower())

    def test_prediction_rejects_decimal_integer_fields(self):
        for field, value in [("grado", 1.5), ("materias_reprobadas", 1.2)]:
            with self.subTest(field=field):
                payload = {**VALID_PAYLOAD, field: value}
                response = self.client.post("/predecir", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("entero", response.get_json()["error"])

    def test_prediction_rejects_boolean_and_non_finite_numbers(self):
        payload = {**VALID_PAYLOAD, "grado": True}
        response = self.client.post("/predecir", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("booleano", response.get_json()["error"])

        payload = {**VALID_PAYLOAD, "promedio": float("nan")}
        response = self.client.post(
            "/predecir",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("finito", response.get_json()["error"])

    def test_prediction_accepts_valid_payload(self):
        response = self.client.post("/predecir", json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn(data["riesgo"], {"BAJO", "ALTO"})
        self.assertIn("probabilidad_bajo", data)
        self.assertIn("probabilidad_alto", data)
        self.assertFalse(data["aprendizaje_realizado"])

    def test_prediction_does_not_change_weights(self):
        before = [weights.copy() for weights in self.app_module.modelo.get_weights()]
        response = self.client.post("/predecir", json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        after = self.app_module.modelo.get_weights()
        for expected, actual in zip(before, after):
            self.assertTrue(np.array_equal(expected, actual))

    def test_batch_prediction_accepts_csv_and_downloads_own_report(self):
        csv_content = _batch_dataframe().to_csv(index=False)
        response = self.client.post(
            "/predecir_lote",
            data={"archivo": (BytesIO(csv_content.encode("utf-8")), "lote.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["total_alumnos"], 2)
        self.assertRegex(data["report_id"], r"^[0-9a-f-]{36}$")
        self.assertNotEqual(data["download_url"], "/descargar_reporte")

        download = self.client.get(data["download_url"])
        self.assertEqual(download.status_code, 200)
        self.assertIn(b"riesgo_predicho", download.data)

    def test_batch_prediction_accepts_xlsx(self):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _batch_dataframe().to_excel(writer, index=False)
        output.seek(0)

        response = self.client.post(
            "/predecir_lote",
            data={"archivo": (output, "lote.xlsx")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["total_alumnos"], 2)

    def test_batch_prediction_rejects_empty_required_values(self):
        csv_content = (
            "grado,grupo,especialidad,horas_semana_totales,asistencia_semanal,"
            "promedio,materias_reprobadas,riesgo_predicho,prob_alto_pct\n"
            "1,A,general,35,,8.5,,BAJO,12.3\n"
        )
        response = self.client.post(
            "/predecir_lote",
            data={"archivo": (BytesIO(csv_content.encode("utf-8")), "lote.csv")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        error = response.get_json()["error"]
        self.assertIn("valores vacios", error)
        self.assertIn("asistencia_semanal", error)
        self.assertIn("materias_reprobadas", error)
        self.assertNotIn("Cannot convert non-finite values", error)

    def test_batch_prediction_rejects_decimal_in_integer_columns(self):
        csv_content = (
            "grado,grupo,especialidad,horas_semana_totales,asistencia_semanal,promedio,materias_reprobadas\n"
            "1.5,A,general,35,88,8.5,0\n"
        )
        response = self.client.post(
            "/predecir_lote",
            data={"archivo": (BytesIO(csv_content.encode("utf-8")), "lote.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("decimales", response.get_json()["error"])

    def test_batch_prediction_rejects_non_positive_hours(self):
        df = _batch_dataframe()
        df.loc[0, "horas_semana_totales"] = 0
        response = self.client.post(
            "/predecir_lote",
            data={"archivo": (BytesIO(df.to_csv(index=False).encode("utf-8")), "lote.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("mayores a 0", response.get_json()["error"])

    def test_batch_prediction_rejects_corrupt_xlsx(self):
        response = self.client.post(
            "/predecir_lote",
            data={"archivo": (BytesIO(b"no es un xlsx valido"), "lote.xlsx")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("corrupto", response.get_json()["error"])

    def test_batch_validator_rejects_duplicate_columns(self):
        df = pd.DataFrame(
            [[1, 1, "A", "general", 35, 88, 8.5, 0]],
            columns=[
                "grado",
                " grado ",
                "grupo",
                "especialidad",
                "horas_semana_totales",
                "asistencia_semanal",
                "promedio",
                "materias_reprobadas",
            ],
        )
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            self.app_module._validar_y_preparar_lote(df)

    def test_feedback_rejects_missing_label(self):
        response = self.client.post(
            "/retroalimentar",
            json=VALID_PAYLOAD,
            headers={"X-Feedback-Token": "test-token"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("etiqueta_real", response.get_json()["error"])

    def test_feedback_rejects_invalid_label(self):
        response = self.client.post(
            "/retroalimentar",
            json={**VALID_PAYLOAD, "etiqueta_real": 2},
            headers={"X-Feedback-Token": "test-token"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("0 (BAJO) o 1 (ALTO)", response.get_json()["error"])

    def test_feedback_requires_authorization(self):
        response = self.client.post(
            "/retroalimentar",
            json={**VALID_PAYLOAD, "etiqueta_real": 0},
        )
        self.assertEqual(response.status_code, 401)

    def test_feedback_rejects_wrong_token(self):
        response = self.client.post(
            "/retroalimentar",
            json={**VALID_PAYLOAD, "etiqueta_real": 0},
            headers={"X-Feedback-Token": "wrong-token"},
        )
        self.assertEqual(response.status_code, 401)

    def test_feedback_rejects_decimal_label(self):
        response = self.client.post(
            "/retroalimentar",
            json={**VALID_PAYLOAD, "etiqueta_real": 0.5},
            headers={"X-Feedback-Token": "test-token"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("sin decimales", response.get_json()["error"])

    def test_feedback_accepts_valid_label_and_updates_weights(self):
        before = self.app_module.modelo.layers[0].get_weights()[0].copy()
        response = self.client.post(
            "/retroalimentar",
            json={**VALID_PAYLOAD, "etiqueta_real": 0},
            headers={"X-Feedback-Token": "test-token"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["aprendizaje_realizado"])
        self.assertIn("loss", data)
        after = self.app_module.modelo.layers[0].get_weights()[0]
        self.assertTrue(np.any(before != after))

    def test_feedback_save_failure_rolls_back_in_memory(self):
        before = [weights.copy() for weights in self.app_module.modelo.get_weights()]
        with patch.object(
            self.app_module,
            "save_model_atomically",
            side_effect=RuntimeError("fallo simulado"),
        ):
            response = self.client.post(
                "/retroalimentar",
                json={**VALID_PAYLOAD, "etiqueta_real": 0},
                headers={"X-Feedback-Token": "test-token"},
            )

        self.assertEqual(response.status_code, 500)
        after = self.app_module.modelo.get_weights()
        for expected, actual in zip(before, after):
            self.assertTrue(np.array_equal(expected, actual))


class TestPersistentModelFlow(unittest.TestCase):
    def test_feedback_persists_across_new_process(self):
        first_process = textwrap.dedent(
            """
            import json
            import numpy as np
            from pathlib import Path

            import app

            payload = {
                "grado": 1,
                "grupo": "A",
                "especialidad": "general",
                "horas_semana_totales": 35,
                "asistencia_semanal": 88.0,
                "promedio": 8.5,
                "materias_reprobadas": 0,
            }
            client = app.app.test_client()
            before = app.modelo.layers[0].get_weights()[0].copy()
            pred = client.post("/predecir", json=payload)
            fb = client.post(
                "/retroalimentar",
                json={**payload, "etiqueta_real": 0},
                headers={"X-Feedback-Token": "persist-token"},
            )
            after = app.modelo.layers[0].get_weights()[0]
            np.save(Path(__import__("os").environ["MODEL_STORAGE_DIR"]) / "after_feedback.npy", after)
            pred2 = client.post("/predecir", json=payload)
            print(json.dumps({
                "pred_status": pred.status_code,
                "feedback_status": fb.status_code,
                "pred2_status": pred2.status_code,
                "changed": bool(np.any(before != after)),
            }))
            """
        )
        second_process = textwrap.dedent(
            """
            import json
            import numpy as np
            import os
            from pathlib import Path

            import app
            from src.config import MODEL_FILE

            expected = np.load(Path(os.environ["MODEL_STORAGE_DIR"]) / "after_feedback.npy")
            loaded = app.modelo.layers[0].get_weights()[0]
            print(json.dumps({
                "preserved": bool(np.array_equal(expected, loaded)),
                "model_file_exists": MODEL_FILE.exists(),
            }))
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["MODEL_STORAGE_DIR"] = tmpdir
            env["FEEDBACK_TOKEN"] = "persist-token"
            first = subprocess.run(
                [sys.executable, "-c", first_process],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.getcwd(),
            )
            second = subprocess.run(
                [sys.executable, "-c", second_process],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.getcwd(),
            )

        first_data = json_from_last_line(first.stdout)
        second_data = json_from_last_line(second.stdout)
        self.assertEqual(first_data["pred_status"], 200)
        self.assertEqual(first_data["feedback_status"], 200)
        self.assertEqual(first_data["pred2_status"], 200)
        self.assertTrue(first_data["changed"])
        self.assertTrue(second_data["preserved"])
        self.assertTrue(second_data["model_file_exists"])


def json_from_last_line(output: str) -> dict:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            import json

            return json.loads(line)
    raise AssertionError(f"No se encontro JSON en la salida:\n{output}")


if __name__ == "__main__":
    unittest.main()
