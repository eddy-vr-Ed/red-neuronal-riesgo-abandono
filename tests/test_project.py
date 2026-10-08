"""Pruebas básicas que no requieren TensorFlow."""

import unittest

import pandas as pd

from src.config import DATASET_FILE, ENCODERS_FILE, FEATURES_RAW, MODEL_FILE, SCALER_FILE, TARGET


class TestProjectFiles(unittest.TestCase):
    def test_dataset_exists(self):
        self.assertTrue(DATASET_FILE.exists(), "No existe el dataset de entrada.")

    def test_dataset_schema(self):
        datos = pd.read_csv(DATASET_FILE)
        self.assertEqual(list(datos.columns), FEATURES_RAW + [TARGET])
        self.assertGreater(len(datos), 0)
        self.assertEqual(datos.isnull().sum().sum(), 0)

    def test_model_artifacts_exist(self):
        self.assertTrue(MODEL_FILE.exists(), "No existe el modelo entrenado.")
        self.assertTrue(SCALER_FILE.exists(), "No existe el escalador.")
        self.assertTrue(ENCODERS_FILE.exists(), "No existe el encoder entrenado.")


class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app

        cls.client = app.app.test_client()

    def test_index_responds(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_prediction_rejects_incomplete_payload(self):
        response = self.client.post("/predecir", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_prediction_accepts_valid_payload(self):
        payload = {
            "grado": 1,
            "grupo": "A",
            "especialidad": "general",
            "horas_semana_totales": 35,
            "asistencia_semanal": 88.0,
            "promedio": 8.5,
            "materias_reprobadas": 0,
        }
        response = self.client.post("/predecir", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn(data["riesgo"], {"BAJO", "ALTO"})
        self.assertIn("probabilidad_bajo", data)
        self.assertIn("probabilidad_alto", data)


if __name__ == "__main__":
    unittest.main()
