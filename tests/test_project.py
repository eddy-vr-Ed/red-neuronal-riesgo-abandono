"""Pruebas básicas que no requieren TensorFlow."""

import unittest

import pandas as pd

from src.config import DATASET_FILE, FEATURES, MODEL_FILE, SCALER_FILE, TARGET


class TestProjectFiles(unittest.TestCase):
    def test_dataset_exists(self):
        self.assertTrue(DATASET_FILE.exists(), "No existe el dataset de entrada.")

    def test_dataset_schema(self):
        datos = pd.read_csv(DATASET_FILE)
        self.assertEqual(list(datos.columns), FEATURES + [TARGET])
        self.assertGreater(len(datos), 0)
        self.assertEqual(datos.isnull().sum().sum(), 0)

    def test_model_artifacts_exist(self):
        self.assertTrue(MODEL_FILE.exists(), "No existe el modelo entrenado.")
        self.assertTrue(SCALER_FILE.exists(), "No existe el escalador.")


if __name__ == "__main__":
    unittest.main()
