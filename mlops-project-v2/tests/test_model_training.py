import unittest
import numpy as np
import pandas as pd
from src.model.model_training import train_model
from src.model.model_evaluation import evaluate_model

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv('data/raw/filtered_file.csv')
        self.data = self.data.dropna()
        self.data['score'] = self.data['score'].round().astype(int)
        self.X = self.data.drop('date', axis=1)
        self.y = self.data['score']

    def test_train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)

if __name__ == '__main__':
    unittest.main()