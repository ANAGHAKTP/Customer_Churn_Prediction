import unittest
import json
import os
import sys
from unittest.mock import MagicMock, patch

# Add api to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api.index import app

class TestChurnAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('api.index.model_data', {
        'numeric_features': ['Monthly Charges'],
        'categorical_features': ['Gender'],
        'num_impute_values': [50.0],
        'scale_mean': [50.0],
        'scale_scale': [10.0],
        'cat_fill_value': 'missing',
        'categories': [['Male', 'Female']],
        'coef': [0.5, -0.5, 0.5], # 1 numeric (scaled), 2 cat (onehot)
        'intercept': 0.0
    })
    def test_predict_success(self):
        # We mocked model_data directly

        payload = {
            "Gender": "Male",
            "Monthly Charges": 60.0
        }

        # Calculation:
        # Numeric: (60 - 50) / 10 = 1.0
        # Cat: Male -> index 0 -> [1.0, 0.0]
        # Features: [1.0, 1.0, 0.0]
        # Coef: [0.5, -0.5, 0.5]
        # Dot: 1*0.5 + 1*-0.5 + 0*0.5 = 0.0
        # Intercept: 0.0
        # Total: 0.0
        # Sigmoid(0) = 0.5

        response = self.app.post('/api/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        # Prediction 1 if >= 0.5
        self.assertEqual(data['churn_prediction'], 1)
        self.assertEqual(data['churn_probability'], 0.5)

    def test_predict_no_data(self):
        response = self.app.post('/api/predict',
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
