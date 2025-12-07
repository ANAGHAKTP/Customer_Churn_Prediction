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

    @patch('api.index.model')
    @patch('api.index.schema')
    def test_predict_success(self, mock_schema, mock_model):
        # Mock schema and model behavior
        mock_schema.return_value = {'numeric': ['Monthly Charges'], 'categorical': ['Gender']}

        # Mock prediction
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]

        payload = {
            "Gender": "Male",
            "Monthly Charges": 50.0
        }

        response = self.app.post('/api/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['churn_prediction'], 1)
        self.assertEqual(data['churn_probability'], 0.8)

    def test_predict_no_data(self):
        response = self.app.post('/api/predict',
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
