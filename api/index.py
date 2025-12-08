from flask import Flask, request, jsonify, render_template

import joblib
import pandas as pd
import os

app = Flask(__name__)

model = None
schema = None

def load_model():
    global model, schema
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.pkl')

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(schema_path):
        schema = joblib.load(schema_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        load_model()

    if model is None:
        return jsonify({'error': 'Model not found. Please train the model first.'}), 500

    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert input dictionary to DataFrame
        # Ensure we have all necessary columns, even if empty/null, based on schema if needed
        # But for now assume input matches

        input_df = pd.DataFrame([data])

        # Ensure numeric columns are numeric
        if schema:
            for col in schema['numeric']:
                if col in input_df.columns:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            'churn_prediction': int(prediction),
            'churn_probability': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
