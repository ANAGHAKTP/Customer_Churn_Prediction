from flask import Flask, request, jsonify, send_from_directory
import json
import math
import os

app = Flask(__name__, static_folder='../public')

model_data = None

def load_model_data():
    global model_data
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.json')
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            model_data = json.load(f)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if model_data is None:
        load_model_data()

    if model_data is None:
        return jsonify({'error': 'Model not found. Please train/export the model first.'}), 500

    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # --- Preprocessing & Feature Extraction ---
        features = []

        # 1. Numeric Features
        numeric_cols = model_data['numeric_features']
        impute_vals = model_data['num_impute_values']
        means = model_data['scale_mean']
        scales = model_data['scale_scale']

        for i, col in enumerate(numeric_cols):
            val = data.get(col)
            # Impute
            if val is None or val == "":
                val = impute_vals[i]
            else:
                try:
                    val = float(val)
                except ValueError:
                    val = impute_vals[i]

            # Scale
            scaled_val = (val - means[i]) / scales[i]
            features.append(scaled_val)

        # 2. Categorical Features
        cat_cols = model_data['categorical_features']
        categories_list = model_data['categories']
        fill_value = model_data['cat_fill_value']

        for i, col in enumerate(cat_cols):
            val = data.get(col)
            # Impute
            if val is None:
                val = fill_value

            val = str(val)

            # OneHotEncode
            # We need to create a binary vector of length len(categories_list[i])
            # If val is in categories_list[i], set that index to 1, else all 0 (handle_unknown='ignore')

            cats = categories_list[i]
            encoded = [0.0] * len(cats)

            try:
                idx = cats.index(val)
                encoded[idx] = 1.0
            except ValueError:
                # Unknown category, all zeros
                pass

            features.extend(encoded)

        # --- Prediction ---
        coef = model_data['coef']
        intercept = model_data['intercept']

        if len(features) != len(coef):
            return jsonify({'error': f'Feature mismatch. Expected {len(coef)}, got {len(features)}'}), 500

        # Dot product
        dot_product = sum(f * c for f, c in zip(features, coef))
        linear_pred = dot_product + intercept

        # Probability
        prob = sigmoid(linear_pred)
        prediction = 1 if prob >= 0.5 else 0

        return jsonify({
            'churn_prediction': prediction,
            'churn_probability': prob
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model_data()
    app.run(debug=True, port=5000)
