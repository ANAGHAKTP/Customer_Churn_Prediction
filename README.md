# Telco Customer Churn Prediction - Vercel Deployment

This project provides a web interface for predicting customer churn. It is optimized for serverless deployment on Vercel by using a lightweight inference engine instead of heavy data science libraries.

## Structure

*   `api/index.py`: The Flask backend that serves the model prediction endpoint. It performs inference using pure Python/NumPy logic.
*   `public/index.html`: The frontend UI.
*   `train_model.py`: Script to train the Scikit-Learn model and save it as `model.pkl`.
*   `export_model.py`: Script to extract parameters from `model.pkl` to `model.json`.
*   `model.json`: The exported model parameters used by the API.

## Local Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `requirements.txt` only contains runtime dependencies (`flask`). To train the model, you need `pandas`, `scikit-learn`, `openpyxl`, `joblib` installed manually.*

2.  Train and Export (if needed):
    ```bash
    pip install pandas scikit-learn openpyxl joblib
    python train_model.py
    python export_model.py
    ```

3.  Run the API locally:
    ```bash
    python api/index.py
    ```
    Open `http://localhost:5000` in your browser.

## Deployment to Vercel

1.  Push this repository to GitHub/GitLab/Bitbucket.
2.  Import the project in Vercel.
3.  Vercel will automatically detect the configuration in `vercel.json`.
4.  Ensure `model.json` is committed to the repository.

## API Usage

Endpoint: `POST /api/predict`

Payload example:
```json
{
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 70.0,
    "Total Charges": 840.0
}
```
