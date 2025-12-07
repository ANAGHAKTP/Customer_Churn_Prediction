# Telco Customer Churn Prediction - Vercel Deployment

This project provides a web interface for predicting customer churn using a Logistic Regression model. It is configured for deployment on Vercel.

## Structure

*   `api/index.py`: The Flask backend that serves the model prediction endpoint.
*   `public/index.html`: The frontend UI.
*   `train_model.py`: Script to train the model and save it as `model.pkl`.
*   `model.pkl`: The trained model.
*   `schema.pkl`: Schema of the features used.

## Local Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Train the model (if `model.pkl` is missing or you want to retrain):
    ```bash
    python train_model.py
    ```

3.  Run the API locally:
    ```bash
    python api/index.py
    ```
    Open `http://localhost:5000` in your browser.

## Deployment to Vercel

1.  Push this repository to GitHub/GitLab/Bitbucket.
2.  Import the project in Vercel.
3.  Vercel will automatically detect the configuration in `vercel.json` and deploy the Python serverless function.
4.  Ensure `model.pkl` and `schema.pkl` are included in the repository (they should be unless ignored).

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
