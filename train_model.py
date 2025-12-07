import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib

def train():
    print("Loading data...")
    try:
        df = pd.read_excel('Telco_customer_churn.xlsx')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Preprocessing
    # 'Total Charges' is object type, coerce to numeric
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    # Target
    target = 'Churn Value'

    # Features
    # We will exclude IDs, location details (too many categories), and churn-related output columns
    drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long',
                 'Latitude', 'Longitude', 'Churn Label', 'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason']

    # Identify numerical and categorical columns
    feature_cols = [c for c in df.columns if c not in drop_cols]

    print(f"Features selected: {feature_cols}")

    X = df[feature_cols]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define transformers
    numeric_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Model - Using Logistic Regression for smaller model size
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(random_state=42, max_iter=1000))])

    print("Training model...")
    clf.fit(X_train, y_train)

    print("Model score: %.3f" % clf.score(X_test, y_test))

    print("Saving model...")
    joblib.dump(clf, 'model.pkl')

    schema = {
        'numeric': numeric_features,
        'categorical': categorical_features
    }
    joblib.dump(schema, 'schema.pkl')

    print("Done.")

if __name__ == '__main__':
    train()
