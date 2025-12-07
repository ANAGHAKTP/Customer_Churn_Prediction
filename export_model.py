import joblib
import json
import numpy as np

def export():
    print("Loading model...")
    model = joblib.load('model.pkl')

    # model is a Pipeline: ['preprocessor', 'classifier']
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    # Preprocessor is ColumnTransformer: ['num', 'cat']
    # 'num' is Pipeline: ['imputer', 'scaler']
    # 'cat' is Pipeline: ['imputer', 'onehot']

    num_pipe = preprocessor.named_transformers_['num']
    cat_pipe = preprocessor.named_transformers_['cat']

    # Numeric stats
    num_imputer = num_pipe.named_steps['imputer']
    scaler = num_pipe.named_steps['scaler']

    # Numeric features used during training
    # We need to know which input fields map to these stats.
    # In train_model.py, we defined numeric_features and categorical_features.
    # We should load schema.pkl to know the names.
    schema = joblib.load('schema.pkl')
    numeric_features = schema['numeric']
    categorical_features = schema['categorical']

    # Imputer statistics (median)
    # SimpleImputer stores statistics_
    num_impute_values = num_imputer.statistics_.tolist()

    # Scaler statistics
    scale_mean = scaler.mean_.tolist()
    scale_scale = scaler.scale_.tolist()

    # Categorical stats
    cat_imputer = cat_pipe.named_steps['imputer']
    onehot = cat_pipe.named_steps['onehot']

    cat_fill_value = cat_imputer.fill_value

    # OneHot categories
    # categories_ is a list of arrays
    categories = [c.tolist() for c in onehot.categories_]

    # Classifier weights
    coef = classifier.coef_.flatten().tolist()
    intercept = classifier.intercept_[0]
    classes = classifier.classes_.tolist()

    export_data = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'num_impute_values': num_impute_values,
        'scale_mean': scale_mean,
        'scale_scale': scale_scale,
        'cat_fill_value': cat_fill_value,
        'categories': categories,
        'coef': coef,
        'intercept': intercept,
        'classes': classes
    }

    with open('model.json', 'w') as f:
        json.dump(export_data, f, indent=4)

    print("Model exported to model.json")

if __name__ == "__main__":
    export()
