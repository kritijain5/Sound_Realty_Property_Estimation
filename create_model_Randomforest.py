import json
import pathlib
import pickle
from typing import List, Tuple
import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import numpy as np

# Import all paths from your config file
from config import SALES_PATH, DEMOGRAPHICS_PATH, MODEL_PATH, FEATURES_PATH, MODEL_DIR

SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode', 'condition', 'yr_built', 'yr_renovated'
]

def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics."""
    data = pd.read_csv(sales_path, usecols=sales_column_selection, dtype={'zipcode': str})
    demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})
    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    
    y = merged_data.pop('price')
    x = merged_data
    return x, y

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers using the Interquartile Range (IQR) method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    # Filter out outliers
    cleaned_data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return cleaned_data

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing ones."""
    data['house_age'] = 2025 - data['yr_built']
    data['is_renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    data = data.drop(columns=['yr_built', 'yr_renovated'])
    return data

def main():
    """Load data, train model, evaluate, and export artifacts."""
    
    with mlflow.start_run(run_name="Random Forest with Engineered Features Depth 20 "):
        x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
        
        # --- Data Cleansing and Feature Engineering ---
        x = engineer_features(x)
        x_cleaned = remove_outliers(x)
        x_cleaned.dropna(inplace=True)
        y_cleaned = y.loc[x_cleaned.index]
        # ----------------------------------------------
        
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x_cleaned, y_cleaned, random_state=42
        )
        
        # --- Random Forest-specific parameters ---
        model_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'random_state': 42
        }
        
        mlflow.log_params(model_params)
        
        model = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            RandomForestRegressor(**model_params)
        ).fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        metrics = {
            "train_r2_score": r2_score(y_train, y_train_pred),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "train_rmse": mean_squared_error(y_train, y_train_pred) ** 0.5,
            "test_r2_score": r2_score(y_test, y_test_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "test_rmse": mean_squared_error(y_test, y_test_pred) ** 0.5
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="random_forest_regressor_model",
            input_example=x_train.head(1)
        )
        
        MODEL_DIR.mkdir(exist_ok=True)
        
        pickle.dump(model, open(MODEL_PATH, 'wb'))
        json.dump(list(x_train.columns), open(FEATURES_PATH, 'w'))
        mlflow.log_artifact(str(FEATURES_PATH))

if __name__ == "__main__":
    main()
