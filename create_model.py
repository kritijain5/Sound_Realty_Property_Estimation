import json
import pathlib
import pickle
from typing import List, Tuple
import pandas
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor # <-- New import
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

# Import all paths from your config file
from config import SALES_PATH, DEMOGRAPHICS_PATH, MODEL_PATH, FEATURES_PATH, MODEL_DIR

SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics."""
    data = pandas.read_csv(sales_path, usecols=sales_column_selection, dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path, dtype={'zipcode': str})
    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data
    return x, y


def main():
    """Load data, train model, evaluate, and export artifacts."""
    
    # --- Start an MLflow run to track this experiment ---
    with mlflow.start_run():
        x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
        
        # Split data into training and test sets for evaluation
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, random_state=42
        )

        # Define the new model and its parameters
        model_params = {
            'n_estimators': 100, # Number of decision trees
            'max_depth': 10,     # Max depth of each tree
            'random_state': 42
        }
        
        # Log model parameters to MLflow
        mlflow.log_params(model_params)

        # Create and train the new model pipeline
        model = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            RandomForestRegressor(**model_params) # <-- New model
        ).fit(x_train, y_train)

        # Make predictions on both training and test sets for evaluation
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        # Calculate and log evaluation metrics to MLflow
        metrics = {
            "train_r2_score": r2_score(y_train, y_train_pred),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "train_rmse": mean_squared_error(y_train, y_train_pred) ** 0.5,
            "test_r2_score": r2_score(y_test, y_test_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "test_rmse": mean_squared_error(y_test, y_test_pred) ** 0.5
        }
        mlflow.log_metrics(metrics)

        # Log the trained model and features file as artifacts to MLflow
        mlflow.sklearn.log_model(model, name = "random_forest_regressor_model") # <-- Updated name
        
        # Ensure the output directory exists
        MODEL_DIR.mkdir(exist_ok=True)
        
        # Save the model and features file locally as well
        pickle.dump(model, open(MODEL_PATH, 'wb'))
        json.dump(list(x_train.columns), open(FEATURES_PATH, 'w'))
        mlflow.log_artifact(str(FEATURES_PATH))


if __name__ == "__main__":
    main()