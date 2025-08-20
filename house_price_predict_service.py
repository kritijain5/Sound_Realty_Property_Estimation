import datetime
import pickle
import json
import pandas as pd
import numpy as np
import os
import logging
import uuid
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from sklearn.impute import SimpleImputer

# --- Configuration Paths ---
from config import MODEL_PATH, FEATURES_PATH, DEMOGRAPHICS_PATH, PREDICTION_LOG_PATH

# --- Retrieve API_VERSION from environment variable ---
API_VERSION = os.getenv("API_VERSION", "unknown")
# ----------------------------------------------------

# --- Set up a logger for predictions ---
def setup_logging():
    log_dir = os.path.dirname(PREDICTION_LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Use WatchedFileHandler for robustness in long-running processes
    # and explicit encoding.
    handler = logging.handlers.WatchedFileHandler(PREDICTION_LOG_PATH, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger = logging.getLogger("prediction_logger")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if called multiple times (e.g., in --reload mode)
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

prediction_logger = setup_logging()
# ----------------------------------------------------

# --- Load model, features, and demographics data ---
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "r") as f:
        model_features = json.load(f)
    
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str}).set_index("zipcode")
    demographics_imputer = SimpleImputer(strategy="mean")
    demographics_imputer.fit(demographics)

except FileNotFoundError as e:
    raise FileNotFoundError(f"Required file not found: {e.filename}. Make sure your model and data files exist in the correct paths.")
except Exception as e:
    raise Exception(f"Error loading models or data: {e}")

# Initialize FastAPI app with version in the title
app = FastAPI(title=f"Housing Price Prediction API - Version {API_VERSION}")

# Define expected input schema for the *full* API
class HouseFeatures(BaseModel):
    bedrooms: int = Field(3, ge=0)
    bathrooms: float = Field(2.0, ge=0)
    sqft_living: int = Field(1800, gt=0)
    sqft_lot: int = Field(7000, gt=0)
    floors: float = Field(1.0, gt=0)
    waterfront: int = Field(0, ge=0, le=1) # No waterfront
    view: int = Field(0, ge=0, le=4)      # No view
    condition: int = Field(3, ge=1, le=5) # Average condition
    grade: int = Field(7, ge=1, le=13)    # Average construction grade
    sqft_above: int = Field(1500, ge=0)
    sqft_basement: int = Field(300, ge=0)
    yr_built: int = Field(1980, ge=1900)  # Typical build year
    yr_renovated: int = Field(0, ge=0)    # Not renovated
    zipcode: str = Field("98103", min_length=5, max_length=5) # A common Seattle zipcode
    lat: float = Field(47.6562, description="Latitude for Seattle area")
    long: float = Field(-122.3486, description="Longitude for Seattle area")
    sqft_living15: int = Field(1800) # Similar to sqft_living for consistency
    sqft_lot15: int = Field(7000)    # Similar to sqft_lot for consistency

# Define expected input schema for the *simplified* API
class SimplifiedHouseFeatures(BaseModel):
    bedrooms: int = Field(3, ge=0)
    bathrooms: float = Field(2.0, ge=0)
    sqft_living: int = Field(1800, gt=0)
    sqft_lot: int = Field(7000, gt=0)
    floors: float = Field(1.0, gt=0)
    sqft_above: int = Field(1500, ge=0)
    sqft_basement: int = Field(300, ge=0)
    zipcode: str = Field("98103", min_length=5, max_length=5, description="5-digit zip code for demographics lookup.")


@app.post("/predict")
def predict_price(data: HouseFeatures):
    """Predicts the selling price of a house using the full set of input features."""
    try:
        # Generate a new unique property_id for each prediction
        current_property_id = str(uuid.uuid4())

        df = pd.DataFrame([data.dict()])
        
        # Merge with demographics
        if data.zipcode not in demographics.index:
            demog_cols = demographics.columns
            imputed_demog_df = pd.DataFrame(np.nan, index=[0], columns=demog_cols)
            imputed_values = demographics_imputer.transform(imputed_demog_df)
            for i, col in enumerate(demog_cols):
                df[col] = imputed_values[0][i]
        else:
            demog_row = demographics.loc[data.zipcode]
            for col, value in demog_row.items():
                df[col] = value

        # Ensure correct feature order and types based on model_features
        if 'house_age' in model_features and 'yr_built' in df.columns:
            df['house_age'] = datetime.datetime.now().year - df['yr_built']
        if 'is_renovated' in model_features and 'yr_renovated' in df.columns:
            df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

        # Drop original year columns (and any other columns not in model_features if they exist in df)
        cols_to_drop = [col for col in ['yr_built', 'yr_renovated'] if col in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Ensure only model_features are selected for prediction
        X_new = df[model_features]
        
        # Cast integer columns to float64 if they might contain NaNs in the future
        for col in X_new.columns:
            if X_new[col].dtype == 'int64':
                if X_new[col].isnull().any():
                    X_new[col] = X_new[col].astype('float64')

        prediction = model.predict(X_new)[0]
        predicted_price_rounded = round(prediction, 2)
        
        response_data = {
            "prediction": {
                "predicted_price": predicted_price_rounded,
                "prediction_units": "USD"
            },
            "metadata": {
                "property_id": current_property_id,
                "valuation": predicted_price_rounded,
                "api_version": API_VERSION,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "disclaimer": "This is an estimate based on a machine learning model and should not be considered a final valuation. Uses all provided features."
            }
        }
        
        # --- Log the prediction details ---
        log_entry = {
            "prediction_id": current_property_id,
            "input_data": data.dict(),
            "processed_features": X_new.to_dict(orient='records')[0],
            "prediction": predicted_price_rounded,
            "model_version": API_VERSION,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "endpoint": "/predict"
        }
        prediction_logger.info(json.dumps(log_entry))
        # ----------------------------------
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during full prediction: {e}")

@app.post("/predict-simplified")
def predict_price_simplified(data: SimplifiedHouseFeatures):
    """
    Predicts the selling price of a house using only the core features.
    """
    try:
        # Generate a new unique property_id for each prediction
        current_property_id = str(uuid.uuid4())

        # Create a DataFrame with all expected model features, initialized to zero or appropriate defaults
        input_df = pd.DataFrame(0.0, index=[0], columns=model_features)

        # Populate with provided simplified features
        for field, value in data.dict().items():
            if field in input_df.columns:
                input_df[field] = value
        
        # Handle demographics for simplified endpoint
        if data.zipcode not in demographics.index:
            imputed_values = demographics_imputer.transform(pd.DataFrame(np.nan, index=[0], columns=demographics.columns))
            for i, col in enumerate(demographics.columns):
                if col in input_df.columns:
                    input_df[col] = imputed_values[0][i]
        else:
            demog_row = demographics.loc[data.zipcode]
            for col, value in demog_row.items():
                if col in input_df.columns:
                    input_df[col] = value

        if 'house_age' in model_features:
            if 'yr_built' in input_df.columns and input_df['yr_built'].iloc[0] != 0:
                 input_df['house_age'] = datetime.datetime.now().year - input_df['yr_built']
            else:
                input_df['house_age'] = datetime.datetime.now().year - 1900
        
        if 'is_renovated' in model_features:
            if 'yr_renovated' in input_df.columns and input_df['yr_renovated'].iloc[0] != 0:
                input_df['is_renovated'] = input_df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
            else:
                input_df['is_renovated'] = 0

        cols_to_drop = [col for col in ['yr_built', 'yr_renovated'] if col in input_df.columns]
        input_df = input_df.drop(columns=cols_to_drop, errors='ignore')

        X_new_simplified = input_df[model_features]

        # Cast integer columns to float64 if they might contain NaNs in the future
        for col in X_new_simplified.columns:
            if X_new_simplified[col].dtype == 'int64':
                if X_new_simplified[col].isnull().any():
                    X_new_simplified[col] = X_new_simplified[col].astype('float64')

        prediction = model.predict(X_new_simplified)[0]
        predicted_price_rounded = round(prediction, 2)
        
        response_data = {
            "prediction": {
                "predicted_price": predicted_price_rounded,
                "prediction_units": "USD"
            },
            "metadata": {
                "property_id": current_property_id,
                "valuation": predicted_price_rounded,
                "api_version": API_VERSION,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "disclaimer": "This is an estimate based on a machine learning model and should not be considered a final valuation. Uses simplified feature set."
            }
        }
        
        # --- Log the prediction details ---
        log_entry = {
            "prediction_id": current_property_id,
            "input_data": data.dict(),
            "processed_features": X_new_simplified.to_dict(orient='records')[0],
            "prediction": predicted_price_rounded,
            "model_version": API_VERSION,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "endpoint": "/predict-simplified"
        }
        prediction_logger.info(json.dumps(log_entry))
        # ----------------------------------
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during simplified prediction: {e}")
