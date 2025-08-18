import datetime
import pickle
import json
import pandas as pd
import numpy as np
import os
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from sklearn.impute import SimpleImputer
from config import MODEL_PATH, FEATURES_PATH, DEMOGRAPHICS_PATH

# --- Retrieve API_VERSION from environment variable ---
API_VERSION = os.getenv("API_VERSION", "unknown")
# ----------------------------------------------------

# Load model, features, and demographics data
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
    raise Exception(f"Error loading model or data: {e}")

# Initialize FastAPI app with version in the title
app = FastAPI(title=f"Housing Price Prediction API - Version {API_VERSION}")

# Define expected input schema for the *full* API
class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., gt=0)
    sqft_lot: int = Field(..., gt=0)
    floors: float = Field(..., gt=0)
    waterfront: int = Field(..., ge=0, le=1)
    view: int = Field(..., ge=0, le=4)
    condition: int = Field(..., ge=1, le=5)
    grade: int = Field(..., ge=1, le=13)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    yr_built: int = Field(..., ge=1900)
    yr_renovated: int = Field(..., ge=0)
    zipcode: str = Field(..., min_length=5, max_length=5)
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

# Define expected input schema for the *simplified* API
class SimplifiedHouseFeatures(BaseModel):
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., gt=0)
    sqft_lot: int = Field(..., gt=0)
    floors: float = Field(..., gt=0)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zip code for demographics lookup.")


@app.post("/predict")
def predict_price(data: HouseFeatures):
    """Predicts the selling price of a house using the full set of input features."""
    try:
        df = pd.DataFrame([data.dict()])
        
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

        X_new = df[model_features]
        prediction = model.predict(X_new)[0]
        
        response_data = {
            "prediction": {
                "predicted_price": round(prediction, 2),
                "prediction_units": "USD"
            },
            "metadata": {
                "api_version": API_VERSION,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "disclaimer": "This is an estimate based on a machine learning model and should not be considered a final valuation. Uses all provided features."
            }
        }
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during full prediction: {e}")

@app.post("/predict-simplified")
def predict_price_simplified(data: SimplifiedHouseFeatures):
    """
    Predicts the selling price of a house using only the core features.
    """
    try:
        input_df = pd.DataFrame(0.0, index=[0], columns=model_features)

        for field, value in data.dict().items():
            if field in input_df.columns:
                input_df[field] = value
        
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

        X_new_simplified = input_df[model_features]

        prediction = model.predict(X_new_simplified)[0]
        
        response_data = {
            "prediction": {
                "predicted_price": round(prediction, 2),
                "prediction_units": "USD"
            },
            "metadata": {
                "api_version": API_VERSION,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "disclaimer": "This is an estimate based on a machine learning model and should not be considered a final valuation. Uses simplified feature set."
            }
        }
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during simplified prediction: {e}")
