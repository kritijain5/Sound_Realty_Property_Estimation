from fastapi import FastAPI, HTTPException
import pickle
import json
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
import numpy as np
from config import MODEL_PATH, FEATURES_PATH, DEMOGRAPHICS_PATH
import datetime # Import for timestamp metadata

# Load model and features
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "r") as f:
        # model_features will contain the exact list of columns the trained model expects
        model_features = json.load(f)
    
    # Load demographics data
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str}).set_index("zipcode")
    
except FileNotFoundError as e:
    raise FileNotFoundError(f"Required file not found: {e.filename}. Make sure model/model.pkl, model/model_features.json, and data/zipcode_demographics.csv exist.")
except Exception as e:
    raise Exception(f"Error loading model or data: {e}")

# Pre-fit an imputer on the demographics data for handling unknown zipcodes
demographics_imputer = SimpleImputer(strategy="mean")
demographics_imputer.fit(demographics)

# Initialize FastAPI app
app = FastAPI(title="Housing Price Prediction API")

# Define expected input schema for the *full* API (your existing endpoint)
class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., gt=0, description="Square footage of the house's living area. Must be a positive value.")
    sqft_lot: int = Field(..., gt=0, description="Square footage of the lot. Must be a positive value.")
    floors: float = Field(..., gt=0)
    waterfront: int = Field(..., ge=0, le=1) # Added validation for 0 or 1
    view: int = Field(..., ge=0, le=4)
    condition: int = Field(..., ge=1, le=5)
    grade: int = Field(..., ge=1, le=13)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    yr_built: int = Field(..., ge=1900, le=datetime.datetime.now().year) # Added validation
    yr_renovated: int = Field(..., ge=0)
    zipcode: str = Field(..., min_length=5, max_length=5) # Added validation
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

# Define expected input schema for the *simplified* API (Bonus)
class SimplifiedHouseFeatures(BaseModel):
    # These are the core features from SALES_COLUMN_SELECTION (excluding price and zipcode)
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., gt=0, description="Square footage of the house's living area. Must be a positive value.")
    sqft_lot: int = Field(..., gt=0, description="Square footage of the lot. Must be a positive value.")
    floors: float = Field(..., gt=0)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zip code for demographics lookup.")


@app.post("/predict")
def predict_price(data: HouseFeatures):
    """
    Predicts the selling price of a house using the full set of input features.
    """
    try:
        df = pd.DataFrame([data.dict()])
        
        # Handle zipcode demographics
        if data.zipcode not in demographics.index:
            # If zipcode is not in our demographics data, impute demographics
            demog_cols = demographics.columns
            imputed_demog_df = pd.DataFrame(np.nan, index=[0], columns=demog_cols)
            imputed_values = demographics_imputer.transform(imputed_demog_df)
            for i, col in enumerate(demog_cols):
                df[col] = imputed_values[0][i]
        else:
            # If zipcode exists, add the corresponding demographic data
            demog_row = demographics.loc[data.zipcode]
            for col, value in demog_row.items():
                df[col] = value

        # Ensure the DataFrame columns match the model's expected features
        # This will filter out any extra columns from HouseFeatures that model_features doesn't need
        X_new = df[model_features]

        # Make prediction
        prediction = model.predict(X_new)[0]
        
        # Return prediction with metadata
        response_data = {
            "prediction": {
                "predicted_price": round(prediction, 2),
                "prediction_units": "USD"
            },
            "metadata": {
                "model_version": "1.0.0", # Placeholder, ideally dynamic
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "disclaimer": "This is an estimate based on a machine learning model and should not be considered a final valuation. Uses all provided features."
            }
        }
        
        return response_data

    except Exception as e:
        # Log the detailed error for debugging purposes
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during full prediction: {e}")

# --- Simplified API Endpoint (Bonus) ---
@app.post("/predict-simplified")
def predict_price_simplified(data: SimplifiedHouseFeatures):
    """
    Predicts the selling price of a house using only the core features.
    Demographics will be looked up or imputed based on zipcode.
    """
    try:
        # Initialize a DataFrame with zeros/defaults for all model_features
        # This ensures all expected columns are present before prediction
        input_df = pd.DataFrame(0.0, index=[0], columns=model_features)

        # Populate the DataFrame with directly provided data
        for field, value in data.dict().items():
            if field in input_df.columns:
                input_df[field] = value
        
        # Handle zipcode demographics
        if data.zipcode not in demographics.index:
            # If zipcode is not in our demographics data, impute demographics
            imputed_values = demographics_imputer.transform(pd.DataFrame(np.nan, index=[0], columns=demographics.columns))
            for i, col in enumerate(demographics.columns):
                if col in input_df.columns: # Ensure demographic column is actually in model_features
                    input_df[col] = imputed_values[0][i]
        else:
            # If zipcode exists, add the corresponding demographic data
            demog_row = demographics.loc[data.zipcode]
            for col, value in demog_row.items():
                if col in input_df.columns: # Ensure demographic column is actually in model_features
                    input_df[col] = value

        # Ensure the final DataFrame for prediction has columns in the exact order
        # as model_features. This is critical for model compatibility.
        X_new_simplified = input_df[model_features]

        # Make prediction
        prediction = model.predict(X_new_simplified)[0]
        
        # Return prediction with metadata
        response_data = {
            "prediction": {
                "predicted_price": round(prediction, 2),
                "prediction_units": "USD"
            },
            "metadata": {
                "model_version": "1.0.0", # Placeholder, ideally dynamic
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "disclaimer": "This is an estimate based on a machine learning model and should not be considered a final valuation. Uses simplified feature set."
            }
        }
        
        return response_data

    except Exception as e:
        # Log the detailed error for debugging purposes
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during simplified prediction: {e}")
