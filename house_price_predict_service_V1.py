from fastapi import FastAPI, HTTPException
import pickle
import json
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
import numpy as np
from config import MODEL_PATH, FEATURES_PATH, DEMOGRAPHICS_PATH

# Load model and features
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_PATH, "r") as f:
    model_features = json.load(f)

# Load demographics data
try:
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str}).set_index("zipcode")
except FileNotFoundError:
    raise FileNotFoundError("Demographics file not found. Please ensure it's in the correct path.")

# Pre-fit an imputer on the demographics data
# This is a crucial step to correctly handle missing values in production
demographics_imputer = SimpleImputer(strategy="mean")
demographics_imputer.fit(demographics)

# Initialize FastAPI app
app = FastAPI(title="Housing Price Prediction API")

# Define expected input schema
class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., gt=0)
    bathrooms: float = Field(..., gt=0)
    sqft_living: int = Field(..., gt=0)
    sqft_lot: int = Field(..., gt=0)
    floors: float = Field(..., gt=0)
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

@app.post("/predict")
def predict_price(data: HouseFeatures):
    try:
        df = pd.DataFrame([data.dict()])
        
        # Check if the zipcode is in our demographics data
        if data.zipcode in demographics.index:
            demog_row = demographics.loc[data.zipcode]
            for col, value in demog_row.items():
                df[col] = value
        else:
            # If zipcode is not found, impute the demographic features
            # Create a DataFrame with NaNs for demographic features
            demog_cols = demographics.columns
            imputed_demog_df = pd.DataFrame(np.nan, index=[0], columns=demog_cols)
            
            # Use the pre-fitted imputer to fill in the missing demographic data
            imputed_values = demographics_imputer.transform(imputed_demog_df)
            
            # Assign the imputed values back to the DataFrame
            for i, col in enumerate(demog_cols):
                df[col] = imputed_values[0][i]

        # Align columns with the model's required features
        X_new = df[model_features]

        # Make prediction and return result
        prediction = model.predict(X_new)[0]
        
        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
