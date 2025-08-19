import pickle
import json
import pandas as pd
import config  # config.py is now in the main directory

# Create prediction folder if it doesn't exist
config.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# Load model and features
with open(config.MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(config.FEATURES_PATH, "r") as f:
    model_features = json.load(f)

print("Model and features loaded successfully")

# Load future examples
new_data = pd.read_csv(config.FUTURE_DATA_PATH, dtype={"zipcode": str})

# Load demographics
demographics = pd.read_csv(config.DEMOGRAPHICS_PATH, dtype={"zipcode": str})

# Merge future examples with demographics
X_new = new_data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

# Ensure columns are in the same order as model_features
X_new = X_new[model_features]

# Make predictions
predictions = model.predict(X_new)
new_data["predicted_price"] = predictions

# Save predictions
new_data.to_csv(config.OUTPUT_PATH, index=False)
print(f"Predictions saved to {config.OUTPUT_PATH}")