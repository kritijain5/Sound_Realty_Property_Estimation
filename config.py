from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
PREDICTION_DIR = DATA_DIR / "prediction"

FUTURE_DATA_PATH = DATA_DIR / "future_unseen_examples.csv"
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"
SALES_PATH = DATA_DIR / "kc_house_data.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "model_features.json"
OUTPUT_PATH = PREDICTION_DIR / "predictions.csv"

# --- Add this line for versioning ---
API_VERSION = "1.1.0"
# -------------------------------------