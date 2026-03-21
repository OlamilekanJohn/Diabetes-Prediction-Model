
""" 
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency 
=========================================================================

This module provides the core inference functionality for the Diabetes Predicton 

Key Responsibitlies: 
1. Load Mlflow-logged model and feature metadata from training
2. Apply identical feature transformation as used during training 
3. Ensure correct feature ordering for model input 
4. Convert model prediction to user-friendly output 

"""

import os
import pandas as pd
import mlflow
import glob

# === Model loading configuration ===
# Use environment variable if set, else default path inside Docker
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "/app/model"  # default path inside Docker container
)

model = None

# === Load MLflow model ===
try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")
    # Fallback: check local mlruns directory (development)
    try:
        local_model_paths = glob.glob("mlruns/**/artifacts", recursive=True)
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"Fallback: Model loaded from local path {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === Load feature columns used during training ===
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === Feature transformation constants ===
BINARY_MAP = {"Gender": {"F": 0, "M": 1}}
NUMERIC_COLS = [
    "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"
]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same transformations as training."""
    df = df.copy()
    df.columns = df.columns.str.strip()  # clean column names

    # Numeric coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Binary mapping
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().map(mapping).astype("Int64")

    # One-hot encode remaining categorical columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Align columns to training schema
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

def predict(input_dict: dict) -> str:
    """Make a prediction for a single input dictionary."""
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    # Generate model prediction
    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0] if isinstance(preds, (list, tuple)) and len(preds) == 1 else preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    # Convert to user-friendly output
    if result == 1:
        return (
            "At risk of type 2 diabetes. To lower your risk, focus on:\n"
            "1. Move More: 150 min/week moderate activity + strength training.\n"
            "2. Smart Swaps: Replace refined carbs with whole grains, cut sugary drinks.\n"
            "3. Weight Management: 5–7% weight loss reduces risk >50%.\n"
            "4. Clinical Checks: HbA1c test & GP guidance."
        )
    else:
        return "Not at risk of type 2 diabetes."