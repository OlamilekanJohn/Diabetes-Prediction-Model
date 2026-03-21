
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os 
import pandas as pd 

# Make sure python finds src package 

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.serving.inference import predict  # core  ML inference logic 

# initialize FastAPI application 

app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="ML API for predicting diabetes risk based on patient data",
    version="1.0.0"
)

# == Health check endpoint ==

@app.get("/")
def root(): 
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "API is running!"}

# === Request Data Schema ===
# pydantic model for automatic validation and API documentation 

class customerData(BaseModel):

    Gender: str # 'M' or 'F'
    AGE: int  
    Urea: int
    Cr: int
    HbA1c: int
    Chol: int
    TG: int 
    HDL: int
    LDL: int
    VLDL: int
    BMI: int

# === Main Prediction API Endpoint ===

@app.post("/predict")
def get_prediction(data: customerData):
    """
    API endpoint to receive patient data and return diabetes risk prediction.

    This endpoint: 
    1. Receives validated patient data via Pydantic model 
    2. Calls the inference pipeline to transform features and predict 
    3. Returns Class 0/1 prediction

    """

    try: 
        # Convert Pydantic model to dict and call inference pipeline 
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error message for debugging (consider logging in production) 
        return {"error": str(e)}


# === Gradio web interface === 

def gradio_interface(
        Gender, AGE, Urea, Cr, 
        HbA1c, Chol, TG, HDL, LDL, 
        VLDL, BMI
):
    """ 
    Gradio interface function that processes user inputs and returns prediction.
    
    This function: 
    1. Takes user inputs from Gradio interface
    2. Constructs the data dictionary matching the API schema
    3. calls the same inference pipeline used by the API
    4. Returns user-friendly prediction output 
    """
    data = {
        "Gender": Gender,
        "AGE": int(AGE),
        "Urea": float(Urea),
        "Cr": float(Cr),
        "HbA1c": float(HbA1c),
        "Chol": float(Chol),
        "TG": float(TG),
        "HDL": float(HDL),
        "LDL": float(LDL),
        "VLDL": float(VLDL),
        "BMI": float(BMI)
    }
    result = predict(data)
    return str(result)

# === Gradio UI cofiguration ===
# Build comprehensive Gradio interface with all customer features 
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(['M', 'F'], label="Gender"), 
        gr.Number(label="AGE", minimum=0, maximum=150),
        gr.Number(label="Urea", minimum=0, maximum=50),
        gr.Number(label="Cr", minimum=0, maximum=800),
        gr.Number(label="HbA1c", minimum=0, maximum=20),
        gr.Number(label="Chol", minimum=0, maximum=15),
        gr.Number(label="TG", minimum=0, maximum=15),
        gr.Number(label="HDL", minimum=0, maximum=15),
        gr.Number(label="LDL", minimum=0, maximum=15),
        gr.Number(label="VLDL", minimum=0, maximum=45),
        gr.Number(label="BMI", minimum=11, maximum=59)
    ], 
    outputs=gr.Textbox(label="Diabetes Risk Prediction", line=2), 
    title=" Diabetes Risk Predictor", 
    description="""
    **Predict patient diabetic risk probability using machine learning**
     
    
    Fill in the patient details below to get a churn prediction. The model uses Logistic regression trained on 
    historical diabetes data to identify patients at risk of diabets. 
     
    """,
    examples=[
        # High risk example
        ["M or F", "Bmi=25","HbA1c=5.7","TG=1.5", "Chol=5"], 

        # Low risk example
        ["M or F", "Bmi=20","HbA1c=3.0","TG=0.6", "Chol=3"]
    ],
    theme=gr.themes.soft() # professional appearance 
)

# === Mount Gradio UI into FastApI ===
# Creates the /ui endpoint that serves the gradio interface
# Important: This must be the final line to properly integrate Gradio with FastAPI 

app = gr.mount_gradio_app(
    app,        # FastAPI application instance 
    demo,       # Gradio interface
    path="/ui"  # URL path where Gradio will be accessible 
)







