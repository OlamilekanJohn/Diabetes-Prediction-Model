
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr 
import os
import sys

# Ensure import from src/serving when running "uvicorn src.app.app:app"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict # source of truth for inference 

app = FastAPI()

@app.get("/")
def root(): 
    return {"status": "ok"}

# Request schema (same fields you collect in the UI)

class diabetesData(BaseModel): 
    AGE: int
    Urea: float
    Cr: float
    HbA1c: float
    Chol: float
    TG: float
    HDL: float
    LDL: float
    VLDL: float
    BMI: float

@app.post("/predict")

def api_predict(data: diabetesData): 
    try: 
        out = predict(data.dict())
        return {"prediction": out}
    except Exception as e: 
        return {"error": str(e)}

# --- Gradio UI wrappers the same predict() --- 

def gradio_interface(
    AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL,
    VLDL, BMI 
): 
    payload = {
        AGE: int(AGE),
        Urea: float(Urea),
        Cr: float(Cr),
        HbA1c: float(HbA1c),
        Chol: float(Chol),
        TG: float(TG),
        HDL: float(HDL),
        LDL: float(LDL),
        VLDL: float(VLDL),
        BMI: float(BMI),
    }
    out = predict(payload)
    return str(out)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(['M', 'F'], label="Gender"), 
        gr.Number(label="AGE"),
        gr.Number(label="Urea"),
        gr.Number(label="Cr"),
        gr.Number(label="HbA1c"),
        gr.Number(label="Chol"),
        gr.Number(label="TG"),
        gr.Number(label="HDL"),
        gr.Number(label="LDL"),
        gr.Number(label="VLDL"),
        gr.Number(label="BMI"), 
    ], 
    outputs=gr.Textbox(label="Diabetes Risk Prediction", lines=2), 
    title=" Diabetes Risk Predictor", 
    description="""
    **Predict patient diabetic risk probability using machine learning**
     
    
    Fill in the patient details below to get a churn prediction. The model uses Logistic regression trained on 
    historical diabetes data to identify patients at risk of diabets. 
     
    """
)

app = gr.mount_gradio_app(app,demo,path="/ui")


