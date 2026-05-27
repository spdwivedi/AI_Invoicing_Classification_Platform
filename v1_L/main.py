import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Invoice Expense Classification Framework - v1_L", 
    description="Engine type: TF-IDF + Logistic Regression Baseline"
)

# Load saved runtime pipeline artifact securely on startup
MODEL_PATH = "model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class InvoiceInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    category: str
    confidence: float

@app.post("/predict", response_model=PredictionOutput)
async def predict_expense(payload: InvoiceInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Operational model file missing.")
    
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Parsed input context contains no strings.")
    
    try:
        # Compute exact category probability mapping
        probabilities = model.predict_proba([clean_text])[0]
        max_index = np.argmax(probabilities)
        
        predicted_category = model.classes_[max_index]
        confidence_score = float(probabilities[max_index])
        
        return {
            "category": predicted_category,
            "confidence": round(confidence_score, 4)
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Runtime evaluation exception: {str(error)}")