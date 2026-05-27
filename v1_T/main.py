import os
import torch
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(
    title="Invoice Expense Classification Framework - v1_T", 
    description="Engine type: Deep Learning Transformer Model (DistilBERT)"
)

MODEL_DIR = "./saved_transformer_model"
ENCODER_PATH = "label_encoder.joblib"

if os.path.exists(MODEL_DIR) and os.path.exists(ENCODER_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    label_encoder = joblib.load(ENCODER_PATH)
    model.eval()
else:
    model, tokenizer, label_encoder = None, None, None

class InvoiceInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    category: str
    confidence: float

@app.post("/predict", response_model=PredictionOutput)
async def predict_expense(payload: InvoiceInput):
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Operational deep learning files missing.")
    
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Parsed input context contains no strings.")
    
    try:
        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
            
        max_index = np.argmax(probabilities)
        predicted_category = label_encoder.inverse_transform([max_index])[0]
        confidence_score = float(probabilities[max_index])
        
        return {
            "category": predicted_category,
            "confidence": round(confidence_score, 4)
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Transformer runtime exception: {str(error)}")