import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Core environment configuration paths
V1_L_PATH = "v1_L/model.joblib"
V1_NB_PATH = "v1_NB/model.joblib"
V1_T_DIR = "v1_T/saved_transformer_model"
V1_T_ENCODER = "v1_T/label_encoder.joblib"
LABELS = ["Cloud/Software", "Inventory", "Logistics", "Office Supplies", "Travel", "Utilities"]

def run_evaluation_suite(csv_path, output_json_name):
    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} not found.")
        return

    print(f"Freezing evaluation metrics for: {csv_path}")
    df = pd.read_csv(csv_path)
    X_test = df['text'].astype(str).tolist()
    y_true = df['category'].astype(str).tolist()

    # Load baseline model architectures
    model_l = joblib.load(V1_L_PATH) if os.path.exists(V1_L_PATH) else None
    model_nb = joblib.load(V1_NB_PATH) if os.path.exists(V1_NB_PATH) else None
    
    # Initialize transformer weights securely
    t_tokenizer, t_model, t_encoder = None, None, None
    if os.path.exists(V1_T_DIR) and os.path.exists(V1_T_ENCODER):
        t_tokenizer = AutoTokenizer.from_pretrained(V1_T_DIR)
        t_model = AutoModelForSequenceClassification.from_pretrained(V1_T_DIR)
        t_encoder = joblib.load(V1_T_ENCODER)
        t_model.eval()

    report_payload = {}

    # 1. Process Logistic Regression Matrix
    if model_l:
        preds = model_l.predict(X_test)
        cm = confusion_matrix(y_true, preds, labels=LABELS).tolist()
        report_payload["v1_L"] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "matrix": cm,
            "metrics": classification_report(y_true, preds, labels=LABELS, output_dict=True, zero_division=0)
        }

    # 2. Process Naive Bayes Matrix
    if model_nb:
        preds = model_nb.predict(X_test)
        cm = confusion_matrix(y_true, preds, labels=LABELS).tolist()
        report_payload["v1_NB"] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "matrix": cm,
            "metrics": classification_report(y_true, preds, labels=LABELS, output_dict=True, zero_division=0)
        }

    # 3. Process Transformer Inference
    if t_model:
        preds = []
        for text in X_test:
            inputs = t_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64, return_token_type_ids=False)
            with torch.no_grad():
                out = t_model(**inputs)
                preds.append(t_encoder.inverse_transform([np.argmax(out.logits.numpy()[0])])[0])
        cm = confusion_matrix(y_true, preds, labels=LABELS).tolist()
        report_payload["v1_T"] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "matrix": cm,
            "metrics": classification_report(y_true, preds, labels=LABELS, output_dict=True, zero_division=0)
        }

    # Persist the processed metrics to the permanent benchmark directory
    os.makedirs("storage/benchmarks", exist_ok=True)
    out_path = os.path.join("storage/benchmarks", output_json_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"[SUCCESS] Metrics frozen at: {out_path}\n")

if __name__ == "__main__":
    run_evaluation_suite("faker_benchmark_v1.csv", "faker_benchmark.json")
    run_evaluation_suite("kaggle_test_data.csv", "uci_retail_shift.json")