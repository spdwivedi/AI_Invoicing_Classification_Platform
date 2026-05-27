import os
import sys
import pandas as pd
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate_pipeline(dataset_path="data.csv"):
    if not os.path.exists(dataset_path):
        print(f"Error: Target evaluation dataset not found at {dataset_path}")
        return

    print(f"=== Beginning Evaluation Engine on: {dataset_path} ===")
    df = pd.read_csv(dataset_path)
    X_test = df['text'].tolist()
    y_true = df['category'].tolist()
    
    # Fixed production-grade layout sequences to avoid matrix dimension errors
    unique_labels = ["Cloud/Software", "Inventory", "Logistics", "Office Supplies", "Travel", "Utilities"]

    # ---------------------------------------------------------
    # 1. Evaluate v1_L (Logistic Regression)
    # ---------------------------------------------------------
    print("\n--- Evaluating v1_L (TF-IDF + Logistic Regression) ---")
    v1_l_path = "v1_L/model.joblib"
    if os.path.exists(v1_l_path):
        model_l = joblib.load(v1_l_path)
        y_pred_l = model_l.predict(X_test)
        print(f"Overall Accuracy: {accuracy_score(y_true, y_pred_l):.4f}")
        print("\nDetailed Classification Metrics:")
        print(classification_report(y_true, y_pred_l, labels=unique_labels, target_names=unique_labels, zero_division=0))
        
        print("Text Confusion Matrix (Rows = Actual, Columns = Predicted):")
        cm_l = confusion_matrix(y_true, y_pred_l, labels=unique_labels)
        cm_df_l = pd.DataFrame(cm_l, index=unique_labels, columns=unique_labels)
        print(cm_df_l.to_string())
    else:
        print("Skipping v1_L: Model artifact missing.")

    # ---------------------------------------------------------
    # 2. Evaluate v1_NB (Naive Bayes)
    # ---------------------------------------------------------
    print("\n--- Evaluating v1_NB (CountVectorizer + Naive Bayes) ---")
    v1_nb_path = "v1_NB/model.joblib"
    if os.path.exists(v1_nb_path):
        model_nb = joblib.load(v1_nb_path)
        y_pred_nb = model_nb.predict(X_test)
        print(f"Overall Accuracy: {accuracy_score(y_true, y_pred_nb):.4f}")
        print("\nDetailed Classification Metrics:")
        # FIXED: Now securely passing y_pred_nb instead of y_pred_l
        print(classification_report(y_true, y_pred_nb, labels=unique_labels, target_names=unique_labels, zero_division=0))
        
        print("Text Confusion Matrix (Rows = Actual, Columns = Predicted):")
        cm_nb = confusion_matrix(y_true, y_pred_nb, labels=unique_labels)
        cm_df_nb = pd.DataFrame(cm_nb, index=unique_labels, columns=unique_labels)
        print(cm_df_nb.to_string())
    else:
        print("Skipping v1_NB: Model artifact missing.")

    # ---------------------------------------------------------
    # 3. Evaluate v1_T (Transformer Model)
    # ---------------------------------------------------------
    print("\n--- Evaluating v1_T (Deep Learning Transformer) ---")
    transformer_dir = "v1_T/saved_transformer_model"
    encoder_path = "v1_T/label_encoder.joblib"
    
    if os.path.exists(transformer_dir) and os.path.exists(encoder_path):
        tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
        model_t = AutoModelForSequenceClassification.from_pretrained(transformer_dir)
        label_encoder = joblib.load(encoder_path)
        model_t.eval()
        
        y_pred_t = []
        for text in X_test:
            inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=64)
            with torch.no_grad():
                outputs = model_t(**inputs)
                max_idx = np.argmax(outputs.logits.numpy()[0])
                predicted_label = label_encoder.inverse_transform([max_idx])[0]
                y_pred_t.append(predicted_label)
                
        print(f"Overall Accuracy: {accuracy_score(y_true, y_pred_t):.4f}")
        print("\nDetailed Classification Metrics:")
        # FIXED: Now securely passing y_pred_t instead of y_pred_l
        print(classification_report(y_true, y_pred_t, labels=unique_labels, target_names=unique_labels, zero_division=0))
        
        print("Text Confusion Matrix (Rows = Actual, Columns = Predicted):")
        cm_t = confusion_matrix(y_true, y_pred_t, labels=unique_labels)
        cm_df_t = pd.DataFrame(cm_t, index=unique_labels, columns=unique_labels)
        print(cm_df_t.to_string())
    else:
        print("Skipping v1_T: Model weights or label encoder missing.")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    evaluate_pipeline(target_file)