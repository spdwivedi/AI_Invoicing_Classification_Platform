import requests
import pandas as pd
import os

# Base API address pointing to active Docker container
BASE_URL = "http://127.0.0.1:8000"

# Define a set of highly distinct, unseen test strings to verify processing live
sample_invoices = [
    "Flipkart internet private ltd order summary for corporate laptop charger",
    "Amazon Web Services data storage node allocation invoice premium",
    "DTDC courier shipping charges for transit of regional document logs",
    "BSNL corporate landline and office fiber broadband link connectivity payment",
    "Makemytrip business class flight itinerary to Mumbai center for audit",
    "Raw aluminum alloy component sheet delivery for manufacturing shop floor"
]

def test_hardcoded_samples():
    print("=== Testing Diverse Invoice Samples Against All Docker Endpoints ===")
    
    endpoints = {
        "v1_L (Linear)": f"{BASE_URL}/v1_L/predict",
        "v1_NB (Bayes)": f"{BASE_URL}/v1_NB/predict",
        "v1_T (BERT)":   f"{BASE_URL}/v1_T/predict"
    }

    for text in sample_invoices:
        print(f"\n📥 Raw Input: '{text}'")
        print("-" * 85)
        
        # Query each model layer sequentially
        for model_name, url in endpoints.items():
            try:
                response = requests.post(url, json={"text": text}, timeout=5)
                if response.status_code == 200:
                    res_data = response.json()
                    pred = res_data.get("category")
                    conf = res_data.get("confidence")
                    print(f" 🤖 {model_name:15} -> Predicted: {pred:16} | Confidence: {conf:.4f}")
                else:
                    print(f" ❌ {model_name:15} -> HTTP Error {response.status_code}")
            except Exception as e:
                print(f" ❌ {model_name:15} -> Connection failure: {str(e)}")

def test_csv_file(csv_path):
    if not os.path.exists(csv_path):
        print(f"\nError: File not found at {csv_path}")
        return

    print(f"\n=== Batch Querying CSV Dataset: {csv_path} ===")
    df = pd.read_csv(csv_path).head(5) # Evaluates the top 5 lines as a quick snapshot test
    
    for idx, row in df.iterrows():
        text = row['text']
        actual = row['category']
        print(f"\n[{idx+1}] Actual Category: {actual} | Text: '{text}'")
        
        # Querying the deep learning transformer route specifically
        url = f"{BASE_URL}/v1_T/predict"
        try:
            res = requests.post(url, json={"text": text})
            if res.status_code == 200:
                data = res.json()
                print(f"    🎯 DistilBERT Prediction -> {data['category']} (Confidence: {data['confidence']})")
        except Exception as e:
            print(f"    ❌ Request failed: {e}")

if __name__ == "__main__":
    # 1. Run immediate comparison on the sample list
    test_hardcoded_samples()
    
    # 2. Run batch comparison on frozen benchmark file
    test_csv_file("faker_benchmark_v1.csv")