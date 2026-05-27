import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def run_training():
    print("Extracting data from root path...")
    # Read the shared dataset located one directory up
    df = pd.read_csv("../data.csv")
    
    # Configure production-ready ML pipeline pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            ngram_range=(1, 2), 
            stop_words='english', 
            sublinear_tf=True
        )),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000))
    ])
    
    print("Fitting Logistic Regression engine...")
    pipeline.fit(df['text'], df['category'])
    
    # Serialize model artifact directly to folder
    joblib.dump(pipeline, "model.joblib")
    print("Artifact successfully saved inside v1_L directory: model.joblib")

if __name__ == "__main__":
    run_training()