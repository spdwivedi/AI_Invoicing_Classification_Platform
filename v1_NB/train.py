import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def run_training():
    print("Extracting data from root path for Naive Bayes baseline...")
    # Reads the same shared dataset located one directory up
    df = pd.read_csv("../data.csv")
    
    # Configure Naive Bayes Pipeline using raw frequency counts
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english')),
        ('classifier', MultinomialNB(alpha=1.0))
    ])
    
    print("Fitting Multinomial Naive Bayes engine...")
    pipeline.fit(df['text'], df['category'])
    
    # Serialize the Naive Bayes model pipeline artifact
    joblib.dump(pipeline, "model.joblib")
    print("Artifact successfully saved inside v1_NB directory: model.joblib")

if __name__ == "__main__":
    run_training()