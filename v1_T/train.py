import os
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

class InvoiceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def run_training():
    print("Loading unified text matrix for Transformer adjustment...")
    df = pd.read_csv("../data.csv")
    
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(df['category'])
    joblib.dump(label_encoder, "label_encoder.joblib")
    
    # Swapped to a native WordPiece architecture to eliminate sentencepiece requirements
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    print("Tokenizing textual inputs via WordPiece engine...")
    encodings = tokenizer(list(df['text']), truncation=True, padding=True, max_length=64)
    dataset = InvoiceDataset(encodings, numerical_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=6)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=7,  # Slight increase to fine-tune DistilBERT layers effectively
        per_device_train_batch_size=16,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    print("Starting Deep Learning weights adjustment loop...")
    trainer.train()
    
    model.save_pretrained("./saved_transformer_model")
    tokenizer.save_pretrained("./saved_transformer_model")
    print("Transformer pipeline successfully persisted inside v1_T/saved_transformer_model")

if __name__ == "__main__":
    run_training()