import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score
import os
from .utils import get_device, load_and_preprocess_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.5):
        super(TextClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.BatchNorm1d(hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_model_with_params(params):
    model_type = params.get('model_type', 'herbert')
    if model_type == 'herbert':
        return train_herbert(params)
    elif model_type == 'mlp':
        return train_mlp(params)
    elif model_type == 'bert':
        return train_bert(params)
    else:
        raise ValueError("Unsupported model type")

def train_herbert(params):
    train_dataset, test_dataset, tokenizer, label_encoder = load_and_preprocess_data(model_type='herbert')
    num_classes = len(label_encoder.classes_)
    device = get_device()

    model = AutoModelForSequenceClassification.from_pretrained(
        "allegro/herbert-base-cased", num_labels=num_classes
    ).to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=params.get('epochs', 3),
        per_device_train_batch_size=params.get('batch_size', 16),
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    callbacks = []
    if params.get('early_stopping', False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Starting HerBERT training...")
    trainer.train()
    print("HerBERT training completed.")

    logs = trainer.state.log_history
    losses = [log['train_loss'] for log in logs if 'train_loss' in log]
    accuracies = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]

    model_dir = './model_herbert'
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_loss = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(accuracies)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_acc = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return model, tokenizer, label_encoder, losses, accuracies, plot_loss, plot_acc

def train_mlp(params):
    df = pd.read_csv('./Scrapper/polsatnews_articles_clean.csv')
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])

    vectorizer = TfidfVectorizer(max_features=params.get('max_features', 2000))
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['category_encoded'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=params.get('batch_size', 32), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = get_device()
    input_size = X.shape[1]
    hidden_size = params.get('hidden_size', 128)
    num_classes = len(label_encoder.classes_)
    num_layers = params.get('num_layers', 2)
    dropout = params.get('dropout', 0.5)

    model = TextClassifier(input_size, hidden_size, num_classes, num_layers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.get('lr', 0.001))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = params.get('epochs', 10)
    losses = []
    accuracies = []

    best_acc = 0
    patience = 3
    counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        losses.append(epoch_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}, Accuracy: {accuracy:.4f}')

        if params.get('early_stopping', False):
            if accuracy > best_acc:
                best_acc = accuracy
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

    model_dir = './model_mlp'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_loss = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(accuracies)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_acc = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return model, vectorizer, label_encoder, losses, accuracies, plot_loss, plot_acc

def train_bert(params):
    train_dataset, test_dataset, tokenizer, label_encoder = load_and_preprocess_data(model_type='bert')
    num_classes = len(label_encoder.classes_)
    device = get_device()

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=num_classes
    ).to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=params.get('epochs', 3),
        per_device_train_batch_size=params.get('batch_size', 16),
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    callbacks = []
    if params.get('early_stopping', False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Starting BERT training...")
    trainer.train()
    print("BERT training completed.")

    logs = trainer.state.log_history
    losses = [log['train_loss'] for log in logs if 'train_loss' in log]
    accuracies = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]

    model_dir = './model_bert'
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_loss = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(accuracies)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_acc = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return model, tokenizer, label_encoder, losses, accuracies, plot_loss, plot_acc

def load_model(model_type='herbert'):
    if model_type == 'herbert':
        model_dir = './model_herbert'
        if os.path.exists(model_dir):
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
            return model, tokenizer, label_encoder
    elif model_type == 'bert':
        model_dir = './model_bert'
        if os.path.exists(model_dir):
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
            return model, tokenizer, label_encoder
    elif model_type == 'mlp':
        model_dir = './model_mlp'
        if os.path.exists(model_dir):
            vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
            label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
            input_size = vectorizer.max_features
            hidden_size = 128  # assuming default
            num_classes = len(label_encoder.classes_)
            model = TextClassifier(input_size, hidden_size, num_classes)
            model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
            model.eval()
            return model, vectorizer, label_encoder
    return None, None, None

def predict_category(text, model, tokenizer_or_vectorizer, label_encoder, model_type='herbert'):
    if model_type in ['herbert', 'bert']:
        inputs = tokenizer_or_vectorizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(get_device())
        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        category = label_encoder.inverse_transform([predicted_class_id])[0]
        confidence = torch.softmax(logits, dim=1).max().item()
    elif model_type == 'mlp':
        X = tokenizer_or_vectorizer.transform([text]).toarray()
        X = torch.tensor(X, dtype=torch.float32).to(get_device())
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            predicted_class_id = predicted.item()
            category = label_encoder.inverse_transform([predicted_class_id])[0]
            confidence = torch.softmax(outputs, dim=1).max().item()
    return category, confidence
