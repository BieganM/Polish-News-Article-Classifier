import io
import base64
import os
import threading
import logging
from dataclasses import asdict
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from datasets import Dataset, ClassLabel

from .config import (
    PATHS,
    DEVICE,
    MODEL_NAMES,
    TransformerConfig,
    MLPConfig,
)

# --- Training State Management ---

_status_lock = threading.Lock()
training_status: Dict = {"running": False, "message": "idle"}
last_train_result: Optional[Dict] = None


def get_training_status() -> Dict:
    with _status_lock:
        return training_status.copy()


def get_last_training_result() -> Optional[Dict]:
    global last_train_result
    return last_train_result


def reset_training_status(model_type: str, params: Dict):
    """Initializes or resets the training status."""
    global training_status, last_train_result
    with _status_lock:
        training_status = {
            "running": True,
            "model_type": model_type,
            "epoch": 0,
            "epochs": params.get("epochs", 0),
            "progress": 0.0,
            "loss": None,
            "accuracy": None,
            "message": "starting",
            "error": None,
            "completed": False,
        }
        last_train_result = None


def set_training_error(error_message: str):
    """Sets the training status to an error state."""
    with _status_lock:
        training_status.update(
            {"running": False, "error": error_message, "message": "failed"}
        )


def set_training_completed():
    """Sets the training status to completed."""
    with _status_lock:
        if training_status.get("running"):
            training_status.update(
                {
                    "running": False,
                    "completed": True,
                    "message": "completed",
                    "progress": 100.0,
                }
            )


# --- Model & Training Logic ---


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


class ProgressCallback(TrainerCallback):
    """Updates training progress for the UI during transformer training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        with _status_lock:
            if "loss" in logs:
                training_status.update(
                    {
                        "loss": float(logs.get("loss")),
                        "epoch": int(state.epoch or 0),
                        "progress": (
                            min(100.0, (state.global_step / state.max_steps) * 100.0)
                            if state.max_steps
                            else 0.0
                        ),
                        "message": "training",
                    }
                )
            if "eval_accuracy" in logs:
                training_status.update({"accuracy": float(logs.get("eval_accuracy"))})


class TextClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _create_plot(values: List[float], title: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values, marker="o", linewidth=2, markersize=4)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return encoded


def _load_dataset_for_transformer(model_type: str, max_length: int = 256):
    df = pd.read_csv(PATHS.data_path)
    label_encoder = LabelEncoder()
    df["labels"] = label_encoder.fit_transform(df["category"])
    dataset = Dataset.from_pandas(df[["text", "labels"]])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_type])

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True).remove_columns(["text"])
    tokenized = tokenized.cast_column(
        "labels",
        ClassLabel(
            num_classes=len(label_encoder.classes_), names=list(label_encoder.classes_)
        ),
    )
    split = tokenized.train_test_split(
        test_size=0.2, seed=42, stratify_by_column="labels"
    )

    return split["train"], split["test"], tokenizer, label_encoder


def _train_transformer(model_type: str, config: TransformerConfig):
    """Train a transformer-based model (HerBERT or BERT)."""
    train_dataset, test_dataset, tokenizer, label_encoder = (
        _load_dataset_for_transformer(model_type, config.max_length)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAMES[model_type], num_labels=len(label_encoder.classes_)
    ).to(DEVICE)

    model_dir = PATHS.herbert_dir if model_type == "herbert" else PATHS.bert_dir

    training_args = TrainingArguments(
        output_dir=str(PATHS.results_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        logging_dir=str(PATHS.logs_dir),
        logging_steps=10,  # Log more frequently for better UI feedback
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
    )

    callbacks = [ProgressCallback()]
    if config.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logging.info(f"Starting {model_type.upper()} training on {DEVICE}...")
    trainer.train()
    logging.info(f"{model_type.upper()} training completed.")

    logs = trainer.state.log_history
    losses = [log["loss"] for log in logs if "loss" in log]
    accuracies = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]

    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")

    global last_train_result
    last_train_result = {
        "losses": losses,
        "accuracies": accuracies,
        "plot_loss": _create_plot(losses, "Training Loss", "Loss"),
        "plot_acc": _create_plot(accuracies, "Validation Accuracy", "Accuracy"),
        "params": asdict(config),
    }
    set_training_completed()


def _train_mlp(config: MLPConfig):
    df = pd.read_csv(PATHS.data_path)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])

    vectorizer = TfidfVectorizer(
        max_features=config.max_features, ngram_range=(1, 2), sublinear_tf=True
    )
    X = vectorizer.fit_transform(df["text"]).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = TextClassifier(
        X.shape[1],
        config.hidden_size,
        len(label_encoder.classes_),
        config.num_layers,
        config.dropout,
    ).to(DEVICE)

    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(y_train), y=y_train),
        dtype=torch.float32,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses, accuracies = [], []
    best_acc, patience_counter = 0, 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch.to(DEVICE))
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted.cpu() == y_batch).sum().item()

        accuracy = correct / total
        accuracies.append(accuracy)
        logging.info(
            f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        with _status_lock:
            training_status.update(
                {
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "accuracy": accuracy,
                    "progress": ((epoch + 1) / config.epochs) * 100,
                }
            )

        if config.early_stopping and accuracy > best_acc:
            best_acc, patience_counter = accuracy, 0
        elif config.early_stopping:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logging.info("Early stopping triggered.")
                break

    os.makedirs(PATHS.mlp_dir, exist_ok=True)
    torch.save(model.state_dict(), PATHS.mlp_dir / "model.pth")
    joblib.dump(vectorizer, PATHS.mlp_dir / "vectorizer.joblib")
    joblib.dump(label_encoder, PATHS.mlp_dir / "label_encoder.joblib")
    joblib.dump(asdict(config), PATHS.mlp_dir / "model_config.joblib")

    global last_train_result
    last_train_result = {
        "losses": losses,
        "accuracies": accuracies,
        "plot_loss": _create_plot(losses, "Training Loss", "Loss"),
        "plot_acc": _create_plot(accuracies, "Validation Accuracy", "Accuracy"),
        "params": asdict(config),
    }
    set_training_completed()


def train_model_with_params(params: dict):
    model_type = params.get("model_type", "herbert")

    try:
        if model_type in ("herbert", "bert"):
            config = TransformerConfig(
                **{k: v for k, v in params.items() if k in asdict(TransformerConfig())}
            )
            _train_transformer(model_type, config)
        elif model_type == "mlp":
            config = MLPConfig(
                **{k: v for k, v in params.items() if k in asdict(MLPConfig())}
            )
            _train_mlp(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        set_training_error(str(e))
        # Re-raise the exception so the background thread in app.py can see it
        raise e


def load_model(model_type: str = "herbert") -> Tuple:
    model_map = {
        "herbert": PATHS.herbert_dir,
        "bert": PATHS.bert_dir,
        "mlp": PATHS.mlp_dir,
    }
    model_dir = model_map.get(model_type)

    if not model_dir or not os.path.exists(model_dir):
        return None, None, None

    try:
        if model_type in ("herbert", "bert"):
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            label_encoder = joblib.load(model_dir / "label_encoder.joblib")
            return model.eval(), tokenizer, label_encoder

        if model_type == "mlp":
            config_dict = joblib.load(model_dir / "model_config.joblib")
            config = MLPConfig(**config_dict)

            vectorizer = joblib.load(model_dir / "vectorizer.joblib")
            label_encoder = joblib.load(model_dir / "label_encoder.joblib")

            # Determine input_size from vectorizer
            input_size = vectorizer.get_feature_names_out().shape[0]

            model = TextClassifier(
                input_size,
                config.hidden_size,
                len(label_encoder.classes_),
                config.num_layers,
                config.dropout,
            )
            model.load_state_dict(
                torch.load(model_dir / "model.pth", map_location=DEVICE)
            )
            return model.eval(), vectorizer, label_encoder
    except Exception as e:
        logging.warning(f"Error loading model {model_type}: {e}")
        return None, None, None

    return None, None, None


def predict_category(
    text: str,
    model,
    tokenizer_or_vectorizer,
    label_encoder: LabelEncoder,
    model_type: str = "herbert",
) -> Tuple[str, float]:
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        if model_type in ("herbert", "bert"):
            inputs = tokenizer_or_vectorizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=256
            ).to(DEVICE)
            logits = model(**inputs).logits
        elif model_type == "mlp":
            X = torch.tensor(
                tokenizer_or_vectorizer.transform([text]).toarray(), dtype=torch.float32
            ).to(DEVICE)
            logits = model(X)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probabilities, dim=1)
        category = label_encoder.inverse_transform([predicted_id.item()])[0]

    return category, confidence.item()
