from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelPaths:
    data_dir: Path = Path("./Scrapper")
    data_file: str = "polsatnews_articles_clean.csv"
    results_dir: Path = Path("./results")
    logs_dir: Path = Path("./logs")
    herbert_dir: Path = Path("./model_herbert")
    bert_dir: Path = Path("./model_bert")
    mlp_dir: Path = Path("./model_mlp")

    @property
    def data_path(self) -> Path:
        return self.data_dir / self.data_file


@dataclass
class TransformerConfig:
    epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    max_length: int = 256
    logging_steps: int = 10
    early_stopping: bool = False
    early_stopping_patience: int = 3
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 2


@dataclass
class MLPConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    max_features: int = 5000
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.4
    early_stopping: bool = False
    early_stopping_patience: int = 3
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5


MODEL_NAMES = {
    "herbert": "allegro/herbert-base-cased",
    "bert": "bert-base-multilingual-cased",
}


def get_device() -> torch.device:
    # Check for environment variable override first
    device_str = os.getenv("DEVICE")
    if device_str:
        print(f"Using device from environment variable: {device_str}")
        return torch.device(device_str)

    # Auto-detection logic
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


PATHS = ModelPaths()
DEVICE = get_device()
