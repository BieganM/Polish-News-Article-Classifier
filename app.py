import os
import threading
import logging
from flask import Flask, request, render_template, jsonify

from models.model import (
    load_model,
    predict_category,
    train_model_with_params,
    get_training_status,
    get_last_training_result,
    reset_training_status,
)
from models.utils import extract_text_from_url, validate_text_input
from models.config import PATHS, TransformerConfig, MLPConfig

# --- Application Setup ---

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


class ModelManager:
    """Thread-safe manager for loading and handling ML models."""

    def __init__(self, default_model_type="herbert"):
        self.model_type = default_model_type
        self.model = None
        self.tokenizer_or_vectorizer = None
        self.label_encoder = None
        self._lock = threading.Lock()
        self.load()

    def load(self, model_type=None):
        """Load the specified model type or the current one."""
        with self._lock:
            if model_type:
                self.model_type = model_type

            (
                self.model,
                self.tokenizer_or_vectorizer,
                self.label_encoder,
            ) = load_model(self.model_type)

            if self.model is None:
                logging.warning(
                    f"Failed to load model '{self.model_type}'. It may need to be trained."
                )
                return False
        return True

    def predict(self, text: str):
        """Predict category for the given text."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please train it first.")

        with self._lock:
            return predict_category(
                text,
                self.model,
                self.tokenizer_or_vectorizer,
                self.label_encoder,
                self.model_type,
            )

    @property
    def is_loaded(self):
        return self.model is not None


# --- Global Instances ---

model_manager = ModelManager()
training_thread = None
training_lock = threading.Lock()


# --- Helper Functions ---


def get_available_models():
    """Checks for the existence of trained model artifacts."""
    return {
        "herbert": os.path.exists(PATHS.herbert_dir),
        "bert": os.path.exists(PATHS.bert_dir),
        "mlp": os.path.exists(PATHS.mlp_dir),
    }


def get_model_labels(available_models):
    """Generates display labels for models, indicating their status."""
    labels = {
        "herbert": "HerBERT (Polski)",
        "bert": "BERT (Wielojęzyczny)",
        "mlp": "MLP (TF-IDF)",
    }
    return {
        key: f"{labels[key]} {'✓' if available else '(Należy wytrenować)'}"
        for key, available in available_models.items()
    }


def extract_text_from_request(form, files):
    """Extracts text from form, file, or URL, raising ValueError on failure."""
    if form.get("text", "").strip():
        return form["text"].strip()

    if "file" in files and files["file"].filename:
        try:
            return files["file"].read().decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                "Nie udało się odczytać pliku. Upewnij się, że jest w formacie UTF-8."
            )

    if form.get("url", "").strip():
        return extract_text_from_url(form["url"].strip())

    return None


# --- Background Training ---


def _background_train(params):
    """
    Runs the training process in a background thread and updates the status.
    Reloads the model on successful completion.
    """
    global model_manager
    # Directly import the model module to access its status variables
    from models import model as model_module

    try:
        train_model_with_params(params)
        # On success, reload the newly trained model
        model_manager.load(params.get("model_type"))

        # Mark training as completed successfully
        with model_module._status_lock:
            model_module.training_status["message"] = "completed"

    except Exception as e:
        # Log the full exception traceback for detailed debugging
        logging.exception("An error occurred during model training.")

        # Update the status to reflect the error
        with model_module._status_lock:
            model_module.training_status.update(
                {"running": False, "error": str(e), "message": "failed"}
            )

    finally:
        # Ensure the 'running' flag is set to False, regardless of outcome
        with model_module._status_lock:
            if model_module.training_status.get("running"):
                model_module.training_status["running"] = False


# --- Flask Routes ---


@app.route("/", methods=["GET", "POST"])
def home():
    available_models = get_available_models()
    model_labels = get_model_labels(available_models)
    template_context = {
        "available_models": available_models,
        "model_labels": model_labels,
        "selected_model": model_manager.model_type,
    }

    if request.method == "POST":
        selected_model = request.form.get("model_type", "herbert")

        if selected_model != model_manager.model_type:
            if not model_manager.load(selected_model):
                template_context["error"] = (
                    f"Model {selected_model} nie został znaleziony. Proszę go wytrenować."
                )
                template_context["selected_model"] = model_manager.model_type
                return render_template("index.html", **template_context)

        template_context["selected_model"] = selected_model

        try:
            text = extract_text_from_request(request.form, request.files)
            if not text:
                template_context["error"] = (
                    "Proszę wprowadzić tekst, wybrać plik lub podać URL."
                )
                return render_template("index.html", **template_context)

            is_valid, validation_result = validate_text_input(text)
            if not is_valid:
                template_context["error"] = validation_result
                return render_template("index.html", **template_context)

            clean_text = validation_result

            if not model_manager.is_loaded:
                template_context["error"] = (
                    "Model nie jest załadowany. Proszę go najpierw wytrenować."
                )
                return render_template("index.html", **template_context)

            category, confidence = model_manager.predict(clean_text)
            template_context["category"] = category
            template_context["confidence"] = confidence

        except (ValueError, RuntimeError) as e:
            template_context["error"] = str(e)

    return render_template("index.html", **template_context)


@app.route("/train", methods=["GET", "POST"])
def train():
    global training_thread
    model_type = request.form.get("model_type", "herbert")
    defaults = TransformerConfig() if model_type in ("herbert", "bert") else MLPConfig()

    if get_training_status().get("running"):
        return render_template(
            "train.html",
            selected_model=model_type,
            training=True,
            status=get_training_status(),
            defaults=defaults,
        )

    if request.method == "POST":
        # Consolidate parameters from form, with defaults from config
        try:
            if model_type in ("herbert", "bert"):
                cfg = TransformerConfig()
                params = {
                    "model_type": model_type,
                    "epochs": int(request.form.get("epochs", cfg.epochs)),
                    "lr": float(request.form.get("lr", cfg.learning_rate)),
                    "batch_size": int(request.form.get("batch_size", cfg.batch_size)),
                    "max_length": int(request.form.get("max_length", cfg.max_length)),
                    "early_stopping": request.form.get("early_stopping") == "on",
                    "early_stopping_patience": int(
                        request.form.get(
                            "early_stopping_patience", cfg.early_stopping_patience
                        )
                    ),
                }
            elif model_type == "mlp":
                cfg = MLPConfig()
                params = {
                    "model_type": model_type,
                    "epochs": int(request.form.get("epochs", cfg.epochs)),
                    "lr": float(request.form.get("lr", cfg.learning_rate)),
                    "batch_size": int(request.form.get("batch_size", cfg.batch_size)),
                    "max_features": int(
                        request.form.get("max_features", cfg.max_features)
                    ),
                    "hidden_size": int(
                        request.form.get("hidden_size", cfg.hidden_size)
                    ),
                    "num_layers": int(request.form.get("num_layers", cfg.num_layers)),
                    "dropout": float(request.form.get("dropout", cfg.dropout)),
                    "early_stopping": request.form.get("early_stopping") == "on",
                    "early_stopping_patience": int(
                        request.form.get(
                            "early_stopping_patience", cfg.early_stopping_patience
                        )
                    ),
                }
            else:
                return "Unsupported model type", 400
        except (ValueError, TypeError) as e:
            return f"Invalid parameter type: {e}", 400

        with training_lock:
            if training_thread is None or not training_thread.is_alive():
                reset_training_status(model_type, params)
                training_thread = threading.Thread(
                    target=_background_train, args=(params,), daemon=True
                )
                training_thread.start()

        return render_template(
            "train.html",
            selected_model=model_type,
            training=True,
            params=params,
            status=get_training_status(),
            defaults=defaults,
        )

    # For GET request
    return render_template(
        "train.html", selected_model=model_type, training=False, defaults=defaults
    )


@app.route("/training_status", methods=["GET"])
def training_status_route():
    return jsonify(get_training_status())


@app.route("/training_result", methods=["GET"])
def training_result_route():
    res = get_last_training_result()
    if not res:
        return jsonify({"ready": False})
    return jsonify({"ready": True, **res})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
