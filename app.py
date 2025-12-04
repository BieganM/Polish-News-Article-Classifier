from flask import Flask, request, render_template
from models.model import (
    load_model,
    predict_category,
    train_model_with_params,
)
from models.utils import extract_text_from_url
import os

app = Flask(__name__)

current_model_type = 'herbert'
model, tokenizer_or_vectorizer, label_encoder = load_model(current_model_type)
if model is None:
    print("No model found, training a new one...")
    params = {'model_type': current_model_type, 'epochs': 3}
    model, tokenizer_or_vectorizer, label_encoder, _, _, _, _ = train_model_with_params(params)
    print("Model training complete.")


@app.route("/", methods=["GET", "POST"])
def home():
    global current_model_type, model, tokenizer_or_vectorizer, label_encoder
    available_models = {
        'herbert': os.path.exists('./model_herbert'),
        'bert': os.path.exists('./model_bert'),
        'mlp': os.path.exists('./model_mlp'),
    }
    model_labels = {
        'herbert': f"HerBERT {'(Dostępny)' if available_models['herbert'] else '(Należy wytrenować)'}",
        'bert': f"BERT {'(Dostępny)' if available_models['bert'] else '(Należy wytrenować)'}",
        'mlp': f"MLP {'(Dostępny)' if available_models['mlp'] else '(Należy wytrenować)'}",
    }
    if request.method == "POST":
        selected_model = request.form.get("model_type", "herbert")
        if selected_model != current_model_type:
            current_model_type = selected_model
            model, tokenizer_or_vectorizer, label_encoder = load_model(current_model_type)
            if model is None:
                return render_template("index.html", error=f"Model {selected_model} not found. Please train it first.", available_models=available_models, model_labels=model_labels)

        text = ""
        if "text" in request.form and request.form["text"].strip():
            text = request.form["text"]
        elif "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            text = file.read().decode("utf-8")
        elif "url" in request.form and request.form["url"].strip():
            url = request.form["url"]
            text = extract_text_from_url(url)

        if text:
            category, confidence = predict_category(
                text, model, tokenizer_or_vectorizer, label_encoder, current_model_type
            )
            return render_template(
                "index.html", category=category, confidence=confidence, selected_model=current_model_type, available_models=available_models, model_labels=model_labels
            )
        else:
            return render_template(
                "index.html",
                error="Proszę wprowadzić tekst, wybrać plik lub podać URL.",
                selected_model=current_model_type, available_models=available_models, model_labels=model_labels
            )
    return render_template("index.html", selected_model=current_model_type, available_models=available_models, model_labels=model_labels)


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        model_type = request.form.get("model_type", "herbert")
        params = {
            "model_type": model_type,
            "epochs": int(request.form.get("epochs", 10)),
            "lr": float(request.form.get("lr", 0.001)),
            "batch_size": int(request.form.get("batch_size", 32)),
            "max_features": int(request.form.get("max_features", 5000)),
            "hidden_size": int(request.form.get("hidden_size", 128)),
            "num_layers": int(request.form.get("num_layers", 2)),
            "dropout": float(request.form.get("dropout", 0.5)),
            "early_stopping": request.form.get("early_stopping") == "on",
        }
        global current_model_type, model, tokenizer_or_vectorizer, label_encoder
        if model_type == 'herbert':
            model, tokenizer_or_vectorizer, label_encoder, losses, accuracies, plot_loss, plot_acc = train_model_with_params(params)
            final_accuracy = accuracies[-1] if accuracies else 0
            current_model_type = model_type
            return render_template(
                "train.html",
                losses=losses,
                accuracies=accuracies,
                final_accuracy=final_accuracy,
                plot_loss=plot_loss,
                plot_acc=plot_acc,
                params=params,
                selected_model=model_type,
            )
        elif model_type == 'bert':
            model, tokenizer_or_vectorizer, label_encoder, losses, accuracies, plot_loss, plot_acc = train_model_with_params(params)
            final_accuracy = accuracies[-1] if accuracies else 0
            current_model_type = model_type
            return render_template(
                "train.html",
                losses=losses,
                accuracies=accuracies,
                final_accuracy=final_accuracy,
                plot_loss=plot_loss,
                plot_acc=plot_acc,
                params=params,
                selected_model=model_type,
            )
        elif model_type == 'mlp':
            model, tokenizer_or_vectorizer, label_encoder, losses, accuracies, plot_loss, plot_acc = train_model_with_params(params)
            final_accuracy = accuracies[-1] if accuracies else 0
            current_model_type = model_type
            return render_template(
                "train.html",
                losses=losses,
                accuracies=accuracies,
                final_accuracy=final_accuracy,
                plot_loss=plot_loss,
                plot_acc=plot_acc,
                params=params,
                selected_model=model_type,
            )
    return render_template("train.html", selected_model="herbert")


if __name__ == "__main__":
    app.run(debug=True)
