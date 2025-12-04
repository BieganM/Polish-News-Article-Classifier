from models.model import load_model

def test_load_model():
    model, vectorizer, label_encoder = load_model()
    assert model is not None, "Model should not be None"
    assert vectorizer is not None, "Vectorizer should not be None"
    assert label_encoder is not None, "LabelEncoder should not be None"
    print("Model loaded successfully!")

if __name__ == "__main__":
    test_load_model()
