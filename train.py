from models.model import train_model_with_params


def main():
    params = {
        "model_type": "herbert",
        "epochs": 10,
        "batch_size": 64,
    }
    train_model_with_params(params)
    print("Model training complete and saved to ./model")


if __name__ == "__main__":
    main()
