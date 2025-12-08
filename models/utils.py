import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from datasets import Dataset
from newspaper import Article

def get_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def load_and_preprocess_data(model_type="herbert", max_len=128):
    df = pd.read_csv('./Scrapper/polsatnews_articles_clean.csv')
    
    if model_type in ['herbert', 'bert']:
        label_encoder = LabelEncoder()
        df['category_encoded'] = label_encoder.fit_transform(df['category'])
        
        dataset = Dataset.from_pandas(df)
        
        if model_type == 'herbert':
            tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
        elif model_type == 'bert':
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text", "category"])
        tokenized_datasets = tokenized_datasets.rename_column("category_encoded", "labels")
        tokenized_datasets.set_format("torch")

        train_dataset, test_dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42).values()
        
        return train_dataset, test_dataset, tokenizer, label_encoder
    elif model_type == 'mlp':
        return None, None, None, None
    else:
        raise ValueError("Unsupported model type")

