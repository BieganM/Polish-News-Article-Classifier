import re
from typing import Optional
from newspaper import Article, ArticleException


def clean_polish_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_text_from_url(url: str) -> str:
    if not url or not url.strip():
        raise ValueError("URL jest pusty.")

    try:
        article = Article(url, language="pl")
        article.download()
        article.parse()

        extracted_text = article.text
        if not extracted_text:
            raise ValueError("Nie udało się wyodrębnić tekstu z artykułu.")

        return clean_polish_text(extracted_text)

    except ArticleException as e:
        raise ValueError(f"Błąd podczas przetwarzania artykułu z URL: {url}") from e


def validate_text_input(text: Optional[str]) -> tuple[bool, str]:
    if not text:
        return False, "Tekst jest pusty."

    text = text.strip()
    if len(text) < 50:
        return False, "Tekst jest zbyt krótki (minimum 50 znaków)."

    if len(text) > 50000:
        return False, "Tekst jest zbyt długi (maksimum 50000 znaków)."

    return True, text
