
import argparse
import logging
import re
import time
from typing import Dict, List, Set

import feedparser
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, detect
from newspaper import Article
from stop_words import get_stop_words
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Ensure consistent language detection results
DetectorFactory.seed = 0

# --- Constants ---

FEEDS = {
    'Polska': [
        'https://www.polsatnews.pl/rss/polska.xml',
        'https://tvn24.pl/polska.xml',
        'https://wiadomosci.wp.pl/rss.xml',
        'https://www.fakt.pl/rss',
        'https://dorzeczy.pl/rss',
        'https://www.wprost.pl/rss'
    ],
    'Świat': [
        'https://www.polsatnews.pl/rss/swiat.xml',
        'https://tvn24.pl/swiat.xml',
        'https://www.rmf24.pl/feed/',
        'https://www.newsweek.pl/rss'
    ],
    'Biznes': [
        'https://www.polsatnews.pl/rss/biznes.xml',
        'https://tvn24.pl/biznes.xml',
        'https://www.pb.pl/rss/najnowsze.xml'
    ],
    'Technologie': [
        'https://www.polsatnews.pl/rss/technologie.xml',
        'https://tvn24.pl/technologie.xml',
        'https://www.computerworld.pl/rss',
        'https://spidersweb.pl/rss'
    ],
    'Moto': [
        'https://www.polsatnews.pl/rss/moto.xml',
        'https://tvn24.pl/moto.xml'
    ],
    'Sport': [
        'https://www.polsatnews.pl/rss/sport.xml',
        'https://tvn24.pl/sport.xml',
        'https://sport.onet.pl/.feed'
    ],
}

RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_NON_LETTER = re.compile(r'[^a-zA-ZąćęłńóśżźĄĆĘŁŃÓŚŻŹ\s-]')
RE_MULTI_WS = re.compile(r'\s+')

# --- Text Extraction and Cleaning ---

def extract_full_text(url: str, timeout: int = 10, user_agent: str = 'Mozilla/5.0') -> str | None:
    """
    Tries to extract full article text using newspaper3k.
    If that fails, falls back to requests + BeautifulSoup.
    """
    headers = {'User-Agent': user_agent}
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if text and len(text) > 100:
            return text
    except Exception as e:
        logging.warning(f"Newspaper3k failed for {url}: {e}")

    # Fallback to requests + BeautifulSoup
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'lxml')
        
        selectors = ["article", "div[class*='article']", "div[class*='content']", "div[id*='article']"]
        for sel in selectors:
            el = soup.select_one(sel)
            if el:
                ps = el.find_all('p')
                text = '\n'.join([p.get_text(strip=True) for p in ps if p.get_text(strip=True)])
                if text and len(text) > 100:
                    return text
        
        # Final fallback: join all <p> tags on page
        ps = soup.find_all('p')
        text = '\n'.join([p.get_text(strip=True) for p in ps if p.get_text(strip=True)])
        if text and len(text) > 100:
            return text
    except requests.RequestException as e:
        logging.error(f"Failed to fetch URL {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"BeautifulSoup failed for {url}: {e}")
        return None
        
    return None


def clean_text(text: str) -> str:
    """Removes extra whitespace from text."""
    return re.sub(r'\s+', ' ', text).strip()


def fetch_articles_from_feeds(
    feeds: Dict[str, List[str]],
    max_articles_per_category: int | None = None,
    min_length: int = 200
) -> pd.DataFrame:
    """
    Fetches articles from a dictionary of RSS feeds.
    """
    records = []
    seen_urls = set()

    for category, urls in feeds.items():
        category_entries = []
        for url in urls:
            logging.info(f"Processing category: {category} -> {url}")
            fp = feedparser.parse(url)
            entries = fp.get('entries', [])
            category_entries.extend(entries)

        if max_articles_per_category:
            category_entries = category_entries[:max_articles_per_category]

        for entry in tqdm(category_entries, desc=f"Scraping '{category}'"):
            link = entry.get('link') or entry.get('guid')
            if not link or link in seen_urls:
                continue

            title = entry.get('title', '')
            published = entry.get('published', entry.get('updated'))
            
            # Add a small delay to be polite to servers
            time.sleep(0.2)
            
            text = extract_full_text(link)
            if not text:
                summary = BeautifulSoup(entry.get('summary', ''), 'lxml').get_text(separator=' ', strip=True)
                text = summary

            if not text:
                continue

            text = clean_text(text)
            if len(text) < min_length:
                continue

            # Detect language to ensure it's Polish
            try:
                if detect(text) != 'pl':
                    continue
            except Exception:
                continue
            
            records.append({
                'category': category,
                'title': title,
                'url': link,
                'published': published,
                'text': text
            })
            seen_urls.add(link)

    df = pd.DataFrame(records)
    if not df.empty:
        df.drop_duplicates(subset=['url', 'text'], inplace=True)
    
    return df

# --- NLP Preprocessing ---

def get_polish_stopwords() -> Set[str]:
    """
    Loads Polish stopwords from stop_words package and spaCy.
    """
    try:
        nlp = spacy.load('pl_core_news_sm')
        spacy_stopwords = set(nlp.Defaults.stop_words)
    except OSError:
        logging.warning("SpaCy 'pl_core_news_sm' model not found. Downloading...")
        spacy.cli.download('pl_core_news_sm')
        nlp = spacy.load('pl_core_news_sm')
        spacy_stopwords = set(nlp.Defaults.stop_words)

    base_stopwords = set(get_stop_words('polish'))
    extra_stops = {'z', 'na', 'i', 'w', 'o', 'że', 'się', 'roku', 'r', 'godz', 'dnia', 'fot', 'foto', 'zdjęcie', 'więcej', 'czytaj'}
    
    return base_stopwords | spacy_stopwords | extra_stops

def normalize_text(text: str) -> str:
    """
    Normalizes text by removing URLs, non-letter characters, and extra whitespace.
    """
    if not isinstance(text, str):
        return ''
    text = text.strip()
    text = RE_URL.sub(' ', text)
    text = text.replace('\xa0', ' ')
    text = RE_NON_LETTER.sub(' ', text)
    text = RE_MULTI_WS.sub(' ', text)
    text = text.lower()
    return text

def lemmatize_and_tokenize(
    text: str,
    nlp_model,
    stopwords: Set[str],
    min_token_len: int = 3
) -> List[str]:
    """
    Lemmatizes and tokenizes text using a spaCy model.
    """
    text_norm = normalize_text(text)
    doc = nlp_model(text_norm)
    tokens = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        lemma = tok.lemma_.lower().strip()
        if len(lemma) < min_token_len or lemma in stopwords or lemma.isdigit():
            continue
        tokens.append(lemma)
    return tokens

def prepare_dataset(
    df_in: pd.DataFrame,
    text_col: str = 'text',
    min_tokens: int = 30
) -> pd.DataFrame:
    """
    Creates a cleaned and preprocessed dataset.
    """
    logging.info("Loading spaCy model for lemmatization...")
    try:
        nlp = spacy.load('pl_core_news_sm')
    except OSError:
        logging.error("SpaCy model 'pl_core_news_sm' not found. Please run 'python -m spacy download pl_core_news_sm'")
        return pd.DataFrame()

    stopwords = get_polish_stopwords()
    df = df_in.copy()
    
    logging.info("Starting text preprocessing (lemmatization and tokenization)...")
    
    # Using apply with a lambda to pass extra arguments
    df['tokens'] = df[text_col].progress_apply(
        lambda t: lemmatize_and_tokenize(t, nlp_model=nlp, stopwords=stopwords)
    )
    df['text_norm'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))
    
    df['n_tokens'] = df['tokens'].apply(len)
    
    # Filter out articles with too few tokens
    df = df[df['n_tokens'] >= min_tokens].reset_index(drop=True)
    
    # Drop duplicates based on the normalized text
    df.drop_duplicates(subset=['text_norm'], inplace=True)
    
    return df


def main():
    """
    Main function to run the scraper and preprocessor.
    """
    parser = argparse.ArgumentParser(description="Scrape and preprocess Polish news articles.")
    parser.add_argument(
        '--raw_output',
        type=str,
        default='Scrapper/polsatnews_articles.csv',
        help="Path to save the raw scraped articles CSV."
    )
    parser.add_argument(
        '--clean_output',
        type=str,
        default='Scrapper/polsatnews_articles_clean.csv',
        help="Path to save the cleaned and preprocessed articles CSV."
    )
    parser.add_argument(
        '--max_per_category',
        type=int,
        default=None,
        help="Maximum number of articles to scrape per category (default: no limit)."
    )
    parser.add_argument(
        '--min_text_length',
        type=int,
        default=200,
        help="Minimum character length of article text to be included."
    )
    parser.add_argument(
        '--min_token_count',
        type=int,
        default=30,
        help="Minimum token count for an article to be included in the clean dataset."
    )
    
    args = parser.parse_args()

    # Add progress_apply to pandas
    tqdm.pandas()

    # --- Step 1: Scrape raw articles ---
    logging.info("Starting article scraping process...")
    df_raw = fetch_articles_from_feeds(
        FEEDS,
        max_articles_per_category=args.max_per_category,
        min_length=args.min_text_length
    )
    
    if df_raw.empty:
        logging.warning("No articles were scraped. Exiting.")
        return

    df_raw.to_csv(args.raw_output, index=False)
    logging.info(f"Saved {len(df_raw)} raw articles to {args.raw_output}")

    # --- Step 2: Preprocess and clean the data ---
    logging.info("Starting dataset preparation process...")
    df_clean = prepare_dataset(
        df_raw,
        min_tokens=args.min_token_count
    )

    if df_clean.empty:
        logging.warning("No articles remained after cleaning. Exiting.")
        return
        
    df_clean.to_csv(args.clean_output, index=False)
    logging.info(f"Saved {len(df_clean)} cleaned articles to {args.clean_output}")


if __name__ == "__main__":
    main()
