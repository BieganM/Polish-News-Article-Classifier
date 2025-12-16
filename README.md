# Automatyczny Klasyfikator Tematyczny Artykułów

## 1. Opis Projektu

Projekt ten to aplikacja webowa do automatycznego tagowania (klasyfikacji tematycznej) polskich artykułów. Użytkownik może wkleić tekst, podać URL do artykułu lub wgrać plik `.txt`, a aplikacja, wykorzystując wybrany model, przewidzi jego kategorię tematyczną (np. "Sport", "Biznes", "Technologie").

Aplikacja umożliwia również trenowanie modeli bezpośrednio z poziomu interfejsu webowego, z możliwością dostosowania hiperparametrów i śledzenia postępów w czasie rzeczywistym.

## 2. Główne Funkcjonalności

- **Klasyfikacja tekstu**: Przewidywanie kategorii dla podanego tekstu, URL lub pliku.
- **Wybór modelu**: Możliwość wyboru jednego z trzech dostępnych modeli:
  - **HerBERT**: Nowoczesny model transformerowy, trenowany na języku polskim.
  - **BERT**: Wielojęzyczny model transformerowy.
  - **MLP**: Prosta sieć neuronowa oparta na wektoryzacji TF-IDF.
- **Trening modeli**: Interfejs do trenowania modeli z personalizacją hiperparametrów (liczba epok, learning rate, etc.).
- **Dynamiczne ładowanie modeli**: Aplikacja automatycznie wykrywa, które modele zostały już wytrenowane i są gotowe do użycia.
- **Przetwarzanie danych w tle**: Trening modeli odbywa się w osobnym wątku, nie blokując aplikacji.

## 3. Struktura Projektu

```
/
├─── app.py                   # Główny plik aplikacji Flask
├─── requirements.txt           # Zależności projektu
├─── .env.example               # Przykładowy plik konfiguracyjny dla zmiennych środowiskowych
├─── .gitignore                 # Pliki ignorowane przez Git
├─── Scrapper/
│    ├─── scrapper.py           # NOWOŚĆ: Skrypt do pobierania i przetwarzania danych
│    ├─── *.csv                 # Wygenerowane pliki z danymi
│    └─── scrapper.ipynb        # Notebook z oryginalną logiką (teraz zrefaktoryzowaną)
├─── models/
│    ├─── model.py              # Logika trenowania, predykcji i zarządzania modelami
│    ├─── config.py             # Konfiguracja ścieżek, modeli i hiperparametrów
│    ├─── utils.py              # Funkcje pomocnicze (np. walidacja tekstu)
├─── tests/                     # NOWOŚĆ: Testy jednostkowe
│    └─── test_utils.py         # Testy dla funkcji pomocniczych
├─── templates/
│    ├─── index.html            # Główny widok do klasyfikacji
│    └─── train.html            # Widok do trenowania modeli
├─── static/
│    ├─── css/style.css
│    └─── js/script.js
└─── model_*/                   # Katalogi na wytrenowane modele (generowane dynamicznie)
```

### Kluczowe komponenty:

- **`app.py`**: Aplikacja Flask, która zarządza routingiem, obsługuje żądania użytkownika, zarządza stanem modeli i uruchamia procesy w tle.
- **`Scrapper/scrapper.py`**: Narzędzie CLI do pobierania artykułów z polskich portali informacyjnych (za pomocą RSS) i ich przetwarzania (lematyzacja, czyszczenie) w celu stworzenia zbioru danych do treningu.
- **`models/model.py`**: Serce logiki ML. Zawiera funkcje do:
  - Trenowania modeli `transformers` (HerBERT, BERT) przy użyciu `Trainer API` z Hugging Face.
  - Trenowania modelu MLP przy użyciu `PyTorch` i `scikit-learn` (TF-IDF).
  - Ładowania zapisanych modeli i robienia predykcji.
  - Zarządzania stanem treningu (postęp, błędy, etc.).
- **`models/config.py`**: Centralny plik konfiguracyjny. Definiuje ścieżki do danych i modeli, domyślne hiperparametry oraz odczytuje konfigurację ze zmiennych środowiskowych (plik `.env`).
- **`tests/`**: Katalog z testami jednostkowymi `pytest`, które weryfikują poprawność działania kluczowych funkcji.

## 4. Instalacja i Uruchomienie

### Krok 1: Klonowanie i przygotowanie środowiska

```bash
# Sklonuj repozytorium
git clone <URL_REPOZYTORIUM>
cd <NAZWA_KATALOGU>

# Utwórz i aktywuj środowisko wirtualne
python3 -m venv .venv
source ./.venv/bin/activate  # macOS/Linux
# lub: .\.venv\Scripts\activate # Windows
```

### Krok 2: Instalacja zależności

Upewnij się, że Twoje środowisko wirtualne jest aktywne.

```bash
pip install -r requirements.txt
```

### Krok 3: Przygotowanie danych treningowych

Aplikacja wymaga zbioru danych do trenowania modeli. Użyj nowego skryptu `scrapper.py`, aby go wygenerować.

```bash
# Pobierz i przetwórz dane (może to zająć kilka minut)
python Scrapper/scrapper.py
```

Po zakończeniu, w katalogu `Scrapper/` pojawią się dwa pliki: `polsatnews_articles.csv` (surowe dane) i `polsatnews_articles_clean.csv` (przetworzone dane, gotowe do treningu).

### Krok 4: Konfiguracja zmiennych środowiskowych

Skopiuj plik `.env.example` do pliku `.env`.

```bash
cp .env.example .env
```

W pliku `.env` możesz dostosować konfigurację, np. wyłączyć tryb debugowania Flaska (`FLASK_DEBUG=False`) w środowisku produkcyjnym.

### Krok 5: Uruchomienie aplikacji

```bash
python app.py
```

Aplikacja będzie dostępna pod adresem **http://127.0.0.1:5000**.

### Krok 6: Trenowanie modelu

Po uruchomieniu aplikacji, żaden model nie jest jeszcze wytrenowany.

1.  Przejdź do zakładki "Train" (`/train`).
2.  Wybierz model, który chcesz wytrenować (np. HerBERT).
3.  Dostosuj hiperparametry lub zostaw domyślne.
4.  Kliknij "Rozpocznij trening".
5.  Możesz śledzić postęp na żywo. Po zakończeniu model będzie dostępny do użytku na stronie głównej.

### Krok 7 (Opcjonalnie): Uruchomienie testów

Aby zweryfikować poprawność działania części logiki, możesz uruchomić testy jednostkowe.

```bash
python -m pytest
```

## 5. Podsumowanie Wprowadzonych Zmian (Senior Developer Refactoring)

- **Refaktoryzacja Logiki `Scrappera`**: Cała logika do pobierania i przetwarzania danych została przeniesiona z notebooka `scrapper.ipynb` do skryptu `Scrapper/scrapper.py`. Skrypt jest teraz narzędziem CLI, co ułatwia automatyzację i ponowne użycie.
- **Zarządzanie Konfiguracją**: Wprowadzono obsługę zmiennych środowiskowych za pomocą pliku `.env` (biblioteka `python-dotenv`). Ustawienia takie jak `FLASK_DEBUG` czy `DEVICE` (CPU/GPU) są teraz zarządzane centralnie i nie są zahardkodowane w kodzie.
- **Poprawa `requirements.txt`**: Usunięto zbędne zależności (`tf-keras`) i dodano brakujące (`spacy`, `python-dotenv`, `pytest`).
- **Wprowadzenie Testów Jednostkowych**: Dodano strukturę testów (`/tests`) z przykładowymi testami dla funkcji pomocniczych, co zwiększa niezawodność i ułatwia dalszy rozwój.
- **Ulepszone Logowanie Błędów**: Poprawiono obsługę błędów w wątku treningowym – teraz błędy są logowane z pełnym `tracebackiem`, co znacząco ułatwia diagnozowanie problemów.
- **Czystość Kodu**: Usunięto zbędne komentarze i fragmenty kodu, a nazewnictwo i struktura zostały ujednolicone zgodnie z najlepszymi praktykami.
