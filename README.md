# Polish News Article Classifier

Aplikacja webowa do automatycznej klasyfikacji tematycznej polskich artykułów newsowych przy użyciu modeli uczenia maszynowego.

## Funkcjonalności

- **Wybór modelu**: MLP (Multi-Layer Perceptron), HerBERT, BERT
- **Trening modeli**: Interfejs webowy do trenowania modeli z parametrami
- **Predykcja**: Klasyfikacja tekstu, plików lub URL artykułów
- **Wizualizacja wyników**: Wykresy straty i dokładności po treningu
- **Early Stopping**: Opcja zatrzymania treningu przy braku poprawy

## Wymagania

- Python 3.8+
- macOS z MPS (dla GPU) lub inne systemy

## Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone <repo-url>
   cd <project-dir>
   ```

2. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```

## Uruchomienie

1. Uruchom aplikację:
   ```bash
   python app.py
   ```

2. Otwórz przeglądarkę i przejdź do `http://127.0.0.1:5000/`

## Użycie

### Klasyfikacja
- Wybierz dostępny model z dropdownu
- Wprowadź tekst, wybierz plik lub podaj URL artykułu
- Kliknij "Klasyfikuj" aby uzyskać kategorię i pewność

### Trening
- Przejdź do `/train`
- Wybierz model i skonfiguruj parametry
- Kliknij "Rozpocznij Trening"
- Po zakończeniu zobacz wyniki i wykresy

## Modele

- **MLP**: Sieć neuronowa z TF-IDF, szybka do treningu
- **HerBERT**: Model językowy dla polskiego, wysoka dokładność
- **BERT**: Wielojęzyczny model, alternatywa dla HerBERT

## Dane

Dane treningowe: Artykuły z polskich serwisów newsowych (Polsat News itp.), rozszerzone o dodatkowe źródła.

## Licencja

MIT License