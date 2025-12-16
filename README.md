# Polish News Article Classifier

Aplikacja webowa do automatycznej klasyfikacji tematycznej polskich artykułów newsowych przy użyciu modeli uczenia maszynowego.

## Funkcjonalności

- **Klasyfikacja tekstu**: Przewidywanie kategorii dla podanego tekstu, URL lub pliku.
- **Wybór modelu**: Możliwość wyboru jednego z trzech dostępnych modeli:
  - **HerBERT**: Nowoczesny model transformerowy, trenowany na języku polskim.
  - **BERT**: Wielojęzyczny model transformerowy.
  - **MLP**: Prosta sieć neuronowa oparta na wektoryzacji TF-IDF.
- **Trening modeli**: Interfejs do trenowania modeli z personalizacją hiperparametrów (liczba epok, learning rate, etc.).
- **Dynamiczne ładowanie modeli**: Aplikacja automatycznie wykrywa, które modele zostały już wytrenowane i są gotowe do użycia.
- **Przetwarzanie danych w tle**: Trening modeli odbywa się w osobnym wątku, nie blokując aplikacji.

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

2. Otwórz przeglądarkę i przejdź do `http://127.0.0.1:10000/`

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

Aby zweryfikować poprawność działania części logiki, możesz uruchomić testy jednostkowe.

```bash
python -m pytest
```

## Modele

- **MLP**: Sieć neuronowa z TF-IDF, szybka do treningu
- **HerBERT**: Model językowy dla polskiego, wysoka dokładność
- **BERT**: Wielojęzyczny model, alternatywa dla HerBERT

## Dane

Dane treningowe: Artykuły z polskich serwisów newsowych (Polsat News itp.), rozszerzone o dodatkowe źródła.

## Licencja

MIT License
