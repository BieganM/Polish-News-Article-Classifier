# INDEKS WSZYSTKICH PLIKÓW ANALIZY

**Katalog:** `analysis_results/`  
**Data wygenerowania:** 2025-01-28  
**Projekt:** Klasyfikacja Polskich Artykułów Newsowych - Analiza Parametrów MLP

---

## 📊 WYKRESY (PNG - 300 DPI)

### Główne analizy parametrów (10 wykresów):

1. **parameter_impact_analysis.png**

   - Przegląd 6 parametrów: Dropout, LR, Batch, Features, Warstwy, Rozmiar vs Czas
   - Użycie: Sekcja Metodologia (ogólny przegląd)

2. **dropout_analysis_detailed.png**

   - 3 panele: Boxplot, Violin plot, Średnie z 95% CI
   - Wniosek: p=0.962 (brak wpływu)

3. **learning_rate_analysis_detailed.png** ⚠️ GŁÓWNY WYKRES

   - 3 panele: Boxplot, Regresja wielomianowa, Skala log
   - Wniosek: p<0.0001 (ISTOTNY wpływ, r=+0.695)
   - Użycie: Sekcja Wyniki (kluczowe odkrycie)

4. **batch_size_analysis_detailed.png**

   - 3 panele: Boxplot, Bar z CI, Scatter z trendem
   - Wniosek: p=0.775 (brak wpływu)

5. **max_features_analysis_detailed.png**

   - 4 panele: Boxplot, Scatter+trend, Features vs Time, Statystyki z CI
   - Wniosek: p=0.920 (brak wpływu)

6. **architecture_analysis_detailed.png**

   - 6 paneli: Głębokość vs F1, Rozmiar vs F1, Time analysis, Efektywność, Heatmap
   - Wniosek: p=0.893 (głębsze sieci NIE są lepsze)
   - Użycie: Sekcja Dyskusji (uzasadnienie prostej architektury)

7. **correlation_matrix_detailed.png**

   - Heatmap 9×9 korelacji Pearsona + Bar chart wpływu na F1
   - Użycie: Sekcja Wyniki (tabela korelacji)

8. **scatter_matrix_parameters.png**

   - Macierz 5×5 scatter plots wszystkich par parametrów
   - Kolor = F1-Score, linie trendu
   - Użycie: Dodatek (szczegółowe interakcje)

9. **interaction_dropout_lr.png**

   - Heatmap + 3D plot: Dropout × LR → F1-Score
   - Wniosek: Optymalna kombinacja Dropout=0.5 × LR=0.001
   - Użycie: Sekcja Dyskusji (analiza interakcji)

10. **best_model_confusion_matrix.png**
    - Znormalizowana macierz pomyłek MLP_LargeBatch (F1=0.8174)
    - Użycie: Sekcja Wyniki (wydajność najlepszego modelu)

---

## 📄 DANE CSV

1. **statistical_tests_summary.csv**

   - Kolumny: Parameter | F-statistic | p-value | Pearson Correlation | Significance
   - 5 wierszy (dla każdego parametru)
   - Użycie: Tabela w sprawozdaniu (wyniki testów ANOVA)

2. **model_comparison_20251228_160804.csv**

   - Wszystkie 12 konfiguracji z metrykami
   - Kolumny: Model, F1-Score, Accuracy, Precision, Recall, Kappa, MCC, Training Time, itd.
   - Użycie: Tabela porównawcza modeli

3. **category_performance_20251228_160804.csv**

   - Wyniki per kategoria dla każdego modelu
   - Kolumny: Model, Category, F1-Score, Precision, Recall, Support
   - Użycie: Analiza per-class performance

4. **inference_times_20251228_160804.csv**
   - Czasy inferecji dla każdego modelu
   - Kolumny: Model, Mean Time, Std, Min, Max, Samples/sec
   - Użycie: Analiza wydajności

---

## 📋 DANE JSON

1. **best_model_report.json**

   - Pełna specyfikacja MLP_LargeBatch
   - Zawiera: hiperparametry, architekturę, wszystkie metryki, czasy
   - Format: JSON (łatwy import do dalszych analiz)

2. **summary_20251228_160804.json**
   - Podsumowanie wszystkich eksperymentów
   - Zawiera: statistyki ogólne, top 3 modele, najgorszy model

---

## 📝 DOKUMENTACJA (Markdown/TXT)

1. **PODSUMOWANIE.md** ⭐ GŁÓWNY DOKUMENT

   - Pełny opis wszystkich 10 wykresów
   - Wnioski statystyczne (co jest istotne, co nie)
   - Specyfikacja najlepszego modelu
   - Zalecenia do sprawozdania (które wykresy gdzie użyć)
   - Statystyki eksperymentów
   - Format: Markdown (czytelny w GitHub/VS Code)

2. **FRAGMENTY_TEKSTU.md**

   - Gotowe fragmenty tekstu do sprawozdania
   - Sekcje: Metodologia, Wyniki, Dyskusja, Wnioski, Dodatek
   - Gotowe zdania na obronę (Q&A)
   - Format: Markdown

3. **TABELA_LATEX.txt**
   - 3 gotowe tabele w LaTeX
   - Przykłady wstawienia wykresów (\\includegraphics)
   - Przykłady cytowania w tekście
   - Gotowe do wklejenia do dokumentu .tex

---

## 🎯 JAK UŻYĆ DO SPRAWOZDANIA?

### Krok 1: Przeczytaj PODSUMOWANIE.md

- Zrozum kluczowe wnioski
- Zobacz które wykresy są najważniejsze
- Sprawdź statystyki

### Krok 2: Wybierz wykresy do raportu

**OBOWIĄZKOWE (Top 5):**

1. `learning_rate_analysis_detailed.png` - GŁÓWNE ODKRYCIE (p<0.0001)
2. `correlation_matrix_detailed.png` - przegląd wszystkich korelacji
3. `best_model_confusion_matrix.png` - wyniki najlepszego modelu
4. `parameter_impact_analysis.png` - przegląd wszystkich parametrów
5. `architecture_analysis_detailed.png` - uzasadnienie prostej architektury

**DODATKOWE (jeśli jest miejsce):** 6. `dropout_analysis_detailed.png` - potwierdzenie braku wpływu dropout 7. `scatter_matrix_parameters.png` - szczegółowe interakcje 8. `interaction_dropout_lr.png` - 3D wizualizacja interakcji

### Krok 3: Użyj FRAGMENTY_TEKSTU.md

- Skopiuj gotowe fragmenty do sekcji sprawozdania
- Dostosuj numerację rysunków/tabel
- Dodaj własne komentarze

### Krok 4: Dodaj tabele z TABELA_LATEX.txt

- Tabela 1: Wyniki ANOVA (statistical_tests_summary.csv)
- Tabela 2: TOP 5 modeli (model*comparison*\*.csv)
- Tabela 3: Specyfikacja najlepszego (best_model_report.json)

### Krok 5: Użyj CSV do tabel numerycznych

- Excel/LibreOffice: Otwórz .csv
- Python/Pandas: `pd.read_csv(...)`
- LaTeX: Convert with pandas.to_latex()

---

## 📐 SPECYFIKACJA TECHNICZNA

### Wykresy:

- Format: PNG
- Rozdzielczość: 300 DPI (publication quality)
- Rozmiar: ~200-500 KB każdy
- Font: DejaVu Sans (czytelny)
- Colormap: viridis, RdYlGn (color-blind friendly)

### CSV:

- Separator: przecinek (,)
- Encoding: UTF-8
- Decimal: kropka (.)
- Header: Yes (pierwszy wiersz)

### JSON:

- Format: Pretty-printed (wcięcia 2 spacje)
- Encoding: UTF-8
- Klucze: lowercase z podkreślnikami

---

## ✅ CHECKLIST DO SPRAWOZDANIA

- [ ] Przeczytane PODSUMOWANIE.md
- [ ] Wybrane 5-8 kluczowych wykresów
- [ ] Wstawione wykresy do dokumentu (z podpisami)
- [ ] Dodane Tabela 1 (ANOVA results)
- [ ] Dodane Tabela 2 (Model comparison)
- [ ] Dodane Tabela 3 (Best model spec)
- [ ] Napisana sekcja Metodologia (użyj FRAGMENTY_TEKSTU.md)
- [ ] Napisana sekcja Wyniki (użyj FRAGMENTY_TEKSTU.md)
- [ ] Napisana sekcja Dyskusja (użyj FRAGMENTY_TEKSTU.md)
- [ ] Napisane Wnioski (5 kluczowych punktów)
- [ ] Sprawdzone numerowanie rysunków/tabel
- [ ] Sprawdzone cytowania (Rysunek X, Tabela Y)
- [ ] Dodane referencje do plików źródłowych (jeśli wymagane)

---

## 🔗 POWIĄZANIA PLIKÓW

```
PODSUMOWANIE.md (czytaj najpierw!)
    ├── Opisuje: wszystkie 10 wykresów .png
    ├── Odnosi się do: statistical_tests_summary.csv
    └── Zawiera: best_model_report.json

FRAGMENTY_TEKSTU.md (użyj do pisania)
    ├── Cytuje: wykresy .png (Rysunek X)
    ├── Cytuje: tabele z .csv (Tabela Y)
    └── Bazuje na: PODSUMOWANIE.md

TABELA_LATEX.txt (gotowe tabele)
    ├── Tabela 1: statistical_tests_summary.csv
    ├── Tabela 2: model_comparison_*.csv
    ├── Tabela 3: best_model_report.json
    └── Przykłady: \includegraphics{wykres.png}
```

---

## 📞 NAJWAŻNIEJSZE LICZBY (do zapamiętania)

| Metryka                    | Wartość        | Znaczenie              |
| -------------------------- | -------------- | ---------------------- |
| **Learning Rate p-value**  | **< 0.0001**   | ISTOTNE statystycznie! |
| **LR korelacja z F1**      | **+0.695**     | Silny pozytywny wpływ  |
| **Najlepszy F1-Score**     | **0.8174**     | MLP_LargeBatch         |
| **Najlepsza Accuracy**     | **0.8252**     | MLP_LargeBatch         |
| **Optymalna architektura** | **[512, 256]** | 2 warstwy wystarczą    |
| **Optymalny LR**           | **0.001**      | Najlepsza wartość      |
| **Liczba konfiguracji**    | **12**         | Przetestowanych        |
| **Dropout p-value**        | **0.962**      | NIE istotne            |
| **Batch Size p-value**     | **0.775**      | NIE istotne            |
| **Głębokość p-value**      | **0.893**      | Głębsze ≠ lepsze       |

---

_Wszystkie pliki wygenerowane automatycznie z notebooka Jupyter_  
_Kod źródłowy: model_comparison_analysis.ipynb_  
_Data: 2025-01-28_
