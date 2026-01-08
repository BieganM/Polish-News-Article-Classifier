# 📊 Analiza Parametrów Modeli MLP - Wyniki

Ten folder zawiera wszystkie wyniki eksperymentów z parametrami modeli MLP dla klasyfikacji polskich artykułów newsowych.

## 🚀 Quick Start

1. **Zacznij od:** `PODSUMOWANIE.md` - pełny opis wszystkich wyników
2. **Do sprawozdania:** `FRAGMENTY_TEKSTU.md` - gotowe fragmenty tekstu
3. **Dla LaTeX:** `TABELA_LATEX.txt` - gotowe tabele i wykresy
4. **Pełny indeks:** `INDEKS.md` - opis wszystkich plików

## 📁 Zawartość

### 📊 Wykresy (10 plików PNG, 300 DPI)

- `parameter_impact_analysis.png` - Przegląd wszystkich 6 parametrów
- `learning_rate_analysis_detailed.png` ⚠️ **GŁÓWNY** - LR ma ISTOTNY wpływ (p<0.0001)
- `dropout_analysis_detailed.png` - Dropout NIE ma wpływu
- `batch_size_analysis_detailed.png` - Batch Size NIE ma wpływu
- `max_features_analysis_detailed.png` - Max Features NIE ma wpływu
- `architecture_analysis_detailed.png` - Głębsze sieci NIE są lepsze
- `correlation_matrix_detailed.png` - Macierz korelacji wszystkich parametrów
- `scatter_matrix_parameters.png` - Pary parametrów (5×5)
- `interaction_dropout_lr.png` - Interakcja Dropout × LR (heatmap + 3D)
- `best_model_confusion_matrix.png` - Macierz pomyłek najlepszego modelu

### 📄 Dane (4 pliki CSV + 2 JSON)

- `statistical_tests_summary.csv` - **Wyniki testów ANOVA** (do tabeli w raporcie)
- `model_comparison_*.csv` - Porównanie wszystkich 12 konfiguracji
- `category_performance_*.csv` - Wyniki per kategoria
- `inference_times_*.csv` - Czasy inferecji
- `best_model_report.json` - Specyfikacja najlepszego modelu
- `summary_*.json` - Ogólne podsumowanie

### 📝 Dokumentacja (4 pliki MD/TXT)

- `PODSUMOWANIE.md` ⭐ - **Czytaj najpierw!** Pełny opis wszystkich wyników
- `INDEKS.md` - Szczegółowy indeks wszystkich plików z instrukcjami użycia
- `FRAGMENTY_TEKSTU.md` - Gotowe fragmenty do wklejenia do sprawozdania
- `TABELA_LATEX.txt` - Gotowe tabele LaTeX + przykłady wstawienia wykresów

## 🎯 Kluczowe Wnioski

### ✅ ISTOTNY statystycznie:

- **Learning Rate** - F=90.75, p<0.0001, korelacja r=+0.695

### ❌ NIE istotne statystycznie:

- **Dropout** - p=0.962
- **Batch Size** - p=0.775
- **Max Features** - p=0.920
- **Liczba Warstw** - p=0.893 (głębsze sieci nie pomagają!)

### 🏆 Najlepszy Model: MLP_LargeBatch

- **F1-Score:** 0.8174
- **Accuracy:** 0.8252
- **Architektura:** [512, 256] (tylko 2 warstwy!)
- **LR:** 0.001
- **Dropout:** 0.5
- **Batch Size:** 64

## 📖 Jak Użyć do Sprawozdania?

### Dla osób piszących w Word/Google Docs:

1. Otwórz `PODSUMOWANIE.md` w VS Code lub przeglądarce
2. Przeczytaj kluczowe wnioski
3. Wstaw wykresy PNG do dokumentu
4. Skopiuj fragmenty z `FRAGMENTY_TEKSTU.md`
5. Otwórz `statistical_tests_summary.csv` w Excel → skopiuj do tabeli

### Dla osób piszących w LaTeX:

1. Przeczytaj `PODSUMOWANIE.md`
2. Skopiuj tabele z `TABELA_LATEX.txt`
3. Użyj `\includegraphics{analysis_results/wykres.png}` dla wykresów
4. Dostosuj `FRAGMENTY_TEKSTU.md` do swojego stylu

### Dla osób robiących prezentację:

1. Użyj wykresów PNG (300 DPI - wysoka jakość)
2. **Must-have slides:**
   - `parameter_impact_analysis.png` - przegląd
   - `learning_rate_analysis_detailed.png` - główne odkrycie
   - `correlation_matrix_detailed.png` - wszystkie korelacje
   - `best_model_confusion_matrix.png` - wyniki
3. Kluczowe liczby z `INDEKS.md` (sekcja "Najważniejsze Liczby")

## 🔬 Metody Statystyczne

- **ANOVA (f_oneway)** - test F dla wielu grup
- **Pearson Correlation** - siła związku liniowego
- **95% Confidence Intervals** - przedziały ufności
- **Polynomial Regression** - krzywa trendu dla LR
- **3D Visualization** - interakcje parametrów

## 💻 Software

- Python 3.10+
- PyTorch 2.0+
- scikit-learn 1.3+
- scipy.stats
- matplotlib 3.7+ (300 DPI)
- seaborn 0.12+

## 📊 Statystyki Eksperymentów

- **Liczba konfiguracji:** 12
- **Łączny czas treningu:** ~13 sekund
- **Zakres F1-Score:** 0.6823 - 0.8174
- **Średni F1:** 0.7858 ± 0.0413
- **Testowane parametry:** 5 (Dropout, LR, Batch, Features, Warstwy)
- **Statystycznie istotne:** 1 (tylko Learning Rate!)

## 🎓 Dodatkowe Informacje

### Gotowe fragmenty dla obrony:

Zobacz sekcję "GOTOWE ZDANIA NA OBRONĘ" w `FRAGMENTY_TEKSTU.md`

### Problemy? FAQ:

- **Q:** Który wykres jest najważniejszy?  
  **A:** `learning_rate_analysis_detailed.png` - pokazuje jedyny istotny parametr

- **Q:** Jak wstawić wykres do LaTeX?  
  **A:** Zobacz przykłady w `TABELA_LATEX.txt`

- **Q:** Skąd wziąć dane do tabeli ANOVA?  
  **A:** Otwórz `statistical_tests_summary.csv` w Excel

- **Q:** Ile wykresów użyć w sprawozdaniu?  
  **A:** Minimum 5 (zobacz sekcja "Top 5" w `INDEKS.md`)

---

**Utworzono:** 2025-01-28  
**Źródło:** model_comparison_analysis.ipynb  
**Kontakt:** EDT Project Team
