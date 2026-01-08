# PODSUMOWANIE ANALIZ PARAMETRÓW MODELI MLP

**Projekt:** Klasyfikacja Polskich Artykułów Newsowych  
**Zespół:** EDT Project  
**Data analizy:** Styczeń 2025

---

## 📊 ZAPISANE WYKRESY DO SPRAWOZDANIA

Wszystkie wykresy są w folderze `analysis_results/` w formacie PNG (300 DPI).

### 1. `parameter_impact_analysis.png`

- **Opis:** 6 paneli pokazujących wpływ każdego parametru na F1-Score
- **Zawartość:** Dropout, Learning Rate, Batch Size, Max Features, Liczba Warstw, Rozmiar vs Czas
- **Użycie w raporcie:** Przegląd wszystkich parametrów

### 2. `dropout_analysis_detailed.png`

- **Opis:** Szczegółowa analiza wpływu Dropout
- **Zawartość:**
  - Boxplot rozkładu F1-Score vs Dropout
  - Violin plot gęstości rozkładu
  - Średnie z 95% przedziałami ufności
- **Wnioski:** Dropout NIE ma istotnego wpływu (p=0.962)

### 3. `learning_rate_analysis_detailed.png`

- **Opis:** Szczegółowa analiza wpływu Learning Rate
- **Zawartość:**
  - Boxplot F1-Score vs Learning Rate
  - Scatter z regresją wielomianową (stopień 2)
  - Wykres w skali logarytmicznej
- **Wnioski:** ⚠️ **Learning Rate MA ISTOTNY WPŁYW** (p<0.0001, korelacja +0.695)

### 4. `batch_size_analysis_detailed.png`

- **Opis:** Szczegółowa analiza wpływu Batch Size
- **Zawartość:**
  - Boxplot F1-Score vs Batch Size
  - Bar plot ze średnimi i przedziałami ufności
  - Scatter z linią trendu liniowego
- **Wnioski:** Batch Size NIE ma istotnego wpływu (p=0.775)

### 5. `max_features_analysis_detailed.png`

- **Opis:** 4 panele analizujące Max Features
- **Zawartość:**
  - Panel A: Boxplot F1-Score vs Max Features
  - Panel B: Scatter z trendem
  - Panel C: Features vs Training Time (kolor = F1-Score)
  - Panel D: Średnie z 95% CI
- **Wnioski:** Max Features NIE ma istotnego wpływu (p=0.920)

### 6. `architecture_analysis_detailed.png`

- **Opis:** 6 paneli analizujących architekturę sieci
- **Zawartość:**
  - Panel A: Głębokość sieci vs F1-Score (boxplot)
  - Panel B: Rozmiar sieci vs F1-Score (scatter, kolor = liczba warstw)
  - Panel C: Rozmiar vs Czas treningu (scatter, kolor = F1-Score)
  - Panel D: Średnie F1-Score per liczba warstw (bar)
  - Panel E: Efektywność (F1/Time) vs Rozmiar
  - Panel F: Heatmap: Architektura vs F1-Score
- **Wnioski:** Głębsze sieci NIE są lepsze (p=0.893)

### 7. `correlation_matrix_detailed.png`

- **Opis:** Pełna macierz korelacji Pearsona
- **Zawartość:**
  - Heatmap 9×9 wszystkich parametrów
  - Bar chart siły wpływu każdego parametru na F1-Score
- **Użycie w raporcie:** Tabela korelacji w sekcji wyników

### 8. `scatter_matrix_parameters.png`

- **Opis:** Macierz scatter plot wszystkich par parametrów
- **Zawartość:**
  - 5×5 kombinacji parametrów
  - Kolor punktów = F1-Score (gradient żółty-czerwony)
  - Linie trendu dla par z F1-Score
- **Użycie w raporcie:** Pokazanie interakcji między parametrami

### 9. `interaction_dropout_lr.png`

- **Opis:** Analiza interakcji Dropout × Learning Rate
- **Zawartość:**
  - Heatmap interakcji: Dropout × LR → F1-Score
  - Wykres 3D: Dropout × LR × F1-Score
- **Wnioski:** Najlepsza kombinacja: Dropout=0.5 × LR=0.001 → F1=0.8174

### 10. `best_model_confusion_matrix.png`

- **Opis:** Macierz pomyłek najlepszego modelu
- **Model:** MLP_LargeBatch
- **F1-Score:** 0.8174

---

## 📄 PLIKI CSV/JSON

### `statistical_tests_summary.csv`

- Testy ANOVA dla wszystkich parametrów
- Kolumny: Parameter, F-statistic, p-value, Pearson Correlation, Significance

### `best_model_report.json`

- Pełna specyfikacja najlepszego modelu
- Wyniki na zbiorze testowym
- Parametry treningu

---

## 🔬 KLUCZOWE WNIOSKI STATYSTYCZNE

### ✅ PARAMETRY Z ISTOTNYM WPŁYWEM:

#### 1. **LEARNING RATE** ⚠️

- **Test ANOVA:** F=90.75, **p<0.0001** ✓
- **Korelacja Pearsona:** +0.695 (silna pozytywna)
- **Interpretacja:** Im wyższy learning rate, tym LEPSZE wyniki (w testowanym zakresie 0.0001-0.002)
- **Optymalna wartość:** 0.001-0.002
- **Zalecenie:** Ten parametr należy priorytetowo optymalizować

---

### ❌ PARAMETRY BEZ ISTOTNEGO WPŁYWU:

#### 2. **DROPOUT**

- **Test ANOVA:** F=0.14, p=0.962 (brak istotności)
- **Korelacja:** -0.133 (słaba negatywna)
- **Interpretacja:** Wartość dropout (0.3, 0.5) nie wpływa na wyniki
- **Zalecenie:** Można użyć dowolnej wartości (0.3-0.5)

#### 3. **BATCH SIZE**

- **Test ANOVA:** F=0.26, p=0.775 (brak istotności)
- **Korelacja:** +0.117 (słaba pozytywna)
- **Interpretacja:** Rozmiar batcha (16, 32, 64) nie wpływa na wyniki
- **Zalecenie:** Można wybrać batch=64 dla szybszego treningu

#### 4. **MAX FEATURES**

- **Test ANOVA:** F=0.08, p=0.920 (brak istotności)
- **Korelacja:** -0.126 (słaba negatywna)
- **Interpretacja:** Liczba cech TF-IDF (5000, 10000) nie wpływa na wyniki
- **Zalecenie:** Użyć 5000 dla mniejszych modeli

#### 5. **LICZBA WARSTW (Głębokość Sieci)**

- **Test ANOVA:** F=0.12, p=0.893 (brak istotności)
- **Korelacja:** -0.082 (słaba negatywna)
- **Interpretacja:** Głębsze sieci (2 vs 3 vs 4 warstwy) NIE są lepsze
- **Zalecenie:** Użyć 2-3 warstw, głębsze sieci zwiększają złożoność bez korzyści

---

## 🏆 NAJLEPSZA KONFIGURACJA

### Model: **MLP_LargeBatch**

#### Hiperparametry:

- **Architektura:** [512, 256] (2 warstwy ukryte)
- **Max Features:** 5000
- **Dropout:** 0.5
- **Learning Rate:** 0.001 ⚠️ (kluczowy parametr!)
- **Batch Size:** 64
- **Epochs:** 20
- **Optimizer:** Adam

#### Wyniki na zbiorze testowym:

| Metryka       | Wartość    |
| ------------- | ---------- |
| **F1-Score**  | **0.8174** |
| **Accuracy**  | **0.8252** |
| Precision     | 0.8344     |
| Recall        | 0.8252     |
| Cohen's Kappa | 0.7724     |
| MCC           | 0.7783     |
| Czas treningu | 1.09s      |

---

## 💡 ZALECENIA DO SPRAWOZDANIA

### Sekcja Metodologii:

1. **Wykres 1** (`parameter_impact_analysis.png`) - przegląd wszystkich testowanych parametrów
2. Opisz 12 testowanych konfiguracji z tabeli `param_df`
3. Użyj `statistical_tests_summary.csv` do tabeli z testami ANOVA

### Sekcja Wyniki:

4. **Wykres 3** (`learning_rate_analysis_detailed.png`) - **GŁÓWNY WYKRES** pokazujący istotny wpływ LR
5. **Wykres 7** (`correlation_matrix_detailed.png`) - macierz korelacji wszystkich parametrów
6. **Wykres 10** (`best_model_confusion_matrix.png`) - wyniki najlepszego modelu

### Sekcja Dyskusji:

7. **Wykres 6** (`architecture_analysis_detailed.png`) - uzasadnienie wyboru płytkiej sieci
8. **Wykres 9** (`interaction_dropout_lr.png`) - pokazanie że optymalna kombinacja to Dropout=0.5 × LR=0.001
9. **Wykresy 2, 4, 5** - potwierdzenie że dropout, batch size, max features nie mają wpływu

### Kluczowe tezy do sprawozdania:

- ✅ **Learning Rate jest JEDYNYM parametrem o statystycznie istotnym wpływie** (p<0.0001)
- ✅ **Głębsze sieci NIE poprawiają wyników** - wystarczą 2 warstwy ukryte
- ✅ **Dropout, Batch Size, Max Features można wybrać dowolnie** w testowanych zakresach
- ✅ **Najlepszy model osiągnął F1=0.8174** z prostą architekturą [512, 256]

---

## 📈 STATYSTYKI EKSPERYMENTÓW

- **Liczba testowanych konfiguracji:** 12
- **Łączny czas treningu:** ~13 sekund
- **Testowane parametry:** 5 (Dropout, LR, Batch, Features, Architektura)
- **Statystycznie istotne:** 1 (Learning Rate)
- **Zakres F1-Score:** 0.6823 - 0.8174
- **Zakres Accuracy:** 0.7056 - 0.8252

---

## 🔧 UŻYTE METODY STATYSTYCZNE

1. **ANOVA (Analysis of Variance)** - test F dla wielu grup
2. **Pearson Correlation** - korelacja liniowa z F1-Score
3. **95% Confidence Intervals** - przedziały ufności dla średnich
4. **Polynomial Regression** (stopień 2) - dla Learning Rate
5. **Linear Regression** - linie trendu w scatter plots
6. **Interaction Analysis** - interakcje Dropout × Learning Rate
7. **3D Visualization** - przestrzenna wizualizacja interakcji

---

_Wszystkie wykresy wygenerowane z matplotlib (300 DPI, publication quality)_  
_Analiza statystyczna wykonana z scipy.stats_
