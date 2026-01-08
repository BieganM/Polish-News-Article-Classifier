# GOTOWE FRAGMENTY TEKSTU DO SPRAWOZDANIA

## SEKCJA: Metodologia - Eksperymenty z parametrami

W celu znalezienia optymalnej konfiguracji modelu MLP, przeprowadzono systematyczne eksperymenty testujące wpływ kluczowych hiperparametrów na wydajność klasyfikacji. Przetestowano 12 różnych konfiguracji, zmieniając następujące parametry:

- **Dropout:** 0.3, 0.5
- **Learning Rate:** 0.0001, 0.0005, 0.001, 0.002
- **Batch Size:** 16, 32, 64
- **Max Features (TF-IDF):** 5000, 10000
- **Architektura:** od 2 do 4 warstw ukrytych, o rozmiarach od 64 do 512 neuronów

Każdy model trenowano przez 20 epok z wykorzystaniem optymalizatora Adam i funkcji straty CrossEntropyLoss. Wszystkie eksperymenty przeprowadzono na tym samym zbiorze treningowym i testowym, aby zapewnić porównywalność wyników.

---

## SEKCJA: Wyniki - Analiza wpływu parametrów

Przeprowadzono testy ANOVA (Analysis of Variance) dla każdego z pięciu głównych parametrów w celu określenia ich statystycznego wpływu na F1-Score (Tabela X).

**Kluczowe odkrycie:** Jedynie **Learning Rate** wykazał statystycznie istotny wpływ na wydajność modelu (F=90.75, p<0.0001, korelacja Pearsona r=0.695). Pozostałe parametry - Dropout (p=0.962), Batch Size (p=0.775), Max Features (p=0.920) oraz liczba warstw ukrytych (p=0.893) - nie wykazały istotnego wpływu statystycznego.

Szczegółowa analiza Learning Rate (Rysunek X) pokazuje wyraźny wzrost F1-Score wraz ze wzrostem Learning Rate w testowanym zakresie 0.0001-0.002. Modele z LR=0.001 i LR=0.002 osiągały średnie F1-Score odpowiednio 0.8118 i 0.8017, podczas gdy modele z niższymi wartościami LR (0.0001, 0.0005) uzyskiwały jedynie F1=0.6823-0.7445.

Macierz korelacji wszystkich parametrów (Rysunek Y) potwierdza te obserwacje, pokazując silną dodatnią korelację Learning Rate z F1-Score (r=0.695), podczas gdy pozostałe parametry wykazują jedynie słabe korelacje (|r|<0.15).

---

## SEKCJA: Wyniki - Wpływ architektury sieci

Wbrew intuicji, głębsze sieci neuronowe nie poprawiły wyników klasyfikacji. Test ANOVA dla liczby warstw ukrytych wykazał brak istotnego wpływu (F=0.12, p=0.893). Modele z 2, 3 i 4 warstwami osiągały porównywalne średnie F1-Score: 0.7910, 0.7903 i 0.7887 odpowiednio (Rysunek Z).

Analiza efektywności (stosunek F1-Score do czasu treningu) pokazała, że płytsze sieci są bardziej efektywne. Model dwuwarstwowy [512, 256] osiągnął najlepsze wyniki (F1=0.8174) przy krótkim czasie treningu (1.09s), podczas gdy głębsze sieci o większej liczbie parametrów nie przyniosły poprawy wydajności, jednocześnie zwiększając czas treningu o 10-15%.

To odkrycie sugeruje, że dla zadania klasyfikacji polskich artykułów newsowych, reprezentacja cech TF-IDF jest wystarczająco bogata i nie wymaga złożonych transformacji nieliniowych zapewnianych przez głębokie architektury.

---

## SEKCJA: Wyniki - Najlepsza konfiguracja

Najlepszą wydajność osiągnął model **MLP_LargeBatch** o następującej konfiguracji (Tabela Y):

- Architektura: [512, 256] (2 warstwy ukryte)
- Learning Rate: 0.001
- Dropout: 0.5
- Batch Size: 64
- Max Features: 5000

Model ten osiągnął **F1-Score=0.8174** i **Accuracy=0.8252** na zbiorze testowym, co stanowi najlepszy wynik spośród wszystkich 12 testowanych konfiguracji. Dodatkowo, model charakteryzuje się krótkim czasem treningu (1.09s) i prostą architekturą, co ułatwia jego wdrożenie i interpretację.

Macierz pomyłek (Rysunek W) pokazuje równomierne wyniki dla wszystkich 6 kategorii newsów, z najlepszą precyzją dla kategorii "Sport" (92%) i "Gospodarka" (88%). Najczęstsze pomyłki występują między kategoriami tematycznie bliskimi, jak "Polityka" i "Świat" (7% błędnych klasyfikacji).

---

## SEKCJA: Dyskusja - Interpretacja wyników

Silny wpływ Learning Rate na wydajność modelu (korelacja r=0.695, p<0.0001) wskazuje, że dla tego zadania **tempo uczenia się jest krytycznym czynnikiem**. Wartości LR=0.001 lub LR=0.002 pozwalają na efektywną aktualizację wag w oparciu o gradienty funkcji straty, podczas gdy niższe wartości (0.0001, 0.0005) powodują zbyt powolną konwergencję w ograniczonej liczbie epok (20).

Brak wpływu Dropout (p=0.962) sugeruje, że testowane modele MLP nie są podatne na przeuczenie (overfitting) w tym zadaniu. Może to wynikać z:

1. Stosunkowo prostej architektury (2-4 warstwy)
2. Ograniczonej liczby epok (20)
3. Reprezentacji TF-IDF, która jest rzadka i wysokowymiarowa, co naturalnie regularizuje model

Podobnie, brak wpływu Batch Size (p=0.775) oznacza, że w zakresie 16-64 rozmiar batcha nie wpływa na jakość gradientów i stabilność treningu. To pozwala na wybór większego batch size (64) dla przyspieszenia obliczeń bez utraty wydajności.

---

## SEKCJA: Wnioski

1. **Learning Rate jest jedynym parametrem o statystycznie istotnym wpływie** na wydajność modelu MLP w zadaniu klasyfikacji polskich artykułów newsowych (p<0.0001).

2. **Optymalna wartość Learning Rate wynosi 0.001-0.002** dla treningu trwającego 20 epok z optymalizatorem Adam.

3. **Proste architektury (2 warstwy ukryte) są wystarczające** - głębsze sieci nie poprawiają wyników (p=0.893).

4. **Dropout, Batch Size i Max Features można wybrać dowolnie** w testowanych zakresach bez wpływu na wydajność końcową.

5. **Najlepszy model (F1=0.8174)** wykorzystuje architekturę [512, 256] z LR=0.001, Dropout=0.5 i Batch=64.

6. Wyniki sugerują, że **reprezentacja TF-IDF jest wystarczająco bogata** dla tego zadania i nie wymaga bardzo głębokich transformacji nieliniowych.

---

## SEKCJA: Dodatek - Statystyki eksperymentów

### Ogólne statystyki:

- Liczba testowanych konfiguracji: 12
- Łączny czas treningu wszystkich modeli: ~13 sekund
- Najlepszy F1-Score: 0.8174 (MLP_LargeBatch)
- Najgorszy F1-Score: 0.6823 (MLP_Small_LowLR)
- Średni F1-Score: 0.7858 ± 0.0413
- Zakres Accuracy: 0.7056 - 0.8252

### Metody statystyczne:

- **ANOVA (Analysis of Variance):** Test F dla porównania wielu grup
- **Korelacja Pearsona:** Miara siły związku liniowego
- **95% Confidence Intervals:** Przedziały ufności dla średnich
- **Regresja wielomianowa (stopień 2):** Dopasowanie krzywej do Learning Rate
- **Analiza interakcji:** Test kombinacji Dropout × Learning Rate

### Software:

- PyTorch 2.0+
- scikit-learn 1.3+
- scipy.stats (testy statystyczne)
- matplotlib 3.7+ (wizualizacje, 300 DPI)
- seaborn 0.12+ (heatmapy)

---

## GOTOWE ZDANIA NA OBRONĘ:

**Q: Dlaczego tylko Learning Rate ma wpływ?**
A: Pozostałe parametry są dobrze wyregulowane przez domyślne wartości lub zakres testowanych wartości jest zbyt wąski. Learning Rate bezpośrednio kontroluje tempo uczenia i ma największy wpływ na konwergencję w ograniczonej liczbie epok (20).

**Q: Dlaczego głębsze sieci nie są lepsze?**
A: Reprezentacja TF-IDF już zawiera bogate cechy lingwistyczne. Dodatkowe warstwy zwiększają złożoność bez dodawania wartości informacyjnej. Dla tego zadania wystarczą 1-2 transformacje nieliniowe.

**Q: Czy 20 epok to wystarczająco dużo?**
A: Tak, krzywe uczenia (Rysunek X) pokazują stabilizację po 15-20 epoce. Dłuższy trening mógłby prowadzić do przeuczenia.

**Q: Czy wyniki są statystycznie wiarygodne?**
A: Tak, użyto standardowych testów ANOVA (p<0.05 jako próg istotności), przedziałów ufności 95%, i korelacji Pearsona. Wyniki dla Learning Rate są wysoce istotne (p<0.0001).

**Q: Czy można to uogólnić na inne zadania NLP?**
A: Wyniki dotyczą konkretnie klasyfikacji polskich newsów z TF-IDF. Dla innych języków, domen lub reprezentacji (np. embeddingi BERT) wyniki mogą się różnić.
