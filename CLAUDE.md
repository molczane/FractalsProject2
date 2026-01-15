# CLAUDE.md - Kontekst projektu dla Claude Code

## Opis projektu

Projekt akademicki na przedmiot **Fraktale** dotyczący **funkcji interpolacji fraktalnych**. Celem jest implementacja, wizualizacja i analiza matematyczna funkcji fraktalnych ze szczególnym uwzględnieniem obliczania wymiaru Minkowskiego (box-counting dimension).

## Kluczowe pojęcia matematyczne

### 1. Funkcja Weierstrassa

Klasyczna funkcja ciągła, która nie jest różniczkowalna w żadnym punkcie:

```
W(x) = Σ(n=0 do N) aⁿ · cos(bⁿ · π · x)
```

Gdzie:
- `0 < a < 1` - współczynnik tłumienia amplitudy
- `b > 1` - współczynnik częstotliwości (zazwyczaj nieparzysta liczba całkowita)
- Warunek `a · b > 1` gwarantuje nieróżniczkowalność

**Wymiar fraktalny:** `D = 2 + log(a)/log(b)` dla `a·b > 1`

### 2. Krzywe Peano (Space-Filling Curves)

Krzywe ciągłe, które wypełniają przestrzeń (przechodzą przez każdy punkt kwadratu jednostkowego):

- **Krzywa Peano** - oryginalna konstrukcja (1890)
- **Krzywa Hilberta** - wariant z prostszą konstrukcją rekurencyjną
- **Wymiar topologiczny:** 1, **Wymiar fraktalny:** 2

### 3. Wymiar Minkowskiego (Box-Counting Dimension)

Metoda obliczania wymiaru fraktalnego poprzez pokrywanie zbioru "pudełkami":

```
dim_box = lim(ε→0) [log(N(ε)) / log(1/ε)]
```

Gdzie `N(ε)` to minimalna liczba pudełek o rozmiarze `ε` potrzebna do pokrycia zbioru.

**W praktyce:** Regresja liniowa dla punktów `(log(1/ε), log(N(ε)))`.

### 4. Fraktalne Funkcje Interpolacyjne (FIF)

Funkcje generowane przez Iterated Function Systems (IFS), które:
- Interpolują zadane punkty danych
- Posiadają fraktalną strukturę między punktami
- Wymiar zależy od parametrów skalujących

## Struktura projektu

```
FractalsProject2/
├── CLAUDE.md              # Ten plik - kontekst dla AI
├── SPEC.md                # Specyfikacja techniczna
├── docs/
│   └── TASK.md            # Oryginalne zadanie
├── src/
│   ├── __init__.py
│   ├── weierstrass.py     # Funkcja Weierstrassa
│   ├── peano.py           # Krzywe Peano/Hilberta
│   ├── box_counting.py    # Algorytm wymiaru Minkowskiego
│   ├── fif.py             # Fraktalne funkcje interpolacyjne
│   └── visualization.py   # Moduł wizualizacji
├── notebooks/
│   └── analysis.ipynb     # Główny notebook z analizą
├── tests/
│   └── test_*.py          # Testy jednostkowe
├── output/
│   ├── figures/           # Wygenerowane wykresy
│   └── data/              # Dane wynikowe
└── requirements.txt       # Zależności Python
```

## Stos technologiczny

- **Python 3.10+**
- **NumPy** - obliczenia numeryczne
- **Matplotlib** - wizualizacje
- **SciPy** - regresja liniowa dla box-counting
- **Jupyter Notebook** - interaktywna analiza

## Komendy deweloperskie

```bash
# Instalacja zależności
pip install -r requirements.txt

# Uruchomienie testów
pytest tests/

# Uruchomienie Jupyter Notebook
jupyter notebook notebooks/analysis.ipynb

# Generowanie wykresów
python -m src.visualization
```

## Kluczowe algorytmy do zaimplementowania

### Box-Counting Algorithm

```python
def box_counting_dimension(points, epsilon_range):
    """
    1. Dla każdego ε w epsilon_range:
       a. Podziel przestrzeń na siatkę pudełek o rozmiarze ε
       b. Policz N(ε) - liczbę pudełek zawierających punkty
    2. Dopasuj prostą do punktów (log(1/ε), log(N(ε)))
    3. Zwróć nachylenie prostej jako wymiar
    """
```

### Weierstrass Function

```python
def weierstrass(x, a, b, n_terms):
    """
    W(x) = Σ(n=0 do n_terms) a^n * cos(b^n * π * x)
    """
```

### Hilbert Curve (L-system)

```python
def hilbert_curve(order):
    """
    Rekurencyjna konstrukcja krzywej Hilberta:
    - order 0: punkt
    - order n: 4 kopie order n-1 połączone w kształt U
    """
```

## Wymagania projektu (z TASK.md)

1. **Raport (50%)** - max 10 stron, forma naukowa
2. **Prezentacja (30%)** - 10 minut, max 10 slajdów
3. **Wkład indywidualny (20%)**

## Cele analizy

1. Zbadać zależność wymiaru Minkowskiego funkcji Weierstrassa od parametrów `a` i `b`
2. Porównać wymiary różnych krzywych wypełniających przestrzeń
3. Zaimplementować i przeanalizować fraktalne funkcje interpolacyjne
4. Zwizualizować strukturę fraktalną na różnych skalach

## Uwagi dla Claude Code

- Kod powinien być modularny i dobrze udokumentowany
- Wizualizacje muszą być czytelne i gotowe do publikacji
- Wyniki numeryczne powinny być weryfikowalne
- Używaj type hints w Pythonie
- Testy jednostkowe dla kluczowych algorytmów
