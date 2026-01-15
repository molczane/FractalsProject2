# SPEC.md - Specyfikacja Techniczna Projektu

## Spis treści

1. [Przegląd projektu](#1-przegląd-projektu)
2. [Architektura systemu](#2-architektura-systemu)
3. [Specyfikacja modułów](#3-specyfikacja-modułów)
4. [Plan implementacji (Taski)](#4-plan-implementacji-taski)
5. [Wymagania testowe](#5-wymagania-testowe)
6. [Deliverables](#6-deliverables)

---

## 1. Przegląd projektu

### 1.1 Cel
Implementacja i analiza matematyczna trzech rodzajów funkcji fraktalnych:
1. **Funkcja Weierstrassa** - funkcja ciągła, wszędzie nieróżniczkowalna
2. **Krzywe wypełniające przestrzeń** - Peano, Hilbert
3. **Fraktalne funkcje interpolacyjne (FIF)** - interpolacja z własnościami fraktalnymi

### 1.2 Zakres analizy
- Wizualizacja funkcji przy różnych parametrach
- Obliczanie wymiaru Minkowskiego metodą box-counting
- Analiza zależności wymiaru od parametrów
- Porównanie teoretycznych i numerycznych wartości wymiaru

### 1.3 Oczekiwane rezultaty
- Biblioteka Python z zaimplementowanymi algorytmami
- Jupyter Notebook z interaktywną analizą
- Zestaw wysokiej jakości wizualizacji
- Raport naukowy (PDF, max 10 stron)
- Prezentacja (PPT/PDF, max 10 slajdów)

---

## 2. Architektura systemu

### 2.1 Struktura katalogów

```
FractalsProject2/
├── CLAUDE.md                    # Kontekst dla AI
├── SPEC.md                      # Ten plik
├── README.md                    # Instrukcja użytkowania
├── requirements.txt             # Zależności
├── pyproject.toml               # Konfiguracja projektu
│
├── docs/
│   ├── TASK.md                  # Oryginalne zadanie
│   └── report/
│       ├── report.tex           # Raport LaTeX (opcjonalnie)
│       └── report.pdf           # Finalny raport
│
├── src/
│   ├── __init__.py
│   ├── weierstrass.py           # Moduł funkcji Weierstrassa
│   ├── space_filling.py         # Krzywe Peano, Hilbert
│   ├── box_counting.py          # Algorytm wymiaru
│   ├── fractal_interpolation.py # FIF
│   ├── visualization.py         # Wizualizacje
│   └── utils.py                 # Funkcje pomocnicze
│
├── notebooks/
│   ├── 01_weierstrass.ipynb     # Analiza Weierstrassa
│   ├── 02_space_filling.ipynb   # Analiza krzywych
│   ├── 03_box_counting.ipynb    # Weryfikacja wymiaru
│   ├── 04_fif.ipynb             # Interpolacja fraktalna
│   └── 05_final_analysis.ipynb  # Podsumowanie
│
├── tests/
│   ├── __init__.py
│   ├── test_weierstrass.py
│   ├── test_space_filling.py
│   ├── test_box_counting.py
│   └── test_fif.py
│
├── output/
│   ├── figures/                 # PNG/PDF z wykresami
│   └── data/                    # CSV z wynikami
│
└── presentation/
    └── slides.pdf               # Prezentacja
```

### 2.2 Zależności (requirements.txt)

```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
jupyter>=1.0.0
pytest>=7.0.0
seaborn>=0.12.0
pandas>=2.0.0
```

---

## 3. Specyfikacja modułów

### 3.1 Moduł `weierstrass.py`

#### Funkcje

```python
def weierstrass_function(
    x: np.ndarray,
    a: float = 0.5,
    b: float = 3.0,
    n_terms: int = 50
) -> np.ndarray:
    """
    Oblicza funkcję Weierstrassa dla podanych punktów.

    Args:
        x: Tablica punktów wejściowych
        a: Współczynnik tłumienia (0 < a < 1)
        b: Współczynnik częstotliwości (b > 1, zazwyczaj nieparzyste)
        n_terms: Liczba wyrazów szeregu

    Returns:
        Wartości funkcji Weierstrassa w punktach x

    Raises:
        ValueError: Gdy parametry nie spełniają warunków
    """

def theoretical_dimension(a: float, b: float) -> float:
    """
    Oblicza teoretyczny wymiar fraktalny wykresu funkcji Weierstrassa.

    D = 2 + log(a)/log(b)  dla a*b > 1

    Returns:
        Teoretyczny wymiar fraktalny (1 < D < 2)
    """

def weierstrass_derivative_approx(
    x: np.ndarray,
    a: float,
    b: float,
    n_terms: int,
    h: float = 1e-8
) -> np.ndarray:
    """
    Przybliżenie pochodnej (demonstracja nieróżniczkowalności).
    """
```

### 3.2 Moduł `space_filling.py`

#### Funkcje

```python
def hilbert_curve(order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generuje krzywą Hilberta zadanego rzędu.

    Args:
        order: Rząd krzywej (0, 1, 2, ...)

    Returns:
        Tuple (x, y) współrzędnych punktów krzywej
    """

def peano_curve(order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generuje oryginalną krzywą Peano zadanego rzędu.

    Args:
        order: Rząd krzywej

    Returns:
        Tuple (x, y) współrzędnych punktów krzywej
    """

def dragon_curve(order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generuje krzywą smoka (Dragon curve).
    """

def gosper_curve(order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generuje krzywą Gospera (flowsnake).
    """
```

### 3.3 Moduł `box_counting.py`

#### Funkcje

```python
def box_count(
    points: np.ndarray,
    epsilon: float
) -> int:
    """
    Liczy liczbę pudełek o rozmiarze epsilon pokrywających punkty.

    Args:
        points: Tablica punktów shape (N, 2)
        epsilon: Rozmiar pudełka

    Returns:
        Liczba niepustych pudełek N(epsilon)
    """

def box_counting_dimension(
    points: np.ndarray,
    epsilon_min: float = 0.001,
    epsilon_max: float = 0.1,
    n_epsilons: int = 20
) -> tuple[float, float, dict]:
    """
    Oblicza wymiar Minkowskiego metodą box-counting.

    Args:
        points: Tablica punktów shape (N, 2)
        epsilon_min: Minimalny rozmiar pudełka
        epsilon_max: Maksymalny rozmiar pudełka
        n_epsilons: Liczba wartości epsilon

    Returns:
        Tuple (dimension, r_squared, details)
        - dimension: Oszacowany wymiar fraktalny
        - r_squared: Współczynnik determinacji R²
        - details: Słownik z danymi do wizualizacji
    """

def function_to_points(
    f: Callable,
    x_range: tuple[float, float],
    n_points: int = 10000
) -> np.ndarray:
    """
    Konwertuje funkcję na zbiór punktów (x, f(x)).
    """
```

### 3.4 Moduł `fractal_interpolation.py`

#### Funkcje

```python
def fractal_interpolation_function(
    data_points: np.ndarray,
    scaling_factors: np.ndarray,
    n_iterations: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generuje fraktalną funkcję interpolacyjną (FIF).

    Args:
        data_points: Punkty do interpolacji shape (N, 2)
        scaling_factors: Współczynniki skalowania dla każdego segmentu
        n_iterations: Liczba iteracji IFS

    Returns:
        Tuple (x, y) interpolowanej funkcji
    """

def fif_dimension(scaling_factors: np.ndarray) -> float:
    """
    Oblicza teoretyczny wymiar FIF na podstawie współczynników.
    """
```

### 3.5 Moduł `visualization.py`

#### Funkcje

```python
def plot_weierstrass(
    a: float,
    b: float,
    x_range: tuple[float, float] = (0, 2),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Wizualizacja funkcji Weierstrassa."""

def plot_weierstrass_zoom(
    a: float,
    b: float,
    center: float,
    zoom_levels: list[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Wizualizacja samopodobieństwa przez przybliżanie."""

def plot_space_filling_curve(
    curve_type: str,
    orders: list[int],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Wizualizacja krzywych wypełniających przestrzeń."""

def plot_box_counting_analysis(
    points: np.ndarray,
    result: dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Wizualizacja analizy box-counting."""

def plot_dimension_vs_parameters(
    param_range: np.ndarray,
    dimensions: np.ndarray,
    theoretical: np.ndarray,
    param_name: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Porównanie wymiarów teoretycznych i numerycznych."""

def create_figure_grid(
    figures: list[plt.Figure],
    titles: list[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Tworzy siatkę wykresów do raportu."""
```

---

## 4. Plan implementacji (Taski)

### Faza 1: Setup i infrastruktura

- [ ] **TASK-001**: Utworzenie struktury katalogów projektu
- [ ] **TASK-002**: Konfiguracja `requirements.txt` i środowiska
- [ ] **TASK-003**: Utworzenie plików `__init__.py`
- [ ] **TASK-004**: Konfiguracja pytest

### Faza 2: Implementacja podstawowych modułów

- [ ] **TASK-010**: Implementacja `weierstrass.py`
  - [ ] Funkcja `weierstrass_function()`
  - [ ] Funkcja `theoretical_dimension()`
  - [ ] Walidacja parametrów

- [ ] **TASK-011**: Testy dla `weierstrass.py`
  - [ ] Test poprawności wartości
  - [ ] Test warunków brzegowych
  - [ ] Test wymiaru teoretycznego

- [ ] **TASK-020**: Implementacja `space_filling.py`
  - [ ] Krzywa Hilberta (rekurencja)
  - [ ] Krzywa Peano
  - [ ] Opcjonalnie: Dragon curve, Gosper curve

- [ ] **TASK-021**: Testy dla `space_filling.py`
  - [ ] Test liczby punktów
  - [ ] Test zakresu współrzędnych
  - [ ] Test ciągłości

- [ ] **TASK-030**: Implementacja `box_counting.py`
  - [ ] Funkcja `box_count()`
  - [ ] Funkcja `box_counting_dimension()`
  - [ ] Optymalizacja wydajności

- [ ] **TASK-031**: Testy dla `box_counting.py`
  - [ ] Test na prostej linii (wymiar = 1)
  - [ ] Test na kwadracie wypełnionym (wymiar = 2)
  - [ ] Test dokładności oszacowania

- [ ] **TASK-040**: Implementacja `fractal_interpolation.py`
  - [ ] Algorytm IFS dla FIF
  - [ ] Funkcja wymiaru teoretycznego

- [ ] **TASK-041**: Testy dla `fractal_interpolation.py`

### Faza 3: Wizualizacje

- [ ] **TASK-050**: Implementacja `visualization.py`
  - [ ] Wykresy funkcji Weierstrassa
  - [ ] Wykresy krzywych wypełniających
  - [ ] Wykresy analizy box-counting
  - [ ] Wspólny styl (publication-ready)

- [ ] **TASK-051**: Konfiguracja stylu matplotlib
  - [ ] Ustawienia fontów
  - [ ] Paleta kolorów
  - [ ] Rozmiary wykresów

### Faza 4: Analiza i Jupyter Notebooks

- [ ] **TASK-060**: Notebook `01_weierstrass.ipynb`
  - [ ] Wizualizacja dla różnych a, b
  - [ ] Demonstracja samopodobieństwa
  - [ ] Obliczenie i porównanie wymiarów

- [ ] **TASK-061**: Notebook `02_space_filling.ipynb`
  - [ ] Animacja budowy krzywych
  - [ ] Porównanie różnych krzywych
  - [ ] Analiza wymiaru

- [ ] **TASK-062**: Notebook `03_box_counting.ipynb`
  - [ ] Weryfikacja algorytmu na znanych przykładach
  - [ ] Analiza czułości na parametry

- [ ] **TASK-063**: Notebook `04_fif.ipynb`
  - [ ] Przykłady interpolacji fraktalnej
  - [ ] Porównanie z interpolacją klasyczną

- [ ] **TASK-064**: Notebook `05_final_analysis.ipynb`
  - [ ] Podsumowanie wszystkich wyników
  - [ ] Tabele porównawcze
  - [ ] Finalne wykresy do raportu

### Faza 5: Dokumentacja i raport

- [ ] **TASK-070**: Napisanie raportu
  - [ ] Wprowadzenie teoretyczne
  - [ ] Opis metod
  - [ ] Prezentacja wyników
  - [ ] Wnioski

- [ ] **TASK-071**: Przygotowanie prezentacji
  - [ ] Max 10 slajdów
  - [ ] Czytelne wizualizacje
  - [ ] Kluczowe wyniki

- [ ] **TASK-072**: Finalizacja README.md
- [ ] **TASK-073**: Przegląd kodu i dokumentacji

---

## 5. Wymagania testowe

### 5.1 Testy jednostkowe

| Moduł | Test | Oczekiwany wynik |
|-------|------|------------------|
| weierstrass | `test_weierstrass_values` | Zgodność z ręcznymi obliczeniami |
| weierstrass | `test_dimension_formula` | D = 2 + log(a)/log(b) |
| space_filling | `test_hilbert_point_count` | 4^order punktów |
| space_filling | `test_curve_bounds` | Punkty w [0,1] × [0,1] |
| box_counting | `test_line_dimension` | D ≈ 1.0 (±0.1) |
| box_counting | `test_square_dimension` | D ≈ 2.0 (±0.1) |
| box_counting | `test_sierpinski_dimension` | D ≈ 1.585 (±0.1) |

### 5.2 Testy integracyjne

- Pipeline: Weierstrass → box_counting → wizualizacja
- Pipeline: Hilbert curve → box_counting → porównanie z teorią

### 5.3 Kryteria akceptacji

- Wszystkie testy przechodzą
- Wymiar box-counting zgadza się z teorią (błąd < 5%)
- Wykresy są czytelne i estetyczne
- Kod jest udokumentowany (docstrings)

---

## 6. Deliverables

### 6.1 Kod źródłowy
- [x] Repozytorium z pełną strukturą
- [ ] Wszystkie moduły zaimplementowane
- [ ] Testy jednostkowe
- [ ] Jupyter Notebooks

### 6.2 Dokumentacja
- [x] CLAUDE.md
- [x] SPEC.md
- [ ] README.md z instrukcją
- [ ] Docstrings w kodzie

### 6.3 Raport
- [ ] PDF, max 10 stron
- [ ] Wprowadzenie teoretyczne
- [ ] Opis implementacji
- [ ] Wyniki i analiza
- [ ] Wnioski

### 6.4 Prezentacja
- [ ] PPT/PDF, max 10 slajdów
- [ ] Czytelne wykresy
- [ ] Podsumowanie kluczowych wyników

---

## Appendix A: Wzory matematyczne

### Funkcja Weierstrassa

$$W(x) = \sum_{n=0}^{\infty} a^n \cos(b^n \pi x)$$

Wymiar fraktalny wykresu:
$$D = 2 + \frac{\log a}{\log b}, \quad \text{dla } ab > 1$$

### Wymiar Minkowskiego

$$\dim_{box}(F) = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$$

### Krzywa Hilberta

Rząd n zawiera $4^n$ segmentów o długości $2^{-n}$ każdy.
Całkowita długość: $L_n = 4^n \cdot 2^{-n} = 2^n \to \infty$

---

## Appendix B: Referencje

1. Peitgen, H.-O., Jürgens, H., & Saupe, D. (2004). *Chaos and Fractals: New Frontiers of Science*
2. Falconer, K. (2014). *Fractal Geometry: Mathematical Foundations and Applications*
3. Barnsley, M. F. (2012). *Fractals Everywhere*
4. Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*
