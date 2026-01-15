# Funkcje Interpolacji Fraktalnych

Projekt akademicki na przedmiot **Fraktale** - analiza i implementacja funkcji interpolacji fraktalnych.

## Opis projektu

Projekt koncentruje się na trzech głównych obszarach:

1. **Funkcja Weierstrassa** - klasyczna funkcja ciągła, wszędzie nieróżniczkowalna
2. **Krzywe wypełniające przestrzeń** - krzywe Peano, Hilberta
3. **Wymiar Minkowskiego** - algorytm box-counting do obliczania wymiaru fraktalnego

## Wymagania

- Python 3.10+
- Biblioteki: NumPy, Matplotlib, SciPy, Jupyter

## Instalacja

```bash
# Klonowanie repozytorium (jeśli używasz git)
cd FractalsProject2

# Utworzenie wirtualnego środowiska
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# lub: .venv\Scripts\activate  # Windows

# Instalacja zależności
pip install -r requirements.txt
```

## Struktura projektu

```
FractalsProject2/
├── src/                    # Moduły źródłowe
│   ├── weierstrass.py      # Funkcja Weierstrassa
│   ├── space_filling.py    # Krzywe Peano, Hilbert
│   ├── box_counting.py     # Algorytm wymiaru
│   └── visualization.py    # Wizualizacje
├── notebooks/              # Jupyter Notebooks z analizą
├── tests/                  # Testy jednostkowe
├── output/                 # Wygenerowane wykresy i dane
├── docs/                   # Dokumentacja i raport
└── presentation/           # Prezentacja
```

## Użycie

### Uruchomienie Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

### Uruchomienie testów

```bash
pytest tests/ -v
```

### Przykład użycia biblioteki

```python
from src import weierstrass_function, box_counting_dimension
import numpy as np

# Generowanie funkcji Weierstrassa
x = np.linspace(0, 2, 10000)
y = weierstrass_function(x, a=0.5, b=3)

# Obliczenie wymiaru fraktalnego
points = np.column_stack([x, y])
dim, r2, details = box_counting_dimension(points)
print(f"Wymiar Minkowskiego: {dim:.4f} (R² = {r2:.4f})")
```

## Dokumentacja

- [TASK.md](docs/TASK.md) - Oryginalne zadanie projektowe
- [CLAUDE.md](CLAUDE.md) - Kontekst projektu
- [SPEC.md](SPEC.md) - Specyfikacja techniczna

## Autorzy

Projekt wykonany w ramach przedmiotu Fraktale.

## Licencja

Projekt edukacyjny - wszystkie prawa zastrzeżone.
