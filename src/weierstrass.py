"""
Moduł funkcji Weierstrassa.

Funkcja Weierstrassa to klasyczny przykład funkcji ciągłej,
która nie jest różniczkowalna w żadnym punkcie.

W(x) = Σ(n=0 do N) a^n * cos(b^n * π * x)

Gdzie:
- 0 < a < 1 (współczynnik tłumienia amplitudy)
- b > 1 (współczynnik częstotliwości)
- Warunek a*b > 1 gwarantuje nieróżniczkowalność
"""

import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, list, float]


def weierstrass_function(
    x: ArrayLike,
    a: float = 0.5,
    b: float = 3.0,
    n_terms: int = 50
) -> np.ndarray:
    """
    Oblicza funkcję Weierstrassa dla podanych punktów.

    Args:
        x: Tablica punktów wejściowych (lub pojedyncza wartość)
        a: Współczynnik tłumienia (0 < a < 1)
        b: Współczynnik częstotliwości (b > 1, zazwyczaj nieparzyste)
        n_terms: Liczba wyrazów szeregu

    Returns:
        Wartości funkcji Weierstrassa w punktach x

    Raises:
        ValueError: Gdy parametry nie spełniają warunków

    Example:
        >>> x = np.linspace(0, 2, 1000)
        >>> y = weierstrass_function(x, a=0.5, b=3)
    """
    # Walidacja parametrów
    if not (0 < a < 1):
        raise ValueError(f"Parametr 'a' musi być w przedziale (0, 1), otrzymano: {a}")
    if b <= 1:
        raise ValueError(f"Parametr 'b' musi być większy od 1, otrzymano: {b}")
    if n_terms < 1:
        raise ValueError(f"Liczba wyrazów musi być >= 1, otrzymano: {n_terms}")

    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    for n in range(n_terms):
        result += (a ** n) * np.cos((b ** n) * np.pi * x)

    return result


def theoretical_dimension(a: float, b: float) -> float:
    """
    Oblicza teoretyczny wymiar fraktalny wykresu funkcji Weierstrassa.

    Dla funkcji Weierstrassa z parametrami a, b spełniającymi a*b > 1,
    wymiar fraktalny wykresu wynosi:

    D = 2 + log(a) / log(b)

    Args:
        a: Współczynnik tłumienia (0 < a < 1)
        b: Współczynnik częstotliwości (b > 1)

    Returns:
        Teoretyczny wymiar fraktalny (wartość między 1 a 2)

    Raises:
        ValueError: Gdy parametry nie spełniają warunków

    Example:
        >>> theoretical_dimension(0.5, 3)
        1.369...
    """
    if not (0 < a < 1):
        raise ValueError(f"Parametr 'a' musi być w przedziale (0, 1), otrzymano: {a}")
    if b <= 1:
        raise ValueError(f"Parametr 'b' musi być większy od 1, otrzymano: {b}")

    if a * b <= 1:
        # Gdy a*b <= 1, funkcja jest różniczkowalna i ma wymiar 1
        return 1.0

    # D = 2 + log(a)/log(b)
    # Ponieważ a < 1, log(a) < 0, więc D < 2
    dimension = 2 + np.log(a) / np.log(b)

    return dimension


def weierstrass_derivative_approx(
    x: ArrayLike,
    a: float = 0.5,
    b: float = 3.0,
    n_terms: int = 50,
    h: float = 1e-8
) -> np.ndarray:
    """
    Przybliżenie pochodnej funkcji Weierstrassa metodą różnic skończonych.

    Służy do demonstracji nieróżniczkowalności - dla a*b > 1
    przybliżenie pochodnej nie zbiega przy h -> 0.

    Args:
        x: Punkty w których obliczamy przybliżenie pochodnej
        a: Współczynnik tłumienia
        b: Współczynnik częstotliwości
        n_terms: Liczba wyrazów szeregu
        h: Krok różnicowy

    Returns:
        Przybliżenie pochodnej (f(x+h) - f(x)) / h
    """
    x = np.asarray(x)
    f_x = weierstrass_function(x, a, b, n_terms)
    f_x_h = weierstrass_function(x + h, a, b, n_terms)

    return (f_x_h - f_x) / h


if __name__ == "__main__":
    # Przykład użycia
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2, 5000)
    y = weierstrass_function(x, a=0.5, b=3, n_terms=50)

    print(f"Teoretyczny wymiar: {theoretical_dimension(0.5, 3):.4f}")

    plt.figure(figsize=(12, 4))
    plt.plot(x, y, 'b-', linewidth=0.5)
    plt.title("Funkcja Weierstrassa (a=0.5, b=3)")
    plt.xlabel("x")
    plt.ylabel("W(x)")
    plt.grid(True, alpha=0.3)
    plt.show()
