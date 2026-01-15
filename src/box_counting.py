"""
Moduł algorytmu box-counting do obliczania wymiaru Minkowskiego.

Wymiar Minkowskiego (box-counting dimension) jest jedną z najpopularniejszych
metod numerycznego wyznaczania wymiaru fraktalnego.

dim_box = lim(ε→0) [log(N(ε)) / log(1/ε)]

Gdzie N(ε) to minimalna liczba "pudełek" o rozmiarze ε potrzebna
do pokrycia zbioru.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Callable, Optional


def box_count(points: np.ndarray, epsilon: float) -> int:
    """
    Liczy liczbę pudełek o rozmiarze epsilon pokrywających punkty.

    Args:
        points: Tablica punktów shape (N, 2) - współrzędne (x, y)
        epsilon: Rozmiar pudełka (boku kwadratu)

    Returns:
        Liczba niepustych pudełek N(epsilon)

    Raises:
        ValueError: Gdy points ma nieprawidłowy kształt
    """
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Points musi mieć kształt (N, 2), otrzymano: {points.shape}")

    if epsilon <= 0:
        raise ValueError(f"Epsilon musi być > 0, otrzymano: {epsilon}")

    # Normalizujemy punkty do [0, 1] x [0, 1]
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals

    # Unikamy dzielenia przez zero
    range_vals = np.where(range_vals == 0, 1, range_vals)
    normalized = (points - min_vals) / range_vals

    # Indeksy pudełek dla każdego punktu
    # Dodajemy małą wartość żeby punkty na krawędzi nie wypadły poza siatkę
    box_indices = (normalized / epsilon).astype(int)
    box_indices = np.clip(box_indices, 0, int(1 / epsilon))

    # Liczymy unikalne pudełka
    unique_boxes = set(map(tuple, box_indices))

    return len(unique_boxes)


def box_counting_dimension(
    points: np.ndarray,
    epsilon_min: float = 0.001,
    epsilon_max: float = 0.1,
    n_epsilons: int = 20
) -> Tuple[float, float, Dict]:
    """
    Oblicza wymiar Minkowskiego metodą box-counting.

    Algorytm:
    1. Dla różnych wartości epsilon, liczymy N(epsilon)
    2. Dopasowujemy prostą do punktów (log(1/epsilon), log(N(epsilon)))
    3. Nachylenie prostej to wymiar fraktalny

    Args:
        points: Tablica punktów shape (N, 2)
        epsilon_min: Minimalny rozmiar pudełka
        epsilon_max: Maksymalny rozmiar pudełka
        n_epsilons: Liczba wartości epsilon do przetestowania

    Returns:
        Tuple (dimension, r_squared, details):
        - dimension: Oszacowany wymiar fraktalny
        - r_squared: Współczynnik determinacji R² (jakość dopasowania)
        - details: Słownik z danymi do wizualizacji:
            - 'epsilons': wartości epsilon
            - 'counts': liczby pudełek N(epsilon)
            - 'log_inv_eps': log(1/epsilon)
            - 'log_counts': log(N(epsilon))
            - 'slope': nachylenie (= wymiar)
            - 'intercept': wyraz wolny

    Example:
        >>> points = np.random.rand(1000, 2)  # Losowe punkty w kwadracie
        >>> dim, r2, _ = box_counting_dimension(points)
        >>> print(f"Wymiar: {dim:.2f} (oczekiwane ~2.0)")
    """
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Points musi mieć kształt (N, 2), otrzymano: {points.shape}")

    # Generujemy wartości epsilon (skala logarytmiczna)
    epsilons = np.logspace(np.log10(epsilon_min), np.log10(epsilon_max), n_epsilons)

    # Liczymy pudełka dla każdego epsilon
    counts = np.array([box_count(points, eps) for eps in epsilons])

    # Filtrujemy wartości gdzie count > 0
    valid = counts > 0
    epsilons = epsilons[valid]
    counts = counts[valid]

    # Przygotowujemy dane do regresji
    log_inv_eps = np.log(1 / epsilons)
    log_counts = np.log(counts)

    # Regresja liniowa
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_inv_eps, log_counts)

    details = {
        'epsilons': epsilons,
        'counts': counts,
        'log_inv_eps': log_inv_eps,
        'log_counts': log_counts,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_value,
        'std_err': std_err
    }

    return slope, r_value ** 2, details


def function_to_points(
    f: Callable,
    x_range: Tuple[float, float] = (0, 1),
    n_points: int = 10000,
    *args,
    **kwargs
) -> np.ndarray:
    """
    Konwertuje funkcję na zbiór punktów (x, f(x)).

    Przydatne do obliczania wymiaru wykresu funkcji.

    Args:
        f: Funkcja jednej zmiennej f(x)
        x_range: Przedział (x_min, x_max)
        n_points: Liczba punktów do wygenerowania
        *args, **kwargs: Dodatkowe argumenty przekazywane do f

    Returns:
        Tablica punktów shape (n_points, 2)

    Example:
        >>> from src.weierstrass import weierstrass_function
        >>> points = function_to_points(weierstrass_function, (0, 2), 10000, a=0.5, b=3)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = f(x, *args, **kwargs)

    return np.column_stack([x, y])


def curve_to_points(
    x: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Konwertuje współrzędne krzywej na tablicę punktów.

    Args:
        x: Tablica współrzędnych x
        y: Tablica współrzędnych y

    Returns:
        Tablica punktów shape (N, 2)
    """
    return np.column_stack([np.asarray(x), np.asarray(y)])


def analyze_dimension_vs_parameter(
    func: Callable,
    param_name: str,
    param_values: np.ndarray,
    x_range: Tuple[float, float] = (0, 2),
    n_points: int = 10000,
    base_params: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analizuje zależność wymiaru od parametru funkcji.

    Args:
        func: Funkcja do analizy
        param_name: Nazwa parametru do zmiany
        param_values: Wartości parametru do przetestowania
        x_range: Przedział x
        n_points: Liczba punktów
        base_params: Bazowe parametry funkcji

    Returns:
        Tuple (param_values, dimensions, r_squared_values)
    """
    if base_params is None:
        base_params = {}

    dimensions = []
    r_squared_values = []

    for param_val in param_values:
        params = base_params.copy()
        params[param_name] = param_val

        points = function_to_points(func, x_range, n_points, **params)
        dim, r2, _ = box_counting_dimension(points)

        dimensions.append(dim)
        r_squared_values.append(r2)

    return param_values, np.array(dimensions), np.array(r_squared_values)


if __name__ == "__main__":
    # Przykład użycia - testowanie na znanych przykładach
    import matplotlib.pyplot as plt

    # Test 1: Prosta linia (wymiar = 1)
    x = np.linspace(0, 1, 1000)
    y = x
    points_line = np.column_stack([x, y])
    dim_line, r2_line, _ = box_counting_dimension(points_line)
    print(f"Prosta linia: wymiar = {dim_line:.3f} (oczekiwane: 1.0), R² = {r2_line:.4f}")

    # Test 2: Kwadrat wypełniony (wymiar = 2)
    points_square = np.random.rand(10000, 2)
    dim_square, r2_square, _ = box_counting_dimension(points_square)
    print(f"Kwadrat: wymiar = {dim_square:.3f} (oczekiwane: 2.0), R² = {r2_square:.4f}")

    # Test 3: Funkcja Weierstrassa
    from weierstrass import weierstrass_function, theoretical_dimension

    points_weier = function_to_points(weierstrass_function, (0, 2), 20000, a=0.5, b=3)
    dim_weier, r2_weier, details = box_counting_dimension(points_weier)
    theo_dim = theoretical_dimension(0.5, 3)
    print(f"Weierstrass: wymiar = {dim_weier:.3f} (teoretyczny: {theo_dim:.3f}), R² = {r2_weier:.4f}")

    # Wykres regresji
    plt.figure(figsize=(8, 6))
    plt.plot(details['log_inv_eps'], details['log_counts'], 'bo', label='Dane')
    plt.plot(details['log_inv_eps'],
             details['slope'] * details['log_inv_eps'] + details['intercept'],
             'r-', label=f'Regresja (D = {dim_weier:.3f})')
    plt.xlabel('log(1/ε)')
    plt.ylabel('log(N(ε))')
    plt.title('Analiza box-counting dla funkcji Weierstrassa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
