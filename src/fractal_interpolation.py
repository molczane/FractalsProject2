"""
Moduł fraktalnych funkcji interpolacyjnych (FIF).

Fraktalne funkcje interpolacyjne to funkcje generowane przez
Iterated Function Systems (IFS), które interpolują zadane punkty
danych i posiadają fraktalną strukturę między punktami.

Teoria oparta na pracach M.F. Barnsleya.
"""

import numpy as np
from typing import Tuple, Optional, List


def fractal_interpolation_function(
    data_points: np.ndarray,
    scaling_factors: Optional[np.ndarray] = None,
    n_iterations: int = 10,
    n_output_points: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generuje fraktalną funkcję interpolacyjną (FIF).

    FIF interpoluje zadane punkty danych, ale między punktami
    ma fraktalną, samopodobną strukturę kontrolowaną przez
    współczynniki skalowania.

    Args:
        data_points: Punkty do interpolacji, shape (N, 2).
                     Pierwsza kolumna to x, druga to y.
                     Punkty muszą być posortowane po x.
        scaling_factors: Współczynniki skalowania dla każdego segmentu.
                        Shape (N-1,). Wartości z przedziału (-1, 1).
                        Jeśli None, używa 0.5 dla wszystkich segmentów.
        n_iterations: Liczba iteracji IFS
        n_output_points: Liczba punktów wyjściowych

    Returns:
        Tuple (x, y) - tablice współrzędnych interpolowanej funkcji

    Raises:
        ValueError: Gdy dane wejściowe są nieprawidłowe

    Example:
        >>> points = np.array([[0, 0], [0.5, 1], [1, 0]])
        >>> x, y = fractal_interpolation_function(points, scaling_factors=np.array([0.3, -0.3]))
    """
    data_points = np.asarray(data_points)

    if data_points.ndim != 2 or data_points.shape[1] != 2:
        raise ValueError(f"data_points musi mieć kształt (N, 2), otrzymano: {data_points.shape}")

    n_points = len(data_points)
    if n_points < 2:
        raise ValueError("Potrzeba co najmniej 2 punktów do interpolacji")

    # Sortujemy po x
    sorted_indices = np.argsort(data_points[:, 0])
    data_points = data_points[sorted_indices]

    x_data = data_points[:, 0]
    y_data = data_points[:, 1]

    # Domyślne współczynniki skalowania
    if scaling_factors is None:
        scaling_factors = np.full(n_points - 1, 0.3)
    else:
        scaling_factors = np.asarray(scaling_factors)
        if len(scaling_factors) != n_points - 1:
            raise ValueError(
                f"Liczba współczynników skalowania ({len(scaling_factors)}) "
                f"musi być równa liczbie segmentów ({n_points - 1})"
            )

    # Sprawdzamy warunek zbieżności
    if np.any(np.abs(scaling_factors) >= 1):
        raise ValueError("Współczynniki skalowania muszą być w przedziale (-1, 1)")

    # Obliczamy parametry transformacji afinicznych
    x0, xn = x_data[0], x_data[-1]
    y0, yn = y_data[0], y_data[-1]

    # Generujemy punkty startowe
    t = np.linspace(0, 1, n_output_points)
    # Początkowa interpolacja liniowa
    current_x = x0 + t * (xn - x0)
    current_y = y0 + t * (yn - y0)

    # Iteracje IFS
    for _ in range(n_iterations):
        new_x = np.zeros_like(current_x)
        new_y = np.zeros_like(current_y)

        for i in range(n_points - 1):
            # Parametry segmentu
            xi, xi1 = x_data[i], x_data[i + 1]
            yi, yi1 = y_data[i], y_data[i + 1]
            d = scaling_factors[i]

            # Transformacja afiniczna dla segmentu i
            ai = (xi1 - xi) / (xn - x0)
            ci = (xi * xn - xi1 * x0) / (xn - x0)

            ei = (yi1 - yi) / (xn - x0) - d * (yn - y0) / (xn - x0)
            fi = (xi1 * y0 - xi * yn) / (xn - x0) - d * (xi1 * y0 - xi * yn) / (xn - x0)
            fi = (xn * yi - x0 * yi1) / (xn - x0) - d * (xn * y0 - x0 * yn) / (xn - x0)

            # Maska dla punktów w tym segmencie
            segment_start = i / (n_points - 1)
            segment_end = (i + 1) / (n_points - 1)
            mask = (t >= segment_start) & (t < segment_end if i < n_points - 2 else t <= segment_end)

            # Stosujemy transformację
            new_x[mask] = ai * current_x[mask] + ci
            new_y[mask] = d * current_y[mask] + ei * current_x[mask] + fi

        current_x = new_x
        current_y = new_y

    return current_x, current_y


def fif_dimension_theoretical(scaling_factors: np.ndarray) -> float:
    """
    Oblicza teoretyczny wymiar fraktalny FIF.

    Dla FIF z N-1 segmentami i współczynnikami skalowania d_i,
    wymiar D jest rozwiązaniem równania:

    Σ |d_i|^D = 1

    Args:
        scaling_factors: Współczynniki skalowania

    Returns:
        Teoretyczny wymiar fraktalny

    Note:
        Dla przypadku gdy wszystkie |d_i| są równe d:
        (N-1) * d^D = 1
        D = -log(N-1) / log(d)
    """
    scaling_factors = np.asarray(scaling_factors)
    abs_d = np.abs(scaling_factors)

    if np.all(abs_d == 0):
        return 1.0  # Interpolacja liniowa

    # Rozwiązujemy równanie numerycznie
    # Σ |d_i|^D = 1
    from scipy.optimize import brentq

    def equation(D):
        return np.sum(abs_d ** D) - 1

    # Szukamy D w przedziale (1, 2)
    try:
        # Sprawdzamy czy istnieje rozwiązanie
        if equation(1.0) * equation(2.0) > 0:
            # Brak rozwiązania w przedziale - zwracamy graniczną wartość
            return 1.0 if equation(1.0) < 0 else 2.0

        dimension = brentq(equation, 1.0, 2.0)
        return dimension
    except ValueError:
        return 1.0


def simple_fif(
    data_points: np.ndarray,
    d: float = 0.3,
    n_iterations: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uproszczona wersja FIF z jednakowym współczynnikiem skalowania.

    Implementacja oparta na algorytmie chaosu (chaos game).

    Args:
        data_points: Punkty do interpolacji, shape (N, 2)
        d: Współczynnik skalowania (jednakowy dla wszystkich segmentów)
        n_iterations: Liczba iteracji

    Returns:
        Tuple (x, y) - tablice współrzędnych
    """
    data_points = np.asarray(data_points)

    if data_points.ndim != 2 or data_points.shape[1] != 2:
        raise ValueError(f"data_points musi mieć kształt (N, 2)")

    n = len(data_points)
    if n < 2:
        raise ValueError("Potrzeba co najmniej 2 punktów")

    # Sortujemy po x
    sorted_indices = np.argsort(data_points[:, 0])
    data_points = data_points[sorted_indices]

    x_data = data_points[:, 0]
    y_data = data_points[:, 1]

    x0, xn = x_data[0], x_data[-1]
    y0, yn = y_data[0], y_data[-1]

    # Obliczamy transformacje dla każdego segmentu
    transforms = []
    for i in range(n - 1):
        xi, xi1 = x_data[i], x_data[i + 1]
        yi, yi1 = y_data[i], y_data[i + 1]

        # Współczynniki transformacji afinicznej
        a = (xi1 - xi) / (xn - x0)
        c = (xn * xi - x0 * xi1) / (xn - x0)
        e = (yi1 - yi - d * (yn - y0)) / (xn - x0)
        f = (xn * yi - x0 * yi1 - d * (xn * y0 - x0 * yn)) / (xn - x0)

        transforms.append((a, c, d, e, f))

    # Generujemy punkty metodą deterministyczną
    n_points = 4 ** n_iterations
    all_x = [x0]
    all_y = [y0]

    # Iteracyjnie generujemy punkty
    current_points = [(x0, y0)]

    for iteration in range(n_iterations):
        new_points = []
        for px, py in current_points:
            for a, c, d_coef, e, f in transforms:
                new_x = a * px + c
                new_y = d_coef * py + e * px + f
                new_points.append((new_x, new_y))
        current_points = new_points

    # Sortujemy po x
    current_points.sort(key=lambda p: p[0])

    x = np.array([p[0] for p in current_points])
    y = np.array([p[1] for p in current_points])

    return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Przykład 1: Trzy punkty
    data = np.array([
        [0.0, 0.0],
        [0.5, 1.0],
        [1.0, 0.0]
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Różne współczynniki skalowania
    scaling_values = [0.0, 0.3, 0.5, -0.4]

    for ax, d in zip(axes.flat, scaling_values):
        if d == 0:
            # Interpolacja liniowa
            x = np.linspace(0, 1, 1000)
            y = np.interp(x, data[:, 0], data[:, 1])
        else:
            x, y = simple_fif(data, d=d, n_iterations=6)

        ax.plot(x, y, 'b-', linewidth=0.5, alpha=0.7)
        ax.plot(data[:, 0], data[:, 1], 'ro', markersize=8)
        ax.set_title(f"FIF z d = {d}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../output/figures/fif_examples.png", dpi=150)
    plt.show()
