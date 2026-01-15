"""
Funkcje pomocnicze dla projektu fraktali.
"""

import numpy as np
from typing import Tuple, Optional
import os


def ensure_directory(path: str) -> None:
    """
    Tworzy katalog jeśli nie istnieje.

    Args:
        path: Ścieżka do katalogu
    """
    os.makedirs(path, exist_ok=True)


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Normalizuje punkty do przedziału [0, 1] x [0, 1].

    Args:
        points: Tablica punktów shape (N, 2)

    Returns:
        Tuple (normalized_points, params)
        - normalized_points: Znormalizowane punkty
        - params: Słownik z parametrami do odwrotnej transformacji
    """
    points = np.asarray(points)

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_vals = max_vals - min_vals

    # Unikamy dzielenia przez zero
    range_vals = np.where(range_vals == 0, 1, range_vals)

    normalized = (points - min_vals) / range_vals

    params = {
        'min': min_vals,
        'max': max_vals,
        'range': range_vals
    }

    return normalized, params


def denormalize_points(
    normalized: np.ndarray,
    params: dict
) -> np.ndarray:
    """
    Odwraca normalizację punktów.

    Args:
        normalized: Znormalizowane punkty
        params: Parametry z normalize_points

    Returns:
        Oryginalne punkty
    """
    return normalized * params['range'] + params['min']


def generate_grid_points(
    n: int,
    x_range: Tuple[float, float] = (0, 1),
    y_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    Generuje siatkę punktów.

    Args:
        n: Liczba punktów na każdą oś
        x_range: Zakres x
        y_range: Zakres y

    Returns:
        Tablica punktów shape (n*n, 2)
    """
    x = np.linspace(x_range[0], x_range[1], n)
    y = np.linspace(y_range[0], y_range[1], n)

    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def estimate_computation_time(
    n_points: int,
    n_epsilons: int = 20
) -> float:
    """
    Szacuje czas obliczeń box-counting w sekundach.

    Args:
        n_points: Liczba punktów
        n_epsilons: Liczba wartości epsilon

    Returns:
        Szacowany czas w sekundach
    """
    # Empiryczne oszacowanie: O(n_points * n_epsilons)
    return n_points * n_epsilons * 1e-7


def progress_bar(
    iteration: int,
    total: int,
    prefix: str = '',
    suffix: str = '',
    length: int = 50
) -> None:
    """
    Wyświetla pasek postępu w konsoli.

    Args:
        iteration: Obecna iteracja
        total: Całkowita liczba iteracji
        prefix: Tekst przed paskiem
        suffix: Tekst po pasku
        length: Długość paska
    """
    percent = f"{100 * iteration / total:.1f}"
    filled = int(length * iteration / total)
    bar = '█' * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)

    if iteration == total:
        print()


def save_results_to_csv(
    filename: str,
    data: dict,
    output_dir: str = 'output/data'
) -> str:
    """
    Zapisuje wyniki do pliku CSV.

    Args:
        filename: Nazwa pliku (bez rozszerzenia)
        data: Słownik z danymi (klucze = nazwy kolumn)
        output_dir: Katalog wyjściowy

    Returns:
        Pełna ścieżka do zapisanego pliku
    """
    import pandas as pd

    ensure_directory(output_dir)
    filepath = os.path.join(output_dir, f"{filename}.csv")

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

    return filepath


def load_results_from_csv(
    filename: str,
    input_dir: str = 'output/data'
) -> dict:
    """
    Wczytuje wyniki z pliku CSV.

    Args:
        filename: Nazwa pliku (bez rozszerzenia)
        input_dir: Katalog wejściowy

    Returns:
        Słownik z danymi
    """
    import pandas as pd

    filepath = os.path.join(input_dir, f"{filename}.csv")
    df = pd.read_csv(filepath)

    return df.to_dict(orient='list')


class Timer:
    """
    Prosty timer do mierzenia czasu wykonania.

    Użycie:
        with Timer("Obliczenia"):
            # kod do zmierzenia
    """

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self.start_time
        print(f"{self.name}: {self.elapsed:.4f} s")


def format_dimension(dim: float, precision: int = 4) -> str:
    """
    Formatuje wymiar fraktalny do wyświetlenia.

    Args:
        dim: Wymiar
        precision: Liczba miejsc po przecinku

    Returns:
        Sformatowany string
    """
    return f"D = {dim:.{precision}f}"


def validate_dimension(dim: float, expected: float, tolerance: float = 0.1) -> bool:
    """
    Sprawdza czy obliczony wymiar jest zgodny z oczekiwanym.

    Args:
        dim: Obliczony wymiar
        expected: Oczekiwany wymiar
        tolerance: Dopuszczalna różnica

    Returns:
        True jeśli wymiar jest w zakresie tolerancji
    """
    return abs(dim - expected) <= tolerance
