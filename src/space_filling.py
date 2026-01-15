"""
Moduł krzywych wypełniających przestrzeń (Space-Filling Curves).

Krzywe wypełniające przestrzeń to ciągłe krzywe, które przechodzą
przez każdy punkt pewnego obszaru (np. kwadratu jednostkowego).

Zaimplementowane krzywe:
- Krzywa Hilberta
- Krzywa Peano
- Krzywa smoka (Dragon curve)
"""

import numpy as np
from typing import Tuple


def hilbert_curve(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generuje krzywą Hilberta zadanego rzędu.

    Krzywa Hilberta jest rekurencyjnie definiowaną krzywą wypełniającą
    przestrzeń. Dla rzędu n, krzywa przechodzi przez 4^n punktów.

    Args:
        order: Rząd krzywej (0, 1, 2, ...). Wyższy rząd = więcej szczegółów.
               Zalecane: 1-8 (dla order > 10 może być wolne)

    Returns:
        Tuple (x, y) - tablice współrzędnych punktów krzywej,
        znormalizowane do przedziału [0, 1]

    Example:
        >>> x, y = hilbert_curve(3)
        >>> len(x)
        64
    """
    if order < 0:
        raise ValueError(f"Rząd krzywej musi być >= 0, otrzymano: {order}")

    if order == 0:
        return np.array([0.5]), np.array([0.5])

    n = 2 ** order  # Rozmiar siatki
    num_points = n * n

    # Tablice na współrzędne
    x = np.zeros(num_points)
    y = np.zeros(num_points)

    for i in range(num_points):
        # Konwersja indeksu na współrzędne Hilberta
        px, py = _hilbert_d2xy(order, i)
        x[i] = (px + 0.5) / n
        y[i] = (py + 0.5) / n

    return x, y


def _hilbert_d2xy(order: int, d: int) -> Tuple[int, int]:
    """
    Konwertuje indeks d na współrzędne (x, y) na krzywej Hilberta.

    Algorytm oparty na iteracyjnej dekompozycji.
    """
    x = y = 0
    s = 1

    for i in range(order):
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)

        # Rotacja
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x

        x += s * rx
        y += s * ry
        d //= 4
        s *= 2

    return x, y


def peano_curve(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generuje oryginalną krzywą Peano zadanego rzędu.

    Krzywa Peano jest pierwszą odkrytą krzywą wypełniającą przestrzeń (1890).
    Dla rzędu n, krzywa przechodzi przez 9^n punktów.

    Args:
        order: Rząd krzywej (0, 1, 2, ...). Zalecane: 1-5

    Returns:
        Tuple (x, y) - tablice współrzędnych punktów krzywej

    Example:
        >>> x, y = peano_curve(2)
        >>> len(x)
        81
    """
    if order < 0:
        raise ValueError(f"Rząd krzywej musi być >= 0, otrzymano: {order}")

    if order == 0:
        return np.array([0.5]), np.array([0.5])

    # Użyjemy L-systemu do generowania krzywej Peano
    # Reguły: L -> LFRFL-F-RFLFR+F+LFRFL
    #         R -> RFLFR+F+LFRFL-F-RFLFR

    # Zaczynamy od prostszej implementacji - rekurencyjna konstrukcja
    n = 3 ** order
    points = []

    def peano_recursive(x0, y0, ax, ay, bx, by, depth):
        """Rekurencyjna konstrukcja krzywej Peano."""
        if depth == 0:
            points.append((x0 + (ax + bx) / 2, y0 + (ay + by) / 2))
            return

        # Dzielimy na 9 części
        ax3, ay3 = ax / 3, ay / 3
        bx3, by3 = bx / 3, by / 3

        # Kolejność przechodzenia przez 9 kwadratów
        peano_recursive(x0, y0, ax3, ay3, bx3, by3, depth - 1)
        peano_recursive(x0 + ax3, y0 + ay3, ax3, ay3, bx3, by3, depth - 1)
        peano_recursive(x0 + 2 * ax3, y0 + 2 * ay3, ax3, ay3, bx3, by3, depth - 1)

        peano_recursive(x0 + 2 * ax3 + bx3, y0 + 2 * ay3 + by3, ax3, ay3, -bx3, -by3, depth - 1)
        peano_recursive(x0 + ax3 + bx3, y0 + ay3 + by3, ax3, ay3, -bx3, -by3, depth - 1)
        peano_recursive(x0 + bx3, y0 + by3, ax3, ay3, -bx3, -by3, depth - 1)

        peano_recursive(x0 + 2 * bx3, y0 + 2 * by3, ax3, ay3, bx3, by3, depth - 1)
        peano_recursive(x0 + ax3 + 2 * bx3, y0 + ay3 + 2 * by3, ax3, ay3, bx3, by3, depth - 1)
        peano_recursive(x0 + 2 * ax3 + 2 * bx3, y0 + 2 * ay3 + 2 * by3, ax3, ay3, bx3, by3, depth - 1)

    peano_recursive(0, 0, 1, 0, 0, 1, order)

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    return x, y


def dragon_curve(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generuje krzywą smoka (Dragon curve) zadanego rzędu.

    Krzywa smoka jest fraktalem powstającym przez wielokrotne
    składanie paska papieru.

    Args:
        order: Rząd krzywej (liczba iteracji). Zalecane: 1-15

    Returns:
        Tuple (x, y) - tablice współrzędnych punktów krzywej

    Example:
        >>> x, y = dragon_curve(10)
    """
    if order < 0:
        raise ValueError(f"Rząd krzywej musi być >= 0, otrzymano: {order}")

    # L-system: F -> F+G, G -> F-G
    # + = skręt w lewo o 90°
    # - = skręt w prawo o 90°

    # Generujemy sekwencję ruchów
    sequence = [1]  # 1 = prawo, 0 = lewo

    for _ in range(order):
        # Nowa sekwencja: stara + 1 + odwrócona(zamieniona) stara
        new_sequence = sequence + [1] + [1 - x for x in reversed(sequence)]
        sequence = new_sequence

    # Konwertujemy sekwencję na współrzędne
    x, y = [0.0], [0.0]
    direction = 0  # 0=prawo, 1=góra, 2=lewo, 3=dół
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    for turn in sequence:
        # Ruch do przodu
        x.append(x[-1] + dx[direction])
        y.append(y[-1] + dy[direction])

        # Skręt
        if turn == 1:  # prawo
            direction = (direction - 1) % 4
        else:  # lewo
            direction = (direction + 1) % 4

    # Ostatni ruch
    x.append(x[-1] + dx[direction])
    y.append(y[-1] + dy[direction])

    # Normalizacja do [0, 1]
    x = np.array(x)
    y = np.array(y)

    x = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x * 0 + 0.5
    y = (y - y.min()) / (y.max() - y.min()) if y.max() != y.min() else y * 0 + 0.5

    return x, y


if __name__ == "__main__":
    # Przykład użycia
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Krzywa Hilberta
    x, y = hilbert_curve(4)
    axes[0].plot(x, y, 'b-', linewidth=0.5)
    axes[0].set_title(f"Krzywa Hilberta (order=4, {len(x)} punktów)")
    axes[0].set_aspect('equal')

    # Krzywa Peano
    x, y = peano_curve(3)
    axes[1].plot(x, y, 'r-', linewidth=0.5)
    axes[1].set_title(f"Krzywa Peano (order=3, {len(x)} punktów)")
    axes[1].set_aspect('equal')

    # Krzywa smoka
    x, y = dragon_curve(12)
    axes[2].plot(x, y, 'g-', linewidth=0.5)
    axes[2].set_title(f"Krzywa smoka (order=12, {len(x)} punktów)")
    axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.show()
