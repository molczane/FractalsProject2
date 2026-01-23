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

    num_points = 9 ** order
    x = np.zeros(num_points)
    y = np.zeros(num_points)

    for i in range(num_points):
        # t to kopia indeksu do wyciągania "cyfr" w systemie dziewiątkowym (3x3)
        t = i
        px, py = 0, 0
        
        # Stany odbicia lustrzanego (False = brak, True = odbicie)
        # To klucz do ciągłości krzywej Peano
        flip_x = False
        flip_y = False

        # Przetwarzamy od najwyższego poziomu (największe kwadraty) do najniższego
        for level in range(order - 1, -1, -1):
            pow3 = 3 ** level
            # Wybieramy jeden z 9 kwadratów w aktualnej skali
            digit = t // (9 ** level)
            t %= (9 ** level)

            # Mapujemy digit (0-8) na bazowe współrzędne (r, c) w siatce 3x3 
            # stosując "zygzak" (snake order):
            # 0 1 2  (wiersz 0, w lewo -> prawo)
            # 5 4 3  (wiersz 1, prawo -> lewo)
            # 6 7 8  (wiersz 2, w lewo -> prawo)
            r_raw = digit // 3
            c_raw = digit % 3
            if r_raw == 1:
                c_raw = 2 - c_raw

            # Nakładamy aktualne skumulowane odbicia lustrzane z wyższych poziomów
            c = (2 - c_raw) if flip_x else c_raw
            r = (2 - r_raw) if flip_y else r_raw

            px += c * pow3
            py += r * pow3

            # Aktualizujemy stany odbicia dla następnego, głębszego poziomu.
            # Reguła Peano: odbijamy oś X, jeśli jesteśmy w środkowym wierszu (r_raw=1)
            # i odbijamy oś Y, jeśli jesteśmy w środkowej kolumnie (c_raw=1).
            if r_raw == 1:
                flip_x = not flip_x
            if c_raw == 1:
                flip_y = not flip_y

        # +0.5 stawia punkt w centrum małego kwadratu
        x[i] = px + 0.5
        y[i] = py + 0.5

    # Normalizacja do zakresu [0, 1]
    max_coord = 3 ** order
    return x / max_coord, y / max_coord


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
