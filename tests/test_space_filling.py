"""
Testy jednostkowe dla modułu space_filling.py
"""

import pytest
import numpy as np
from src.space_filling import hilbert_curve, peano_curve, dragon_curve


class TestHilbertCurve:
    """Testy dla krzywej Hilberta."""

    def test_order_0(self):
        """Test dla rzędu 0."""
        x, y = hilbert_curve(0)

        assert len(x) == 1
        assert len(y) == 1
        assert x[0] == 0.5
        assert y[0] == 0.5

    def test_point_count(self):
        """Test liczby punktów dla różnych rzędów."""
        for order in range(1, 6):
            x, y = hilbert_curve(order)
            expected_points = 4 ** order

            assert len(x) == expected_points
            assert len(y) == expected_points

    def test_bounds(self):
        """Test zakresu współrzędnych [0, 1]."""
        for order in range(1, 5):
            x, y = hilbert_curve(order)

            assert np.all(x >= 0) and np.all(x <= 1)
            assert np.all(y >= 0) and np.all(y <= 1)

    def test_continuity(self):
        """Test ciągłości - sąsiednie punkty są blisko siebie."""
        x, y = hilbert_curve(4)

        # Obliczamy odległości między kolejnymi punktami
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)

        # Maksymalna odległość powinna być mała
        n = 2 ** 4  # rozmiar siatki
        max_expected_dist = np.sqrt(2) / n + 0.01  # diagonal + tolerance

        assert np.all(distances <= max_expected_dist)

    def test_invalid_order(self):
        """Test nieprawidłowego rzędu."""
        with pytest.raises(ValueError, match="Rząd krzywej musi być >= 0"):
            hilbert_curve(-1)

    def test_coverage(self):
        """Test pokrycia przestrzeni - punkty powinny być rozproszone."""
        x, y = hilbert_curve(5)

        # Dzielimy na siatkę i sprawdzamy pokrycie
        n_bins = 10
        hist, _, _ = np.histogram2d(x, y, bins=n_bins, range=[[0, 1], [0, 1]])

        # Większość komórek powinna być niepusta
        non_empty = np.sum(hist > 0)
        assert non_empty >= n_bins * n_bins * 0.8


class TestPeanoCurve:
    """Testy dla krzywej Peano."""

    def test_order_0(self):
        """Test dla rzędu 0."""
        x, y = peano_curve(0)

        assert len(x) == 1
        assert len(y) == 1

    def test_point_count(self):
        """Test liczby punktów."""
        for order in range(1, 4):
            x, y = peano_curve(order)
            expected_points = 9 ** order

            assert len(x) == expected_points

    def test_bounds(self):
        """Test zakresu współrzędnych."""
        x, y = peano_curve(3)

        assert np.all(x >= 0) and np.all(x <= 1)
        assert np.all(y >= 0) and np.all(y <= 1)

    def test_invalid_order(self):
        """Test nieprawidłowego rzędu."""
        with pytest.raises(ValueError):
            peano_curve(-1)


class TestDragonCurve:
    """Testy dla krzywej smoka."""

    def test_basic_generation(self):
        """Test podstawowego generowania."""
        x, y = dragon_curve(5)

        assert len(x) > 0
        assert len(y) > 0
        assert len(x) == len(y)

    def test_point_count(self):
        """Test liczby punktów - rosnąca liczba z rzędem."""
        prev_len = 0
        for order in range(1, 10):
            x, y = dragon_curve(order)
            # Liczba punktów rośnie z każdym rzędem
            assert len(x) > prev_len, f"Liczba punktów nie rośnie dla order={order}"
            prev_len = len(x)

    def test_normalization(self):
        """Test normalizacji do [0, 1]."""
        x, y = dragon_curve(10)

        assert np.all(x >= 0) and np.all(x <= 1)
        assert np.all(y >= 0) and np.all(y <= 1)

    def test_invalid_order(self):
        """Test nieprawidłowego rzędu."""
        with pytest.raises(ValueError):
            dragon_curve(-1)

    def test_continuity(self):
        """Test ciągłości - krzywa jest połączona."""
        x, y = dragon_curve(8)

        # Sprawdzamy, że punkty są połączone (brak skoków)
        dx = np.diff(x)
        dy = np.diff(y)

        # Żaden skok nie powinien być większy niż połowa zakresu
        assert np.all(np.abs(dx) < 0.5)
        assert np.all(np.abs(dy) < 0.5)
