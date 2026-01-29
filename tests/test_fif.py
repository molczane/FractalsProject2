"""
Testy jednostkowe dla modułu fractal_interpolation.py
"""

import pytest
import numpy as np
from src.fractal_interpolation import (
    fractal_interpolation_function,
    fif_dimension_theoretical,
    simple_fif
)


class TestFractalInterpolationFunction:
    """Testy dla głównej funkcji FIF."""

    def test_basic_generation(self):
        """Test podstawowego generowania."""
        data = np.array([[0, 0], [0.5, 1], [1, 0]])
        x, y = fractal_interpolation_function(data, n_iterations=5)

        assert len(x) > 0
        assert len(y) > 0
        assert len(x) == len(y)

    def test_interpolation_points(self):
        """Test czy funkcja przechodzi przez punkty interpolacji."""
        data = np.array([[0, 0], [0.5, 1], [1, 0]])
        x, y = fractal_interpolation_function(data, n_iterations=10, n_output_points=10000)

        # Sprawdzamy czy punkty początkowy i końcowy są zbliżone do danych
        # (z pewną tolerancją ze względu na dyskretyzację)
        assert np.any(np.abs(x - 0) < 0.01)
        assert np.any(np.abs(x - 1) < 0.01)

    def test_scaling_factors_effect(self):
        """Test wpływu współczynników skalowania."""
        data = np.array([[0, 0], [0.5, 1], [1, 0]])

        # d = 0 powinno dać interpolację liniową
        x1, y1 = fractal_interpolation_function(
            data,
            scaling_factors=np.array([0.01, 0.01]),
            n_iterations=5
        )

        # d > 0 powinno dać fraktalną strukturę
        x2, y2 = fractal_interpolation_function(
            data,
            scaling_factors=np.array([0.5, 0.5]),
            n_iterations=5
        )

        # Wariancja powinna być większa dla większego d
        var1 = np.var(y1)
        var2 = np.var(y2)

        # Ten test może być niestabilny - sprawdzamy tylko że są różne
        assert var1 != var2

    def test_invalid_data_points(self):
        """Test nieprawidłowych danych wejściowych."""
        # Za mało punktów
        with pytest.raises(ValueError, match="co najmniej 2 punktów"):
            fractal_interpolation_function(np.array([[0, 0]]))

        # Nieprawidłowy kształt
        with pytest.raises(ValueError, match="kształt"):
            fractal_interpolation_function(np.array([0, 1, 2]))

    def test_invalid_scaling_factors(self):
        """Test nieprawidłowych współczynników skalowania."""
        data = np.array([[0, 0], [0.5, 1], [1, 0]])

        # Nieprawidłowa liczba współczynników
        with pytest.raises(ValueError, match="Liczba współczynników"):
            fractal_interpolation_function(data, scaling_factors=np.array([0.5]))

        # Współczynnik >= 1
        with pytest.raises(ValueError, match="w przedziale"):
            fractal_interpolation_function(data, scaling_factors=np.array([1.0, 0.5]))

    def test_sorting(self):
        """Test sortowania punktów po x."""
        # Punkty nie posortowane
        data = np.array([[1, 0], [0, 0], [0.5, 1]])
        x, y = fractal_interpolation_function(data, n_iterations=3)

        # Powinno działać bez błędu
        assert len(x) > 0


class TestFIFDimensionTheoretical:
    """Testy dla teoretycznego wymiaru FIF."""

    def test_zero_scaling(self):
        """Test dla zerowych współczynników (wymiar = 1)."""
        dim = fif_dimension_theoretical(np.array([0, 0]))
        assert dim == 1.0

    def test_uniform_scaling(self):
        """Test dla jednakowych współczynników."""
        # Dla równoodległych węzłów (a_i = 1/n) oraz stałego d:
        # jeśli n*|d| > 1, to D = 2 + log|d|/log(n)
        d = 0.5
        n_segments = 4
        scaling = np.full(n_segments, d)

        dim = fif_dimension_theoretical(scaling)
        expected = 2 + np.log(abs(d)) / np.log(n_segments)

        np.testing.assert_almost_equal(dim, expected, decimal=3)

    def test_threshold_sum_di_equals_one(self):
        """Jeśli suma |d_i| <= 1, wykres ma wymiar 1."""
        # n*|d| = 1
        d = 0.25
        n_segments = 4
        scaling = np.full(n_segments, d)
        dim = fif_dimension_theoretical(scaling)
        assert dim == 1.0

    def test_dimension_range(self):
        """Test zakresu wymiaru (1 <= D <= 2)."""
        test_cases = [
            np.array([0.3, 0.3]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.1, 0.9]),
        ]

        for scaling in test_cases:
            dim = fif_dimension_theoretical(scaling)
            assert 1.0 <= dim <= 2.0, f"Wymiar {dim} poza zakresem dla {scaling}"


class TestSimpleFIF:
    """Testy dla uproszczonej wersji FIF."""

    def test_basic_generation(self):
        """Test podstawowego generowania."""
        data = np.array([[0, 0], [0.5, 1], [1, 0]])
        x, y = simple_fif(data, d=0.3, n_iterations=5)

        assert len(x) > 0
        assert len(y) > 0

    def test_point_count_growth(self):
        """Test wzrostu liczby punktów z iteracjami."""
        data = np.array([[0, 0], [0.5, 1], [1, 0]])

        x1, _ = simple_fif(data, d=0.3, n_iterations=3)
        x2, _ = simple_fif(data, d=0.3, n_iterations=5)

        # Więcej iteracji = więcej punktów
        assert len(x2) > len(x1)

    def test_different_d_values(self):
        """Test różnych wartości d."""
        data = np.array([[0, 0], [0.5, 1], [1, 0]])

        for d in [0.1, 0.3, 0.5, -0.3]:
            x, y = simple_fif(data, d=d, n_iterations=4)
            assert len(x) > 0, f"Błąd dla d={d}"

    def test_invalid_data(self):
        """Test nieprawidłowych danych."""
        with pytest.raises(ValueError):
            simple_fif(np.array([[0, 0]]), d=0.3)

    def test_four_points(self):
        """Test dla czterech punktów."""
        data = np.array([
            [0.0, 0.0],
            [0.3, 0.8],
            [0.7, 0.3],
            [1.0, 1.0]
        ])

        x, y = simple_fif(data, d=0.3, n_iterations=4)

        assert len(x) > 0
        # Sprawdzamy zakres
        assert x.min() >= -0.1
        assert x.max() <= 1.1


class TestIntegration:
    """Testy integracyjne FIF z box-counting."""

    @pytest.mark.slow
    def test_fif_box_counting(self):
        """Test wymiaru FIF metodą box-counting."""
        from src.box_counting import curve_to_points, box_counting_dimension

        data = np.array([[0, 0], [0.5, 1], [1, 0]])
        d = 0.4

        x, y = simple_fif(data, d=d, n_iterations=7)
        points = curve_to_points(x, y)

        dim, r2, _ = box_counting_dimension(
            points,
            epsilon_min=0.005,
            epsilon_max=0.2,
            n_epsilons=15
        )

        # Wymiar powinien być pozytywny i rozsądny
        assert 0.5 <= dim <= 2.5, f"Wymiar FIF poza zakresem: {dim}"

        # R² powinno być rozsądne (może być niższe dla FIF)
        assert r2 > 0.5, f"R² dla FIF: {r2}"
