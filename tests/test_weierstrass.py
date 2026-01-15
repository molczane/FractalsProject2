"""
Testy jednostkowe dla modułu weierstrass.py
"""

import pytest
import numpy as np
from src.weierstrass import (
    weierstrass_function,
    theoretical_dimension,
    weierstrass_derivative_approx
)


class TestWeierstrassFunction:
    """Testy dla funkcji Weierstrassa."""

    def test_basic_computation(self):
        """Test podstawowych obliczeń."""
        x = np.array([0.0, 0.5, 1.0])
        result = weierstrass_function(x, a=0.5, b=3, n_terms=10)

        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert not np.any(np.isnan(result))

    def test_single_value(self):
        """Test dla pojedynczej wartości."""
        result = weierstrass_function(0.5, a=0.5, b=3, n_terms=10)
        assert isinstance(result, np.ndarray)

    def test_manual_calculation(self):
        """Test zgodności z ręcznymi obliczeniami dla n_terms=3."""
        x = 0.0
        a, b = 0.5, 3

        # W(0) = Σ a^n * cos(b^n * π * 0) = Σ a^n * 1 = 1 + 0.5 + 0.25 = 1.75
        expected = 1 + 0.5 + 0.25
        result = weierstrass_function(x, a=a, b=b, n_terms=3)

        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_invalid_a_parameter(self):
        """Test walidacji parametru a."""
        with pytest.raises(ValueError, match="Parametr 'a' musi być w przedziale"):
            weierstrass_function([0, 1], a=1.5, b=3)

        with pytest.raises(ValueError, match="Parametr 'a' musi być w przedziale"):
            weierstrass_function([0, 1], a=0, b=3)

        with pytest.raises(ValueError, match="Parametr 'a' musi być w przedziale"):
            weierstrass_function([0, 1], a=-0.5, b=3)

    def test_invalid_b_parameter(self):
        """Test walidacji parametru b."""
        with pytest.raises(ValueError, match="Parametr 'b' musi być większy od 1"):
            weierstrass_function([0, 1], a=0.5, b=0.5)

        with pytest.raises(ValueError, match="Parametr 'b' musi być większy od 1"):
            weierstrass_function([0, 1], a=0.5, b=1)

    def test_invalid_n_terms(self):
        """Test walidacji liczby wyrazów."""
        with pytest.raises(ValueError, match="Liczba wyrazów musi być >= 1"):
            weierstrass_function([0, 1], a=0.5, b=3, n_terms=0)

    def test_convergence(self):
        """Test zbieżności szeregu przy zwiększaniu n_terms."""
        x = np.array([0.5])

        result_10 = weierstrass_function(x, n_terms=10)
        result_50 = weierstrass_function(x, n_terms=50)
        result_100 = weierstrass_function(x, n_terms=100)

        # Różnice powinny maleć
        diff_10_50 = np.abs(result_10 - result_50)
        diff_50_100 = np.abs(result_50 - result_100)

        assert diff_50_100 < diff_10_50

    def test_output_range(self):
        """Test zakresu wartości wyjściowych."""
        x = np.linspace(0, 10, 1000)
        y = weierstrass_function(x, a=0.5, b=3, n_terms=50)

        # Suma szeregu geometrycznego: |W(x)| <= Σ a^n = 1/(1-a) = 2
        max_bound = 1 / (1 - 0.5)
        assert np.all(np.abs(y) <= max_bound + 0.01)


class TestTheoreticalDimension:
    """Testy dla funkcji theoretical_dimension."""

    def test_known_values(self):
        """Test dla znanych wartości."""
        # Dla a=0.5, b=3: D = 2 + log(0.5)/log(3) ≈ 1.369
        dim = theoretical_dimension(0.5, 3)
        expected = 2 + np.log(0.5) / np.log(3)

        np.testing.assert_almost_equal(dim, expected, decimal=10)

    def test_dimension_range(self):
        """Test zakresu wymiaru (1 < D < 2)."""
        test_cases = [
            (0.5, 3),
            (0.3, 5),
            (0.7, 7),
            (0.9, 11),
        ]

        for a, b in test_cases:
            dim = theoretical_dimension(a, b)
            assert 1 < dim < 2, f"Wymiar {dim} poza zakresem dla a={a}, b={b}"

    def test_boundary_case(self):
        """Test przypadku granicznego a*b = 1."""
        # Gdy a*b <= 1, funkcja jest różniczkowalna i wymiar = 1
        dim = theoretical_dimension(0.3, 3)  # a*b = 0.9 < 1
        assert dim == 1.0

    def test_invalid_parameters(self):
        """Test walidacji parametrów."""
        with pytest.raises(ValueError):
            theoretical_dimension(1.5, 3)

        with pytest.raises(ValueError):
            theoretical_dimension(0.5, 0.5)


class TestWeierstrassDerivativeApprox:
    """Testy dla przybliżenia pochodnej."""

    def test_output_shape(self):
        """Test kształtu wyjścia."""
        x = np.linspace(0, 1, 100)
        result = weierstrass_derivative_approx(x, a=0.5, b=3)

        assert result.shape == x.shape

    def test_non_convergence(self):
        """Test niezbieżności pochodnej dla a*b > 1."""
        x = np.array([0.5])

        # Dla różnych h, wartości powinny być różne (nie zbiegać)
        deriv_h1 = weierstrass_derivative_approx(x, h=1e-4)[0]
        deriv_h2 = weierstrass_derivative_approx(x, h=1e-6)[0]
        deriv_h3 = weierstrass_derivative_approx(x, h=1e-8)[0]

        # Sprawdzamy, że wartości się znacznie różnią
        diffs = [abs(deriv_h1 - deriv_h2), abs(deriv_h2 - deriv_h3)]

        # Dla funkcji nieróżniczkowalnej różnice nie powinny maleć systematycznie
        # Ten test jest orientacyjny
        assert max(diffs) > 0.1 or min(diffs) > 0.001
