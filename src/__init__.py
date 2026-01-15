"""
Fractal Interpolation Functions Library

Biblioteka do analizy funkcji interpolacji fraktalnych:
- Funkcja Weierstrassa
- Krzywe wypełniające przestrzeń (Peano, Hilbert)
- Algorytm box-counting dla wymiaru Minkowskiego
- Fraktalne funkcje interpolacyjne (FIF)
"""

__version__ = "0.1.0"
__author__ = "Zespół Projektowy"

from .weierstrass import weierstrass_function, theoretical_dimension
from .space_filling import hilbert_curve, peano_curve
from .box_counting import box_counting_dimension, box_count
from .fractal_interpolation import fractal_interpolation_function

__all__ = [
    "weierstrass_function",
    "theoretical_dimension",
    "hilbert_curve",
    "peano_curve",
    "box_counting_dimension",
    "box_count",
    "fractal_interpolation_function",
]
