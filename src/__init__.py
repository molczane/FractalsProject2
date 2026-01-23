"""
Fractal Interpolation Functions Library

Biblioteka do analizy funkcji interpolacji fraktalnych:
- Funkcja Weierstrassa
- Krzywe wypełniające przestrzeń (Peano, Hilbert)
- Algorytm box-counting dla wymiaru Minkowskiego
- Fraktalne funkcje interpolacyjne (FIF)

Użycie:
    from src.weierstrass import weierstrass_function
    from src.space_filling import hilbert_curve
    from src.box_counting import box_counting_dimension
"""

__version__ = "0.1.0"
__author__ = "Rafał Filarecki, Ernest Mołczan"

__all__ = [
    "weierstrass",
    "space_filling",
    "box_counting",
    "fractal_interpolation",
    "visualization",
    "utils",
]
