"""
Moduł wizualizacji dla projektu fraktali.

Zawiera funkcje do tworzenia wysokiej jakości wykresów
gotowych do publikacji w raporcie.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List, Tuple, Dict
import os

# Konfiguracja stylu wykresów
STYLE_CONFIG = {
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
}

# Paleta kolorów
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'success': '#3A7D44',
}


def setup_style():
    """Konfiguruje globalny styl matplotlib."""
    plt.rcParams.update(STYLE_CONFIG)


def plot_weierstrass(
    a: float = 0.5,
    b: float = 3.0,
    x_range: Tuple[float, float] = (0, 2),
    n_points: int = 5000,
    n_terms: int = 50,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Wizualizacja funkcji Weierstrassa.

    Args:
        a: Współczynnik tłumienia
        b: Współczynnik częstotliwości
        x_range: Przedział x
        n_points: Liczba punktów
        n_terms: Liczba wyrazów szeregu
        save_path: Ścieżka do zapisu (opcjonalna)
        show: Czy wyświetlić wykres

    Returns:
        Obiekt Figure
    """
    from .weierstrass import weierstrass_function, theoretical_dimension

    setup_style()

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = weierstrass_function(x, a, b, n_terms)
    dim = theoretical_dimension(a, b)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, color=COLORS['primary'], linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('W(x)')
    ax.set_title(f'Funkcja Weierstrassa (a={a}, b={b}, D={dim:.3f})')

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_weierstrass_zoom(
    a: float = 0.5,
    b: float = 3.0,
    center: float = 1.0,
    zoom_levels: List[float] = None,
    n_points: int = 5000,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Wizualizacja samopodobieństwa funkcji Weierstrassa przez przybliżanie.

    Args:
        a: Współczynnik tłumienia
        b: Współczynnik częstotliwości
        center: Punkt centralny przybliżenia
        zoom_levels: Lista poziomów przybliżenia (szerokości okna)
        n_points: Liczba punktów
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić

    Returns:
        Obiekt Figure
    """
    from .weierstrass import weierstrass_function

    setup_style()

    if zoom_levels is None:
        zoom_levels = [2.0, 0.5, 0.1, 0.02]

    n_levels = len(zoom_levels)
    fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4))

    if n_levels == 1:
        axes = [axes]

    for ax, width in zip(axes, zoom_levels):
        x_min, x_max = center - width / 2, center + width / 2
        x = np.linspace(x_min, x_max, n_points)
        y = weierstrass_function(x, a, b)

        ax.plot(x, y, color=COLORS['primary'], linewidth=0.5)
        ax.set_title(f'Szerokość = {width}')
        ax.set_xlabel('x')
        ax.set_ylabel('W(x)')

    fig.suptitle(f'Samopodobieństwo funkcji Weierstrassa (a={a}, b={b})', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_space_filling_curves(
    curve_type: str = 'hilbert',
    orders: List[int] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Wizualizacja krzywych wypełniających przestrzeń dla różnych rzędów.

    Args:
        curve_type: Typ krzywej ('hilbert', 'peano', 'dragon')
        orders: Lista rzędów do wyświetlenia
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić

    Returns:
        Obiekt Figure
    """
    from .space_filling import hilbert_curve, peano_curve, dragon_curve

    setup_style()

    curve_functions = {
        'hilbert': hilbert_curve,
        'peano': peano_curve,
        'dragon': dragon_curve
    }

    if curve_type not in curve_functions:
        raise ValueError(f"Nieznany typ krzywej: {curve_type}")

    if orders is None:
        orders = [1, 2, 3, 4] if curve_type != 'dragon' else [5, 8, 10, 12]

    func = curve_functions[curve_type]
    n_orders = len(orders)

    fig, axes = plt.subplots(1, n_orders, figsize=(4 * n_orders, 4))

    if n_orders == 1:
        axes = [axes]

    for ax, order in zip(axes, orders):
        x, y = func(order)
        ax.plot(x, y, color=COLORS['secondary'], linewidth=0.5)
        ax.set_title(f'Rząd {order} ({len(x)} pkt)')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    curve_names = {
        'hilbert': 'Hilberta',
        'peano': 'Peano',
        'dragon': 'smoka'
    }
    fig.suptitle(f'Krzywa {curve_names[curve_type]}', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_box_counting_analysis(
    points: np.ndarray,
    title: str = "Analiza box-counting",
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Wizualizacja analizy box-counting.

    Args:
        points: Punkty do analizy, shape (N, 2)
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić

    Returns:
        Obiekt Figure
    """
    from .box_counting import box_counting_dimension

    setup_style()

    dim, r2, details = box_counting_dimension(points)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lewy wykres - punkty
    axes[0].scatter(points[:, 0], points[:, 1], s=0.1, alpha=0.5, color=COLORS['primary'])
    axes[0].set_title('Analizowany zbiór punktów')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')

    # Prawy wykres - regresja
    axes[1].scatter(details['log_inv_eps'], details['log_counts'],
                    color=COLORS['primary'], s=50, label='Dane')
    axes[1].plot(details['log_inv_eps'],
                 details['slope'] * details['log_inv_eps'] + details['intercept'],
                 color=COLORS['secondary'], linewidth=2,
                 label=f'Regresja: D = {dim:.4f}')
    axes[1].set_xlabel('log(1/ε)')
    axes[1].set_ylabel('log(N(ε))')
    axes[1].set_title(f'Box-counting: D = {dim:.4f}, R² = {r2:.4f}')
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_dimension_comparison(
    param_values: np.ndarray,
    numerical_dims: np.ndarray,
    theoretical_dims: np.ndarray,
    param_name: str,
    title: str = "Porównanie wymiarów",
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Porównanie wymiarów teoretycznych i numerycznych.

    Args:
        param_values: Wartości parametru
        numerical_dims: Wymiary numeryczne
        theoretical_dims: Wymiary teoretyczne
        param_name: Nazwa parametru
        title: Tytuł wykresu
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić

    Returns:
        Obiekt Figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(param_values, numerical_dims, 'o-',
            color=COLORS['primary'], linewidth=2, markersize=8,
            label='Wymiar numeryczny (box-counting)')
    ax.plot(param_values, theoretical_dims, 's--',
            color=COLORS['secondary'], linewidth=2, markersize=8,
            label='Wymiar teoretyczny')

    ax.set_xlabel(param_name)
    ax.set_ylabel('Wymiar fraktalny D')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(1, 2)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_fif_comparison(
    data_points: np.ndarray,
    scaling_factors_list: List[float],
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Porównanie FIF dla różnych współczynników skalowania.

    Args:
        data_points: Punkty do interpolacji
        scaling_factors_list: Lista współczynników do porównania
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić

    Returns:
        Obiekt Figure
    """
    from .fractal_interpolation import simple_fif

    setup_style()

    n_plots = len(scaling_factors_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

    if n_plots == 1:
        axes = [axes]

    for ax, d in zip(axes, scaling_factors_list):
        if d == 0:
            x = np.linspace(data_points[:, 0].min(), data_points[:, 0].max(), 1000)
            y = np.interp(x, data_points[:, 0], data_points[:, 1])
        else:
            x, y = simple_fif(data_points, d=d, n_iterations=6)

        ax.plot(x, y, color=COLORS['primary'], linewidth=0.5, alpha=0.7)
        ax.scatter(data_points[:, 0], data_points[:, 1],
                   color=COLORS['secondary'], s=50, zorder=5)
        ax.set_title(f'd = {d}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle('Fraktalna funkcja interpolacyjna (FIF)', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def create_summary_figure(
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Tworzy podsumowującą figurę ze wszystkimi typami fraktali.

    Args:
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić

    Returns:
        Obiekt Figure
    """
    from .weierstrass import weierstrass_function
    from .space_filling import hilbert_curve
    from .fractal_interpolation import simple_fif

    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Funkcja Weierstrassa
    x = np.linspace(0, 2, 5000)
    y = weierstrass_function(x, a=0.5, b=3)
    axes[0, 0].plot(x, y, color=COLORS['primary'], linewidth=0.5)
    axes[0, 0].set_title('Funkcja Weierstrassa (a=0.5, b=3)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('W(x)')

    # 2. Krzywa Hilberta
    x, y = hilbert_curve(5)
    axes[0, 1].plot(x, y, color=COLORS['secondary'], linewidth=0.5)
    axes[0, 1].set_title('Krzywa Hilberta (rząd 5)')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # 3. FIF
    data = np.array([[0, 0], [0.3, 0.8], [0.7, 0.3], [1, 1]])
    x, y = simple_fif(data, d=0.4, n_iterations=6)
    axes[1, 0].plot(x, y, color=COLORS['tertiary'], linewidth=0.5, alpha=0.7)
    axes[1, 0].scatter(data[:, 0], data[:, 1], color=COLORS['quaternary'], s=50, zorder=5)
    axes[1, 0].set_title('Fraktalna interpolacja (d=0.4)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')

    # 4. Placeholder dla box-counting lub innego wykresu
    from .box_counting import function_to_points, box_counting_dimension
    points = function_to_points(weierstrass_function, (0, 2), 10000, a=0.5, b=3)
    dim, r2, details = box_counting_dimension(points)

    axes[1, 1].scatter(details['log_inv_eps'], details['log_counts'],
                       color=COLORS['primary'], s=50)
    axes[1, 1].plot(details['log_inv_eps'],
                    details['slope'] * details['log_inv_eps'] + details['intercept'],
                    color=COLORS['secondary'], linewidth=2)
    axes[1, 1].set_xlabel('log(1/ε)')
    axes[1, 1].set_ylabel('log(N(ε))')
    axes[1, 1].set_title(f'Box-counting: D = {dim:.3f}')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    # Test wizualizacji
    create_summary_figure(save_path='../output/figures/summary.png', show=True)
