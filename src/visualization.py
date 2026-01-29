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


def plot_fif_spectrum(
    data_points: np.ndarray = None,
    d_values: List[float] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Wizualizacja spektrum FIF od interpolacji liniowej (d=0) do quasi-wypełniania przestrzeni (d→1).

    Pokazuje jak zmienia się struktura FIF wraz ze wzrostem współczynnika skalowania d,
    demonstrując koncepcyjny związek z krzywymi wypełniającymi przestrzeń.

    Args:
        data_points: Punkty do interpolacji. Domyślnie trzy punkty tworzące trójkąt.
        d_values: Lista wartości d do pokazania. Domyślnie [0, 0.3, 0.5, 0.7, 0.9].
        save_path: Ścieżka do zapisu (opcjonalna)
        show: Czy wyświetlić wykres

    Returns:
        Obiekt Figure
    """
    from .fractal_interpolation import simple_fif, fif_dimension_theoretical

    setup_style()

    if data_points is None:
        data_points = np.array([[0, 0], [0.5, 1], [1, 0]])

    if d_values is None:
        d_values = [0.0, 0.3, 0.5, 0.7, 0.9]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, d in enumerate(d_values):
        if i >= len(axes) - 1:
            break

        ax = axes[i]

        if d == 0:
            # Interpolacja liniowa
            x = np.linspace(data_points[:, 0].min(), data_points[:, 0].max(), 1000)
            y = np.interp(x, data_points[:, 0], data_points[:, 1])
            title = "d = 0 (interpolacja liniowa)\nD = 1.0"
        else:
            x, y = simple_fif(data_points, d=d, n_iterations=7)
            scaling = np.array([d] * (len(data_points) - 1))
            dim_theo = fif_dimension_theoretical(scaling)
            title = f"d = {d}\nD (teoretyczny) = {dim_theo:.2f}"

        ax.plot(x, y, color=COLORS['primary'], linewidth=0.3, alpha=0.8)
        ax.scatter(data_points[:, 0], data_points[:, 1],
                   color=COLORS['quaternary'], s=80, zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Ostatni panel: porównanie z krzywą Peano (koncepcyjne)
    from .space_filling import hilbert_curve
    ax = axes[-1]
    x_h, y_h = hilbert_curve(5)
    ax.plot(x_h, y_h, color=COLORS['secondary'], linewidth=0.5)
    ax.set_title("Krzywa Hilberta\nD = 2.0 (granica)", fontsize=11)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.suptitle('Spektrum FIF: od interpolacji liniowej do wypełniania przestrzeni',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_weierstrass_interpolation(
    data_points: np.ndarray = None,
    a_values: List[float] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Wizualizacja interpolacji z perturbacją Weierstrassa.

    Pokazuje jak funkcja Weierstrassa może być użyta do dodania fraktalnej
    struktury do interpolacji liniowej, demonstrując związek między
    Weierstrassem a FIF.

    Args:
        data_points: Punkty do interpolacji
        a_values: Lista wartości parametru a Weierstrassa
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić wykres

    Returns:
        Obiekt Figure
    """
    from .fractal_interpolation import weierstrass_interpolation

    setup_style()

    if data_points is None:
        data_points = np.array([[0, 0.2], [0.3, 0.8], [0.6, 0.3], [1, 0.7]])

    if a_values is None:
        a_values = [0.3, 0.5, 0.7]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, a in zip(axes, a_values):
        x, y = weierstrass_interpolation(data_points, a=a, amplitude=0.15)

        # Interpolacja liniowa dla porównania
        x_lin = np.linspace(data_points[:, 0].min(), data_points[:, 0].max(), 1000)
        y_lin = np.interp(x_lin, data_points[:, 0], data_points[:, 1])

        ax.plot(x_lin, y_lin, '--', color='gray', linewidth=1, alpha=0.7,
                label='interpolacja liniowa')
        ax.plot(x, y, color=COLORS['primary'], linewidth=0.5, alpha=0.8,
                label=f'z perturbacją Weierstrassa')
        ax.scatter(data_points[:, 0], data_points[:, 1],
                   color=COLORS['quaternary'], s=80, zorder=5, label='punkty danych')
        ax.set_title(f'a = {a} (szorstkość: {"niska" if a < 0.5 else "wysoka" if a > 0.6 else "średnia"})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='best', fontsize=9)

    plt.suptitle('Interpolacja z perturbacją Weierstrassa: związek z FIF',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_fif_practical(
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Wizualizacja praktycznych zastosowań FIF.

    Pokazuje jak FIF może być używana do modelowania danych o nieregularnej
    strukturze: linia brzegowa, profil terenu, dane pomiarowe.

    Args:
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić wykres

    Returns:
        Obiekt Figure
    """
    from .fractal_interpolation import simple_fif

    setup_style()
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Symulacja linii brzegowej
    ax = axes[0]
    coastline_data = np.array([[0, 0.5], [0.2, 0.6], [0.4, 0.4], [0.6, 0.7], [0.8, 0.5], [1, 0.6]])
    x_lin = np.linspace(0, 1, 500)
    y_lin = np.interp(x_lin, coastline_data[:, 0], coastline_data[:, 1])
    x_fif, y_fif = simple_fif(coastline_data, d=0.4, n_iterations=6)

    ax.plot(x_lin, y_lin, '--', color='gray', linewidth=1.5, alpha=0.7, label='Interpolacja liniowa')
    ax.plot(x_fif, y_fif, color=COLORS['primary'], linewidth=0.5, label='FIF (d=0.4)')
    ax.scatter(coastline_data[:, 0], coastline_data[:, 1], color=COLORS['quaternary'], s=60, zorder=5)
    ax.set_title('Model linii brzegowej')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best', fontsize=9)

    # 2. Profil terenu
    ax = axes[1]
    terrain_data = np.array([[0, 0.3], [0.15, 0.5], [0.35, 0.8], [0.5, 0.6], [0.7, 0.9], [0.85, 0.4], [1, 0.5]])
    x_fif, y_fif = simple_fif(terrain_data, d=0.35, n_iterations=6)

    ax.fill_between(x_fif, 0, y_fif, alpha=0.3, color=COLORS['success'])
    ax.plot(x_fif, y_fif, color=COLORS['success'], linewidth=0.8)
    ax.scatter(terrain_data[:, 0], terrain_data[:, 1], color=COLORS['quaternary'], s=60, zorder=5)
    ax.set_title('Profil terenu')
    ax.set_xlabel('Pozycja')
    ax.set_ylabel('Wysokość')

    # 3. Porównanie różnych d dla tych samych danych
    ax = axes[2]
    sample_data = np.array([[0, 0], [0.3, 0.7], [0.6, 0.4], [1, 0.8]])
    d_compare = [0.2, 0.5, 0.7]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    for d, color in zip(d_compare, colors):
        x_fif, y_fif = simple_fif(sample_data, d=d, n_iterations=6)
        ax.plot(x_fif, y_fif, color=color, linewidth=0.5, alpha=0.8, label=f'd = {d}')

    ax.scatter(sample_data[:, 0], sample_data[:, 1], color=COLORS['quaternary'], s=80, zorder=5)
    ax.set_title('Wpływ parametru d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best', fontsize=9)

    plt.suptitle('Praktyczne zastosowania fraktalnej interpolacji',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_interpolation_comparison(
    data_points: np.ndarray = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Porównanie klasycznych metod interpolacji z FIF.

    Args:
        data_points: Punkty do interpolacji
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić wykres

    Returns:
        Obiekt Figure
    """
    from .fractal_interpolation import simple_fif
    from scipy.interpolate import CubicSpline

    setup_style()

    if data_points is None:
        data_points = np.array([[0, 0.3], [0.25, 0.7], [0.5, 0.4], [0.75, 0.8], [1, 0.5]])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_dense = np.linspace(data_points[:, 0].min(), data_points[:, 0].max(), 1000)

    # 1. Interpolacja liniowa
    ax = axes[0, 0]
    y_lin = np.interp(x_dense, data_points[:, 0], data_points[:, 1])
    ax.plot(x_dense, y_lin, color=COLORS['primary'], linewidth=1.5)
    ax.scatter(data_points[:, 0], data_points[:, 1], color=COLORS['quaternary'], s=80, zorder=5)
    ax.set_title('Interpolacja liniowa')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 2. Spline kubiczny
    ax = axes[0, 1]
    cs = CubicSpline(data_points[:, 0], data_points[:, 1])
    y_spline = cs(x_dense)
    ax.plot(x_dense, y_spline, color=COLORS['secondary'], linewidth=1.5)
    ax.scatter(data_points[:, 0], data_points[:, 1], color=COLORS['quaternary'], s=80, zorder=5)
    ax.set_title('Spline kubiczny')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 3. FIF (d=0.3)
    ax = axes[1, 0]
    x_fif, y_fif = simple_fif(data_points, d=0.3, n_iterations=6)
    ax.plot(x_fif, y_fif, color=COLORS['tertiary'], linewidth=0.5)
    ax.scatter(data_points[:, 0], data_points[:, 1], color=COLORS['quaternary'], s=80, zorder=5)
    ax.set_title('FIF (d = 0.3, umiarkowana szorstkość)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 4. FIF (d=0.6)
    ax = axes[1, 1]
    x_fif, y_fif = simple_fif(data_points, d=0.6, n_iterations=6)
    ax.plot(x_fif, y_fif, color=COLORS['success'], linewidth=0.3)
    ax.scatter(data_points[:, 0], data_points[:, 1], color=COLORS['quaternary'], s=80, zorder=5)
    ax.set_title('FIF (d = 0.6, wysoka szorstkość)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.suptitle('Porównanie metod interpolacji',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_hilbert_interpolation(
    data_points: np.ndarray = None,
    hilbert_order: int = 6,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Wizualizacja krzywej Hilberta jako formy interpolacji parametrycznej.

    Pokazuje jak krzywa Hilberta "odwiedza" punkty danych w określonej kolejności,
    demonstrując związek między krzywymi wypełniającymi przestrzeń a interpolacją.

    Args:
        data_points: Punkty danych 2D do "interpolacji". Domyślnie losowe punkty.
        hilbert_order: Rząd krzywej Hilberta (wyższy = dokładniejsza aproksymacja)
        save_path: Ścieżka do zapisu
        show: Czy wyświetlić wykres

    Returns:
        Obiekt Figure
    """
    from .space_filling import hilbert_curve
    from .fractal_interpolation import simple_fif

    setup_style()

    if data_points is None:
        # Punkty danych rozłożone w kwadracie jednostkowym
        np.random.seed(42)
        data_points = np.array([
            [0.1, 0.2], [0.3, 0.8], [0.5, 0.4],
            [0.7, 0.9], [0.9, 0.3], [0.2, 0.6],
            [0.8, 0.7], [0.4, 0.1]
        ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Krzywa Hilberta z punktami danych
    ax = axes[0, 0]
    x_h, y_h = hilbert_curve(hilbert_order)
    ax.plot(x_h, y_h, color=COLORS['primary'], linewidth=0.5, alpha=0.7, label='Krzywa Hilberta')
    ax.scatter(data_points[:, 0], data_points[:, 1],
               color=COLORS['quaternary'], s=120, zorder=5, edgecolors='black', linewidth=1.5)

    # Znajdź najbliższy punkt na krzywej dla każdego punktu danych
    hilbert_points = np.column_stack([x_h, y_h])
    point_order = []
    for i, dp in enumerate(data_points):
        distances = np.sqrt(np.sum((hilbert_points - dp)**2, axis=1))
        nearest_idx = np.argmin(distances)
        point_order.append((nearest_idx, i))
        # Linia łącząca punkt z krzywą
        ax.plot([dp[0], x_h[nearest_idx]], [dp[1], y_h[nearest_idx]],
                'k--', linewidth=1, alpha=0.5)

    # Sortuj punkty według kolejności na krzywej Hilberta
    point_order.sort(key=lambda x: x[0])
    visit_order = [p[1] for p in point_order]

    # Numeruj punkty według kolejności odwiedzin
    for rank, point_idx in enumerate(visit_order):
        ax.annotate(str(rank + 1), data_points[point_idx],
                    fontsize=10, fontweight='bold', ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')

    ax.set_title(f'Krzywa Hilberta (rząd {hilbert_order}) "odwiedza" punkty danych\n'
                 f'Numery pokazują kolejność parametryczną', fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Panel 2: Ścieżka przez punkty w kolejności Hilberta
    ax = axes[0, 1]
    ordered_points = data_points[visit_order]
    ax.plot(ordered_points[:, 0], ordered_points[:, 1],
            color=COLORS['secondary'], linewidth=2, marker='o', markersize=10,
            markerfacecolor=COLORS['quaternary'], markeredgecolor='black')

    for rank, point_idx in enumerate(visit_order):
        ax.annotate(str(rank + 1), data_points[point_idx],
                    fontsize=10, fontweight='bold', ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')

    ax.set_title('Interpolacja parametryczna: połączenie punktów\n'
                 'w kolejności wyznaczonej przez krzywą Hilberta', fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Panel 3: FIF z wysokim d (quasi-wypełniające)
    ax = axes[1, 0]
    fif_points = np.array([[0, 0], [0.5, 1], [1, 0]])
    x_fif, y_fif = simple_fif(fif_points, d=0.85, n_iterations=8)
    ax.plot(x_fif, y_fif, color=COLORS['tertiary'], linewidth=0.2, alpha=0.8)
    ax.scatter(fif_points[:, 0], fif_points[:, 1],
               color=COLORS['quaternary'], s=100, zorder=5, edgecolors='black')
    ax.set_title('FIF z d = 0.85\nZachowanie quasi-wypełniające przestrzeń', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Panel 4: Porównanie - FIF vs Hilbert
    ax = axes[1, 1]
    # Hilbert przeskalowany do podobnego rozmiaru
    x_h_scaled = x_h
    y_h_scaled = y_h * 0.8 + 0.1  # Skalowanie do zakresu podobnego do FIF
    ax.plot(x_h_scaled, y_h_scaled, color=COLORS['primary'], linewidth=0.3,
            alpha=0.5, label='Krzywa Hilberta (D=2)')

    # FIF z bardzo wysokim d
    x_fif_high, y_fif_high = simple_fif(fif_points, d=0.9, n_iterations=8)
    ax.plot(x_fif_high, y_fif_high, color=COLORS['tertiary'], linewidth=0.3,
            alpha=0.8, label='FIF d=0.9 (D≈1.95)')
    ax.scatter(fif_points[:, 0], fif_points[:, 1],
               color=COLORS['quaternary'], s=100, zorder=5, edgecolors='black')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Porównanie: FIF przy d→1 zbliża się\ndo zachowania krzywej wypełniającej', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.suptitle('Krzywe Peano/Hilberta jako forma interpolacji parametrycznej',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    # Test wizualizacji
    create_summary_figure(save_path='../output/figures/summary.png', show=True)
