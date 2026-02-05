import sys
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from functools import wraps
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import logging
logger = logging.getLogger(__name__)


def resource_path(relative_path):
    """Zwraca absolutną ścieżkę do zasobu (działa i w exe, i w .py)"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f">>> {func.__name__}() took {end - start:.4f} seconds")
        return result
    return wrapper


def compute_offset_global(reference, target):
    """
    Oblicza średni offset tam, gdzie obie siatki mają dane (ignoruje NaN).
    """
    mask = ~np.isnan(reference) & ~np.isnan(target)
    diff = reference - target
    masked_diff = diff[mask]

    if masked_diff.size == 0:
        raise ValueError("Lack of common valid data across the grid")

    offset = np.mean(masked_diff)
    return offset


def compute_offset_in_center(reference, target, window_size=100):
    # Rozmiary obrazów
    rows, cols = reference.shape
    # Środek
    center_row = rows // 2
    center_col = cols // 2
    half = window_size // 2
    # Wytnij okno centralne
    ref_central = reference[center_row-half:center_row+half, center_col-half:center_col+half]
    target_central = target[center_row-half:center_row+half, center_col-half:center_col+half]
    # Maska: tylko tam, gdzie oba są nie NaN
    mask = ~np.isnan(ref_central) & ~np.isnan(target_central)
    diff = ref_central - target_central
    masked_diff = diff[mask]
    if masked_diff.size == 0:
        raise ValueError("No valid data in the central window")
    offset = np.mean(masked_diff)
    # print(f'Offset w centralnym obszarze {window_size}x{window_size}: {offset:.2f}')
    return offset

def remove_relative_offset(reference, target, mask):
    #offset = compute_offset_in_center(reference, target, window_size=1000)
    offset = compute_offset_global(reference, target)
    return target + offset

# def remove_relative_offset(reference, target, mask):
#     #mask = ~np.isnan(reference) & ~np.isnan(target)
#     difference = reference - target
#     masked_diff = difference[mask]
#     if masked_diff.size == 0:
#         raise ValueError("Brak ważnych danych do obliczenia offsetu")
#     offset = np.mean(masked_diff)
#     print('Wyznaczony offset:', offset)
#     return target + offset

def remove_relative_tilt(reference, target, mask):
    difference = reference - target
    rows, cols = difference.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    XX = X[mask].flatten()
    YY = Y[mask].flatten()
    ZZ = difference[mask].flatten()
    valid_mask = ~np.isnan(ZZ)
    XX, YY, ZZ = XX[valid_mask], YY[valid_mask], ZZ[valid_mask]
    if len(ZZ) == 0:
        raise ValueError("No valid data for regression - all points contained NaN")
    features = np.vstack((XX, YY)).T
    model = LinearRegression().fit(features, ZZ)
    tilt_plane = model.predict(np.vstack((X.flatten(), Y.flatten())).T).reshape(difference.shape)
    return target + tilt_plane


def fill_holes(grid, mask=None):
    if grid is None:
        return None

    grid = grid.copy()
    tst = np.isnan(grid)

    if mask is not None:
        # interpoluj tylko tam, gdzie maska jest True
        tst = tst & mask  # tylko dziury w zaznaczonym obszarze

    if not np.any(tst):
        return grid  # nic do wypełniania

    grid_x, grid_y = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))

    interp_points = np.column_stack((grid_x[tst], grid_y[tst]))

    valid = ~np.isnan(grid)
    interp_values = griddata(
        (grid_x[valid], grid_y[valid]),
        grid[valid],
        interp_points,
        method='nearest'
    )

    grid[tst] = interp_values
    return grid

def nan_aware_gaussian(grid, sigma, mask=None):
    """
    Filtr Gaussa, który ignoruje NaN-y i opcjonalnie ogranicza się do maski.
    Zwraca wynik w pełnym rozmiarze siatki.
    """
    if grid is None:
        return None

    nan_mask = np.isnan(grid)
    filled = np.where(nan_mask, 0, grid)
    weights = (~nan_mask).astype(float)

    # jeśli maska istnieje – przytnij do maski
    if mask is not None:
        filled = np.where(mask, filled, 0.0)
        weights = np.where(mask, weights, 0.0)

    smoothed = gaussian_filter(filled, sigma=sigma)
    weight_sum = gaussian_filter(weights, sigma=sigma)

    with np.errstate(invalid='ignore', divide='ignore'):
        result = smoothed / weight_sum
        result[weight_sum == 0] = np.nan

    return result

def remove_outliers(original_grid, smoothed_grid, threshold, mask=None):
    """
    Zamienia outliery w original_grid na wartości z siatki wygładzonej,
    jeśli różnica przekracza próg (threshold). Opcjonalnie ogranicza do maski.
    """
    diff = np.abs(original_grid - smoothed_grid)
    mask_outlier = diff > threshold

    if mask is not None:
        mask_outlier = mask_outlier & mask

    cleaned = original_grid.copy()
    cleaned[mask_outlier] = smoothed_grid[mask_outlier]  # lub +200 jeśli to był tylko test
    return cleaned

