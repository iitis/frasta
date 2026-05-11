"""Shared utilities for exported 2D and 3D colorbar layout.

This module centralizes tick generation, numeric formatting, font sizing, and
label layout so the 2D and 3D exporters use the same rules.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui


def get_colorbar_font_sizes(
    width: int,
    height: int,
    font_size: int | None = None,
) -> tuple[int, int]:
    """Return title and label font sizes for an exported colorbar.

    Args:
        width: Export image width in pixels.
        height: Export image height in pixels.
        font_size: Optional explicit label size in points.
    """
    if font_size is not None:
        label_size = int(np.clip(round(float(font_size)), 8, 24))
        title_size = int(np.clip(label_size + 1, 9, 26))
        return title_size, label_size

    scale_base = max(64, min(int(width), int(height)))
    title_size = int(np.clip(round(scale_base * 0.055), 10, 18))
    label_size = int(np.clip(round(scale_base * 0.05), 9, 16))
    return title_size, label_size


def build_colorbar_tick_layout(
    vmin: float,
    vmax: float,
    target_count: float = 6.0,
    minor_subdivisions: int = 3,
) -> dict[str, list[dict[str, float | bool]]]:
    """Build rounded major and minor ticks for a vertical colorbar."""
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return {"major": [], "minor": []}
    if vmax <= vmin:
        vmax = vmin + 1e-9

    major_step = select_colorbar_tick_step(vmin, vmax, target_count=target_count)
    major_ticks = build_ticks_for_step(vmin, vmax, major_step)
    minor_ticks = build_minor_ticks(vmin, vmax, major_step, subdivisions=minor_subdivisions)
    return {"major": major_ticks, "minor": minor_ticks}


def select_colorbar_tick_step(vmin: float, vmax: float, target_count: float = 6.0) -> float:
    """Choose a rounded major-tick step for exported colorbars."""
    extent = max(float(vmax) - float(vmin), 1e-9)
    raw_step = extent / max(float(target_count), 1.0)
    exponent = np.floor(np.log10(raw_step))
    base = 10.0 ** exponent
    normalized = raw_step / base
    if normalized <= 1.0:
        nice = 1.0
    elif normalized <= 2.0:
        nice = 2.0
    elif normalized <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * base)


def build_ticks_for_step(vmin: float, vmax: float, step: float) -> list[dict[str, float | bool]]:
    """Build major ticks for a given regular step and include zero if visible."""
    if step <= 0:
        return []
    tolerance = max(1e-9, abs(step) * 1e-6)
    start_index = int(np.ceil((vmin - tolerance) / step))
    end_index = int(np.floor((vmax + tolerance) / step))
    ticks: list[dict[str, float | bool]] = []
    for index in range(start_index, end_index + 1):
        value = float(index * step)
        if value < vmin - tolerance or value > vmax + tolerance:
            continue
        ticks.append(
            {
                "value": value,
                "fraction": (value - vmin) / max(vmax - vmin, 1e-9),
                "is_zero": abs(value) <= tolerance,
            }
        )

    if vmin <= 0.0 <= vmax and not any(bool(tick["is_zero"]) for tick in ticks):
        ticks.append(
            {
                "value": 0.0,
                "fraction": (0.0 - vmin) / max(vmax - vmin, 1e-9),
                "is_zero": True,
            }
        )

    if not ticks:
        midpoint = 0.5 * (vmin + vmax)
        ticks = [
            {"value": float(vmax), "fraction": 1.0, "is_zero": abs(vmax) <= tolerance},
            {"value": float(midpoint), "fraction": 0.5, "is_zero": abs(midpoint) <= tolerance},
            {"value": float(vmin), "fraction": 0.0, "is_zero": abs(vmin) <= tolerance},
        ]

    ticks.sort(key=lambda item: float(item["fraction"]))
    return ticks


def build_minor_ticks(vmin: float, vmax: float, major_step: float, subdivisions: int) -> list[dict[str, float]]:
    """Build minor ticks between neighboring major tick values."""
    if major_step <= 0 or subdivisions < 1:
        return []
    tolerance = max(1e-9, abs(major_step) * 1e-6)
    step = major_step / float(subdivisions + 1)
    start_index = int(np.ceil((vmin - tolerance) / step))
    end_index = int(np.floor((vmax + tolerance) / step))
    minor_ticks: list[dict[str, float]] = []
    for index in range(start_index, end_index + 1):
        value = float(index * step)
        if value < vmin - tolerance or value > vmax + tolerance:
            continue
        if abs((value / major_step) - round(value / major_step)) <= 1e-6:
            continue
        minor_ticks.append(
            {
                "value": value,
                "fraction": (value - vmin) / max(vmax - vmin, 1e-9),
            }
        )
    return minor_ticks


def format_colorbar_value(
    value: float,
    decimals: int | None = None,
    trim_trailing_zeros: bool = False,
) -> str:
    """Format one colorbar label value.

    Args:
        value: Numeric value to format.
        decimals: Fixed decimal count. ``None`` uses compact automatic mode.
        trim_trailing_zeros: Remove redundant trailing zeroes for fixed-decimal
            formatting when enabled.
    """
    if not np.isfinite(value):
        return "nan"
    if decimals is not None:
        decimals = max(0, int(decimals))
        formatted = f"{value:.{decimals}f}"
        if trim_trailing_zeros:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted
    magnitude = abs(float(value))
    if magnitude >= 1e4 or (0 < magnitude < 1e-3):
        return f"{value:.3e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_reference_label_specs(
    scale_vmin: float,
    scale_vmax: float,
    color_vmin: float,
    color_vmax: float,
    decimals: int | None = None,
) -> list[dict[str, float | str | bool]]:
    """Build right-side labels for data limits and threshold limits."""
    if not np.isfinite(scale_vmin) or not np.isfinite(scale_vmax) or scale_vmax <= scale_vmin:
        return []

    span = max(scale_vmax - scale_vmin, 1e-9)
    specs: list[dict[str, float | str | bool]] = [
        {
            "fraction": 0.0,
            "text": format_colorbar_value(scale_vmin, decimals=decimals),
            "bold": False,
        },
        {
            "fraction": 1.0,
            "text": format_colorbar_value(scale_vmax, decimals=decimals),
            "bold": False,
        },
    ]

    tolerance = max(1e-9, span * 1e-6)
    if abs(color_vmin - scale_vmin) > tolerance:
        specs.append(
            {
                "fraction": (float(color_vmin) - scale_vmin) / span,
                "text": format_colorbar_value(color_vmin, decimals=decimals),
                "bold": False,
            }
        )
    if abs(color_vmax - scale_vmax) > tolerance:
        specs.append(
            {
                "fraction": (float(color_vmax) - scale_vmin) / span,
                "text": format_colorbar_value(color_vmax, decimals=decimals),
                "bold": False,
            }
        )

    specs.sort(key=lambda item: float(item["fraction"]))
    return specs


def layout_reference_label_positions(
    bar_rect: QtCore.QRect,
    sorted_specs: list[dict[str, float | str | bool]],
    font_height: float,
    min_gap: float | None = None,
) -> list[float]:
    """Lay out right-side labels from top to bottom with a minimum gap."""
    if not sorted_specs:
        return []

    resolved_gap = max(float(font_height) * 0.9, 12.0) if min_gap is None else float(min_gap)
    top_limit = float(bar_rect.top()) + max(float(font_height) * 0.45, 8.0)
    bottom_limit = float(bar_rect.bottom()) - max(float(font_height) * 0.45, 8.0)
    target_positions = [
        float(bar_rect.bottom()) - float(spec["fraction"]) * bar_rect.height()
        for spec in sorted_specs
    ]

    placed_positions: list[float] = []
    for index, target_y in enumerate(target_positions):
        if index == 0:
            placed_y = max(top_limit, min(bottom_limit, target_y))
        else:
            placed_y = max(target_y, placed_positions[-1] + resolved_gap)
            placed_y = min(placed_y, bottom_limit)
        placed_positions.append(placed_y)

    for index in range(len(placed_positions) - 2, -1, -1):
        placed_positions[index] = min(placed_positions[index], placed_positions[index + 1] - resolved_gap)
        placed_positions[index] = max(top_limit, placed_positions[index])

    return placed_positions


def build_tick_label_rect(
    x_left: float,
    width: float,
    anchor_y: float,
    metrics: QtGui.QFontMetricsF,
    placement: str = "center",
) -> QtCore.QRectF:
    """Build a text rectangle anchored relative to a tick position."""
    text_height = max(float(metrics.height()), 1.0)
    if placement == "above":
        top = anchor_y - text_height * 0.88
    else:
        top = anchor_y - text_height * 0.5
    return QtCore.QRectF(float(x_left), float(top), float(width), float(text_height))
