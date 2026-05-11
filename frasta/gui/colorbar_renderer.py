"""Shared renderer for exported 2D and 3D colorbar images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5 import QtCore, QtGui

from .colorbar_common import (
    build_colorbar_tick_layout,
    build_reference_label_specs,
    build_tick_label_rect,
    format_colorbar_value,
    get_colorbar_font_sizes,
    layout_reference_label_positions,
)


@dataclass(slots=True)
class ColorbarRenderConfig:
    """Parameters needed to render one standalone colorbar image."""

    width: int
    height: int
    title_text: str
    gradient_stops: list[tuple[float, QtGui.QColor]]
    color_vmin: float
    color_vmax: float
    scale_vmin: float
    scale_vmax: float
    hist_values: np.ndarray
    transparent_background: bool = True
    include_histogram: bool = True
    decimals: int | None = None
    font_size: int | None = None
    title_alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter


class ExportColorbarRenderer:
    """Render a publication-oriented vertical colorbar with optional histogram."""

    def __init__(self, config: ColorbarRenderConfig):
        self.config = config

    def render(self) -> QtGui.QImage:
        """Render the configured colorbar into a standalone ARGB image."""
        width = max(32, int(self.config.width))
        height = max(64, int(self.config.height))
        image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32_Premultiplied)
        background = QtGui.QColor(0, 0, 0, 0) if self.config.transparent_background else QtGui.QColor(255, 255, 255, 255)
        image.fill(background)

        painter = QtGui.QPainter(image)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

            text_color = QtGui.QColor(255, 255, 255) if self.config.transparent_background else QtGui.QColor(20, 20, 20)
            border_color = QtGui.QColor(255, 255, 255, 180) if self.config.transparent_background else QtGui.QColor(40, 40, 40)
            hist_color = QtGui.QColor(180, 180, 180, 170)
            title_font_size, label_font_size = get_colorbar_font_sizes(width, height, font_size=self.config.font_size)

            title_font = QtGui.QFont()
            title_font.setBold(True)
            title_font.setPointSize(title_font_size)
            painter.setFont(title_font)
            painter.setPen(text_color)
            title_rect = QtCore.QRectF(12, 8, width - 24, 28)
            painter.drawText(title_rect, self.config.title_alignment, self.config.title_text)

            label_font = QtGui.QFont()
            label_font.setPointSize(label_font_size)
            label_metrics = QtGui.QFontMetricsF(label_font)

            bar_top = 48
            bar_bottom = max(bar_top + 40, height - 24)
            bar_height = bar_bottom - bar_top
            left_tick_label_width = max(44, min(96, width // 3))
            bar_width = max(28, min(40, width // 5))
            bar_left = 12 + left_tick_label_width
            bar_rect = QtCore.QRect(bar_left, bar_top, bar_width, bar_height)

            gradient = self._build_gradient(bar_rect)
            painter.fillRect(bar_rect, gradient)
            painter.setPen(QtGui.QPen(border_color, 1.0))
            painter.drawRect(bar_rect)

            if self.config.include_histogram and self.config.hist_values.size > 0:
                hist_left = bar_rect.right() + 4
                hist_width = max(20, min(60, width // 4))
                hist_rect = QtCore.QRect(hist_left, bar_top, hist_width, bar_height)
                self._draw_histogram(painter, hist_rect, hist_color)

            self._draw_ticks(
                painter=painter,
                bar_rect=bar_rect,
                left_label_left=12,
                left_label_width=left_tick_label_width,
                right_label_left=bar_rect.right() + 12,
                image_width=width,
                text_color=text_color,
                tick_color=border_color,
                label_font=label_font,
                label_metrics=label_metrics,
            )
        finally:
            painter.end()

        return image

    def _build_gradient(self, bar_rect: QtCore.QRect) -> QtGui.QLinearGradient:
        """Build a vertical gradient with plateau colors outside the color range."""
        gradient = QtGui.QLinearGradient(
            float(bar_rect.left()),
            float(bar_rect.bottom()),
            float(bar_rect.left()),
            float(bar_rect.top()),
        )
        if not self.config.gradient_stops:
            gradient.setColorAt(0.0, QtGui.QColor(0, 0, 0))
            gradient.setColorAt(1.0, QtGui.QColor(255, 255, 255))
            return gradient

        span = max(self.config.scale_vmax - self.config.scale_vmin, 1e-9)
        lower_fraction = float(np.clip((self.config.color_vmin - self.config.scale_vmin) / span, 0.0, 1.0))
        upper_fraction = float(np.clip((self.config.color_vmax - self.config.scale_vmin) / span, 0.0, 1.0))
        first_color = self.config.gradient_stops[0][1]
        last_color = self.config.gradient_stops[-1][1]

        gradient.setColorAt(0.0, first_color)
        if lower_fraction > 0.0:
            gradient.setColorAt(lower_fraction, first_color)

        active_span = max(upper_fraction - lower_fraction, 1e-9)
        for position, color in self.config.gradient_stops:
            mapped_position = lower_fraction + float(position) * active_span
            gradient.setColorAt(float(np.clip(mapped_position, 0.0, 1.0)), color)

        if upper_fraction < 1.0:
            gradient.setColorAt(upper_fraction, last_color)
        gradient.setColorAt(1.0, last_color)
        return gradient

    def _draw_histogram(self, painter: QtGui.QPainter, hist_rect: QtCore.QRect, color: QtGui.QColor) -> None:
        """Draw a normalized histogram beside the colorbar."""
        values = self.config.hist_values
        vmin = self.config.scale_vmin
        vmax = self.config.scale_vmax
        if values.size < 1 or vmax <= vmin:
            return

        counts, edges = np.histogram(values, bins=min(128, max(16, hist_rect.height() // 4)), range=(vmin, vmax))
        if counts.size < 1 or np.max(counts) <= 0:
            return

        counts = counts.astype(np.float32)
        counts /= float(np.max(counts))
        if counts.size >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
            kernel /= np.sum(kernel)
            counts = np.convolve(counts, kernel, mode="same")
            counts /= max(float(np.max(counts)), 1e-9)

        path = QtGui.QPainterPath()
        path.moveTo(hist_rect.left(), hist_rect.bottom())
        for idx, count in enumerate(counts):
            y0 = hist_rect.bottom() - (edges[idx] - vmin) / (vmax - vmin) * hist_rect.height()
            y1 = hist_rect.bottom() - (edges[idx + 1] - vmin) / (vmax - vmin) * hist_rect.height()
            x = hist_rect.left() + count * hist_rect.width()
            if idx == 0:
                path.lineTo(x, y0)
            path.lineTo(x, y1)
        path.lineTo(hist_rect.left(), hist_rect.top())
        path.closeSubpath()
        painter.fillPath(path, QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(color.darker(130), 1.0))
        painter.drawPath(path)

    def _draw_ticks(
        self,
        painter: QtGui.QPainter,
        bar_rect: QtCore.QRect,
        left_label_left: int,
        left_label_width: int,
        right_label_left: int,
        image_width: int,
        text_color: QtGui.QColor,
        tick_color: QtGui.QColor,
        label_font: QtGui.QFont,
        label_metrics: QtGui.QFontMetricsF,
    ) -> None:
        """Draw left-side regular ticks and right-side reference labels."""
        tick_layout = build_colorbar_tick_layout(
            self.config.scale_vmin,
            self.config.scale_vmax,
            target_count=6.0,
            minor_subdivisions=3,
        )
        major_ticks = tick_layout["major"]
        minor_ticks = tick_layout["minor"]

        for tick in major_ticks:
            value = float(tick["value"])
            fraction = float(tick["fraction"])
            is_zero = bool(tick["is_zero"])
            y = bar_rect.bottom() - fraction * bar_rect.height()
            tick_length = 12 if is_zero else 10
            painter.setPen(QtGui.QPen(tick_color, 1.2))
            painter.drawLine(bar_rect.left() - 1, int(y), bar_rect.left() - tick_length, int(y))

            font = QtGui.QFont(label_font)
            font.setBold(is_zero)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(text_color, 1.0))
            text_rect = build_tick_label_rect(
                left_label_left,
                left_label_width - 6,
                y,
                label_metrics,
                placement="above",
            )
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                format_colorbar_value(value, decimals=self.config.decimals, trim_trailing_zeros=True),
            )
        painter.setFont(label_font)

        for tick in minor_ticks:
            fraction = float(tick["fraction"])
            minor_y = bar_rect.bottom() - fraction * bar_rect.height()
            painter.setPen(QtGui.QPen(tick_color, 1.0))
            painter.drawLine(bar_rect.left() - 1, int(minor_y), bar_rect.left() - 6, int(minor_y))

        self._draw_reference_labels(
            painter=painter,
            bar_rect=bar_rect,
            label_left=right_label_left,
            image_width=image_width,
            text_color=text_color,
            tick_color=tick_color,
            label_font=label_font,
        )

    def _draw_reference_labels(
        self,
        painter: QtGui.QPainter,
        bar_rect: QtCore.QRect,
        label_left: int,
        image_width: int,
        text_color: QtGui.QColor,
        tick_color: QtGui.QColor,
        label_font: QtGui.QFont,
    ) -> None:
        """Draw right-side labels for visible data and threshold limits."""
        label_specs = build_reference_label_specs(
            scale_vmin=self.config.scale_vmin,
            scale_vmax=self.config.scale_vmax,
            color_vmin=self.config.color_vmin,
            color_vmax=self.config.color_vmax,
            decimals=self.config.decimals,
        )
        if not label_specs:
            return

        sorted_specs = sorted(
            label_specs,
            key=lambda item: float(bar_rect.bottom()) - float(item["fraction"]) * bar_rect.height(),
        )
        tick_y_positions = [
            float(bar_rect.bottom()) - float(spec["fraction"]) * bar_rect.height()
            for spec in sorted_specs
        ]
        metrics = QtGui.QFontMetricsF(label_font)
        text_y_positions = layout_reference_label_positions(
            bar_rect,
            sorted_specs,
            font_height=float(metrics.height()),
            min_gap=max(float(metrics.height()) * 0.9, 12.0),
        )

        painter.setFont(label_font)
        for spec, tick_y, text_y in zip(sorted_specs, tick_y_positions, text_y_positions):
            painter.setPen(QtGui.QPen(tick_color, 1.2))
            painter.drawLine(bar_rect.right() + 1, int(tick_y), bar_rect.right() + 10, int(tick_y))
            painter.setPen(QtGui.QPen(text_color, 1.0))
            text_rect = build_tick_label_rect(
                label_left,
                image_width - label_left - 8,
                text_y,
                metrics,
                placement="center",
            )
            painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, str(spec["text"]))
