from PyQt5 import QtCore
import pyqtgraph as pg

class ResponsiveInfiniteLine(pg.InfiniteLine):
    """An InfiniteLine with smooth dragging and throttled update callbacks.

    This class emits updates at a controlled rate while the line is being moved, and triggers an immediate update when dragging ends.
    """
    def __init__(self, update_callback=None, delay_ms=200, *args, **kwargs):
        """Initializes the responsive infinite line with throttled update callbacks.

        Sets up the timer, connects signals for position changes, and configures the cursor and callback behavior.

        Args:
            update_callback (callable, optional): Function to call when the line value changes.
            delay_ms (int, optional): Throttle delay in milliseconds for update callbacks.
            *args: Additional positional arguments for InfiniteLine.
            **kwargs: Additional keyword arguments for InfiniteLine.
        """
        super().__init__(*args, **kwargs)

        self.setCursor(QtCore.Qt.SizeHorCursor if self.angle == 90 else QtCore.Qt.SizeVerCursor)

        # Timer do throttlowania update'ów
        self._update_callback = update_callback
        self._update_timer = QtCore.QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._handle_throttled_update)

        self._last_value = self.value()
        self._throttle_delay = delay_ms

        self.sigPositionChanged.connect(self._on_position_changed)

        # Dodatkowy sygnał po zakończeniu przeciągania (jeśli dostępny)
        if hasattr(self, "sigPositionChangeFinished"):
            self.sigPositionChangeFinished.connect(self._on_change_finished)

    def _on_position_changed(self):
        """Handles throttled update when the line position changes."""
        if self._update_callback:
            self._update_timer.start(self._throttle_delay)

    def _handle_throttled_update(self):
        """Calls the update callback if the value has changed after throttling."""
        value = self.value()
        if value != self._last_value:
            self._last_value = value
            if self._update_callback:
                self._update_callback(value)

    def _on_change_finished(self):
        """Immediately calls the update callback when dragging is finished."""
        self._update_timer.stop()
        value = self.value()
        self._last_value = value
        if self._update_callback:
            self._update_callback(value)
