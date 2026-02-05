import pytest
from src.responsiveInfiniteLine import ResponsiveInfiniteLine

def test_responsive_infinite_line_init(qapp):
    # Sprawdza, czy widget się tworzy i nie rzuca wyjątku
    line = ResponsiveInfiniteLine()
    assert line is not None

def test_callback_invocation(qapp):
    called = []
    def cb(val): called.append(val)
    line = ResponsiveInfiniteLine(update_callback=cb)
    line._last_value = 0
    line.setValue(1)
    line._handle_throttled_update()
    assert called and called[0] == 1
