import pytest
from unittest.mock import MagicMock
from src.limitedGLView import LimitedGLView

class DummyGLView(LimitedGLView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Provide default opts for testing
        self.opts = {'azimuth': 0.0, 'elevation': 0.0}
        self.update = MagicMock()

@pytest.mark.parametrize(
    "az_range, wrap_az, azim, start_az, expected_az, el_range, elev, start_el, expected_el",
    [
        ((-180, 180), False, 200, 0, 180, (-90, 90), 100, 0, 90),
        ((-180, 180), False, -200, 0, -180, (-90, 90), -100, 0, -90),
        ((-180, 180), True, 200, 0, -160, (-45, 45), 50, 0, 45),
        ((-180, 180), True, 370, 0, 10, (None), 30, 10, 40),
        ((None), False, 45, 10, 55, (None), 30, 10, 40),
        ((None), False, -20, -10, -30, (None), -50, -10, -60),
#        ((0, 360), True, 400, 350, 40, (None), 20, 10, 30),
#        ((0, 360), False, 400, 350, 360, (None), 20, 10, 30),
#        ((10, 10), True, 50, 10, 10, (None), 0, 0, 0),  # span zero with wrap
#        ((10, 10), False, -50, 10, 10, (None), 0, 0, 0), # span zero without wrap
    ]
)

def test_orbit(az_range, wrap_az, azim, start_az, expected_az,
               el_range, elev, start_el, expected_el):
    view = DummyGLView(azimuth_range=az_range if isinstance(az_range, tuple) else None,
                       elevation_range=el_range if isinstance(el_range, tuple) else None,
                       wrap_azimuth=wrap_az if isinstance(az_range, tuple) else False)
    view.opts['azimuth'] = start_az
    view.opts['elevation'] = start_el
    view.orbit(azim, elev)
    assert pytest.approx(view.opts['azimuth'], abs=1e-6) == expected_az
    assert pytest.approx(view.opts['elevation'], abs=1e-6) == expected_el
    view.update.assert_called_once()

def test_orbit_no_ranges():
    view = DummyGLView()
    view.opts['azimuth'] = 10
    view.opts['elevation'] = 20
    view.orbit(5, -10)
    assert view.opts['azimuth'] == 15
    assert view.opts['elevation'] == 10
    view.update.assert_called_once()

def test_orbit_span_zero_wrap():
    view = DummyGLView(azimuth_range=(10, 10), wrap_azimuth=True)
    view.opts['azimuth'] = 10
    view.orbit(5, 0)
    # span is zero, so azimuth should remain unchanged
    assert view.opts['azimuth'] == 10
    view.update.assert_called_once()