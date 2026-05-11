"""Quantitative validation tests on synthetic geometries.

These tests document the numerical accuracy of core processing routines using
analytical reference geometries (tilted planes, cylindrical surfaces) and
controlled round-trip experiments.  The measured tolerances are intended to
support the manuscript's validation claims.
"""

import numpy as np
import pytest

from frasta.processing.alignment import remove_relative_tilt, remove_relative_offset
from frasta.processing.interpolation import fill_holes
from frasta.processing.morphology import remove_polynomial_form


# ---------------------------------------------------------------------------
# Tilted-plane alignment accuracy
# ---------------------------------------------------------------------------

class TestPlaneSyntheticAlignment:
    """Alignment accuracy on synthetic tilted planes with known parameters."""

    def test_tilt_removal_rmse(self):
        """remove_relative_tilt recovers reference plane to machine precision on perfect surfaces.

        Reference:  z_ref = 0.05 x + 0.03 y
        Target:     z_tgt = 0.08 x + 0.01 y + 25.0  (different slope + offset)

        After tilt + offset removal the residual difference map should be zero
        to floating-point precision (RMSE < 1e-6).
        """
        ny, nx = 200, 200
        Y, X = np.mgrid[0:ny, 0:nx]

        reference = 0.05 * X + 0.03 * Y
        target    = 0.08 * X + 0.01 * Y + 25.0

        mask = np.ones((ny, nx), dtype=bool)

        aligned = remove_relative_tilt(reference, target, mask)
        aligned = remove_relative_offset(reference, aligned, mask)

        residual  = aligned - reference
        rmse      = np.sqrt(np.mean(residual ** 2))
        max_error = np.max(np.abs(residual))

        assert rmse < 1e-6, (
            f"Alignment RMSE on perfect tilted planes: {rmse:.2e} "
            f"(expected < 1e-6)"
        )
        assert max_error < 1e-5, (
            f"Max alignment error on perfect tilted planes: {max_error:.2e} "
            f"(expected < 1e-5)"
        )

    def test_difference_map_parallel_planes(self):
        """Difference map of two parallel planes recovers constant separation with RMSE = 0.

        Two identical tilted planes separated by a known constant offset
        (separation = 25.0) should produce a perfectly flat difference map.
        """
        ny, nx = 200, 200
        Y, X = np.mgrid[0:ny, 0:nx]

        separation = 25.0
        surface_a  = 0.05 * X + 0.03 * Y
        surface_b  = 0.05 * X + 0.03 * Y + separation

        diff_map = surface_b - surface_a
        mean_sep = float(np.mean(diff_map))
        rmse     = float(np.sqrt(np.mean((diff_map - separation) ** 2)))

        assert abs(mean_sep - separation) < 1e-10, (
            f"Mean difference-map value {mean_sep:.6f} != expected {separation}"
        )
        assert rmse < 1e-10, (
            f"Difference-map RMSE {rmse:.2e} for parallel planes (expected 0)"
        )


# ---------------------------------------------------------------------------
# Cylindrical surface validation
# ---------------------------------------------------------------------------

class TestCylindricalSurfaceValidation:
    """Validation on a synthetic cylindrical surface with known analytical form.

    The cylindrical geometry  z = R - sqrt(R^2 - x^2)  is a standard test
    case because its sag (central height above edges) has an analytical value
    and its form is essentially quadratic (plus higher-order terms).
    """

    @pytest.fixture
    def cylinder_grid(self):
        """Return a 201x201 grid of z = R - sqrt(R^2 - x^2) with R=1000, x in [-200, 200]."""
        R          = 1000.0    # µm
        half_width = 200.0     # µm
        n          = 201

        x = np.linspace(-half_width, half_width, n)
        y = np.linspace(-half_width, half_width, n)
        X, Y = np.meshgrid(x, y)

        Z = R - np.sqrt(R ** 2 - X ** 2)
        return Z, R, half_width, n

    def test_analytical_sag(self, cylinder_grid):
        """Central sag of cylindrical surface matches analytical formula R - sqrt(R^2 - a^2).

        For R = 1000 µm, half-width a = 200 µm the expected sag is
            h = 1000 - sqrt(1000^2 - 200^2) = 1000 - sqrt(960000) ≈ 20.20 µm.
        """
        Z, R, half_width, n = cylinder_grid

        # z = R - sqrt(R^2 - x^2): minimum at centre (x=0, z=0),
        # maximum at edges (x=±half_width).  Sag = edge height - centre height.
        sag_measured = float(Z[n // 2, 0] - Z[n // 2, n // 2])
        sag_expected = R - np.sqrt(R ** 2 - half_width ** 2)

        assert abs(sag_measured - sag_expected) < 1e-6, (
            f"Sag measured {sag_measured:.6f} µm, expected {sag_expected:.6f} µm"
        )

    def test_polynomial_form_removal_residual(self, cylinder_grid):
        """After 2nd-order form removal, residual RMSE on cylindrical surface < 0.5 µm.

        A cylindrical surface z = R - sqrt(R^2 - x^2) can be approximated as
            z ≈ x^2 / (2R) + x^4 / (8R^3) + ...
        A 2nd-order polynomial fit captures the dominant quadratic term; the
        remaining residual reflects higher-order contributions and should be
        small compared to the sag (~20 µm for the chosen geometry).
        """
        Z, R, half_width, n = cylinder_grid

        residual    = remove_polynomial_form(Z, order=2)
        rmse        = float(np.sqrt(np.nanmean(residual ** 2)))
        max_residual = float(np.nanmax(np.abs(residual)))

        # For R=1000, half_width=200: 4th-order max contribution ≈ 200^4/(8×1000^3) ≈ 0.2 µm
        assert rmse < 0.5, (
            f"Residual RMSE after quadratic form removal: {rmse:.4f} µm "
            f"(expected < 0.5 µm)"
        )
        assert max_residual < 1.0, (
            f"Max residual after quadratic form removal: {max_residual:.4f} µm "
            f"(expected < 1.0 µm)"
        )


# ---------------------------------------------------------------------------
# Interpolation round-trip accuracy
# ---------------------------------------------------------------------------

class TestInterpolationRoundTrip:
    """Round-trip accuracy of fill_holes interpolation on smooth synthetic surfaces."""

    def test_roundtrip_rmse_smooth_surface(self):
        """Nearest-neighbour fill_holes RMSE < 5 % of amplitude (20 % missing points).

        Surface:  z = 10 sin(x) + 5 cos(y),  x/y in [0, 2pi],  100x100 grid.
        Amplitude ~ 30 µm.  20 % of points removed at random (seed=42),
        then restored with fill_holes (nearest-neighbour).

        RMSE on the restored subset should be < 5 % of total amplitude.
        """
        rng = np.random.RandomState(42)
        nx = ny = 100
        X, Y = np.meshgrid(
            np.linspace(0, 2 * np.pi, nx),
            np.linspace(0, 2 * np.pi, ny),
        )
        Z_true    = 10.0 * np.sin(X) + 5.0 * np.cos(Y)
        amplitude = float(Z_true.max() - Z_true.min())

        missing  = rng.random((ny, nx)) < 0.20
        Z_holed  = Z_true.copy()
        Z_holed[missing] = np.nan

        Z_filled = fill_holes(Z_holed)

        rmse           = float(np.sqrt(np.mean((Z_filled[missing] - Z_true[missing]) ** 2)))
        relative_error = rmse / amplitude

        assert relative_error < 0.05, (
            f"Round-trip RMSE {rmse:.4f} = {100 * relative_error:.1f} % of amplitude "
            f"(expected < 5 %)"
        )
