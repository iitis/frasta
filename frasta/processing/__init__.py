"""Processing algorithms for scan data analysis."""

from .alignment import (
    remove_relative_offset,
    remove_relative_tilt,
    compute_offset_global,
    compute_offset_in_center
)
from .filtering import (
    nan_aware_gaussian,
    remove_outliers
)
from .interpolation import fill_holes

# Advanced filtering
from .advanced_filtering import (
    bilateral_filter,
    median_filter_nan_aware,
    morphological_opening,
    morphological_closing,
    robust_gaussian_filter
)

# Morphological operations and leveling
from .morphology import (
    fit_plane_least_squares,
    fit_plane_robust,
    level_by_plane,
    level_by_three_points,
    remove_polynomial_form,
    threshold_grid
)

# Geometric transformations
from .transforms import (
    rotate_grid,
    rescale_grid,
    crop_to_valid_region,
    auto_register_surfaces,
    apply_registration
)

# Local plane fitting (for interactive tilt correction)
from .plane_fitting import (
    fit_plane_local_least_squares,
    fit_plane_local_ransac,
    fit_plane_local_median_filter
)
from .roughness import (
    surface_roughness_parameters,
    profile_roughness_parameters
)

__all__ = [
    # Alignment
    'remove_relative_offset',
    'remove_relative_tilt',
    'compute_offset_global',
    'compute_offset_in_center',
    # Basic filtering
    'nan_aware_gaussian',
    'remove_outliers',
    # Interpolation
    'fill_holes',
    # Advanced filtering
    'bilateral_filter',
    'median_filter_nan_aware',
    'morphological_opening',
    'morphological_closing',
    'robust_gaussian_filter',
    # Morphology and leveling
    'fit_plane_least_squares',
    'fit_plane_robust',
    'level_by_plane',
    'level_by_three_points',
    'remove_polynomial_form',
    'threshold_grid',
    # Transforms
    'rotate_grid',
    'rescale_grid',
    'crop_to_valid_region',
    'auto_register_surfaces',
    'apply_registration',
    # Local plane fitting
    'fit_plane_local_least_squares',
    'fit_plane_local_ransac',
    'fit_plane_local_median_filter',
    # Roughness summaries
    'surface_roughness_parameters',
    'profile_roughness_parameters',
]
