"""Decorators for utility functions.

This module provides decorators for measuring performance and other
cross-cutting concerns.
"""

import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def measure_time(func):
    """Decorator for measuring and logging function execution time.
    
    Args:
        func (callable): Function to measure.
        
    Returns:
        callable: Wrapped function that logs execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f">>> {func.__name__}() took {end - start:.4f} seconds")
        return result
    return wrapper
