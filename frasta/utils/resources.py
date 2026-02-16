"""Resource path resolution utilities.

This module provides functions for resolving paths to resource files in both
development and PyInstaller executable environments.
"""

import sys
import os


def resource_path(relative_path):
    """Returns absolute path to a resource file.
    
    Works both in development (.py) and PyInstaller executable (.exe) environments.
    
    Args:
        relative_path (str): Relative path to the resource.
        
    Returns:
        str: Absolute path to the resource.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
