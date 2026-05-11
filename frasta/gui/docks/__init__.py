"""Dockable widgets for FRASTA analysis panels.

These QDockWidget subclasses can be added to QMainWindow dock areas and
floated as independent windows. They communicate through signals rather than
direct method calls, which makes them loosely coupled and easy to rearrange.
"""

from .frasta_binary_dock import FrastaBinaryDock
from .frasta_profile_dock import FrastaProfileDock
from .frasta_controller import FrastaController

__all__ = [
    "FrastaBinaryDock",
    "FrastaProfileDock",
    "FrastaController",
]
