"""Tests for frasta.utils module (resources, decorators, and colormaps)."""

import pytest
import time
import sys
from pathlib import Path
from unittest.mock import patch
from frasta.utils import (
    resource_path,
    measure_time,
    get_colormap,
    get_lookup_table,
    get_gradient_brush,
    get_brushes_for_values,
)


class TestResourcePath:
    """Test suite for resource_path function."""
    
    def test_resource_path_normal_mode(self):
        """Test resource_path in normal (non-frozen) mode."""
        result = resource_path("test.txt")
        
        # Should return a path that exists in the project structure
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_resource_path_with_subdirectory(self):
        """Test resource_path with subdirectory."""
        result = resource_path("icons/test.png")
        
        assert "icons" in result or "icons" in result.replace("\\", "/")
    
    def test_resource_path_frozen_mode(self, monkeypatch):
        """Test resource_path when running as frozen executable."""
        # Mock sys._MEIPASS to simulate PyInstaller frozen mode
        mock_meipass = "/fake/frozen/path"
        monkeypatch.setattr(sys, '_MEIPASS', mock_meipass, raising=False)
        
        result = resource_path("test.txt")
        
        # Should use _MEIPASS path
        assert mock_meipass in result
    
    def test_resource_path_multiple_calls(self):
        """Test that multiple calls work correctly."""
        path1 = resource_path("file1.txt")
        path2 = resource_path("file2.txt")
        
        assert path1 != path2
        assert "file1.txt" in path1
        assert "file2.txt" in path2


class TestMeasureTimeDecorator:
    """Test suite for measure_time decorator."""
    
    def test_measure_time_returns_result(self):
        """Test that decorated function returns correct result."""
        @measure_time
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        assert result == 5
    
    def test_measure_time_with_sleep(self):
        """Test that decorator measures time correctly."""
        @measure_time
        def slow_function():
            time.sleep(0.01)  # Sleep 10ms
            return 42
        
        start = time.time()
        result = slow_function()
        elapsed = time.time() - start
        
        assert result == 42
        # Should take at least 10ms
        assert elapsed >= 0.01
    
    def test_measure_time_preserves_function_name(self):
        """Test that decorator preserves function metadata."""
        @measure_time
        def test_func():
            """Test docstring."""
            return "test"
        
        assert test_func.__name__ == "test_func"
        assert "Test docstring" in test_func.__doc__
    
    def test_measure_time_with_args_and_kwargs(self):
        """Test decorated function with various arguments."""
        @measure_time
        def complex_func(a, b, c=10, d=20):
            return a + b + c + d
        
        result1 = complex_func(1, 2)
        result2 = complex_func(1, 2, c=5)
        result3 = complex_func(1, 2, c=5, d=15)
        
        assert result1 == 33  # 1+2+10+20
        assert result2 == 28  # 1+2+5+20
        assert result3 == 23  # 1+2+5+15
    
    def test_measure_time_with_exception(self):
        """Test that decorator doesn't interfere with exceptions."""
        @measure_time
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
    
    def test_measure_time_multiple_decorators(self):
        """Test that multiple decorated functions work independently."""
        @measure_time
        def func1():
            return 1
        
        @measure_time
        def func2():
            return 2
        
        assert func1() == 1
        assert func2() == 2


class TestColormaps:
    """Test suite for custom colormap helpers."""

    def test_get_metrology_colormap(self):
        """Custom metrology colormap should be available."""
        cmap = get_colormap("metrology")
        assert cmap is not None

    def test_get_lookup_table_shape(self):
        """Lookup table should have requested shape and RGB channels."""
        lut = get_lookup_table("metrology", 64)
        assert lut.shape == (64, 3)

    def test_get_builtin_colormap(self):
        """Builtin pyqtgraph colormaps still work through helper."""
        cmap = get_colormap("viridis")
        assert cmap is not None

    def test_get_difference_colormap(self):
        """Custom diverging colormap should be available."""
        cmap = get_colormap("difference")
        assert cmap is not None

    def test_get_gradient_brush(self):
        """Gradient brush helper should build a brush."""
        brush = get_gradient_brush("metrology")
        assert brush is not None

    def test_get_brushes_for_values(self):
        """Per-bin brush helper should return one brush per sample."""
        brushes = get_brushes_for_values("difference", [0.0, 0.5, 1.0])
        assert len(brushes) == 3
