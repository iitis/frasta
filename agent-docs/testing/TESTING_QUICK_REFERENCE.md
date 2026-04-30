# Quick Test Reference Guide

This guide provides common patterns and examples for writing tests for FRASTA GUI modules.

## Table of Contents
1. [Basic Test Structure](#basic-test-structure)
2. [Mocking PyQt5 Components](#mocking-pyqt5-components)
3. [Testing Manager Classes](#testing-manager-classes)
4. [Testing Data Processing](#testing-data-processing)
5. [Testing Error Handling](#testing-error-handling)
6. [Common Fixtures](#common-fixtures)

---

## Basic Test Structure

### Standard Test Class
```python
class TestMyManager:
    """Test suite for MyManager."""
    
    @pytest.fixture
    def manager(self):
        """Create MyManager instance with mocks."""
        return MyManager(mock_dependency)
    
    def test_method_basic_behavior(self, manager):
        """Test that method does expected thing."""
        result = manager.method(input_data)
        assert result == expected_output
```

### Test Naming Convention
```python
def test_<what>_<condition>_<expected>():
    """
    Format: test_<method_name>_<specific_condition>_<expected_result>
    
    Examples:
    - test_compute_range_with_nan_returns_valid_values()
    - test_create_mask_no_roi_returns_none()
    - test_flip_scan_up_down_inverts_rows()
    """
    pass
```

---

## Mocking PyQt5 Components

### Mock QWidget
```python
@pytest.fixture
def mock_widget():
    """Create mock PyQt5 widget."""
    from unittest.mock import Mock
    widget = Mock()
    widget.setText = Mock()
    widget.text = Mock(return_value="")
    widget.isVisible = Mock(return_value=True)
    return widget
```

### Mock MainWindow
```python
@pytest.fixture
def mock_main_window():
    """Create mock MainWindow with essential components."""
    window = Mock()
    
    # Mock tab widget
    window.tabs = Mock()
    window.tabs.count = Mock(return_value=0)
    window.tabs.addTab = Mock()
    window.tabs.currentWidget = Mock(return_value=None)
    
    # Mock status bar
    window.statusBar = Mock(return_value=Mock())
    window.statusBar().showMessage = Mock()
    
    # Mock current_tab method
    window.current_tab = Mock(return_value=None)
    
    return window
```

### Mock QDialog with Accept/Reject
```python
def test_dialog_accepted():
    """Test behavior when dialog is accepted."""
    with patch.object(QtWidgets.QDialog, 'exec_', 
                     return_value=QtWidgets.QDialog.Accepted):
        result = my_function_showing_dialog()
        assert result is not None

def test_dialog_rejected():
    """Test behavior when dialog is cancelled."""
    with patch.object(QtWidgets.QDialog, 'exec_', 
                     return_value=QtWidgets.QDialog.Rejected):
        result = my_function_showing_dialog()
        assert result is None
```

### Mock QMessageBox
```python
def test_shows_warning_message():
    """Test that warning message is displayed."""
    with patch.object(QtWidgets.QMessageBox, 'warning') as mock_warning:
        controller.method_that_shows_warning()
        
        # Verify warning was called
        mock_warning.assert_called_once()
        
        # Check specific message
        call_args = mock_warning.call_args
        assert "Expected text" in str(call_args)
```

### Mock QFileDialog
```python
def test_open_file_dialog():
    """Test file selection dialog."""
    test_file = "/path/to/test.h5"
    
    with patch.object(QtWidgets.QFileDialog, 'getOpenFileName',
                     return_value=(test_file, "")):
        controller.open_file()
        
        # Verify file was processed
        assert controller.current_file == test_file
```

---

## Testing Manager Classes

### Test Initialization
```python
def test_initialization(self, manager):
    """Test that manager initializes with correct defaults."""
    assert manager.some_attribute == expected_default
    assert manager.some_list == []
    assert manager.some_flag is False
```

### Test State Changes
```python
def test_method_updates_state(self, manager):
    """Test that method correctly updates internal state."""
    initial_state = manager.get_state()
    
    manager.update_something(new_value)
    
    final_state = manager.get_state()
    assert final_state != initial_state
    assert manager.attribute == new_value
```

### Test Method Delegation
```python
def test_delegates_to_dependency(self, manager):
    """Test that manager delegates to dependency correctly."""
    mock_dependency = Mock()
    manager.dependency = mock_dependency
    
    manager.high_level_operation()
    
    # Verify dependency was called
    mock_dependency.low_level_operation.assert_called_once()
    
    # Check arguments
    call_args = mock_dependency.low_level_operation.call_args
    assert call_args[0][0] == expected_arg
```

---

## Testing Data Processing

### Test with numpy Arrays
```python
def test_processes_grid_correctly(self):
    """Test grid processing with numpy array."""
    input_grid = np.arange(100, dtype=float).reshape(10, 10)
    
    result = process_grid(input_grid)
    
    # Check shape preserved
    assert result.shape == input_grid.shape
    
    # Check data transformed correctly
    assert result.dtype == np.float32
    assert not np.array_equal(result, input_grid)
```

### Test NaN Handling
```python
def test_handles_nan_values(self):
    """Test that NaN values are handled correctly."""
    grid = np.ones((10, 10), dtype=float)
    grid[3:5, 4:6] = np.nan  # Create NaN region
    
    result = process_grid(grid)
    
    # NaN should be preserved or handled appropriately
    assert np.any(np.isnan(result))
    # OR
    assert not np.any(np.isnan(result))  # If filled
```

### Test Edge Cases
```python
def test_handles_empty_data(self):
    """Test behavior with no valid data."""
    empty_grid = np.full((10, 10), np.nan, dtype=float)
    
    result = process_grid(empty_grid)
    
    # Should handle gracefully
    assert result is not None
    # OR should return default
    assert result == default_value

def test_handles_constant_data(self):
    """Test behavior with constant values."""
    constant_grid = np.ones((10, 10), dtype=float) * 5.0
    
    result = calculate_range(constant_grid)
    
    # Should create reasonable range
    lo, hi = result
    assert lo < hi
    assert lo < 5.0 < hi
```

---

## Testing Error Handling

### Test Exception Handling
```python
def test_handles_error_gracefully(self):
    """Test that method handles errors without crashing."""
    with patch('module.function', side_effect=ValueError("Error")):
        # Should not raise exception
        result = manager.method_that_might_fail()
        
        # Should return safe value or None
        assert result is None
        # OR
        assert result == default_value
```

### Test Warning Display
```python
def test_shows_error_dialog_on_failure(self):
    """Test that error dialog is shown on failure."""
    with patch.object(QtWidgets.QMessageBox, 'critical') as mock_critical:
        with patch('module.function', side_effect=Exception("Failed")):
            manager.method()
            
            # Should show error dialog
            mock_critical.assert_called_once()
            
            # Check error message contains useful info
            args = mock_critical.call_args[0]
            assert "Failed" in str(args)
```

### Test Validation
```python
def test_validates_input(self):
    """Test that invalid input is rejected."""
    invalid_data = None  # or invalid value
    
    with patch.object(QtWidgets.QMessageBox, 'warning'):
        result = manager.process(invalid_data)
        
        # Should return early without processing
        assert result is None
        QtWidgets.QMessageBox.warning.assert_called_once()
```

---

## Common Fixtures

### Add to conftest.py

```python
@pytest.fixture
def sample_grid_with_holes():
    """Create grid with NaN holes for testing."""
    grid = np.arange(100, dtype=float).reshape(10, 10)
    grid[2:4, 3:5] = np.nan  # Rectangle hole
    grid[7, 8] = np.nan      # Single point
    return grid

@pytest.fixture
def mock_qapplication(qapp):
    """Provide QApplication for widget tests."""
    return qapp

@pytest.fixture
def sample_surface():
    """Create Surface object for testing."""
    from frasta.core import Surface
    grid = np.random.randn(20, 20) + 100
    return Surface(height=grid, dx=1.5, dy=2.0, x0=0, y0=0)

@pytest.fixture
def mock_scan_tab():
    """Create mock ScanTab with standard methods."""
    tab = Mock()
    tab.grid = np.ones((10, 10), dtype=float)
    tab.dx = 1.0
    tab.dy = 1.0
    tab.update_image = Mock()
    tab.update_histogram = Mock()
    return tab
```

---

## Advanced Patterns

### Testing Callbacks
```python
def test_callback_is_invoked(self):
    """Test that callback is called with correct arguments."""
    callback = Mock()
    
    manager = MyManager(callback)
    manager.do_something()
    
    # Verify callback was called
    callback.assert_called_once()
    
    # Check callback arguments
    args, kwargs = callback.call_args
    assert args[0] == expected_first_arg
    assert kwargs['param'] == expected_value
```

### Testing Signals (PyQt)
```python
def test_signal_emitted(self):
    """Test that Qt signal is emitted."""
    from pytestqt.qtbot import QtBot
    
    widget = MyWidget()
    
    with qtbot.waitSignal(widget.my_signal, timeout=1000):
        widget.trigger_signal()
    
    # Signal was emitted if we reach here
```

### Parametrized Tests
```python
@pytest.mark.parametrize("direction,expected_first,expected_last", [
    ('UD', 9, 0),  # Up-down flips rows
    ('LR', 9, 0),  # Left-right flips columns
])
def test_flip_scan_directions(direction, expected_first, expected_last):
    """Test flip_scan with different directions."""
    grid = np.arange(100).reshape(10, 10)
    
    result = TransformOperations.flip_scan(grid, direction)
    
    if direction == 'UD':
        assert result[0, 0] == expected_first * 10
    else:
        assert result[0, 0] == expected_first
```

---

## Debugging Tips

### Print Actual vs Expected
```python
def test_calculation(self):
    """Test numerical calculation."""
    result = calculate_something()
    expected = 42.0
    
    print(f"Result: {result}, Expected: {expected}")  # Debug output
    assert np.isclose(result, expected, rtol=0.01)
```

### Inspect Mock Calls
```python
def test_mock_usage(self):
    """Test with detailed mock inspection."""
    mock_obj = Mock()
    
    my_function(mock_obj)
    
    # See all calls made to mock
    print(mock_obj.method_calls)
    
    # Check specific call
    print(mock_obj.method.call_args_list)
```

### Use pytest's -vv and --tb=short
```bash
# Verbose output with short tracebacks
pytest tests/test_my_module.py -vv --tb=short

# Show print statements
pytest tests/test_my_module.py -s

# Run specific test
pytest tests/test_my_module.py::TestClass::test_method -vv
```

---

## Quick Reference Commands

```bash
# Run all GUI tests
pytest tests/test_gui_*.py -v

# Run with coverage
pytest tests/test_gui_*.py --cov=frasta.gui --cov-report=html

# Run single test file
pytest tests/test_gui_viewers.py -v

# Run single test
pytest tests/test_gui_viewers.py::TestLODManager::test_initialization -vv

# Run tests matching pattern
pytest -k "test_flip" -v

# Show all output (including prints)
pytest tests/test_gui_scan_tab.py -s

# Stop at first failure
pytest tests/test_gui_*.py -x

# Run last failed tests
pytest --lf -v
```

---

## Best Practices Checklist

- [ ] Test names are descriptive and follow convention
- [ ] Each test has a docstring explaining what it tests
- [ ] Fixtures are used for common setup
- [ ] Mocks are used instead of real UI components
- [ ] Edge cases are tested (None, empty, NaN, etc.)
- [ ] Error conditions are tested
- [ ] Tests are independent (don't rely on order)
- [ ] Assertions include helpful failure messages
- [ ] Tests run quickly (< 1 second each)
- [ ] Code coverage is measured and tracked

---

**For more examples, see:**
- [test_gui_viewers.py](../../tests/test_gui_viewers.py) - Manager class testing
- [test_gui_scan_tab.py](../../tests/test_gui_scan_tab.py) - Data processing testing
- [test_gui_main_window.py](../../tests/test_gui_main_window.py) - Controller testing
- [test_gui_profile_viewer.py](../../tests/test_gui_profile_viewer.py) - Integration patterns
