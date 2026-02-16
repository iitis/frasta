# FRASTA Advanced Processing - Quick Reference

## 🚀 Szybka ściągawka

### Filtracja

```python
from frasta.processing import (
    bilateral_filter,           # edge-preserving
    median_filter_nan_aware,    # outlier removal
    robust_gaussian_filter,     # robust smoothing
    morphological_opening,      # remove peaks
    morphological_closing       # fill valleys
)

# Edge-preserving smoothing (najlepsze dla pęknięć!)
smoothed = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0, 
                           px_x=px_x, px_y=px_y)

# Usuń spike'y
cleaned = median_filter_nan_aware(grid, size=5.0, px_x=px_x, px_y=px_y)

# Robust Gaussian
filtered = robust_gaussian_filter(grid, sigma=10.0, px_x=px_x, 
                                 iterations=3, threshold=3.0)
```

---

### Levelowanie i korekcje

```python
from frasta.processing import (
    level_by_plane,             # remove tilt
    remove_polynomial_form,     # remove curvature
    threshold_grid             # value masking
)

# Usuń przechyłkę (tilt)
leveled = level_by_plane(grid, method='least_squares')  # szybka
leveled = level_by_plane(grid, method='robust')         # odporna

# Usuń krzywizny, wypaczenia
flattened = remove_polynomial_form(grid, order=2)  # quadratic
flattened = remove_polynomial_form(grid, order=3)  # cubic

# Threshold outliers
mean, std = np.nanmean(grid), np.nanstd(grid)
filtered = threshold_grid(grid, low=mean-3*std, high=mean+3*std)
```

---

### Transformacje

```python
from frasta.processing import (
    rotate_grid,                # rotate
    rescale_grid,              # change resolution
    crop_to_valid_region,      # crop
    auto_register_surfaces,    # auto-align
    apply_registration         # apply transform
)

# Obrót
rotated, xi, yi, px_x, px_y = rotate_grid(grid, 45, xi, yi, px_x, px_y)

# Zmiana rozdzielczości
high_res, xi, yi, px_x, px_y = rescale_grid(grid, 2.0, xi, yi, px_x, px_y)  # 2x
low_res, xi, yi, px_x, px_y = rescale_grid(grid, 0.5, xi, yi, px_x, px_y)   # 0.5x

# Przytnij do valid data
cropped, xi, yi, px_x, px_y = crop_to_valid_region(grid, xi, yi, px_x, px_y)

# Auto-wyrównanie
params = auto_register_surfaces(surf1, surf2, method='correlation')
aligned, xi, yi, px_x, px_y = apply_registration(
    surf2, xi, yi, px_x, px_y,
    translation=params['translation'],
    rotation=params['rotation']
)
```

---

## 🎯 Typowe scenariusze

### Scenario 1: Czyszczenie surowych danych

```python
# Pipeline: median -> level -> threshold
cleaned = median_filter_nan_aware(raw, size=5.0, px_x=px_x)
leveled = level_by_plane(cleaned, method='robust')
mean, std = np.nanmean(leveled), np.nanstd(leveled)
final = threshold_grid(leveled, low=mean-3*std, high=mean+3*std)
```

---

### Scenario 2: Pre-processing dla analizy chropowatości

```python
# Usuń formy geometryczne, zachowaj chropowatość
leveled = level_by_plane(grid)                          # usuń tilt
flattened = remove_polynomial_form(leveled, order=2)    # usuń bending
# Teraz możesz policzyć Sa, Sq itp.
```

---

### Scenario 3: Smoothing z zachowaniem krawędzi

```python
# Bilateral zamiast Gaussian dla powierzchni pęknięć
smoothed = bilateral_filter(grid, sigma_spatial=5.0, sigma_range=10.0,
                           px_x=px_x, px_y=px_y)
# Krawędzie pęknięcia pozostają ostre!
```

---

### Scenario 4: Automatyczne wyrównanie dwóch powierzchni

```python
# 1. Znajdź parametry (quick correlation)
params = auto_register_surfaces(surf1, surf2, method='correlation')

# 2. Zastosuj
aligned, xi, yi, px_x, px_y = apply_registration(
    surf2, xi, yi, px_x, px_y,
    translation=params['translation']
)

# 3. Dodaj fine-tuning z ICP jeśli potrzeba
params_fine = auto_register_surfaces(surf1, aligned, method='icp')
```

---

## ⚙️ Parametry - kiedy co używać?

### Bilateral filter
- `sigma_spatial`: ~5-10 × pixel size (skala przestrzenna)
- `sigma_range`: ~1-2 × noise level (tolerancja wysokości)
- **Mniejsze sigma_range** = ostrzejsze krawędzie

### Median filter
- `size`: 3-5 × pixel size dla spike removal
- Zwiększaj size jeśli szum jest większy

### Polynomial removal
- `order=1`: tylko tilt (równoważne plane leveling)
- `order=2`: standardowa korekcja (bending, warping)
- `order=3`: tylko jeśli widzisz złożone krzywe
- `order>3`: rzadko potrzebne, może usunąć rzeczywiste cechy!

### Auto-registration methods
- `'correlation'`: tylko translacja, SZYBKA, dobra na start
- `'icp'`: translacja + rotacja, WOLNIEJSZA, lepsza precyzja

---

## 💡 Pro Tips

1. **Zawsze sprawdzaj wynik wizualnie** - nie ufaj ślepo algorytmom
2. **Zapisuj parametry** dla reprodukowalności
3. **Bilateral filter jest wolny** - rozważ downsampling dla dużych danych
4. **Robust methods** (RANSAC, robust gaussian) są wolniejsze ale lepsze przy outlierach
5. **Order polynomial** - rozpocznij od 2, zwiększaj tylko gdy potrzeba
6. **ICP wymaga dobrego overlap** - użyj correlation najpierw dla rough alignment

---

## 📊 Porównanie wydajności (dla 500×500 grid)

| Funkcja | Czas | Uwagi |
|---------|------|-------|
| `median_filter` | ~0.1s | Szybka |
| `bilateral_filter` | ~30s | Wolna (Python impl) |
| `level_by_plane` | <0.01s | Bardzo szybka |
| `remove_polynomial_form` (order=2) | ~0.05s | Szybka |
| `auto_register` (correlation) | ~0.5s | Średnia |
| `auto_register` (ICP) | ~2-5s | Wolniejsza |

---

## 📝 Przykłady w akcji

Zobacz: `examples_advanced_processing.py`

```bash
python examples_advanced_processing.py
```

---

## 🔗 Pełna dokumentacja

[docs/ADVANCED_PROCESSING.md](ADVANCED_PROCESSING.md)
