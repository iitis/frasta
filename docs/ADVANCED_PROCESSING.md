# Zaawansowane przetwarzanie danych w FRASTA

Moduły `advanced_filtering`, `morphology` i `transforms` zawierają zaawansowane funkcje przetwarzania adaptowane z projektu EFS-toolbox.

## Moduły

### 1. Zaawansowana filtracja (`advanced_filtering.py`)

#### Filtr bilateralny - `bilateral_filter()`
**Zachowuje krawędzie podczas wygładzania**

```python
from frasta.processing import bilateral_filter

filtered = bilateral_filter(
    grid,
    sigma_spatial=5.0,  # skala przestrzenna wygładzania (jednostki fizyczne)
    sigma_range=10.0,   # tolerancja różnic wysokości
    px_x=grid_data.px_x,
    px_y=grid_data.px_y
)
```

**Zastosowanie:**
- Wygładzanie szumu przy zachowaniu ostrych krawędzi pęknięć
- Pre-processing przed analizą cech
- Alternatywa dla Gaussian gdy ważne są krawędzie

**Zalety:** Nie rozmywa granic między regionami powierzchni

---

#### Filtr medianowy - `median_filter_nan_aware()`
**Odporne usuwanie outlierów**

```python
from frasta.processing import median_filter_nan_aware

filtered = median_filter_nan_aware(
    grid,
    size=10.0,  # rozmiar kernela w jednostkach fizycznych
    px_x=1.0,
    px_y=1.0
)
```

**Zastosowanie:**
- Usuwanie skoków pomiarowych (spikes)
- Redukcja szumu zachowując krawędzie
- Pre-processing przed analizą fraktalną

**Zalety:** Odporność na pojedyncze ekstremalne wartości

---

#### Operacje morfologiczne

**Opening** - usuwa małe szczyty:
```python
from frasta.processing import morphological_opening

filtered = morphological_opening(grid, size=5.0, px_x=1.0)
```

**Closing** - wypełnia małe doliny:
```python
from frasta.processing import morphological_closing

filtered = morphological_closing(grid, size=5.0, px_x=1.0)
```

**Zastosowanie:**
- Usuwanie artefaktów pomiarowych
- Strukturalne przetwarzanie powierzchni
- Czyszczenie danych przed ekstrakcją cech

---

#### Robust Gaussian - `robust_gaussian_filter()`
**Wygładzanie z iteracyjnym odrzucaniem outlierów**

```python
from frasta.processing import robust_gaussian_filter

filtered = robust_gaussian_filter(
    grid,
    sigma=10.0,
    px_x=1.0,
    iterations=3,      # liczba iteracji odrzucania outlierów
    threshold=3.0      # prog odrzucenia (wielokrotność std)
)
```

**Zastosowanie:**
- Wygładzanie danych z outlierami
- Bardziej odporne niż standardowy Gaussian
- Gdy median filter za bardzo zmienia strukturę

---

### 2. Morfologia i korekcje geometryczne (`morphology.py`)

#### Levelowanie płaszczyzny - `level_by_plane()`
**Usuwa przechyłki powierzchni**

```python
from frasta.processing import level_by_plane

# Metoda najmniejszych kwadratów (szybka)
leveled = level_by_plane(grid, method='least_squares')

# Metoda robust z RANSAC (odporna na outliers)
leveled = level_by_plane(grid, method='robust')
```

**Zastosowanie:**
- Korekcja przechyłki próbki
- Usuwanie sistematycznego nachylenia
- Pre-processing przed analizą chropowatości

---

#### Usuwanie form wielomianowych - `remove_polynomial_form()`
**Korekcja krzywizny, wygięcia, wypaczenia**

```python
from frasta.processing import remove_polynomial_form

# Usuwanie formy kwadratowej (bending, warping)
corrected = remove_polynomial_form(grid, order=2)

# Usuwanie formy kubicznej (bardziej złożone krzywe)
corrected = remove_polynomial_form(grid, order=3)
```

**Parametr `order`:**
- `order=1`: płaszczyzna (równoważne levelowaniu)
- `order=2`: paraboloida (zgięcie, wypaczenie)
- `order=3`: kubiczna (złożone krzywe)
- `order=4-5`: bardzo złożone formy (ostrożnie!)

**Zastosowanie:**
- Korekcja wypaczenia stołu pomiarowego
- Usuwanie krzywizny próbki
- Usuwanie systematycznych artefaktów pomiarowych
- Zgodne z ISO 25178 (form removal)

---

#### Levelowanie przez 3 punkty - `level_by_three_points()`

```python
from frasta.processing import level_by_three_points

leveled = level_by_three_points(
    grid,
    p1=(0, 0),      # współrzędne fizyczne trzech punktów
    p2=(100, 0),
    p3=(50, 50),
    xi, yi
)
```

**Zastosowanie:** Gdy znasz referencyjne punkty powierzchni

---

#### Thresholding - `threshold_grid()`
**Maskowanie na podstawie wartości**

```python
from frasta.processing import threshold_grid

# Usuń ekstremalne wartości
mean = np.nanmean(grid)
std = np.nanstd(grid)
filtered = threshold_grid(grid, low=mean-3*std, high=mean+3*std)
```

---

### 3. Transformacje geometryczne (`transforms.py`)

#### Rotacja - `rotate_grid()`

```python
from frasta.processing import rotate_grid

rotated, xi, yi, px_x, px_y = rotate_grid(
    grid, 
    angle_degrees=45,  # kąt obrotu
    xi, yi, px_x, px_y,
    order=3           # interpolacja (0=nearest, 1=linear, 3=cubic)
)
```

**Zastosowanie:**
- Obrót powierzchni do pożądanej orientacji
- Wyrównanie przed porównaniem

---

#### Zmiana rozdzielczości - `rescale_grid()`

```python
from frasta.processing import rescale_grid

# Zwiększenie rozdzielczości 2x
high_res, xi, yi, px_x, px_y = rescale_grid(
    grid, 2.0, xi, yi, px_x, px_y
)

# Zmniejszenie rozdzielczości 2x (downsampling)
low_res, xi, yi, px_x, px_y = rescale_grid(
    grid, 0.5, xi, yi, px_x, px_y
)
```

**Zastosowanie:**
- Unified resolution przed porównaniem powierzchni
- Upsampling dla wizualizacji
- Downsampling dla szybszych obliczeń

---

#### Przycinanie - `crop_to_valid_region()`

```python
from frasta.processing import crop_to_valid_region

cropped, xi, yi, px_x, px_y = crop_to_valid_region(
    grid, xi, yi, px_x, px_y,
    margin=10  # piksele marginesu wokół valid data
)
```

**Zastosowanie:**
- Usuwanie pustych obramowań
- Zmniejszenie rozmiaru danych
- Optymalizacja przed obliczeniami

---

#### Automatyczna rejestracja - `auto_register_surfaces()`
**Automatyczne wyrównanie dwóch powierzchni**

```python
from frasta.processing import auto_register_surfaces, apply_registration

# Znajdź parametry wyrównania
params = auto_register_surfaces(
    reference_grid,
    target_grid,
    method='correlation'  # lub 'icp' dla rotacji + translacji
)

print(f"Translacja: {params['translation']}")
print(f"Rotacja: {params['rotation']} deg")
print(f"RMSE: {params['rmse']}")

# Zastosuj wyrównanie
aligned, xi, yi, px_x, px_y = apply_registration(
    grid, xi, yi, px_x, px_y,
    translation=params['translation'],
    rotation=params['rotation']
)
```

**Metody:**
- `'correlation'`: Cross-correlation (tylko translacja, szybka)
- `'icp'`: Iterative Closest Point (translacja + rotacja, wolniejsza)

**Opcje ICP:**
- `refine=True`: wolniejsze końcowe dopracowanie pod RMSE wysokości
- `stable_region=True`: drugi przebieg ICP na automatycznie wybranym obszarze o małym mismatchu

**Zastosowanie:**
- Automatyczne wyrównanie przeciwnych powierzchni pęknięcia
- Pre-processing przed analizą mismatch
- Znajdowanie przesunięć między skanami

---

## Typowe workflow

### Workflow 1: Czyszczenie danych surowych

```python
from frasta.processing import (
    median_filter_nan_aware,
    level_by_plane,
    threshold_grid
)

# 1. Usuń outliers
cleaned = median_filter_nan_aware(raw_grid, size=5.0, px_x=px_x)

# 2. Usuń przechyłkę
leveled = level_by_plane(cleaned, method='robust')

# 3. Odrzuć ekstremalne wartości
mean, std = np.nanmean(leveled), np.nanstd(leveled)
final = threshold_grid(leveled, low=mean-3*std, high=mean+3*std)
```

---

### Workflow 2: Korekcja form geometrycznych

```python
from frasta.processing import (
    level_by_plane,
    remove_polynomial_form
)

# 1. Usuń narost liniowy (tilt)
leveled = level_by_plane(grid)

# 2. Usuń formę kwadratową (bending)
flattened = remove_polynomial_form(leveled, order=2)
```

---

### Workflow 3: Edge-preserving denoising

```python
from frasta.processing import bilateral_filter

# Wygładź szum ale zachowaj krawędzie pęknięcia
smoothed = bilateral_filter(
    grid,
    sigma_spatial=5.0,   # przestrzenna skala
    sigma_range=10.0,    # tolerancja wysokości
    px_x=px_x,
    px_y=px_y
)
```

---

### Workflow 4: Automatyczne wyrównanie powierzchni

```python
from frasta.processing import (
    auto_register_surfaces,
    apply_registration,
    remove_relative_offset
)

# 1. Znajdź transformację
params = auto_register_surfaces(surf1, surf2, method='icp')

# 2. Zastosuj transformację
aligned = apply_registration(
    surf2, xi, yi, px_x, px_y,
    translation=params['translation'],
    rotation=params['rotation']
)

# 3. Usuń offset wysokościowy
aligned_corrected = remove_relative_offset(surf1, aligned[0], mask)
```

---

## Porównanie z istniejącymi funkcjami

| Funkcja | Stara (basic) | Nowa (advanced) | Zaleta |
|---------|---------------|-----------------|--------|
| Smoothing | `nan_aware_gaussian` | `bilateral_filter` | Zachowuje krawędzie |
| Outlier removal | `remove_outliers` | `median_filter_nan_aware` | Bardziej robust |
| Leveling | `remove_relative_tilt` | `level_by_plane` + `remove_polynomial_form` | Więcej opcji korekcji |
| Alignment | Manual translation/rotation | `auto_register_surfaces` | Automatyczne, ICP |

---

## WARNING Uwagi

1. **Wydajność bilateral filter**: Python implementation jest wolniejszy. Dla dużych danych rozważ downsampling lub podziel na tile'y.

2. **Wybór order dla polynomial removal**: 
   - Zbyt wysoki order może usunąć rzeczywiste cechy powierzchni
   - Zalecane: `order=2` dla większości przypadków
   - `order=3` tylko gdy rzeczywiście widzisz kubiczną formę

3. **ICP registration**:
   - Wymaga dobrze nakładających się regionów
   - Dla dużych różnic użyj najpierw `'correlation'`
   - Może być wolniejszy dla dużych powierzchni (subsamplinguje do 5000 punktów)

4. **Interpolation order**:
   - `order=0` (nearest): najszybsza, najmniej smooth
   - `order=1` (linear): dobry kompromis
   - `order=3` (cubic): najgładsza, może wprowadzić artefakty przy NaN

---

## Źródła

Funkcje adaptowane z projektu **EFS-toolbox** (Enhanced Fracture Surface Toolbox):
- `efs/filters/` - advanced filtering
- `efs/preprocess/` - morphology and leveling  
- Zgodność z ISO 25178 (form removal, filtering)

---

## Zobacz przykłady

Pełne działające przykłady znajdują się w pliku
[`examples/advanced_processing.py`](../examples/advanced_processing.py).

```bash
python examples/advanced_processing.py
```
