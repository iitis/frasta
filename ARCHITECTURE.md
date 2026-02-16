# FRASTA-toolbox - Struktura Projektu

## Przegląd Refaktoryzacji

Projekt został zreorganizowany w modułową strukturę, która oddziela odpowiedzialności i ułatwia dalszy rozwój. Stara struktura `src/` została zastąpiona nowym pakietem `frasta/` z wyraźnym podziałem na moduły funkcjonalne.

## Nowa Struktura

```
frasta/
├── __init__.py                 # Główny pakiet
│
├── core/                       # Podstawowe struktury danych
│   ├── __init__.py
│   └── grid_data.py            # Klasa GridData
│
├── processing/                 # Algorytmy analizy danych
│   ├── __init__.py
│   ├── alignment.py            # Wyrównywanie skanów (offset, tilt)
│   ├── filtering.py            # Filtrowanie i usuwanie outlierów
│   └── interpolation.py        # Wypełnianie dziur w danych
│
├── io/                         # Wejście/Wyjście
│   ├── __init__.py
│   ├── loaders.py              # Ładowanie CSV, NPZ, H5
│   └── exporters.py            # Zapisywanie do NPZ, H5
│
├── utils/                      # Narzędzia ogólne
│   ├── __init__.py
│   ├── resources.py            # Rozwiązywanie ścieżek zasobów
│   └── decorators.py           # Dekoratory (measure_time)
│
└── gui/                        # Interfejs graficzny
    ├── __init__.py
    ├── main_window.py          # Główne okno aplikacji
    ├── scan_tab.py             # Widget pojedynczego skanu
    │
    ├── dialogs/                # Okna dialogowe
    │   ├── __init__.py
    │   ├── about.py            # Dialog "O programie"
    │   ├── overlay_viewer.py   # Nakładanie i porównywanie skanów
    │   └── profile_viewer.py   # Analiza profili przekrojowych
    │
    ├── viewers/                # Widoki 3D
    │   ├── __init__.py
    │   ├── grid_3d_viewer.py   # Wizualizacja 3D
    │   ├── limited_gl_view.py  # Widok OpenGL z ograniczeniami
    │   └── lod_surface.py      # Renderowanie LOD
    │
    └── widgets/                # Komponenty GUI
        ├── __init__.py
        └── responsive_infinite_line.py  # Linia nieskończona z throttlingiem
```

## Mapowanie Starych → Nowych Modułów

| Stara lokalizacja | Nowa lokalizacja | Uwagi |
|-------------------|------------------|-------|
| `src/gridData.py` | `frasta/core/grid_data.py` | Tylko klasa GridData |
| `src/helpers.py` (alignment) | `frasta/processing/alignment.py` | Funkcje wyrównywania |
| `src/helpers.py` (filtering) | `frasta/processing/filtering.py` | Filtrowanie i outliers |
| `src/helpers.py` (interpolation) | `frasta/processing/interpolation.py` | Wypełnianie dziur |
| `src/helpers.py` (resource_path) | `frasta/utils/resources.py` | Ścieżki zasobów |
| `src/helpers.py` (measure_time) | `frasta/utils/decorators.py` | Dekoratory |
| `src/frasta_gui.py` (GridWorker) | `frasta/io/loaders.py` | Ładowanie danych |
| `src/frasta_gui.py` (save) | `frasta/io/exporters.py` | Zapisywanie danych |
| `src/frasta_gui.py` (MainWindow) | `frasta/gui/main_window.py` | Główne okno |
| `src/scanTab.py` | `frasta/gui/scan_tab.py` | Widget skanu |
| `src/aboutDialog.py` | `frasta/gui/dialogs/about.py` | Dialog O programie |
| `src/overlayViewer.py` | `frasta/gui/dialogs/overlay_viewer.py` | Nakładanie skanów |
| `src/profileViewer.py` | `frasta/gui/dialogs/profile_viewer.py` | Analiza profili |
| `src/grid3DViewer.py` | `frasta/gui/viewers/grid_3d_viewer.py` | Wizualizacja 3D |
| `src/limitedGLView.py` | `frasta/gui/viewers/limited_gl_view.py` | Widok OpenGL |
| `src/lodSurface.py` | `frasta/gui/viewers/lod_surface.py` | LOD rendering |
| `src/responsiveInfiniteLine.py` | `frasta/gui/widgets/responsive_infinite_line.py` | Widget linii |

## Przykłady Importów

### Stare importy (src/)
```python
from src.gridData import GridData
from src.helpers import fill_holes, remove_relative_offset
from src.frasta_gui import MainWindow
from src.scanTab import ScanTab
```

### Nowe importy (frasta/)
```python
from frasta.core import GridData
from frasta.processing import fill_holes, remove_relative_offset
from frasta.gui import MainWindow, ScanTab
```

## Korzyści z Refaktoryzacji

### 1. **Separacja Odpowiedzialności**
- **core/**: Tylko struktury danych, bez logiki biznesowej
- **processing/**: Algorytmy analityczne, niezależne od GUI
- **io/**: Obsługa I/O oddzielona od logiki aplikacji
- **gui/**: Wszystkie komponenty UI w jednym miejscu

### 2. **Łatwiejsze Testowanie**
- Moduły `processing/` i `io/` można testować bez Qt/GUI
- Jasne granice między modułami ułatwiają mock-owanie
- Mniejsze, bardziej skupione moduły

### 3. **Możliwość Ponownego Użycia**
- Logika przetwarzania dostępna niezależnie od GUI
- Możliwość stworzenia CLI/API bez zmian w core logic
- Łatwiejsze integrowanie w inne projekty

### 4. **Skalowalność**
- Jasna struktura dla nowych funkcjonalności:
  - Nowy algorytm → `processing/`
  - Nowy format pliku → `io/`
  - Nowy widget → `gui/widgets/`
- Łatwiejsze onboarding nowych programistów

### 5. **Czytelność**
- Natychmiastowe zrozumienie, gdzie szukać konkretnej funkcjonalności
- Nazwy modułów opisują ich przeznaczenie
- Mniejsze pliki = łatwiejsza nawigacja

## Migracja Kodu

### Dla deweloperów
Wszystkie importy powinny używać nowej struktury `frasta/`:
```python
# Poprawne importy:
from frasta.core import GridData
from frasta.processing import fill_holes
from frasta.gui import MainWindow
```

### Status Migracji
✅ **Migracja zakończona** - Stary katalog `src/` został usunięty.  
✅ Wszystkie moduły przeniesione do `frasta/`  
✅ Wszystkie testy zaktualizowane  
✅ Aplikacja w pełni funkcjonalna

## Następne Kroki

### Możliwe Rozszerzenia
1. **CLI Module** (`frasta/cli/`) - Interfejs linii poleceń
2. **API Module** (`frasta/api/`) - Programowy dostęp do funkcji
3. **Plugins** (`frasta/plugins/`) - System wtyczek
4. **Export Formats** (`frasta/io/exporters.py`) - Więcej formatów eksportu

### Zalecenia
- Regularnie sprawdzaj spójność dokumentacji z kodem
- Dodawaj testy jednostkowe dla nowych modułów
- Rozważ dodanie type hints (mypy) dla lepszej statycznej analizy
- Aktualizuj `__all__` w plikach `__init__.py` przy dodawaniu nowych funkcji

## Wsparcie

W przypadku pytań dotyczących nowej struktury:
1. Sprawdź ten dokument
2. Zobacz przykłady importów w testach (`tests/`)
3. Przejrzyj pliki `__init__.py` w każdym module


