# Quick Start - GUI Advanced Processing

Ten krótki przewodnik pokazuje jak używać nowych funkcji przetwarzania w GUI.

## Menu "Processing"

Po uruchomieniu FRASTA-toolbox zobaczysz nowe menu między "Scan Actions" a "Tools":

```
+-------------------------------------------------+
| File | Edit | Scan Actions | Processing | Tools|
|                                  |               |
|                          +-------v--------------+|
|                          | Advanced Filtering...||
|                          | Morphology & Leveling||
|                          | Geometric Transforms.||
|                          | ---------------------||
|                          | Auto-Register...     ||
|                          +----------------------+|
+-------------------------------------------------+
```

## Przykład 1: Filtrowanie bilateralne

**Cel**: Wygładzenie powierzchni z zachowaniem krawędzi pęknięcia

```
1. Wczytaj skan (File -> Open)
2. Kliknij: Processing -> Advanced Filtering
3. W oknie dialogowym:
   +--------------------------------+
   | Advanced Filtering             |
   +--------------------------------+
   | Filter Type:                   |
   | +--------------------------+   |
   | | Bilateral Filter       v |   |
   | +--------------------------+   |
   |                                |
   | Parameters:                    |
   | Spatial Sigma: [5.0   ] px    |
   | Range Sigma:   [10.0  ]       |
   |                                |
   | info Edge-preserving smoothing   |
   |                                |
   |         [ OK ] [ Cancel ]      |
   +--------------------------------+
4. Kliknij OK
5. Poczekaj (kursor zmieni się na timer)
6. Pojawi się komunikat: "Filter applied successfully!"
```

**Rezultat**: Skan będzie wygładzony, ale krawędzie pęknięć zachowane.

---

## Przykład 2: Poziomowanie powierzchni

**Cel**: Usunięcie pochylenia (tilt) z powierzchni

```
1. Processing -> Morphology & Leveling
2. W oknie dialogowym:
   +--------------------------------+
   | Morphology & Leveling          |
   +--------------------------------+
   | Operation:                     |
   | +--------------------------+   |
   | | Level by Plane (Robust) v|   |
   | +--------------------------+   |
   |                                |
   | Parameters:                    |
   | Max Iterations: [1000    ]    |
   | Inlier Threshold: [10.0  ] nm |
   |                                |
   | info RANSAC robust plane fitting |
   |                                |
   | [x] Show preview                 |
   |                                |
   |         [ OK ] [ Cancel ]      |
   +--------------------------------+
3. Kliknij OK
```

**Rezultat**: Powierzchnia będzie wypoziomowana (usunięty tilt).

---

## Przykład 3: Obrót powierzchni

**Cel**: Obrócenie powierzchni o 45 deg

```
1. Processing -> Geometric Transforms
2. W oknie dialogowym:
   +--------------------------------+
   | Geometric Transforms           |
   +--------------------------------+
   | Transform Type:                |
   | +--------------------------+   |
   | | Rotate Grid            v |   |
   | +--------------------------+   |
   |                                |
   | Parameters:                    |
   | Angle: [45.0  ] degrees       |
   | Interpolation: [Cubic      v] |
   |                                |
   | info Rotate grid by angle        |
   |                                |
   |         [ OK ] [ Cancel ]      |
   +--------------------------------+
3. Kliknij OK
4. Komunikat:
   "Transform applied successfully!
    New shape: (512, 512)
    Pixel size: 1.000 x 1.000"
```

---

## Przykład 4: Automatyczna rejestracja

**Cel**: Automatyczne dopasowanie dwóch powierzchni pęknięcia

```
1. Wczytaj dwa skany (będą w osobnych zakładkach)
2. Processing -> Auto-Register Surfaces
3. W oknie dialogowym:
   +--------------------------------+
   | Automatic Surface Registration |
   +--------------------------------+
   | Select Surfaces:               |
   | Reference: [Scan 1          v] |
   | Moving:    [Scan 2          v] |
   |                                |
   | Registration Method:           |
   | +--------------------------+   |
   | | ICP (Iterative Closest   |   |
   | | Point, 3D rigid)       v |   |
   | +--------------------------+   |
   |                                |
   | info Moving surface will be       |
   |   transformed to match         |
   |   reference surface.           |
   |                                |
   |         [ OK ] [ Cancel ]      |
   +--------------------------------+
4. Kliknij OK
5. Komunikat:
   "Registration completed!
    Method: ICP
    Translation: (-3.0, -5.0) pixels
    Rotation: 2.34 deg
    RMSE: 0.2500"
```

**Rezultat**: Zakładka "Scan 2" zostanie zaktualizowana - powierzchnia będzie dopasowana do "Scan 1".

---

## Wskazówki

### Kolejność operacji
Zalecana kolejność przetwarzania:

```
1. Wczytaj dane (File -> Open)
   v
2. Usuń outliers (Scan Actions -> Remove holes and outliers)
   v
3. Poziomuj (Processing -> Morphology & Leveling)
   v
4. Filtruj (Processing -> Advanced Filtering)
   v
5. Obróć jeśli potrzeba (Processing -> Geometric Transforms)
   v
6. Zapisz (File -> Save current scan)
```

### Przyciski paska narzędzi

Na pasku narzędzi dostępne są przyciski dla głównych operacji przetwarzania:

- Advanced Filtering
- Morphology and Leveling
- Geometric Transforms

Najedź kursorem na przycisk, żeby zobaczyć tooltip.

### Skróty klawiszowe

Obecnie brak domyślnych skrótów. Możesz dodać w `main_window.py`:

```python
self.actions["filter"].setShortcut("Ctrl+Shift+F")
self.actions["morphology"].setShortcut("Ctrl+Shift+L")
self.actions["transform"].setShortcut("Ctrl+Shift+T")
```

---

## Ważne

1. **Brak cofnij (undo)** - Operacje modyfikują skan bezpośrednio. Zapisz kopię przed przetwarzaniem!
2. **Długie operacje** - Filtr bilateralny może trwać 5-10 sekund dla dużych skanów
3. **Pixel size** - Pamiętaj o zachowaniu prawidłowych rozmiarów pikseli
4. **NaN values** - Wszystkie funkcje obsługują NaN (brakujące dane)

---

## Rozwiązywanie problemów

### Problem: "No data" warning
**Rozwiązanie**: Najpierw wczytaj skan (File -> Open)

### Problem: "Not enough scans" (rejestracja)
**Rozwiązanie**: Wczytaj co najmniej 2 skany do osobnych zakładek

### Problem: Długi czas przetwarzania
**Rozwiązanie**: 
- Filtr bilateralny jest powolny - użyj mniejszych sigma
- Rescale do mniejszej rozdzielczości przed przetwarzaniem

### Problem: Błąd "Failed to apply..."
**Rozwiązanie**: 
- Sprawdź console output (szczegóły błędu)
- Upewnij się że dane są prawidłowe
- Sprawdź czy scipy i scikit-learn są zainstalowane

---

## Więcej informacji

- **Szczegółowa dokumentacja**: [GUI_INTEGRATION.md](GUI_INTEGRATION.md)
- **API dokumentacja**: [ADVANCED_PROCESSING.md](ADVANCED_PROCESSING.md)
- **Przykłady Python**: [examples/](../examples/)
- **Quick reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

---

**Gotowe do użycia!** 

Uruchom: `python main.py` i zacznij eksperymentować z nowymi funkcjami.
