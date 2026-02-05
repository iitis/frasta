# FRASTA-toolbox

FRASTA-toolbox is an open-source desktop application for fracture-surface topography analysis based on the FRASTA (Fracture Surface Topography Analysis) methodology. The software supports interactive import, preprocessing, alignment, and comparative analysis of opposing fracture surfaces represented as structured 3D grids.

The toolbox provides tools for masking, interpolation-based hole filling, manual surface alignment with live difference maps, and cross-sectional profile analysis with synchronized 2D and 3D visualization. It is designed to support reproducible fracture-surface analysis workflows and to translate established FRASTA concepts into a practical, accessible research tool.

FRASTA-toolbox is implemented in Python using PyQt5 and pyqtgraph, and is intended for use in materials science, fracture mechanics, tribology, biomedical engineering, and related research domains.

## Configuration

### Windows

* create virtual environment:
`python -m venv .venv`

* activate:
`.venv\Scripts\activate.bat`

* instal packages:
`.venv\Scripts\pip.exe install -r requirements.txt`

* generating of requirements.txt:
`.venv\Scripts\pip.exe freeze > requirements.txt`

### Linux

* create virtual environment:
`python -m venv .venv`

* activate:
`sh .venv/bin/activate`

* instal packages:
`./.venv/bin/pip install -r requirements.txt`

* generating of requirements.txt:
`./.venv/bin/pip freeze > requirements.txt`

## Other useful commands:

* creating distribution package:
`./.venv/bin/python -m PyInstaller --add-data "icons;icons" main.py`

* running tests:
`./.venv/bin/python -m pytest -v -s`

