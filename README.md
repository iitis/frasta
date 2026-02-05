# Frasta

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
