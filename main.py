import sys
from PyQt5 import QtWidgets
from src.frasta_gui import MainWindow

def set_logger():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("src").setLevel(logging.DEBUG)

def run():
    set_logger()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()
