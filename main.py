import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from frasta.gui import MainWindow

def set_logger():
    import logging
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("frasta").setLevel(logging.DEBUG)

def run():
    set_logger()
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QtCore.Qt, "AA_UseDesktopOpenGL"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseDesktopOpenGL, True)
    if hasattr(QtWidgets.QApplication, "setHighDpiScaleFactorRoundingPolicy") and hasattr(
        QtCore.Qt,
        "HighDpiScaleFactorRoundingPolicy",
    ):
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    surface_format = QtGui.QSurfaceFormat()
    surface_format.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
    surface_format.setVersion(2, 1)
    surface_format.setProfile(QtGui.QSurfaceFormat.NoProfile)
    surface_format.setDepthBufferSize(24)
    surface_format.setStencilBufferSize(8)
    surface_format.setSamples(0)
    surface_format.setSwapBehavior(QtGui.QSurfaceFormat.DoubleBuffer)
    QtGui.QSurfaceFormat.setDefaultFormat(surface_format)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()
