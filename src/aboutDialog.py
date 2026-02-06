"""About dialog for FRASTA-toolbox application.

This module provides a simple about dialog displaying application information,
version, author, and licensing details.
"""

from PyQt5 import QtWidgets
from PyQt5 import QtCore

class AboutDialog(QtWidgets.QDialog):
    """Dialog window displaying application information and credits.
    
    Shows application name, version, author, copyright, and third-party
    attribution information in a modal dialog.
    """
    
    def __init__(self, parent=None):
        """Initialize the about dialog.
        
        Args:
            parent (QWidget, optional): Parent widget for the dialog. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setMinimumWidth(350)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        # Treść okna
        label = QtWidgets.QLabel("""
        <b>FRASTA-toolbox</b><br>
        <br>
        Author: Dariusz Pojda<br>
        Version: 1.0.0<br>
        <br>
        This software uses icons from <a href='https://icons8.com/icons/set/eyedropper'>Icons8.com</a>.<br>
        <small>Some icons are from Google Material Icons (Apache 2.0) and/or FontAwesome (CC BY 4.0).</small>
        <br><br>
        &copy; 2025-2026 IITiS PAN
        """)
        label.setOpenExternalLinks(True)
        layout.addWidget(label)

        btn = QtWidgets.QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn, alignment=QtCore.Qt.AlignRight)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    dlg = AboutDialog()
    dlg.exec_()
