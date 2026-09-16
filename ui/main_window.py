from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget
from ahamdcode.runtime import RUNTIME

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AhamdCode — AhamdCode 0.1 Mini")
        w = QWidget(); layout = QVBoxLayout(w)
        layout.addWidget(QLabel("AhamdCode 0.1 Mini"))
        status = "Loaded" if RUNTIME.model is not None else "Not loaded"
        layout.addWidget(QLabel(f"Model status: {status}"))
        self.setCentralWidget(w)
        self.resize(900, 600)
