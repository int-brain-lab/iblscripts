from pathlib import Path
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QFileDialog

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import ibllib.ephys.ephysqc as ephysqc


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.title = 'PyQt5 matplotlib example - pythonspot.com'
        self.width = 640
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        m = PlotCanvas(self, width=6, height=4)
        session_path = Path(QFileDialog.getExistingDirectory(None, "Select Directory"))
        if str(session_path) == '.':
            sys.exit()
        if session_path.exists():
            try:
                ephysqc.validate_ttl_test(session_path, display=m.axes)
            except ValueError:
                pass
        self.show()


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
