"""
Application to perform Fiber Photometry related tasks
"""
import sys

import qt_designer_util
import PyQt5
from PyQt5 import QtWidgets, QtCore, uic

try:
    UI_FILE = "fiber_photometry_form.ui"
    qt_designer_util.convert_ui_file_to_py(UI_FILE, UI_FILE[:-3] + "_ui.py")
    import fiber_photometry_form_ui
except ImportError:
    raise


class FiberPhotometryForm(QtWidgets.QMainWindow):
    """
    Controller
    """
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args
        kwargs
        """
        super(FiberPhotometryForm, self).__init__(*args, *kwargs)
        # TODO: figure out how to properly load up UI_MainWindow
        self.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    fiber = FiberPhotometryForm()
    sys.exit(app.exec_())