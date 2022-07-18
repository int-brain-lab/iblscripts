"""
Application to perform Fiber Photometry related tasks

TODO:
- Configure date_edit field to be in YYYY-mm-dd format
- Use QSettings to store widget data
- parse up csv file when adding item to queue
- call ibllib when initiating the transfer

QtSettings values:
    path_fiber_photometry: str - last path of fiber photometry csv file
    path_server_sessions: str - destination path for local lab server, i.e. Y:\Subjects\something
    subjects: list[str] = field(default_factory=list) - list of subjects should carry over between sessions
"""
import sys

from qt_designer_util import convert_ui_file_to_py
from pathlib import Path
from PyQt5 import QtWidgets, QtCore

try:
    UI_FILE = "fiber_photometry_form.ui"
    convert_ui_file_to_py(UI_FILE, UI_FILE[:-3] + "_ui.py")
    from fiber_photometry_form_ui import Ui_MainWindow
except ImportError:
    raise


class FiberPhotometryForm(QtWidgets, UI_FILE):
    """
    Controller
    """
    def __init__(self):
        """

        """
        super(FiberPhotometryForm, self).__init__()
        self.setupUI(self)

        # init QSettings
        self.settings = QtCore.QSettings('int-brain-lab', 'FiberCopy')

        self.action_load_csv.triggered.connect(self.load_csv)

    def load_csv(self, file=None):
        """
        Called from file menu of the application, launches a QtWidget.QFileDialog, defaulting to the last known file location
        directory; this information is stored by QtCore.QSettings.

        Parameters
        ----------
        file
            specify file location when performing tests, otherwise a QtWidget.QFileDialog is launched

        """
        print("load_csv called")
        # if file is None:
        #     file, _ = QtWidgets.QFileDialog.getOpenFileName(
        #         parent=self, caption="Select Raw Fiber Photometry recording",
        #         directory=self.settings.value("path_fiber_photometry"), filter="*.csv")
        # if file == '':
        #     return
        # file = Path(file)
        # self.settings.setValue("path_fiber_photometry", str(file.parent))
        # self.model = Model(pd.read_csv(file))
        #
        # # Change status bar text
        # self.status_bar_message.setText(f"CSV file loaded: {file}")
        # self.statusBar().addWidget(self.status_bar_message)
        #
        # # Enable actionable widgets now that CSV is loaded
        # self.enable_actionable_widgets()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    fiber = FiberPhotometryForm()
    fiber.show()
    sys.exit(app.exec_())