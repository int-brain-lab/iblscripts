"""
Application to perform Fiber Photometry related tasks

TODO:
- use QSettings to store additional widget data
- parse up csv file when adding item to queue
    - create additional pd file for each 'item'
- call ibllib when initiating the transfer

QtSettings values:
    path_last_loaded_csv: str - path to the parent dir of the last loaded csv
    path_server_sessions: str - destination path for local lab server, i.e  \\mainenlab_server\Subjects
    subjects: list[str] = field(default_factory=list) - list of subjects should carry over between sessions
"""
import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from PyQt5 import QtWidgets, QtCore

from qt_designer_util import convert_ui_file_to_py

try:
    UI_FILE = "fiber_photometry_form.ui"
    convert_ui_file_to_py(UI_FILE, UI_FILE[:-3] + "_ui.py")
    from fiber_photometry_form_ui import Ui_MainWindow
except ImportError:
    raise


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Controller
    """
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.model = None

        # init QSettings
        self.settings = QtCore.QSettings("int-brain-lab", "fiber_photometry_form")

        # connect methods to triggers
        self.action_load_csv.triggered.connect(self.load_csv)
        self.action_add_item_to_queue.triggered.connect(self.add_item_to_queue)
        self.action_transfer_items_to_server.triggered.connect(self.transfer_items_to_server)

        # Set status bar message prior to csv file being loaded
        self.status_bar_message = QtWidgets.QLabel(self)
        self.status_bar_message.setText("No CSV file loaded")
        self.statusBar().addWidget(self.status_bar_message)

        # Populate widgets
        self.date_edit.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
        self.populate_widgets_with_dummy_data()  # using dummy data for now

    def populate_widgets_with_dummy_data(self):
        # subject_combo_box
        subject_dummy_data = ["mouse1", "mouse2", "mouse3"]
        self.settings.setValue("subjects", subject_dummy_data)
        for value in self.settings.value("subjects"):
            self.subject_combo_box.addItem(value)

        # session_number
        self.session_number.setText("001")

        # server_path
        self.server_path.addItem("\\\\mainen_lab_server\\Subjects")

    def transfer_items_to_server(self):
        print("transfer button press, call ibllib rsync transfer")

    def add_item_to_queue(self):
        """
        Verifies that all entered values are present. A cleaner implementation is desirable.
        """
        # Pull data
        checked_rois = []
        checked_rois.append("Region0R") if self.cb_region0r.isChecked() else None
        checked_rois.append("Region1G") if self.cb_region1g.isChecked() else None
        checked_rois.append("Region2R") if self.cb_region2r.isChecked() else None
        checked_rois.append("Region3R") if self.cb_region3r.isChecked() else None
        checked_rois.append("Region4R") if self.cb_region4r.isChecked() else None
        checked_rois.append("Region5G") if self.cb_region5g.isChecked() else None
        checked_rois.append("Region6G") if self.cb_region6g.isChecked() else None
        checked_rois.append("Region7R") if self.cb_region7r.isChecked() else None
        checked_rois.append("Region8G") if self.cb_region8g.isChecked() else None

        checked_patches = []
        checked_patches.append("Patch1") if self.cb_patch1.isChecked() else None
        checked_patches.append("Patch2") if self.cb_patch2.isChecked() else None
        checked_patches.append("Patch3") if self.cb_patch3.isChecked() else None

        self.items_to_transfer = {
            "subject": self.subject_combo_box.currentText(),
            "date": self.date_edit.text(),
            "session_number": self.session_number.text(),
            "rois": checked_rois,
            "patches": checked_patches,
            "server_path": self.server_path.currentText()
        }

        stringified_items_to_transfer = self.items_to_transfer["subject"] + " - " + self.items_to_transfer["date"] + " - " + \
                                        self.items_to_transfer["session_number"] + " - "
        for values in self.items_to_transfer["rois"]:
            stringified_items_to_transfer += values + " "
        for values in self.items_to_transfer["patches"]:
            stringified_items_to_transfer += values + " "
        stringified_items_to_transfer += "-> " + self.items_to_transfer["server_path"]

        # Display data that will be added to the queue
        self.item_list_queue.addItem(stringified_items_to_transfer)

    def load_csv(self, file=None):
        """
        Called from file menu of the application, launches a QtWidget.QFileDialog, defaulting to the last known file location
        directory; this information is stored by QtCore.QSettings.

        Parameters
        ----------
        file
            specify file location when performing tests, otherwise a QtWidget.QFileDialog is launched

        """
        if file is None or file is False:
            file, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent=self, caption="Select Raw Fiber Photometry Recording",
                directory=self.settings.value("path_last_loaded_csv"), filter="*.csv")
        if file == "":
            return
        file = Path(file)
        self.settings.setValue("path_last_loaded_csv", str(file.parent))
        self.model = Model(pd.read_csv(file))

        # Change status bar text
        self.status_bar_message.setText(f"CSV file loaded: {file}")
        self.statusBar().addWidget(self.status_bar_message)

        # # Enable actionable widgets now that CSV is loaded
        self.enable_actionable_widgets()

    def enable_actionable_widgets(self, enable: bool = True):
        """
        Enables or disables various widgets to prevent user interaction

        Parameters
        ----------
        enable : bool
            used to determine if we are enabling or disabling the widgets
        """
        self.subject_combo_box.setEnabled(enable)
        self.date_edit.setEnabled(enable)
        self.button_add_item_to_queue.setEnabled(enable)
        self.item_list_queue.setEnabled(enable)
        self.session_number.setEnabled(enable)
        self.cb_region0r.setEnabled(enable)
        self.cb_region1g.setEnabled(enable)
        self.cb_region2r.setEnabled(enable)
        self.cb_region3r.setEnabled(enable)
        self.cb_region4r.setEnabled(enable)
        self.cb_region5g.setEnabled(enable)
        self.cb_region6g.setEnabled(enable)
        self.cb_region7r.setEnabled(enable)
        self.cb_region8g.setEnabled(enable)
        self.cb_patch1.setEnabled(enable)
        self.cb_patch2.setEnabled(enable)
        self.cb_patch3.setEnabled(enable)
        self.server_path.setEnabled(enable)
        self.button_transfer_items_to_server.setEnabled(enable)

@dataclass
class Model:
    """Class to store the necessary data"""
    dataframe: pd.DataFrame

    @property
    def regions(self):
        regions = list(set(self.dataframe.keys()).difference(set(['FrameCounter', 'Timestamp', 'Flags'])))
        regions.sort()
        return regions


def test_model(file_test):
    """
    This test does not require instantiating a GUI, and tests only the model logic
    """
    model = Model(pd.read_csv(file_test))
    assert(model.regions == [
        'Region0R', 'Region1G', 'Region2R', 'Region3R', 'Region4R', 'Region5G', 'Region6G', 'Region7R', 'Region8G'])


def test_controller(fiber_form_test, file_test):
    """
    This requires a GUI instance
    """
    fiber_form_test.load_csv(file=file_test)
    assert (fiber_form_test.model is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fiber Photometry Form")
    parser.add_argument("-t", "--test", default=False, required=False, action="store_true", help="Run tests")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    fiber_form = MainWindow()
    if args.test:
        csv_file = Path("fiber_copy_test_fixture.csv")
        test_model(csv_file)
        test_controller(fiber_form, csv_file)
    else:
        # Disable actionable widgets until CSV is loaded
        fiber_form.enable_actionable_widgets(False)
        fiber_form.show()
        sys.exit(app.exec_())
