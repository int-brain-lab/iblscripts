"""
Application to perform Fiber Photometry related tasks

TODO:
- parse up csv file when adding item to queue
    - create additional pd file for each 'item'
- call ibllib when initiating the transfer

QtSettings values:
    last_loaded_csv_path: str - path to the parent dir of the last loaded csv
    server_path: str - destination path for local lab server, i.e  \\mainenlab_server\Subjects
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
        self.items_to_transfer = []

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
        self.populate_widgets()

    def populate_widgets(self):
        """
        Pulls from stored QSettings values where appropriate, and sets reasonable defaults to others
        """
        # date
        self.date_edit.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
        # subject
        if self.settings.value("subjects") is not None:
            for value in self.settings.value("subjects"):  # TODO: reverse loop to get last entry as default?
                self.subject_combo_box.addItem(value)
        else:
            self.settings.setValue("subjects", ["mouse1"])  # dummy data
        # server_path
        if self.settings.value("path_server_session") is not None:
            self.server_path.setText(self.settings.value("server_path"))
        else:
            self.server_path.setText("\\\\path_to_server\\Subjects")  # dummy data

        # session_number
        self.session_number.setText("001")

    def transfer_items_to_server(self):
        # if subject value in items_to_transfer list dict is not in self.settings.value("subject"), then add it to the settings
        print("transfer button press, call ibllib rsync transfer")
        transfer_success = True

        if transfer_success:
            # Add subject to QSettings if it is not already present
            subject_list = self.settings.value("subjects")
            if self.subject_combo_box.currentText() not in subject_list:
                subject_list.append(self.subject_combo_box.currentText())
                self.settings.setValue("subjects", subject_list)

            # Set server path to QSettings
            self.settings.setValue("server_path", self.server_path.text())

    def add_item_to_queue(self):
        """
        Verifies that all entered values are present. A cleaner implementation is desirable for this function.
        """
        # Pull data for ROIs
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

        # Pull data for patch cord
        checked_patches = []
        checked_patches.append("Patch1") if self.cb_patch1.isChecked() else None
        checked_patches.append("Patch2") if self.cb_patch2.isChecked() else None
        checked_patches.append("Patch3") if self.cb_patch3.isChecked() else None

        # Build out items to transfer list
        self.items_to_transfer.append({
            "subject": self.subject_combo_box.currentText(),
            "date": self.date_edit.text(),
            "session_number": self.session_number.text(),
            "rois": checked_rois,
            "patches": checked_patches,
            "server_path": self.server_path.text()
        })

        stringified_item_to_transfer = self.stringify_items_to_transfer(self.items_to_transfer[-1])

        # Display data that will be added to the queue
        self.item_list_queue.addItem(stringified_item_to_transfer)

    def stringify_items_to_transfer(self, item_to_transfer: dict) -> str:
        """
        Method to build out a string representation for the item that we are preparing to transfer

        Parameters
        ----------
        item_to_transfer: dict
            The following keys are exected: subject, date, session_number, rois, patches, server_path
        Returns
        -------
        str
            representation for the item we are preparing to transfer
        """
        return_string = item_to_transfer["subject"] + " - " + item_to_transfer["date"] + " - " + \
                        item_to_transfer["session_number"] + " - "
        for values in item_to_transfer["rois"]:
            return_string += values + " "
        for values in item_to_transfer["patches"]:
            return_string += values + " "
        return_string += "-> " + item_to_transfer["server_path"]
        return return_string

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
                directory=self.settings.value("last_loaded_csv_path"), filter="*.csv")
        if file == "":
            return
        file = Path(file)
        self.settings.setValue("last_loaded_csv_path", str(file.parent))
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
        # fiber_form.enable_actionable_widgets(False)
        fiber_form.show()
        sys.exit(app.exec_())
