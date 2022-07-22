"""
Application to perform Fiber Photometry related tasks

TODO:
- call ibllib when initiating the transfer
- Display dialog box with summary of transfers
- Disable regions and patch cords that have already been selected
  - Create reset button/function in case mistakes were made

QtSettings values:
    last_loaded_csv_path: str - path to the parent dir of the last loaded csv
    server_path: str - destination path for local lab server, i.e  \\mainenlab_server\Subjects
    subjects: list[str] = field(default_factory=list) - list of subjects should carry over between sessions
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from PyQt5 import QtWidgets, QtCore

from qt_designer_util import convert_ui_file_to_py

try:  # specify ui file output by Qt Designer, call function to convert to py for efficiency and ease of imports
    UI_FILE = "fiber_photometry_form.ui"
    convert_ui_file_to_py(UI_FILE, UI_FILE[:-3] + "_ui.py")
    from fiber_photometry_form_ui import Ui_MainWindow
except ImportError:
    raise

# Ensure data folders exist for local storage of fiber photometry data
if os.name == "nt":  # check on OS platform
    FIBER_PHOTOMETRY_DATA_FOLDER = "C:\\ibl_fiber_photometry_data\\Subjects"
    try:  # to create local data folder
        os.makedirs(FIBER_PHOTOMETRY_DATA_FOLDER, exist_ok=True)
    except OSError:
        raise
else:
    import tempfile  # cleaner implementation desired
    FIBER_PHOTOMETRY_DATA_FOLDER = tempfile.TemporaryDirectory()
    FIBER_PHOTOMETRY_DATA_FOLDER = FIBER_PHOTOMETRY_DATA_FOLDER.name
    Path(FIBER_PHOTOMETRY_DATA_FOLDER).joinpath("Subjects").mkdir(parents=True)
    print(f"Not a Windows OS, will only create temp files for data output in dir: {FIBER_PHOTOMETRY_DATA_FOLDER}")
    FIBER_PHOTOMETRY_DATA_FOLDER += "/Subjects"


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Controller
    """
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.data_path = None
        self.data_file_path = None
        self.settings_file_path = None
        self.bonsai_file_path = None
        self.model = None
        self.setupUi(self)
        self.items_to_transfer = []

        # init QSettings
        self.settings = QtCore.QSettings("int-brain-lab", "fiber_photometry_form")

        # connect methods to triggers
        self.action_load_csv.triggered.connect(self.load_csv)
        self.action_add_item_to_queue.triggered.connect(self.add_item_to_queue)
        self.action_transfer_items_to_server.triggered.connect(self.transfer_items_to_server)
        self.action_reset_form.triggered.connect(self.reset_form)

        # Set status bar message prior to csv file being loaded
        self.status_bar_message = QtWidgets.QLabel(self)
        self.status_bar_message.setText("No CSV file loaded")
        self.statusBar().addWidget(self.status_bar_message)

        # Populate default qsetting values for subjects and server_path
        self.populate_default_subjects_and_server_path()

        # Populate widgets
        self.populate_widgets()

    def populate_widgets(self):
        """
        Pulls from stored QSettings values where appropriate, and sets reasonable defaults to others
        """
        # date
        self.date_edit.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))

        # subject
        for value in reversed(self.settings.value("subjects")):  # list the most recent entries first
            self.subject_combo_box.addItem(value)

        # server_path
        self.server_path.setText(self.settings.value("server_path"))

        # session_number
        self.session_number.setText("001")

    def transfer_items_to_server(self):
        # if subject value in items_to_transfer list dict is not in self.settings.value("subject"), then add it to the settings
        print("transfer button press, call ibllib rsync transfer")  # TODO: ibllib rsync transfer call

        # Determine if transfer was successful
        transfer_success = True  # TODO: replace with logic for determining successful transfer

        if transfer_success:
            # Add subject to QSettings if it is not already present
            subject_list = self.settings.value("subjects")
            if self.subject_combo_box.currentText() not in subject_list:
                subject_list.append(self.subject_combo_box.currentText())
                self.settings.setValue("subjects", subject_list)

            # Display dialog box with summary of transfers
            # TODO

            # Set server path to QSettings as the new default if it has changed
            if self.server_path.text() is not self.settings.value("server_path"):
                self.settings.setValue("server_path", self.server_path.text())

    def add_item_to_queue(self):
        """Verifies that all entered values are present. A cleaner implementation is desired."""
        checked_rois = self.get_selected_rois()
        checked_patch_cords = self.get_selected_patch_cords()

        # Ensure at least one ROI and one Patch Cord is selected
        if not checked_rois or not checked_patch_cords:
            print("No ROI or no Patch Cord selected, returning")  # TODO: create dialog box for no ROIs or Patch Cord selection
            return

        # Create local directory structure for Subject/Date/SessionNumber/raw_fiber_photometry_data/{subject}_data_file.csv
        self.data_path = Path(
            Path(FIBER_PHOTOMETRY_DATA_FOLDER) /
            self.subject_combo_box.currentText() /
            self.date_edit.text() /
            self.session_number.text() /
            "raw_fiber_photometry_data")

        # dst Path for data_file.csv file for selected regions
        self.data_file_path = Path(self.data_path / f"{self.subject_combo_box.currentText()}_data_file.csv")

        # Build out item to transfer
        item = {
            "subject": self.subject_combo_box.currentText(),
            "date": self.date_edit.text(),
            "session_number": self.session_number.text(),
            "rois": checked_rois,
            "patches": checked_patch_cords,
            "server_path": self.server_path.text(),
            "data_file_loc": self.data_file_path.name
        }
        self.items_to_transfer.append(item)

        # dst Path for settings.json file
        self.settings_file_path = Path(self.data_path / "settings.json")

        # dst Path for bonsai.workflow file
        self.bonsai_file_path = Path(self.data_path / "bonsai.workflow")

        try:  # to perform OS write operations
            os.makedirs(self.data_path, exist_ok=True)
            self.model.dataframe[checked_rois].to_csv(self.data_file_path, encoding='utf-8', index=False)
            self.settings_file_path.write_text(json.dumps(item))
            Path(self.bonsai_file_path).touch()  # TODO: modify to shutil.copyfile(src, self.bonsai_file_path)
        except (OSError, TypeError):
            raise

        # Display data that has been added to the queue
        stringified_item_to_transfer = self.stringify_item_to_transfer(self.items_to_transfer[-1])
        self.item_list_queue.addItem(stringified_item_to_transfer)

        # Disable check boxes
        self.disable_selected_check_boxes()
        self.uncheck_check_boxes()

    def stringify_item_to_transfer(self, item_to_transfer: dict) -> str:
        """
        Method to build out a string representation for the item that we are preparing to transfer

        Parameters
        ----------
        item_to_transfer: dict
            The following keys are extracted: subject, date, session_number, rois, patches, server_path
        Returns
        -------
        str
            representation for the item we are preparing to transfer
        """
        return_string = "Subject: " + item_to_transfer["subject"] + "\n" +\
                        "Date: " + item_to_transfer["date"] + "\n" +\
                        "Session Number: " + item_to_transfer["session_number"] + "\n"
        return_string += "ROIs: "
        for values in item_to_transfer["rois"]:
            return_string += values + " "
        return_string += "\nPatch Cord: "
        for values in item_to_transfer["patches"]:
            return_string += values + " "
        return_string += "\nServer Path: " + item_to_transfer["server_path"] +\
            "\n------------------------------------------------------------"
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

        # read csv file into Model's panda dataframe
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

    def get_selected_rois(self) -> list:
        """Pull data for ROI check box selection"""
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
        return checked_rois

    def get_selected_patch_cords(self) -> list:
        """Pull data for patch cord check box selection"""
        checked_patch_cords = []
        checked_patch_cords.append("Patch1") if self.cb_patch1.isChecked() else None
        checked_patch_cords.append("Patch2") if self.cb_patch2.isChecked() else None
        checked_patch_cords.append("Patch3") if self.cb_patch3.isChecked() else None
        return checked_patch_cords

    def reset_form(self):
        """Resets the form in case mistakes were made"""
        # Populates widgets with default values
        self.populate_widgets()

        # Uncheck all check boxes
        self.uncheck_check_boxes()

        # Re-enable all check boxes
        self.enable_actionable_widgets()

        # Clear item_list_queue
        self.item_list_queue.clear()

        # Cleanup of local files TODO: delete all directories as well
        if self.data_file_path:
            os.unlink(self.data_file_path)
            self.data_file_path = None
        if self.settings_file_path:
            os.unlink(self.settings_file_path)
            self.settings_file_path = None
        if self.bonsai_file_path:
            os.unlink(self.bonsai_file_path)
            self.bonsai_file_path = None
        if self.data_path:
            os.rmdir(self.data_path)
            self.data_path = None

    def populate_default_subjects_and_server_path(self):
        """Populate QSettings with default values, typically for a first run"""
        if not self.settings.value("subjects"):
            self.settings.setValue("subjects", ["mouse1"])

        if not self.settings.value("server_path"):
            self.settings.setValue("server_path", "\\\\path_to_server\\Subjects")

    def disable_selected_check_boxes(self):
        """Disable the selected checkboxes, useful after adding an item to the queue as more than one subject can not have the
        same ROIs or Patch Cords"""
        self.cb_region0r.setDisabled(True) if self.cb_region0r.isChecked() else None
        self.cb_region1g.setDisabled(True) if self.cb_region1g.isChecked() else None
        self.cb_region2r.setDisabled(True) if self.cb_region2r.isChecked() else None
        self.cb_region3r.setDisabled(True) if self.cb_region3r.isChecked() else None
        self.cb_region4r.setDisabled(True) if self.cb_region4r.isChecked() else None
        self.cb_region5g.setDisabled(True) if self.cb_region5g.isChecked() else None
        self.cb_region6g.setDisabled(True) if self.cb_region6g.isChecked() else None
        self.cb_region7r.setDisabled(True) if self.cb_region7r.isChecked() else None
        self.cb_region8g.setDisabled(True) if self.cb_region8g.isChecked() else None
        self.cb_patch1.setDisabled(True) if self.cb_patch1.isChecked() else None
        self.cb_patch2.setDisabled(True) if self.cb_patch2.isChecked() else None
        self.cb_patch3.setDisabled(True) if self.cb_patch3.isChecked() else None
        self.uncheck_check_boxes()

    def uncheck_check_boxes(self):
        """Unchecks all the checkboxes"""
        self.cb_region0r.setChecked(False)
        self.cb_region1g.setChecked(False)
        self.cb_region2r.setChecked(False)
        self.cb_region3r.setChecked(False)
        self.cb_region4r.setChecked(False)
        self.cb_region5g.setChecked(False)
        self.cb_region6g.setChecked(False)
        self.cb_region7r.setChecked(False)
        self.cb_region8g.setChecked(False)
        self.cb_patch1.setChecked(False)
        self.cb_patch2.setChecked(False)
        self.cb_patch3.setChecked(False)

@dataclass
class Model:
    """Class to store the necessary data"""
    dataframe: pd.DataFrame

    @property
    def regions(self):
        regions = list(set(self.dataframe.keys()).difference({'FrameCounter', 'Timestamp', 'Flags'}))
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
    # arg parsing
    parser = argparse.ArgumentParser(description="Fiber Photometry Form")
    parser.add_argument("-t", "--test", default=False, required=False, action="store_true", help="Run tests")
    clear_qsettings_help = "Resets the Qt QSetting values; useful for when there are many unused values in the subject combo box."
    parser.add_argument("-c", "--clear-qsettings", default=False, required=False, action="store_true", help=clear_qsettings_help)
    args = parser.parse_args()

    # Create application and main window
    app = QtWidgets.QApplication(sys.argv)
    fiber_form = MainWindow()

    # determine test situation or production
    if args.test:
        csv_file = Path("fiber_copy_test_fixture.csv")
        test_model(csv_file)
        test_controller(fiber_form, csv_file)
    elif args.clear_qsettings:
        fiber_form.settings.clear()
    else:
        # Disable actionable widgets until CSV is loaded
        fiber_form.enable_actionable_widgets(False)
        fiber_form.show()
        sys.exit(app.exec_())
