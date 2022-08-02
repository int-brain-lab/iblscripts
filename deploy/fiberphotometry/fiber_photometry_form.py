"""
Application to perform Fiber Photometry related tasks

Development machine details:
- Ubuntu 22.04
- Anaconda 4.13.0
- opencv-python 4.3.0.36
- PyQt5 5.15.7
- ibllib widefield2 branch

TODO:
- use shutil.copyfile(src, bonsai_file) to get workflow file into appropriate location
- add error checking for patch cord link/ROI selection between items selected for transfer?
    - verify that this would not be a hindrance

QtSettings values:
    last_loaded_csv_path: str - path to the parent dir of the last loaded csv
    server_path: str - destination path for local lab server, i.e  \\mainenlab_server\Subjects
    subjects: list[str] = field(default_factory=list) - list of subjects should carry over between sessions
"""
import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from PyQt5 import QtWidgets, QtCore
from ibllib.pipes.misc import rsync_paths

from qt_designer_util import convert_ui_file_to_py

try:  # specify ui file(s) output by Qt Designer, call function to convert to py for runtime efficiency and ease of imports
    # main ui file
    main_ui_file = "fiber_photometry_form.ui"
    convert_ui_file_to_py(main_ui_file, main_ui_file[:-3] + "_ui.py")

    # dialog box ui file
    dialog_box_ui_file = "fiber_photometry_dialog_box.ui"
    convert_ui_file_to_py(dialog_box_ui_file, dialog_box_ui_file[:-3] + "_ui.py")

    from fiber_photometry_form_ui import Ui_MainWindow
    from fiber_photometry_dialog_box_ui import Ui_Dialog
except ImportError:
    raise

# Ensure data folders exist for local storage of fiber photometry data
FIBER_PHOTOMETRY_DATA_FOLDER_TEST_REMOTE = None  # used for testing transfer function
if os.name == "nt":  # check on OS platform
    FIBER_PHOTOMETRY_DATA_FOLDER = "C:\\ibl_fiber_photometry_data\\Subjects"
    FIBER_PHOTOMETRY_DATA_FOLDER_QUEUED = "C:\\Temp\\ibl_fiber_photometry_data_queued"
    try:  # to create local data folder
        os.makedirs(FIBER_PHOTOMETRY_DATA_FOLDER, exist_ok=True)
        os.makedirs(FIBER_PHOTOMETRY_DATA_FOLDER_QUEUED, exist_ok=True)
    except OSError:
        raise
else:
    import tempfile  # cleaner implementation desired
    # Create temp dir structure
    TEMP_DIR = tempfile.TemporaryDirectory()
    FIBER_PHOTOMETRY_DATA_FOLDER = TEMP_DIR.name + "/local/Subjects"
    FIBER_PHOTOMETRY_DATA_FOLDER_TEST_REMOTE = TEMP_DIR.name + "/remote/Subjects"
    FIBER_PHOTOMETRY_DATA_FOLDER_QUEUED = TEMP_DIR.name + "/queued/Subjects"
    try:
        os.makedirs(FIBER_PHOTOMETRY_DATA_FOLDER, exist_ok=True)
        os.makedirs(FIBER_PHOTOMETRY_DATA_FOLDER_TEST_REMOTE, exist_ok=True)
        os.makedirs(FIBER_PHOTOMETRY_DATA_FOLDER_QUEUED, exist_ok=True)
    except OSError:
        raise
    print(f"Not a Windows OS, will only create temp files\nlocal data dir: {FIBER_PHOTOMETRY_DATA_FOLDER}\nremote data dir: "
          f"{FIBER_PHOTOMETRY_DATA_FOLDER_TEST_REMOTE}\nqueued dir: {FIBER_PHOTOMETRY_DATA_FOLDER_QUEUED}")


class Dialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(Dialog, self).__init__(parent=parent)
        self.setupUi(self)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Controller
    """
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.model = None
        self.setupUi(self)
        self.dialog_box = Dialog()
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

        # List of the default ROIs to display in the combo boxes
        self.roi_defaults = [
            "", "Region0R", "Region1G", "Region2R", "Region3R", "Region4R", "Region5G", "Region6G", "Region7R", "Region8G"]

        # Populate widgets
        self.populate_widgets()

        # Disable actionable widgets until CSV is loaded
        self.enable_actionable_widgets(False)
        # Disable transfer button widget until items in are in the queue
        self.button_transfer_items_to_server.setEnabled(False)

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

        # patch cord link to ROI combo boxes
        self.link_a_combo_box.addItems(self.roi_defaults)
        self.link_b_combo_box.addItems(self.roi_defaults)
        self.link_c_combo_box.addItems(self.roi_defaults)

    def transfer_items_to_server(self):
        """Transfer queued items to server using ibllib rsync_paths function"""
        print("Transfer button pressed, please wait...")

        if FIBER_PHOTOMETRY_DATA_FOLDER_TEST_REMOTE:  # if var is set, we should be in testing mode
            remote_folder = Path(FIBER_PHOTOMETRY_DATA_FOLDER_TEST_REMOTE)
        else:
            remote_folder = Path(self.server_path.text())

        try:  # Copy items from queue_path to data_path for every item in queue
            [shutil.copytree(item["queue_path"], item["data_path"], dirs_exist_ok=True) for item in self.items_to_transfer]
        except shutil.Error:
            raise

        # Iterate over the queued items to transfer
        transfer_success = False
        for item in self.items_to_transfer:
            # create remote directory structure .../Subject/Date/SessionNumber
            remote_data_path = Path(
                Path(remote_folder) /
                Path(item["subject"]) /
                Path(item["date"]) /
                Path(item["session_number"]) /
                "raw_fiber_photometry_data")
            try:
                os.makedirs(remote_data_path, exist_ok=True)
            except OSError:
                raise

            # Call rsync from ibllib
            transfer_success = rsync_paths(item["data_path"], remote_data_path)
            if not transfer_success:
                self.dialog_box.label.setText("Something went wrong during the transfer, please carefully review log messages in "
                                              "the terminal.")
                self.dialog_box.exec_()

        if transfer_success:
            # Add subject to QSettings if it is not already present
            subject_list = self.settings.value("subjects")
            if self.subject_combo_box.currentText() not in subject_list:
                subject_list.append(self.subject_combo_box.currentText())
                self.settings.setValue("subjects", subject_list)

            # Set server path to QSettings as the new default if it has changed
            if self.server_path.text() is not self.settings.value("server_path"):
                self.settings.setValue("server_path", self.server_path.text())

            # Display dialog box with success message
            self.dialog_box.label.setText("The transfer has completed. Please review the log messages in the terminal for "
                                          "details. Pressing OK will reset the application, but keep the current CSV loaded.")
            self.dialog_box.exec_()

            self.reset_form()

    def add_item_to_queue(self):
        """Verifies that all entered values are present."""

        if not self.validate_rois():  # Check for at least a single ROI has been selected; no duplicate ROIs
            return

        # local directory structure .../Subject/Date/SessionNumber/raw_fiber_photometry_data
        data_path = Path(
            Path(FIBER_PHOTOMETRY_DATA_FOLDER) /
            self.subject_combo_box.currentText() /
            self.date_edit.text() /
            self.session_number.text() /
            "raw_fiber_photometry_data")
        queue_path = Path(
            Path(FIBER_PHOTOMETRY_DATA_FOLDER_QUEUED) /
            self.subject_combo_box.currentText() /
            self.date_edit.text() /
            self.session_number.text() /
            "raw_fiber_photometry_data")

        # dst Path for data_file.csv file for selected regions
        data_file = Path(queue_path / f"{self.subject_combo_box.currentText()}_data_file.csv")

        # dst Path for settings.json file
        settings_file = Path(queue_path / "settings.json")

        # dst Path for bonsai.workflow file
        bonsai_file = Path(queue_path / "bonsai.workflow")

        # Build out item to transfer
        item = {
            "subject": self.subject_combo_box.currentText(),
            "date": self.date_edit.text(),
            "session_number": self.session_number.text(),
            "link_a_roi": self.link_a_combo_box.currentText(),
            "link_b_roi": self.link_b_combo_box.currentText(),
            "link_c_roi": self.link_c_combo_box.currentText(),
            "server_path": self.server_path.text(),
            "queue_path": str(queue_path),
            "data_path": str(data_path),
            "data_file": data_file.name,
            "settings_file": settings_file.name,
            "bonsai_file": bonsai_file.name
        }
        self.items_to_transfer.append(item)

        try:  # to perform OS write operations
            os.makedirs(queue_path, exist_ok=True)
            self.model.dataframe[self.get_selected_rois()].to_csv(data_file, encoding='utf-8', index=False)
            settings_file.write_text(json.dumps(item))
            Path(bonsai_file).touch()  # TODO: modify to shutil.copyfile(src, bonsai_file) once we have confirmation on src
        except (OSError, TypeError):
            raise

        # Display data that has been added to the queue
        stringified_item_to_transfer = self.stringify_item_to_transfer(self.items_to_transfer[-1])
        self.item_list_queue.addItem(stringified_item_to_transfer)

        # Enable the transfer button
        self.button_transfer_items_to_server.setEnabled(True)

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
                        "Session Number: " + item_to_transfer["session_number"] + "\n" +\
                        "Patch Cord:\n"
        if item_to_transfer["link_a_roi"] != "":
            return_string += "- Link A: " + item_to_transfer["link_a_roi"] + "\n"
        if item_to_transfer["link_b_roi"] != "":
            return_string += "- Link B: " + item_to_transfer["link_b_roi"] + "\n"
        if item_to_transfer["link_c_roi"] != "":
            return_string += "- Link C: " + item_to_transfer["link_c_roi"] + "\n"
        return_string += "Server Path: " + item_to_transfer["server_path"] +\
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

        # Enable actionable widgets now that CSV is loaded
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
        self.link_a_combo_box.setEnabled(enable)
        self.link_b_combo_box.setEnabled(enable)
        self.link_c_combo_box.setEnabled(enable)
        self.server_path.setEnabled(enable)
        self.button_reset_form.setEnabled(enable)

    def get_selected_rois(self) -> list:
        """
        Pull data for ROI check box selection

        Returns
        -------
        list
            ROIs that have been selected for the current session

        """
        selected_rois = []
        if self.link_a_combo_box.currentText() != "":
            selected_rois.append(self.link_a_combo_box.currentText())
        if self.link_b_combo_box.currentText() != "":
            selected_rois.append(self.link_b_combo_box.currentText())
        if self.link_c_combo_box.currentText() != "":
            selected_rois.append(self.link_c_combo_box.currentText())
        return selected_rois

    def reset_form(self):
        """Resets the form in case mistakes were made"""
        # Clear combo boxes
        self.subject_combo_box.clear()
        self.link_a_combo_box.clear()
        self.link_b_combo_box.clear()
        self.link_c_combo_box.clear()

        # Populates widgets with default values
        self.populate_widgets()

        # Clear item_list_queue
        self.item_list_queue.clear()

        # Cleanup local queue_path files and empty self.items_to_transfer list
        for item in self.items_to_transfer:
            if item["queue_path"]:
                print(f"Deleting {Path(item['queue_path']).parent}")
                shutil.rmtree(Path(item["queue_path"]).parent)
        self.items_to_transfer = []

        # Disable transfer button
        self.button_transfer_items_to_server.setDisabled(True)

        # Dialog box for reset notification
        self.dialog_box.label.setText("Form has been reset. CSV file is still loaded.")
        self.dialog_box.exec_()

    def populate_default_subjects_and_server_path(self):
        """Populate QSettings with default values, typically for a first run"""
        if not self.settings.value("subjects"):
            self.settings.setValue("subjects", ["mouse1"])

        if not self.settings.value("server_path"):
            self.settings.setValue("server_path", "\\\\path_to_server\\Subjects")

    def validate_rois(self) -> bool:
        """
        Ensure at least a single ROI has been selected and that there are no duplicate ROIs selected. Will cause a dialog box to
        appear if we hit an error state.

        Returns
        -------
        bool
            True if all validations pass, False if any validations fail

        """
        # Ensure at least a single ROI has been selected
        if (self.link_a_combo_box.currentText() == "") \
                and (self.link_b_combo_box.currentText() == "") \
                and (self.link_c_combo_box.currentText() == ""):
            # Display dialog box to end user regarding no ROI selected
            self.dialog_box.label.setText("No ROIs selected. Please select at least one ROI.")
            self.dialog_box.exec_()
            return False

        # Ensure there are no duplicate ROIs selected
        duplicates = False
        if self.link_a_combo_box.currentText() != "":
            if (self.link_a_combo_box.currentText() == self.link_b_combo_box.currentText()) \
                    or (self.link_a_combo_box.currentText() == self.link_c_combo_box.currentText()):
                duplicates = True
        if self.link_b_combo_box.currentText() != "":
            if (self.link_b_combo_box.currentText() == self.link_a_combo_box.currentText()) \
                    or (self.link_b_combo_box.currentText() == self.link_c_combo_box.currentText()):
                duplicates = True
        if self.link_c_combo_box.currentText() != "":
            if (self.link_c_combo_box.currentText() == self.link_a_combo_box.currentText()) \
                    or (self.link_c_combo_box.currentText() == self.link_b_combo_box.currentText()):
                duplicates = True

        # Disply dialog box to end user about duplicate ROIs
        if duplicates:
            self.dialog_box.label.setText("The same ROI has been selected multiple times. Please select only one ROI per patch "
                                          "cord link.")
            self.dialog_box.exec_()
            return False
        return True


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
        fiber_form.show()
        sys.exit(app.exec_())
