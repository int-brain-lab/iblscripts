"""
Application to perform Fiber Photometry related tasks

Development machine details:
- Ubuntu 22.04
- Anaconda 4.13.0
- Python 3.8
- opencv-python 4.3.0.36
- PyQt5 5.15.7
- ibllib develop branch

QtSettings values:
    subjects: list[str] = field(default_factory=list) - list of subjects should carry over between sessions
    local_data_path: str - destination path for parent data directory on local machine, i.e C:\ibl_fiber_photometry_data\Subjects
    server_data_path: str - destination path for local lab server, i.e  \\mainenlab_server\Subjects

TODO:
    - add run numbers to exported data
    - validation for daq produced tdms when compared to bonsai produced csv file, read in peaks/troughs for TTLs, timestamps, +/- 100 frames
    - validation for selected ROI with headers available in bonsai produced csv file
"""
import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from PyQt5 import QtWidgets, QtCore
from dateutil.relativedelta import relativedelta
from ibllib.atlas import BrainRegions
from ibllib.pipes.misc import rsync_paths

from qt_designer_util import convert_ui_file_to_py

try:  # specify ui file(s) output by Qt Designer, call function to convert to py for runtime efficiency and ease of imports
    # main ui file
    main_ui_file = "fiber_photometry_form.ui"
    convert_ui_file_to_py(main_ui_file, main_ui_file[:-3] + "_ui.py")

    # dialog box ui file
    dialog_box_ui_file = "fiber_photometry_dialog_box.ui"
    convert_ui_file_to_py(dialog_box_ui_file, dialog_box_ui_file[:-3] + "_ui.py")

    # confirm box ui file
    confirm_box_ui_file = "fiber_photometry_confirm_box.ui"
    convert_ui_file_to_py(confirm_box_ui_file, confirm_box_ui_file[:-3] + "_ui.py")

    from fiber_photometry_form_ui import Ui_MainWindow
    from fiber_photometry_dialog_box_ui import Ui_Dialog
    from fiber_photometry_confirm_box_ui import Ui_Dialog as Ui_Confirm
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


class ConfirmBox(QtWidgets.QDialog, Ui_Confirm):
    def __init__(self, parent=None):
        super(ConfirmBox, self).__init__(parent=parent)
        self.setupUi(self)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Controller
    """
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.dialog_box = Dialog()
        self.confirm_box = ConfirmBox()
        self.items_to_transfer = []

        # init QSettings
        self.settings = QtCore.QSettings("int-brain-lab", "fiber_photometry_form")

        # connect triggers to class methods
        self.action_clear_qsetting_values.triggered.connect(self.clear_qsetting_values)
        self.action_remove_old_sessions.triggered.connect(self.remove_old_sessions)
        self.action_add_item_to_queue.triggered.connect(self.add_item_to_queue)
        self.action_transfer_items_to_server.triggered.connect(self.transfer_items_to_server)
        self.action_reset_form.triggered.connect(self.reset_form)

        # Populate default qsetting values for subjects and server_data_path
        self.populate_default_qsetting_values()

        # List of the default Patch Cords and ROIs to display in combo boxes
        self.patch_cord_defaults = ["", "Patch Cord A", "Patch Cord B", "Patch Cord C"]
        self.roi_defaults = [
            "", "Region0R", "Region1G", "Region2R", "Region3R", "Region4R", "Region5G", "Region6G", "Region7R", "Region8G"]
        self.run_numbers = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # hashtag_programming

        # Populate widgets
        self.populate_widgets()

        # Disable widgets until needed
        self.button_transfer_items_to_server.setEnabled(False)

    def remove_old_sessions(self):
        """Calls confirmation box for user, then will remove any session older than six months in the local data folder"""
        self.confirm_box.label.setText(
            "Pressing OK will remove any local sessions older than 6 months. Please be sure this is your intention.")
        if self.confirm_box.exec_() == 1:  # OK button pressed
            # get list of all subjects and create a datetime obj 6 months in the past
            subject_list = os.listdir(FIBER_PHOTOMETRY_DATA_FOLDER)
            six_month_old_date = datetime.now() - relativedelta(months=6)

            for subject in subject_list:
                subject = str(subject)
                # get list of all date directories for a subject
                date_list = os.listdir(Path(FIBER_PHOTOMETRY_DATA_FOLDER).joinpath(subject))

                for date_dir in date_list:
                    date_dir = str(date_dir)
                    date_dir_as_datetime = datetime.strptime(date_dir, "%Y-%m-%d")
                    if date_dir_as_datetime < six_month_old_date:
                        dir_to_remove = Path(FIBER_PHOTOMETRY_DATA_FOLDER).joinpath(subject, date_dir)
                        print(f"Removing {dir_to_remove} ...")
                        try:
                            shutil.rmtree(dir_to_remove)
                        except FileNotFoundError:
                            raise

            self.dialog_box.label.setText(
                "Local sessions older than 6 months have been removed. Review the terminal for details on the operation.")
            self.dialog_box.exec_()
        else:
            self.dialog_box.label.setText("Operation cancelled, local sessions have not been removed.")
            self.dialog_box.exec_()

    def clear_qsetting_values(self):
        self.confirm_box.label.setText(
            "Pressing OK will reset the QSetting values (subject names, local data path, and server data path). Please be sure "
            "this is your intention.")
        if self.confirm_box.exec_() == 1:  # OK button pressed
            self.settings.clear()
            self.dialog_box.label.setText("QSetting values have been cleared, application will now close.")
            self.dialog_box.exec_()
            exit(0)
        else:
            self.dialog_box.label.setText("Operation cancelled, QSetting values have NOT been changed.")
            self.dialog_box.exec_()

    def populate_widgets(self):
        """
        Pulls from stored QSettings values where appropriate, and sets reasonable defaults to others
        """
        # date
        self.date_edit.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))

        # subject
        for value in reversed(self.settings.value("subjects")):  # list the most recent entries first
            self.subject_combo_box.addItem(value)

        # local_data_path
        self.local_data_path.setText(self.settings.value("local_data_path"))

        # server_data_path
        self.server_data_path.setText(self.settings.value("server_data_path"))

        # session_number
        self.session_number.setText("001")

        # patch cord, ROI, run selector combo boxes
        self.patch_cord_selector_01.addItems(self.patch_cord_defaults)
        self.patch_cord_selector_02.addItems(self.patch_cord_defaults)
        self.patch_cord_selector_03.addItems(self.patch_cord_defaults)
        self.roi_selector_01.addItems(self.roi_defaults)
        self.roi_selector_02.addItems(self.roi_defaults)
        self.roi_selector_03.addItems(self.roi_defaults)
        self.run_selector_01.addItems(self.run_numbers)
        self.run_selector_02.addItems(self.run_numbers)
        self.run_selector_03.addItems(self.run_numbers)

    def transfer_items_to_server(self):
        """Transfer queued items to server using ibllib rsync_paths function"""
        print("Transfer button pressed, please wait...")

        if FIBER_PHOTOMETRY_DATA_FOLDER_TEST_REMOTE:  # if var is set, we should be in testing mode
            remote_folder = Path(FIBER_PHOTOMETRY_DATA_FOLDER_TEST_REMOTE)
        else:
            remote_folder = Path(self.server_data_path.text())

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

            # Set server data path to QSettings as the new default if it has changed
            if self.server_data_path.text() is not self.settings.value("server_data_path"):
                self.settings.setValue("server_data_path", self.server_data_path.text())

            # Display dialog box with success message
            self.dialog_box.label.setText("The transfer has completed. Please review the log messages in the terminal for "
                                          "details. Pressing OK will reset the form.")
            self.dialog_box.exec_()

            self.reset_form()

    def validate_brain_area(self) -> bool:
        """
        Attempts to match text input (case-insensitive) into the "Brain Area" QLineEdit field with the acronyms listed in ibllib
        BrainRegions class.

        Returns
        -------
        bool
            True - text input matches the given acronyms
            False - the text does not match the given acronyms
        """

        # TODO: Finish generalizing this check
        # def _private_brain_area_validate(text_input: QtWidgets.QLineEdit) -> bool:
        #     text = text_input.text()
        #     if text != "":
        #         text = "".join(text.split()).lower()
        #         try:
        #             region_index = lower_case_acronym_list.index(text)
        #         except ValueError:
        #             self.dialog_box.label.setText(f"Brain Area text for input {text} could not be validated. Please verify what "
        #                                           f"was typed.")
        #             self.dialog_box.exec_()
        #             return False
        #         text_input.setText(acronym_list[region_index])

        # Check if any text has been input and requires validation
        ba01 = self.brain_area_01.text()
        ba02 = self.brain_area_02.text()
        ba03 = self.brain_area_03.text()
        if ba01 == "" and ba02 == "" and ba03 == "":  # No regions input, nothing to validate
            return True

        # Brain Region
        acronym_list = BrainRegions().acronym.tolist()
        # TODO: simplify with list comprehension
        lower_case_acronym_list = []
        for acronym in acronym_list:
            lower_case_acronym_list.append("".join(acronym.split()).lower())

        # TODO: generalize this
        # Pull out all white spaces and lowercase the input for ease of text comparison
        if ba01 != "":
            ba01 = "".join(ba01.split()).lower()
            try:
                region_index = lower_case_acronym_list.index(ba01)
            except ValueError:
                self.dialog_box.label.setText("Brain Area text for input 01 could not be validated. Please verify what was "
                                              "typed.")
                self.dialog_box.exec_()
                return False
            self.brain_area_01.setText(acronym_list[region_index])

        if ba02 != "":
            ba02 = "".join(ba02.split()).lower()
            try:
                region_index = lower_case_acronym_list.index(ba02)
            except ValueError:
                self.dialog_box.label.setText("Brain Area text for input 02 could not be validated. Please verify what was "
                                              "typed.")
                self.dialog_box.exec_()
                return False
            self.brain_area_02.setText(acronym_list[region_index])

        if ba03 != "":
            ba03 = "".join(ba03.split()).lower()
            try:
                region_index = lower_case_acronym_list.index(ba03)
            except ValueError:
                self.dialog_box.label.setText("Brain Area text for input 03 could not be validated. Please verify what was "
                                              "typed.")
                self.dialog_box.exec_()
                return False
            self.brain_area_03.setText(acronym_list[region_index])

        return True

    def add_item_to_queue(self):
        """Verifies that all entered values are valid and present. Adds item to the queue."""

        # Validations
        if not self.validate_patch_cord_and_roi():  # Check for at least a single ROI has been selected; no duplicate ROIs
            return
        if not self.validate_brain_area():  # check if text input for brain area is valid
            return
        # Validate local_data_path

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

        # Ensure Subject / Date / Session Number are unique
        for item in self.items_to_transfer:
            if queue_path == Path(item["queue_path"]):
                self.dialog_box.label.setText("Subject / Date / Session Number combination already queued.")
                self.dialog_box.exec_()
                return

        # set data file name
        data_file = f"{self.subject_combo_box.currentText()}_data_file.parquet"

        # dst Path for settings.json and bonsai.workflow file
        settings_file = Path(queue_path / "settings.json")
        bonsai_file = Path(queue_path / "bonsai.workflow")

        # Build out item to transfer
        item = {
            "subject": self.subject_combo_box.currentText(),
            "date": self.date_edit.text(),
            "session_number": self.session_number.text(),
            "patch_cord_selection_01": self.patch_cord_selector_01.currentText(),
            "patch_cord_selection_02": self.patch_cord_selector_02.currentText(),
            "patch_cord_selection_03": self.patch_cord_selector_03.currentText(),
            "roi_selection_01": self.roi_selector_01.currentText(),
            "roi_selection_02": self.roi_selector_02.currentText(),
            "roi_selection_03": self.roi_selector_03.currentText(),
            "brain_area_01": self.brain_area_01.text(),
            "brain_area_02": self.brain_area_02.text(),
            "brain_area_03": self.brain_area_03.text(),
            "local_data_path": self.local_data_path.text(),
            "server_data_path": self.server_data_path.text(),
            "queue_path": str(queue_path),
            "data_path": str(data_path),
            "data_file": data_file,
            "settings_file": settings_file.name,
            "bonsai_file": bonsai_file.name
        }
        self.items_to_transfer.append(item)

        try:  # to perform OS write operations for dir structure and settings file
            os.makedirs(queue_path, exist_ok=True)
            settings_file.write_text(json.dumps(item))
            Path(bonsai_file).touch()  # TODO: modify to shutil.copyfile(src, bonsai_file) once we have confirmation on src
        except (OSError, TypeError):
            raise

        # Display data that has been added to the queue
        stringified_item_to_transfer = self.stringify_item_to_transfer(self.items_to_transfer[-1])
        self.prepare_item_for_transfer(stringified_item_to_transfer)

        # Reset patch cord link ROI selectors
        self.reset_patch_cord_roi_combo_boxes_and_brain_area()

        # Validate if transfer button should be enabled or not
        self.validate_transfers_ready()

    def prepare_item_for_transfer(self, text: str):
        """
        Adds the stringified text to the next available text box for a queued item

        Parameters
        ----------
        text
            to be displayed in the form
        """
        if self.item_queue_01.toPlainText() == "":
            self.item_queue_01.setText(text)
        elif self.item_queue_02.toPlainText() == "":
            self.item_queue_02.setText(text)
        elif self.item_queue_03.toPlainText() == "":
            self.item_queue_03.setText(text)
        elif self.item_queue_04.toPlainText() == "":
            self.item_queue_04.setText(text)
        elif self.item_queue_05.toPlainText() == "":
            self.item_queue_05.setText(text)
        elif self.item_queue_06.toPlainText() == "":
            self.item_queue_06.setText(text)
        elif self.item_queue_07.toPlainText() == "":
            self.item_queue_07.setText(text)
        elif self.item_queue_08.toPlainText() == "":
            self.item_queue_08.setText(text)
        elif self.item_queue_09.toPlainText() == "":
            self.item_queue_09.setText(text)
        elif self.item_queue_10.toPlainText() == "":
            self.item_queue_10.setText(text)
            self.button_add_item_to_queue.setDisabled(True)

    def reset_patch_cord_roi_combo_boxes_and_brain_area(self):
        """Sets the default values to the patch cord and roi selector combo boxes"""
        # Patch Cords
        self.patch_cord_selector_01.clear()
        self.patch_cord_selector_01.addItems(self.patch_cord_defaults)
        self.patch_cord_selector_02.clear()
        self.patch_cord_selector_02.addItems(self.patch_cord_defaults)
        self.patch_cord_selector_03.clear()
        self.patch_cord_selector_03.addItems(self.patch_cord_defaults)
        # ROI
        self.roi_selector_01.clear()
        self.roi_selector_01.addItems(self.roi_defaults)
        self.roi_selector_02.clear()
        self.roi_selector_02.addItems(self.roi_defaults)
        self.roi_selector_03.clear()
        self.roi_selector_03.addItems(self.roi_defaults)
        # Brain Area
        self.brain_area_01.clear()
        self.brain_area_02.clear()
        self.brain_area_03.clear()

    def stringify_item_to_transfer(self, item_to_transfer: dict) -> str:
        """
        Method to build out a string representation for the item that we are preparing to transfer

        Parameters
        ----------
        item_to_transfer: dict
            The following keys are extracted: subject, date, session_number, rois, patches, server_data_path
        Returns
        -------
        str
            representation for the item we are preparing to transfer
        """
        return_string = "Subject: " + item_to_transfer["subject"] + " | " + \
                        "Date: " + item_to_transfer["date"] + " | " + \
                        "Session Number: " + item_to_transfer["session_number"] + "\n"
        # Patch Cord Selectors
        if item_to_transfer["patch_cord_selection_01"] != "":
            return_string += "Patch Cord Selection 01: " + item_to_transfer["patch_cord_selection_01"] + "\n"
        if item_to_transfer["patch_cord_selection_02"] != "":
            return_string += "Patch Cord Selection 02: " + item_to_transfer["patch_cord_selection_02"] + "\n"
        if item_to_transfer["patch_cord_selection_03"] != "":
            return_string += "Patch Cord Selection 03: " + item_to_transfer["patch_cord_selection_03"] + "\n"
        # ROI Selectors
        if item_to_transfer["roi_selection_01"] != "":
            return_string += "ROI Selection 01: " + item_to_transfer["roi_selection_01"] + "\n"
        if item_to_transfer["roi_selection_02"] != "":
            return_string += "ROI Selection 02: " + item_to_transfer["roi_selection_02"] + "\n"
        if item_to_transfer["roi_selection_03"] != "":
            return_string += "ROI Selection 03: " + item_to_transfer["roi_selection_03"] + "\n"
        # Brain Areas
        if item_to_transfer["brain_area_01"] != "":
            return_string += "Brain Area 01: " + item_to_transfer["brain_area_01"] + "\n"
        if item_to_transfer["brain_area_02"] != "":
            return_string += "Brain Area 02: " + item_to_transfer["brain_area_02"] + "\n"
        if item_to_transfer["brain_area_03"] != "":
            return_string += "Brain Area 03: " + item_to_transfer["brain_area_03"] + "\n"
        # Local and Server Data Paths
        return_string += "Local Data Path: " + item_to_transfer["local_data_path"] + "\n" + \
                         "Server Data Path: " + item_to_transfer["server_data_path"]
        return return_string

    def validate_transfers_ready(self):
        """Validates every queued item has a csv file attached before enabling transfer button"""
        self.button_transfer_items_to_server.setEnabled(False)
        transfer_ready = True
        for item in self.items_to_transfer:
            data_file_loc = Path(item["queue_path"]) / Path(item["data_file"])
            if not data_file_loc.exists():
                transfer_ready = False
                break
        if transfer_ready:
            self.button_transfer_items_to_server.setEnabled(True)

    def attach_csv(self, queue_item_num: int) -> str:
        """
        ------ DEPRECATED ------
        Attaches CSV file to given queued item value

        Parameters
        ----------
        queue_item_num
            number of queue item slot

        Returns
        -------
        str
            file name that is attached to the queued item
        """
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self, caption="Select Raw Fiber Photometry Recording",
            directory=self.settings.value("last_loaded_csv_path"), filter="*.csv")
        if file == "":
            return "No CSV Loaded"
        file = Path(file)
        self.settings.setValue("last_loaded_csv_path", str(file.parent))

        # read csv file into Model's panda dataframe and output to parquet
        item = self.items_to_transfer[queue_item_num - 1]
        Model(pd.read_csv(file)).dataframe.to_parquet(Path(item["queue_path"]) / Path(item["data_file"]))

        # Attempt to enable transfer button
        self.validate_transfers_ready()

        return f"Attached:\n{file.name}"

    def get_selected_rois(self) -> list:
        """
        Pull data for ROI check box selection

        Returns
        -------
        list
            ROIs that have been selected for the current session

        """
        selected_rois = []
        if self.roi_selector_01.currentText() != "":
            selected_rois.append(self.roi_selector_01.currentText())
        if self.roi_selector_02.currentText() != "":
            selected_rois.append(self.roi_selector_02.currentText())
        if self.roi_selector_03.currentText() != "":
            selected_rois.append(self.roi_selector_03.currentText())
        return selected_rois

    def reset_form(self):
        """Resets the form, called after a successful transfer to server or explicitly by user"""
        # Clear combo boxes
        self.subject_combo_box.clear()
        self.patch_cord_selector_01.clear()
        self.patch_cord_selector_02.clear()
        self.patch_cord_selector_03.clear()
        self.roi_selector_01.clear()
        self.roi_selector_02.clear()
        self.roi_selector_03.clear()

        # Populates widgets with default values
        self.populate_widgets()

        # Clear item_queues
        self.item_queue_01.clear()
        self.item_queue_02.clear()
        self.item_queue_03.clear()
        self.item_queue_04.clear()
        self.item_queue_05.clear()
        self.item_queue_06.clear()
        self.item_queue_07.clear()
        self.item_queue_08.clear()
        self.item_queue_09.clear()
        self.item_queue_10.clear()

        # Cleanup local queue_path files and empty self.items_to_transfer list
        for item in self.items_to_transfer:
            if item["queue_path"]:
                print(f"Deleting {Path(item['queue_path']).parent.parent.parent}")
                try:
                    shutil.rmtree(Path(item["queue_path"]).parent.parent.parent)
                except FileNotFoundError:
                    print(f"{Path(item['queue_path']).parent.parent.parent} was not found. Directory was likely already deleted.")
        self.items_to_transfer = []

        # Disable transfer button
        self.button_transfer_items_to_server.setDisabled(True)

        # Dialog box for reset notification
        self.dialog_box.label.setText("Form has been reset.")
        self.dialog_box.exec_()

    def populate_default_qsetting_values(self):
        """Populate QSettings with default values, used for first run of application or after a qsetting reset"""
        if not self.settings.value("subjects"):
            self.settings.setValue("subjects", ["mouse1"])

        if not self.settings.value("local_data_path") or os.name != "nt":  # os name check for testing on linux with temp dirs
            self.settings.setValue("local_data_path", FIBER_PHOTOMETRY_DATA_FOLDER)

        if not self.settings.value("server_data_path"):
            self.settings.setValue("server_data_path", "\\\\path_to_server\\Subjects")

    def validate_patch_cord_and_roi(self) -> bool:
        """
        Ensure at least a single ROI has been selected and that there are no duplicate ROIs selected. Will cause a dialog box to
        appear if we hit an error state.

        Returns
        -------
        bool
            True if all validations pass, False if any validations fail

        """
        # Ensure at least a single patch cord and a single ROI has been selected
        if ((self.patch_cord_selector_01.currentText() == "") and (self.roi_selector_02.currentText() == "") and
            (self.roi_selector_03.currentText() == "")) or \
                ((self.roi_selector_01.currentText() == "") and (self.roi_selector_02.currentText() == "")
                 and (self.roi_selector_03.currentText() == "")):
            self.dialog_box.label.setText("Patch Cord or ROI selection is missing. Please select at least one Patch Cord and one"
                                          " ROI.")
            self.dialog_box.exec_()
            return False

        # Ensure Patch Cord selection lines up with ROI selection
        if (self.patch_cord_selector_01.currentText() != "" and self.roi_selector_01.currentText() == "") or \
                (self.patch_cord_selector_01.currentText() == "" and self.roi_selector_01.currentText() != "") or \
                (self.patch_cord_selector_02.currentText() != "" and self.roi_selector_02.currentText() == "") or \
                (self.patch_cord_selector_02.currentText() == "" and self.roi_selector_02.currentText() != "") or \
                (self.patch_cord_selector_03.currentText() != "" and self.roi_selector_03.currentText() == "") or \
                (self.patch_cord_selector_03.currentText() == "" and self.roi_selector_03.currentText() != ""):
            self.dialog_box.label.setText("Patch Cord and ROI selections do not match up. Please make sure each Patch Cord "
                                          "selection has an ROI selection.")
            self.dialog_box.exec_()
            return False

        # Ensure there are no duplicate ROIs selected
        duplicates = False
        if self.roi_selector_01.currentText() != "":
            if (self.roi_selector_01.currentText() == self.roi_selector_02.currentText()) \
                    or (self.roi_selector_01.currentText() == self.roi_selector_03.currentText()):
                duplicates = True
        if self.roi_selector_02.currentText() != "":
            if (self.roi_selector_02.currentText() == self.roi_selector_01.currentText()) \
                    or (self.roi_selector_02.currentText() == self.roi_selector_03.currentText()):
                duplicates = True
        if self.roi_selector_03.currentText() != "":
            if (self.roi_selector_03.currentText() == self.roi_selector_01.currentText()) \
                    or (self.roi_selector_03.currentText() == self.roi_selector_02.currentText()):
                duplicates = True

        # Display dialog box to end user about duplicate ROIs
        if duplicates:
            self.dialog_box.label.setText("The same ROI has been selected multiple times. Please select only one ROI per patch "
                                          "cord link.")
            self.dialog_box.exec_()
            return False

        # If all of the above tests pass, return True
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
