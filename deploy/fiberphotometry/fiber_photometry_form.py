"""
Application to perform Fiber Photometry related tasks

Development machine details:
- Ubuntu 22.04
- Anaconda 4.13.0
- Python 3.8
- opencv-python 4.3.0.36
- PyQt5 5.15.7
- ibllib fiberphotometry branch

QtSettings values:
    subjects: list[str] = field(default_factory=list) - list of subjects should carry over between sessions
    local_data_path: str - destination path for parent data directory on local machine, i.e C:\fp_data
    server_data_path: str - destination path for local lab server, i.e  \\mainenlab_server\Subjects
    local_bkup_path: str - local path mirrors server dir structure, ensures backup exists, i.e. C:\fp_data_bkup\Subjects

TODO:
    - implement validation for daq produced tdms, compared to bonsai/fp produced csv file (ibllib fp keyword error resolution)
    - add FP3002Config.01.xml data to parquet metadata
    - validation for selected ROI with headers available in bonsai produced csv file (modify available ROI combobox selection)
        - run number selection must occur prior to this, modify view to place run selection higher up
    - replace print statements with proper logging
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml
from PyQt5 import QtWidgets, QtCore
from dateutil.relativedelta import relativedelta
from ibllib.atlas import BrainRegions
from ibllib.io.extractors import fibrephotometry as fp_extractor
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

    # imports
    from fiber_photometry_form_ui import Ui_MainWindow
    from fiber_photometry_dialog_box_ui import Ui_Dialog
    from fiber_photometry_confirm_box_ui import Ui_Dialog as Ui_Confirm
except ImportError:
    raise

# Ensure data folders exist for local storage of fiber photometry data
FP_REMOTE_PATH_TEST = None  # used for testing transfer function, TODO: better implementation for testing desired
if os.name == "nt":  # check on OS platform
    FP_LOCAL_DATA_PATH = "D:\\ibl_fp_data"  # folder that will contain the fp data output from bonsai and daq
    FP_LOCAL_BKUP_PATH = "D:\\ibl_fp_data_bkup\\Subjects"  # folder mirroring server dir structure
    FP_LOCAL_QUEUED_PATH = "C:\\Temp\\ibl_fp_data_queued\\Subjects"  # temp folder created for queued items
    try:  # to create local queue data folder
        os.makedirs(FP_LOCAL_BKUP_PATH, exist_ok=True)
        os.makedirs(FP_LOCAL_QUEUED_PATH, exist_ok=True)
    except OSError:
        raise
else:
    import tempfile  # cleaner implementation desired
    # Create temp dir structure
    TEMP_DIR = tempfile.TemporaryDirectory()
    FP_LOCAL_DATA_PATH = TEMP_DIR.name + "/local/fp_data"
    FP_LOCAL_BKUP_PATH = TEMP_DIR.name + "/local/Subjects"
    FP_REMOTE_PATH_TEST = TEMP_DIR.name + "/remote/Subjects"
    FP_LOCAL_QUEUED_PATH = TEMP_DIR.name + "/queued/Subjects"
    try:
        os.makedirs(FP_LOCAL_DATA_PATH, exist_ok=True)
        os.makedirs(FP_LOCAL_BKUP_PATH, exist_ok=True)
        os.makedirs(FP_REMOTE_PATH_TEST, exist_ok=True)
        os.makedirs(FP_LOCAL_QUEUED_PATH, exist_ok=True)
    except OSError:
        raise
    print(f"Not a Windows OS, will only create temp files\nlocal subject dir: {FP_LOCAL_DATA_PATH}\nremote subject dir: "
          f"{FP_REMOTE_PATH_TEST}\nqueued subject dir: {FP_LOCAL_QUEUED_PATH}")

# Used to copy test data, remove once ready for production with proper tests in place, functionality finalized
# -----------------------------------------------------------------------------
TEST_DATA_LOC = Path.home() / "Downloads/FP_Extraction_Prototypes/rigs_data/photometry/2022-09-06/"
TEST_LOCAL_DATE_LOC = Path(FP_LOCAL_DATA_PATH) / Path(str(date.today()))
shutil.copytree(TEST_DATA_LOC, TEST_LOCAL_DATE_LOC)
print("TEST DATA LOADED FROM: ", TEST_DATA_LOC, "\nTEST DATA LOADED TO: ", TEST_LOCAL_DATE_LOC)
# -----------------------------------------------------------------------------


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
        self.runs_available = []
        self.available_data_files = {}
        self.run_selector_index = None
        self.stringified_latest_item = ""

        # init QSettings
        self.settings = QtCore.QSettings("int-brain-lab", "fiber_photometry_form")

        # connect triggers to class methods
        self.action_clear_qsetting_values.triggered.connect(self.clear_qsetting_values)
        self.action_remove_old_sessions.triggered.connect(self.remove_old_sessions)
        self.action_add_item_to_queue.triggered.connect(self.add_item_to_queue)
        self.action_transfer_items_to_server.triggered.connect(self.transfer_items_to_server)
        self.action_reset_form.triggered.connect(self.reset_form)
        self.action_verify_local_data_files.triggered.connect(self.verify_local_data_files)
        self.action_run_selector_updated.triggered.connect(self.run_selector_updated)

        # Populate default qsetting values for subjects and server_data_path
        self.populate_default_qsetting_values()

        # List of the default Patch Cords and ROIs to display in combo boxes
        self.patch_cord_defaults = ["", "Patch Cord A", "Patch Cord B", "Patch Cord C"]
        self.roi_defaults = [
            "", "Region0R", "Region1G", "Region2R", "Region3R", "Region4R", "Region5G", "Region6G", "Region7R", "Region8G"]

        # Populate widgets
        self.populate_widgets()

        # Disable various widgets until needed
        self.enable_insertion_user_input_widgets(enable=False)
        self.button_transfer_items_to_server.setEnabled(False)

    def enable_insertion_user_input_widgets(self, enable: bool = True):
        """Disables the various insertion input widgets to prevent undesired user interaction"""
        # patch cords
        self.patch_cord_selector_01.setEnabled(enable)
        self.patch_cord_selector_02.setEnabled(enable)
        self.patch_cord_selector_03.setEnabled(enable)
        # roi
        self.roi_selector_01.setEnabled(enable)
        self.roi_selector_02.setEnabled(enable)
        self.roi_selector_03.setEnabled(enable)
        # brain area
        self.brain_area_01.setEnabled(enable)
        self.brain_area_02.setEnabled(enable)
        self.brain_area_03.setEnabled(enable)
        # notes
        self.notes_01.setEnabled(enable)
        self.notes_02.setEnabled(enable)
        self.notes_03.setEnabled(enable)
        # run selector
        self.run_selector.setEnabled(enable)
        self.run_selector_display.setEnabled(enable)
        # add item to queue buttons
        self.button_add_item_to_queue.setEnabled(enable)

    def verify_local_data_files(self):
        """
        Evaluates the local data directory to determine the available runs that took place for the subject/date specified and
        allow for additional information to be input.
        """
        # Determine what data files are available for the currently selected date and set self.available_data_files
        self.scan_local_date_folder()
        if self.available_data_files:  # Evaluate the available files
            # Enable the user insertion input widgets
            self.enable_insertion_user_input_widgets(enable=True)

            # TODO: ideal location to populate the ROI combo boxes based on content of the available data files
            # Populate run numbers as strings, starting with 1 instead of 0
            self.runs_available = [str(x + 1) for x in range(len(self.available_data_files["daq_files"]))]
            self.run_selector.addItems(self.runs_available) if self.runs_available else print("Something went wrong identifying "
                                                                                              "run numbers.")

    def run_selector_updated(self):
        """Called when the run_selector combo box text is changed"""
        if self.run_selector.currentText() != "":
            if self.available_data_files:
                self.run_selector_index = int(self.run_selector.currentText()) - 1
                display_str = "DAQ File: "
                display_str += str(self.available_data_files["daq_files"][self.run_selector_index])
                display_str += "\nPhotometry File: "
                display_str += str(self.available_data_files["photometry_files"][self.run_selector_index])
                display_str += "\nFP Configuration File: "  # TODO: wrap into the parquet metadata
                display_str += str(self.available_data_files["fp_config_files"][self.run_selector_index])
                self.run_selector_display.setText(display_str)

    def scan_local_date_folder(self):
        """
        Method to scan the local date directory to find available data files based on what is currently input for the date_edit
        widget. This generates a dict with three lists, referencable by the following keys: "daq_files", "photometry_files", and
        "fp_config_files"
        """
        date_folder = Path(FP_LOCAL_DATA_PATH) / Path(self.date_edit.text())
        if not date_folder.is_dir():
            self.dialog_box.label.setText(f"Date folder was not found:\n{date_folder}")
            self.dialog_box.exec_()
            return
        daq_files = list(date_folder.glob("sync_*.tdms"))
        photometry_files = list(date_folder.glob("raw_photometry*.csv"))
        fp_config_files = list(date_folder.glob("FP3002Config*.xml"))
        if not (len(daq_files) == len(photometry_files) == len(fp_config_files)):
            self.dialog_box.label.setText(f"Number of found output files for DAQ and Photometry do not match.")
            self.dialog_box.exec_()
            return
        # sort the files alphabetically to (hopefully) determine which run was first
        daq_files.sort()
        photometry_files.sort()
        fp_config_files.sort()

        # store available data file locations to dict
        self.available_data_files = {
            "daq_files": daq_files, "photometry_files": photometry_files, "fp_config_files": fp_config_files}

    def remove_old_sessions(self):
        """Calls confirmation box for user, then will remove any session older than six months in the local data folder"""
        self.confirm_box.label.setText(
            "Pressing OK will remove any local sessions older than 6 months. Please be sure this is your intention.")
        if self.confirm_box.exec_() == 1:  # OK button pressed
            # get list of all subjects and create a datetime obj 6 months in the past
            subject_list = os.listdir(FP_LOCAL_DATA_PATH)
            six_month_old_date = datetime.now() - relativedelta(months=6)

            for subject in subject_list:
                subject = str(subject)
                # get list of all date directories for a subject
                date_list = os.listdir(Path(FP_LOCAL_DATA_PATH).joinpath(subject))

                for date_dir in date_list:
                    date_dir = str(date_dir)
                    date_dir_as_datetime = datetime.strptime(date_dir, "%Y-%m-%d")
                    if date_dir_as_datetime < six_month_old_date:
                        dir_to_remove = Path(FP_LOCAL_DATA_PATH).joinpath(subject, date_dir)
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
        self.date_edit.setDate(date.today())
        self.date_edit.setMaximumDate(date.today())  # disallow future date picks
        # subject, list the most recent entries first
        [self.subject_combo_box.addItem(value) for value in reversed(self.settings.value("subjects"))]
        # data paths
        self.local_data_path.setText(self.settings.value("local_data_path"))
        self.server_data_path.setText(self.settings.value("server_data_path"))
        self.local_bkup_path.setText(self.settings.value("local_bkup_path"))
        # session_number
        self.session_number.setText("001")
        # patch cord
        self.patch_cord_selector_01.addItems(self.patch_cord_defaults)
        self.patch_cord_selector_02.addItems(self.patch_cord_defaults)
        self.patch_cord_selector_03.addItems(self.patch_cord_defaults)
        # ROI TODO: this may change if we decide to populate these widgets based on what is available in the data file
        self.roi_selector_01.addItems(self.roi_defaults)
        self.roi_selector_02.addItems(self.roi_defaults)
        self.roi_selector_03.addItems(self.roi_defaults)
        # Run selector
        self.run_selector.addItems([""])

    def transfer_items_to_server(self):
        """Transfer queued items to server using ibllib rsync_paths function"""
        if FP_REMOTE_PATH_TEST:  # if var is set, we should be in testing mode TODO: cleaner implementation desired
            remote_folder = Path(FP_REMOTE_PATH_TEST)
        else:
            remote_folder = Path(self.server_data_path.text())

        try:  # Copy items from queue_data_raw_path to local_bkup_raw_path for every item in queue
            [shutil.copytree(
                item["queue_data_raw_path"], item["local_bkup_raw_path"], dirs_exist_ok=True) for item in self.items_to_transfer]
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
            transfer_success = rsync_paths(item["local_bkup_raw_path"], remote_data_path)
            if not transfer_success:
                self.dialog_box.label.setText("Something went wrong during the transfer, please carefully review log messages in "
                                              "the terminal.")
                self.dialog_box.exec_()

        if transfer_success:
            self.store_qsetting_values()  # saves new subject names, changes to data paths
            self.append_or_create_experiment_description()

            # Display dialog box with success message
            self.dialog_box.label.setText("The transfer has completed. Please review the log messages in the terminal for "
                                          "details. Pressing OK will reset the form.")
            self.dialog_box.exec_()

            self.reset_form()

    def append_or_create_experiment_description(self):
        """
        Append to or creates the experiment description file. Appending will occur if an existing file is found on the local lab
        server. Method will also create a local copy of the file in case anything goes wrong writing to the network location.
        """
        if FP_REMOTE_PATH_TEST:  # if var is set, we should be in testing mode TODO: cleaner implementation desired
            remote_folder = Path(FP_REMOTE_PATH_TEST)
        else:
            remote_folder = Path(self.server_data_path.text())

        for item in self.items_to_transfer:
            # set experiment description file path
            exp_desc_path = Path(
                Path(remote_folder) /
                Path(item["subject"]) /
                Path(item["date"]) /
                Path(item["session_number"]) /
                Path("_ibl_experiment.description.yaml")
            )

            if exp_desc_path.exists():
                # load data from experiment description file into dict
                with open(exp_desc_path, "r") as ed:
                    data = yaml.safe_load(ed)
                # error check that fiber photometry information does not already exist
                if "photometry" in data["devices"]:
                    self.dialog_box.label.setText("The 'photometry' key value is already in the experiment description file "
                                                  "found on the server. Please review log messages in the terminal.")
                    self.dialog_box.exec_()
                    print(f"Unknown state for experiment description file, nothing written to disk. Experiment description file "
                          f"location that requires further investigation: {exp_desc_path}")
                    return
                else:  # append fiber photometry data into data dict
                    data["devices"]["photometry"] = {
                        "collection": "raw_photometry_data",
                        "sync_label": "frame_trigger",
                    }
                    data["procedures"] = data["procedures"].append("Fiber photometry")
                    data["projects"] = data["projects"].append("ibl_fiberphotometry")
                    for i in range(1, 4):
                        if item[f"roi_selection_0{i}"] != "":
                            data["devices"]["photometry"]["regions"] = {
                                item[f"roi_selection_0{i}"]: item[f"patch_cord_selection_0{i}"],
                                "acronym": item[f"brain_area_0{i}"],
                                "notes": item[f"notes_0{i}"]
                            }

            else:  # the experiment description file does not exist on server
                data = {
                    "devices": {
                        "photometry": {
                            "collection": "raw_photometry_data",
                            "sync_label": "frame_trigger",
                        }
                    },
                    "procedures": ["Fiber photometry"],
                    "projects": ["ibl_fiberphotometry"]
                }
                for i in range(1, 4):
                    if item[f"roi_selection_0{i}"] != "":
                        data["devices"]["photometry"]["regions"] = {
                            item[f"roi_selection_0{i}"]: item[f"patch_cord_selection_0{i}"],
                            "acronym": item[f"brain_area_0{i}"],
                            "notes": item[f"notes_0{i}"]
                        }

            # attempt to write the experiment description file locally
            with open((Path(item["local_bkup_raw_path"]).parent / "_ibl_experiment.description.yaml"), "w") as ed:
                try:
                    yaml.safe_dump(data, ed)
                except OSError as msg:
                    print("Something went wrong writing the experiment description file locally.", msg)

            # attempt to (over)write the server file, if file is in use, wait a randomized time period and attempt to write again
            with open(exp_desc_path, "w") as ed:
                exp_desc_write_success = False
                for i in range(0, 3):
                    try:
                        yaml.safe_dump(data, ed)
                        exp_desc_write_success = True
                        break
                    except OSError as msg:
                        print("Could not write experiment description file to server: ", msg, "\nReattempting write operation...")
                        time.sleep(random.randint(0, 5))
                if exp_desc_write_success:
                    print("Experiment description file successfully wrote to server.")
                else:
                    print("Failed to write experiment description file to server after several attempts.")

    def store_qsetting_values(self):
        """Store QSetting values (subject list, server data path, local data path)"""
        # Add subject to QSettings if it is not already present
        subject_list = self.settings.value("subjects")
        if self.subject_combo_box.currentText() not in subject_list:
            subject_list.append(self.subject_combo_box.currentText())
            self.settings.setValue("subjects", subject_list)

        # Store server data path to QSettings as the new default if it has changed
        if self.server_data_path.text() is not self.settings.value("server_data_path"):
            self.settings.setValue("server_data_path", self.server_data_path.text())

        # Store local data path to QSettings as the new default if it has changed
        if self.local_data_path.text() is not self.settings.value("local_data_path"):
            self.settings.setValue("local_data_path", self.local_data_path.text())

    def validate_brain_area(self) -> bool:
        """
        Attempts to match text input (case-insensitive) into the "Brain Area" QLineEdit field with the acronyms listed in ibllib
        BrainRegions class.

        Returns
        -------
        bool
            True - text input matches the given acronyms
            False - no text input or the text does not match any given acronyms
        """
        # Check if any text has been input that requires validation
        if self.brain_area_01.text() == "" and self.brain_area_02.text() == "" and self.brain_area_03.text() == "":
            self.dialog_box.label.setText("No text input into the Brain Area field. Please input a value.")
            self.dialog_box.exec_()
            return False  # No brain area input, nothing to validate

        # Brain Regions acronym list
        acronym_list = BrainRegions().acronym.tolist()
        lower_case_acronym_list = ["".join(x.split()).lower() for x in acronym_list]

        # Simple helper function to match user input with the available acronyms found in the ibllib BrainRegions class
        def _match_user_input_to_atlas_acronym(text_input: QtWidgets.QLineEdit) -> bool:
            text = text_input.text()
            if text != "":
                text = "".join(text.split()).lower()
                try:
                    list_region_index = lower_case_acronym_list.index(text)
                except ValueError:
                    self.dialog_box.label.setText(f"Brain Area text for input {text} could not be validated. Please verify what "
                                                  f"was typed.")
                    self.dialog_box.exec_()
                    return False
                text_input.setText(acronym_list[list_region_index])  # Update the QtWidgets.QLineEdit text field
                return True
            else:
                return True  # No brain area input, nothing to validate

        # Match user input with the available acronyms
        if not _match_user_input_to_atlas_acronym(self.brain_area_01):
            return False
        if not _match_user_input_to_atlas_acronym(self.brain_area_02):
            return False
        if not _match_user_input_to_atlas_acronym(self.brain_area_03):
            return False

        return True

    def add_item_to_queue(self):
        """Verifies that all entered values are valid and present. Adds item to the queue."""

        # Initial validations
        if not self.validate_patch_cord_and_roi():  # check at least a single ROI has been selected; no duplicate ROIs
            return
        if not self.validate_brain_area():  # check if text input for brain area is valid
            return
        if not self.validate_run_selector():  # check if user has selected a run number
            return

        # local directory structures, create to mirror server side .../Subject/Date/SessionNumber/raw_fiber_photometry_data
        local_bkup_raw_path = Path(
            Path(FP_LOCAL_BKUP_PATH) /
            self.subject_combo_box.currentText() /
            self.date_edit.text() /
            self.session_number.text() /
            "raw_photometry_data")
        queue_data_raw_path = Path(
            Path(FP_LOCAL_QUEUED_PATH) /
            self.subject_combo_box.currentText() /
            self.date_edit.text() /
            self.session_number.text() /
            "raw_photometry_data")

        # Ensure Subject / Date / Session Number are unique when compared to previously entered items
        for item in self.items_to_transfer:
            if queue_data_raw_path == Path(item["queue_data_raw_path"]):
                self.dialog_box.label.setText("Subject / Date / Session Number combination already queued.")
                self.dialog_box.exec_()
                return

        # set source/destination data file paths for item
        src_fp_data_file_path = self.available_data_files["photometry_files"][self.run_selector_index]
        src_daq_data_file_path = self.available_data_files["daq_files"][self.run_selector_index]
        src_fp_config_file_path = self.available_data_files["fp_config_files"][self.run_selector_index]
        dst_fp_data_file = queue_data_raw_path / "_neurophotometrics_fpData.raw.pqt"
        dst_daq_data_file = queue_data_raw_path / "_mcc_DAQdata.raw.tdms"
        dst_fp_config_file_path = queue_data_raw_path / "_fp_config.xml"
        item_settings = queue_data_raw_path / "_item_settings.json"

        try:  # to validate timestamps, comparison between FP and DAQ data files
            fp_extractor.check_timestamps(daq_file=src_daq_data_file_path, photometry_file=src_fp_data_file_path)
        except (AssertionError, TypeError) as msg:
            self.dialog_box.label.setText("Validation during the comparison of TTLs between DAQ and Fiber Photometry produced "
                                          "csv file has failed. Check the terminal for a more detailed error message.")
            self.dialog_box.exec_()
            print(msg)
            return

        # Build out item to transfer as dict, to be output to json/yml
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
            "notes_01": self.notes_01.text(),
            "notes_02": self.notes_02.text(),
            "notes_03": self.notes_03.text(),
            "run_selection": self.run_selector.currentText(),
            "local_data_path": self.local_data_path.text(),
            "server_data_path": self.server_data_path.text(),
            "local_bkup_path": self.local_bkup_path.text(),
            "queue_data_raw_path": str(queue_data_raw_path),
            "local_bkup_raw_path": str(local_bkup_raw_path),
            "src_fp_data_file_path": str(src_fp_data_file_path),
            "src_daq_data_file_path": str(src_daq_data_file_path),
            "src_fp_config_file_path": str(src_fp_config_file_path),
            "dst_fp_data_file": str(dst_fp_data_file),
            "dst_daq_data_file": str(dst_daq_data_file)
        }
        self.items_to_transfer.append(item)

        try:  # to perform OS write operations for queue_data_raw_path structure and create data files
            os.makedirs(queue_data_raw_path, exist_ok=True)
            Model(pd.read_csv(src_fp_data_file_path)).dataframe.to_parquet(dst_fp_data_file)
            shutil.copy(src_daq_data_file_path, dst_daq_data_file)
            shutil.copy(src_fp_config_file_path, dst_fp_config_file_path)  # TODO: configuration to be moved into parquet metadata
            item_settings.write_text(json.dumps(item))
        except (OSError, TypeError) as msg:
            print(msg)
            return

        # Display data that has been added to the queue
        self.stringify_latest_item()
        self.display_item_in_queue()

        # Reset insertion information widgets
        self.reset_user_insertion_widgets()

        # Enable transfer button if it is not already
        self.button_transfer_items_to_server.setEnabled(True)

    def stringify_latest_item(self):
        """Builds out a string representation for the latest item added to the transfer queue"""
        item = self.items_to_transfer[-1]
        si = "Subject: " + item["subject"] + " | " + "Date: " + item["date"] + " | " + "Session Number: " + \
             item["session_number"] + "\n"
        for i in range(1, 4):
            # Patch Cord Selectors
            if item[f"patch_cord_selection_0{i}"] != "":
                si += f"Patch Cord Selection 0{i}: " + item[f"patch_cord_selection_0{i}"] + "\n"
            # ROI Selectors
            if item[f"roi_selection_0{i}"] != "":
                si += f"ROI Selection 0{i}: " + item[f"roi_selection_0{i}"] + "\n"
            # Brain Areas
            if item[f"brain_area_0{i}"] != "":
                si += f"Brain Area 0{i}: " + item[f"brain_area_0{i}"] + "\n"
            # Notes
            if item[f"notes_0{i}"] != "":
                si += f"Note 0{i}: " + item[f"notes_0{i}"] + "\n"
        # Run Selector
        if item["run_selection"] != "":
            si += "Run Selection: " + item["run_selection"] + "\n"
        # Data Paths
        si += "Local Data Path: " + item["local_data_path"] + "\nServer Data Path: " + item["server_data_path"] + \
              "\nLocal Backup Path: " + item["local_bkup_path"]
        self.stringified_latest_item = si

    def display_item_in_queue(self):
        """Adds the latest stringified item to the next available text box, disables add_item_to_queue when no more available
        'slots'"""
        if self.item_queue_01.toPlainText() == "":
            self.item_queue_01.setText(self.stringified_latest_item)
        elif self.item_queue_02.toPlainText() == "":
            self.item_queue_02.setText(self.stringified_latest_item)
        elif self.item_queue_03.toPlainText() == "":
            self.item_queue_03.setText(self.stringified_latest_item)
        elif self.item_queue_04.toPlainText() == "":
            self.item_queue_04.setText(self.stringified_latest_item)
        elif self.item_queue_05.toPlainText() == "":
            self.item_queue_05.setText(self.stringified_latest_item)
        elif self.item_queue_06.toPlainText() == "":
            self.item_queue_06.setText(self.stringified_latest_item)
        elif self.item_queue_07.toPlainText() == "":
            self.item_queue_07.setText(self.stringified_latest_item)
        elif self.item_queue_08.toPlainText() == "":
            self.item_queue_08.setText(self.stringified_latest_item)
        elif self.item_queue_09.toPlainText() == "":
            self.item_queue_09.setText(self.stringified_latest_item)
        elif self.item_queue_10.toPlainText() == "":
            self.item_queue_10.setText(self.stringified_latest_item)
            self.button_add_item_to_queue.setDisabled(True)

    def reset_user_insertion_widgets(self):
        """Sets the default values to the various insertion information widgets"""
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
        # Notes
        self.notes_01.clear()
        self.notes_02.clear()
        self.notes_03.clear()
        # Run Number
        self.run_selector.clear()
        self.run_selector_display.clear()

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
        # subject
        self.subject_combo_box.clear()
        # patch cord
        self.patch_cord_selector_01.clear()
        self.patch_cord_selector_02.clear()
        self.patch_cord_selector_03.clear()
        # roi
        self.roi_selector_01.clear()
        self.roi_selector_02.clear()
        self.roi_selector_03.clear()
        # brain area
        self.brain_area_01.clear()
        self.brain_area_02.clear()
        self.brain_area_03.clear()
        # notes
        self.notes_01.clear()
        self.notes_02.clear()
        self.notes_03.clear()
        # runs
        self.run_selector.clear()
        self.run_selector_display.clear()
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

        # Cleanup local queue_data_raw_path files, empty self.items_to_transfer list and self.available_data_files
        for item in self.items_to_transfer:
            subject_name_path = Path(item['queue_data_raw_path']).parent.parent.parent
            if item["queue_data_raw_path"]:
                print(f"Deleting {subject_name_path}")
                try:
                    shutil.rmtree(subject_name_path)
                except FileNotFoundError:
                    print(f"{subject_name_path} was not found. Directory was likely already deleted.")
        self.items_to_transfer = []
        self.runs_available = []
        self.available_data_files = {}

        # Disable insertion input widgets and transfer button
        self.enable_insertion_user_input_widgets(enable=False)
        self.button_transfer_items_to_server.setDisabled(True)

        # Dialog box for reset notification
        self.dialog_box.label.setText("Form has been reset.")
        self.dialog_box.exec_()

    def populate_default_qsetting_values(self):
        """Populate QSettings with default values, used for first run of application or after a qsetting reset"""
        if not self.settings.value("subjects"):
            self.settings.setValue("subjects", ["mouse1"])

        if not self.settings.value("local_data_path") or os.name != "nt":  # os name check for testing on linux with temp dirs
            self.settings.setValue("local_data_path", FP_LOCAL_DATA_PATH)

        if not self.settings.value("server_data_path"):
            self.settings.setValue("server_data_path", "\\\\path_to_server\\Subjects")

        if not self.settings.value("local_bkup_path") or os.name != "nt":  # os name check for testing on linux with temp dirs
            self.settings.setValue("local_bkup_path", FP_LOCAL_BKUP_PATH)

    def validate_run_selector(self) -> bool:
        """
        Ensure run_selector is not an empty string (i.e. the user did not select a run)

        Returns
        -------
        bool
            True if a run has been selected, False if no run has been selected
        """
        if self.run_selector.currentText() == "":
            self.dialog_box.label.setText("No value has been selected for 'Run Number'")
            self.dialog_box.exec_()
            return False
        return True

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
