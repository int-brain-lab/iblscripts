from pathlib import Path
from functools import partial
import warnings
import argparse

import yaml
from one.api import ONE
from PyQt5 import QtWidgets, QtCore, uic

PROCEDURES = ['Behavior training/tasks',
              'Ephys recording with acute probe(s)',
              'Ephys recording with chronic probe(s)',
              'Fiber photometry',
              'Imaging',
              'Optical stimulation']

DEFAULT_PROJECTS = ['practice', 'ibl_neuropixel_brainwide_01']


class MainWindow(QtWidgets.QMainWindow):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]

    @staticmethod
    def _get_or_create(title=None, **kwargs):
        av = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         MainWindow._instances()), None)
        if av is None:
            av = MainWindow(**kwargs)
            av.setWindowTitle(title)
        return av

    def __init__(self, subject, user, session_path=None, one=None):
        if not subject or not isinstance(subject, str):
            raise ValueError(f'Invalid subject: "{subject}"')
        super(MainWindow, self).__init__()
        uic.loadUi(Path(__file__).parent.joinpath('project_protocol_form.ui'), self)
        self.settings = QtCore.QSettings('int-brain-lab', 'project_protocol_form')
        self.subject_settings = QtCore.QSettings('int-brain-lab', f'project_protocol_form_{subject}')
        self.session_path = session_path
        self.subject = subject

        # Load the previous experiment description file for this subject
        self.previously_selected_description_path = self.subject_settings.value('selected_description_path') or ''
        self.session_info = self.subject_settings.value('selected_description') or {}
        prev_projects = self.session_info.get('projects', [])
        prev_procedures = self.session_info.get('projects', [])

        try:
            one = one or ONE()
            users = list({user, one.alyx.user})
            projects = one.alyx.rest('projects', 'list')
            self.projects = [p['name'] for p in projects]
            self.user_projects = [p['name'] for p in projects if any(u in p['users'] for u in users)]
            self.settings.setValue('projects', self.projects)
            self.settings.setValue('user_projects', self.user_projects)
        except ConnectionError:
            # If we can't connect to alyx see if we can get the projects from the previously stored settings
            self.projects = self.settings.value('projects')
            self.user_projects = self.settings.value('user_projects') or []

            if self.projects is None:
                # If these are None then see if we can get the projects from either the previously selected projects
                # or the user projects

                self.projects = prev_projects or self.settings.value('user_projects')
                if self.projects is None:
                    # If still None resort to the default projects
                    self.projects = DEFAULT_PROJECTS

        self.descriptionPath.setText(self.previously_selected_description_path)

        self.descriptionPath.editingFinished.connect(self.validate_description_path)
        self.saveButton.accepted.connect(self.on_save_button_pressed)
        self.filterprojectButton.clicked.connect(self.on_filter_button_pressed)
        self.browseButton.clicked.connect(self.on_browse_button_pressed)
        self.loadButton.clicked.connect(self.on_load_button_pressed)
        self.validateButton.clicked.connect(self.validate_yaml)
        self.subjectLabel.setText(subject)
        self.projectList.itemChanged.connect(partial(self.on_item_clicked, 'projects'))
        self.procedureList.itemChanged.connect(partial(self.on_item_clicked, 'procedures'))
        self.populate_lists(self.projectList, self.projects, prev_projects)
        self.populate_lists(self.procedureList, PROCEDURES, prev_procedures)
        # Update experiment description text field with previous data
        self.validate_yaml(data=self.session_info)

    @staticmethod
    def populate_lists(listView, options, defaults=()):
        listView.clear()
        for opt in options:
            item = QtWidgets.QListWidgetItem()
            item.setText(opt)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            if opt in defaults:
                item.setCheckState(QtCore.Qt.Checked)
            else:
                item.setCheckState(QtCore.Qt.Unchecked)
            listView.addItem(item)

    def on_item_clicked(self, list_name):
        """Callback for when a list item is (un)ticked"""
        if list_name not in ('projects', 'procedures'):
            raise ValueError(f'Unknown list "{list_name}"')
        list_widget = self.projectList if list_name == 'projects' else self.procedureList
        self.session_info[list_name] = self.get_selected_items(list_widget)
        self.validate_yaml(data=self.session_info)

    @staticmethod
    def get_selected_items(list_view):
        """Return the data of all ticked list items"""
        n_items = list_view.count()
        items = map(list_view.item, range(n_items))
        ticked = filter(lambda x: x.checkState() == QtCore.Qt.Checked, items)
        return list(map(QtWidgets.QListWidgetItem.text, ticked))

    def on_filter_button_pressed(self):
        previously_selected_projects = self.session_info.get('projects', [])
        if self.filterprojectButton.isChecked():
            self.populate_lists(self.projectList, self.user_projects, previously_selected_projects)
        else:
            self.populate_lists(self.projectList, self.projects, previously_selected_projects)

    def on_save_button_pressed(self):
        self.validate_yaml()  # Check the YAML is OK, update session_info field
        self.save_to_yaml()
        self.subject_settings.setValue('selected_description', self.session_info)

        selected_procedures = self.session_info.get('procedures', [])
        if 'Fiber photometry' in selected_procedures:
            self.fp_dialog = FibrePhotometryDialog(self, self.subject, self.session_path)
            self.fp_dialog.open()

        # Should this be the case?
        # self.close()

    def validate_description_path(self):
        """Callback for when experiment description path has been edited"""
        self.previously_selected_description_path = self.descriptionPath.text()

    def on_browse_button_pressed(self):
        prev_path = self.previously_selected_description_path  # From single line editor
        if prev_path and (prev_path := Path(prev_path)).parent.exists():
            browse_path = str(prev_path if prev_path.exists() else prev_path.parent)
        else:
            browse_path = self.settings.value('selected_description') or 'C:\\'
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', browse_path, 'YAML files (*.yml *.yaml)')
        if fileName:
            self.descriptionPath.setText(fileName)
            self.previously_selected_description_path = fileName
            self.on_load_button_pressed()

    def update_lists_from_loaded(self):
        """Update the project/procedure lists with contents of the loaded experiment description file"""
        for list_view, key in ((self.projectList, 'projects'), (self.procedureList, 'procedures')):
            values = self.session_info.get(key, []).copy()  # setCheckState callback will edit
            # First deselect all in list view
            for item in map(list_view.item, range(list_view.count())):
                item.setCheckState(QtCore.Qt.Unchecked)
            # Then update all the necessary items
            for value in map(str.strip, values):
                items = list_view.findItems(value, QtCore.Qt.MatchExactly)
                if not items:  # add to list
                    item = QtWidgets.QListWidgetItem()
                    item.setText(value)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    item.setCheckState(QtCore.Qt.Checked)
                    list_view.addItem(item)
                else:  # Tick item in list
                    assert len(items) == 1
                    items[0].setCheckState(QtCore.Qt.Checked)

    def on_load_button_pressed(self):
        """Callback for when experiment description load button is pressed"""
        if not self.previously_selected_description_path:
            return
        file_path = Path(self.previously_selected_description_path)
        if not file_path.exists():
            QtWidgets.QMessageBox.critical(self, 'File Not Found', f'{file_path} not found!')
            return
        with open(file_path, 'r') as fp:
            self.validate_yaml(data=fp)
        self.update_lists_from_loaded()
        self.subject_settings.setValue('selected_description_path', str(file_path))
        self.subject_settings.setValue('selected_description', self.session_info)

    def validate_yaml(self, *_, data=None):
        """Validate and update yaml data"""
        if data is None:
            data = self.plainTextEdit.toPlainText()
        if isinstance(data, dict):
            self.session_info.update(data)
            self.plainTextEdit.setPlainText(yaml.dump(self.session_info))
        else:
            try:
                self.session_info.update(yaml.safe_load(data))
            except yaml.YAMLError as ex:
                QtWidgets.QMessageBox.critical(self, 'YAML Parse Error', f'{type(ex).__name__} {ex}')
                return
            self.plainTextEdit.setPlainText(yaml.dump(self.session_info))

    def save_to_yaml(self):
        if not self.session_path:
            print(self.session_info)
            warnings.warn('File not written: no session path set')
            return
        session_path = Path(self.session_path)
        session_path.mkdir(parents=True, exist_ok=True)
        print(f'Saving experiment description file to {session_path}')
        with open(session_path / '_ibl_experiment.description.yaml', 'w') as fp:
            yaml.safe_dump(self.session_info, fp)


class FibrePhotometryDialog(QtWidgets.QDialog):

    def __init__(self, qmain, subject, session_path):
        super(FibrePhotometryDialog, self).__init__()
        uic.loadUi(Path(__file__).parent.joinpath('fibrephotometry_form.ui'), self)
        self.qmain = qmain
        self.subject = subject
        self.session_path = session_path
        self.subjectLabel.setText(subject)
        self.previously_selected_fibres = self.qmain.settings.value(f'selected_fibres_{subject}') or []
        self.qmain.populate_lists(self.roiList, [f'ROI_{i}' for i in range(5)], self.previously_selected_fibres)
        self.saveButton.accepted.connect(self.on_save_button_pressed)

    def on_save_button_pressed(self):
        selected_fibers = self.qmain.get_selected_items(self.roiList)
        self.qmain.settings.setValue(f'selected_fibres_{self.subject}', selected_fibers)
        fiber_info = {
            'fiberphotometry': {f'fiber0{fib.split("_")[-1]}': {} for fib in selected_fibers}
        }

        if self.qmain.session_info.get('devices', None):
            self.qmain.session_info['devices'].update(fiber_info)
        else:
            self.qmain.session_info['devices'] = fiber_info

        self.qmain.save_to_yaml(self.qmain.session_info)


if __name__ == '__main__':
    r"""Experimental session parameter GUI.
    A GUI for managing the devices, procedures and projects associated with a given session.

    python experiment_form.py <subject> <user> --session-path <session path>

    Examples:
      python experiment_form.py SW_023 Nate
    """
    # Parse parameters
    parser = argparse.ArgumentParser(description='Integration tests for ibllib.')
    parser.add_argument('subject', help='subject name')
    parser.add_argument('user', help='an Alyx username')
    parser.add_argument('--session-path', '-p',
                        help='a session path in which to save the experiment description file')
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = MainWindow(args.subject, args.user, session_path=args.session_path)
    mainapp.show()
    app.exec_()
