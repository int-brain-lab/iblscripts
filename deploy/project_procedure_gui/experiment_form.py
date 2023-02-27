"""A GUI for selecting devices, projects and procedures.

Notes:
    - Currently if a device i
"""
from pathlib import Path
from functools import partial
import warnings
import argparse
import datetime

import yaml
from iblutil.io import params
from one.webclient import AlyxClient
from ibllib.io import session_params
from one.alf.io import next_num_folder
from one.alf.files import get_alf_path
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import Qt

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

    def __init__(self, subject, user=None, session_path=None, computer=None, one=None):

        if not subject or not isinstance(subject, str):
            raise ValueError(f'Invalid subject: "{subject}"')
        super(MainWindow, self).__init__()
        uic.loadUi(Path(__file__).parent.joinpath('project_protocol_form.ui'), self)

        # Load in the remote and local paths from the transfer_params dict
        self.remote_data_path = (params.as_dict(params.read('transfer_params', {})) or {}).get('REMOTE_DATA_FOLDER_PATH', None)
        self.local_data_path = (params.as_dict(params.read('transfer_params', {})) or {}).get('DATA_FOLDER_PATH', None)

        # If the params are not setup we cannot continue
        if not self.remote_data_path or not self.local_data_path:
            raise IOError('No local data path found.  Please run ibllib.pipes.misc.create_basic_transfer_params')

        # Get subject and session path
        self.subject = subject
        self.session_path = session_path or self.get_session_path(self.subject, Path(self.local_data_path))

        # Load the previous experiment description file for this subject
        self.settings = QtCore.QSettings('int-brain-lab', 'project_protocol_form')
        self.subject_settings = QtCore.QSettings('int-brain-lab', f'project_protocol_form_{self.subject}')
        self.previously_selected_description_path = self.subject_settings.value('selected_description_path') or ''
        self.session_info = self.subject_settings.value('selected_description') or {}
        prev_projects = self.session_info.get('projects', [])
        prev_procedures = self.session_info.get('procedures', [])
        prev_devices = self.subject_settings.value('selected_devices') or ()

        try:
            # one = one or ONE(mode='remote')
            alyx = AlyxClient()
            users = list({user, alyx.user})
            projects = alyx.rest('projects', 'list')
            self.projects = [p['name'] for p in projects]
            self.user_projects = [p['name'] for p in projects if any(u in p['users'] for u in users)]
            self.settings.setValue('projects', self.projects)
            self.settings.setValue('user_projects', self.user_projects)
        except (ConnectionError, TimeoutError):
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
        self.clearButton.clicked.connect(self.on_clear_button_pressed)
        self.subjectLabel.setText(subject)
        self.projectList.itemChanged.connect(partial(self.on_item_clicked, 'projects'))
        self.procedureList.itemChanged.connect(partial(self.on_item_clicked, 'procedures'))
        self.deviceList.itemChanged.connect(self.on_device_clicked)
        self.populate_lists(self.projectList, self.projects, prev_projects)
        self.populate_lists(self.procedureList, PROCEDURES, prev_procedures)

        if computer and Path(__file__).parent.parent.joinpath(f'{computer}pc').exists():
            device_stub_folders = Path(__file__).parent.parent.joinpath(f'{computer}pc').glob('device_stubs')
        else:
            device_stub_folders = Path(__file__).parent.parent.rglob('device_stubs')

        self.device_stub_files = dict()
        for fold in device_stub_folders:
            files = fold.glob('*')
            for file in files:
                self.device_stub_files[file.with_suffix('').name] = file

        self.populate_lists(self.deviceList, [key for key, _ in self.device_stub_files.items()], prev_devices)

        # Load remote devices file
        if Path(self.remote_data_path).joinpath('remote_devices.yaml').exists():
            self.populate_table(self.remote_data_path)
            self.remoteDeviceTable.itemChanged.connect(self.on_table_changed)
        else:
            self.centralLayout.removeWidget(self.remoteWidget)

        # Update experiment description text field with previous data
        self.validate_yaml(data=self.session_info)

    @staticmethod
    def get_session_path(subject, local_path):
        date = datetime.datetime.now().date().isoformat()
        num = next_num_folder(local_path.joinpath(subject, date))
        return local_path.joinpath(subject, date, num)

    @staticmethod
    def populate_lists(listView, options, defaults=()):
        listView.clear()
        for opt in options:
            item = QtWidgets.QListWidgetItem()
            item.setText(opt)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if opt in defaults:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            listView.addItem(item)

    def populate_table(self, remote_device_path):
        self.remoteDeviceTable.clear()
        self.remoteDeviceTable.setColumnCount(3)
        self.remoteDeviceTable.setHorizontalHeaderLabels(['Device Name', 'URI', 'Enabled'])

        if not remote_device_path:
            # TODO Elevate to error
            warnings.warn('No remote data path found.  Please run ibllib.pipes.misc.create_basic_transfer_params')
            self.remoteDeviceTable.setRowCount(0)
            return

        remote_devices_file = Path(remote_device_path, 'remote_devices.yaml')
        remote_devices = {}
        if remote_devices_file.exists():
            with open(remote_devices_file, 'r') as fp:
                remote_devices = yaml.safe_load(fp)

        self.remoteDeviceTable.setRowCount(len(remote_devices))
        selected_devices = self.session_info.get('devices', {}).keys()
        # Populate table
        for i, (name, uri) in enumerate(remote_devices.items()):
            device_name = QtWidgets.QTableWidgetItem(name)
            device_name.setFlags(Qt.NoItemFlags)
            self.remoteDeviceTable.setItem(i, 0, device_name)
            device_uri = QtWidgets.QTableWidgetItem(uri)
            device_uri.setFlags(Qt.NoItemFlags)
            self.remoteDeviceTable.setItem(i, 1, device_uri)
            tickbox = QtWidgets.QTableWidgetItem()
            tickbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            value = Qt.Checked if name in selected_devices else Qt.Unchecked
            tickbox.setCheckState(value)
            self.remoteDeviceTable.setItem(i, 2, tickbox)

    def _get_state_map(self,):
        remote_device_states = {}
        for i in range(self.remoteDeviceTable.rowCount()):
            key = self.remoteDeviceTable.item(i, 0).text()
            value = self.remoteDeviceTable.item(i, 2).checkState() == Qt.Checked
            remote_device_states[key] = value
        return remote_device_states

    def on_item_clicked(self, list_name):
        """Callback for when a list item is (un)ticked"""
        if list_name not in ('projects', 'procedures'):
            raise ValueError(f'Unknown list "{list_name}"')
        list_widget = self.projectList if list_name == 'projects' else self.procedureList
        self.session_info[list_name] = self.get_selected_items(list_widget)
        self.validate_yaml(data=self.session_info)

    @staticmethod
    def clear_list(list):
        n_items = list.count()
        # temporarily block signals to device list widget
        list.blockSignals(True)
        for i in range(n_items):
            item = list.item(i)
            item.setCheckState(Qt.Unchecked)
        # unblock signals
        list.blockSignals(False)

    def on_device_clicked(self, selected_item):

        # if the selected_item has been deselected
        if selected_item.checkState() == Qt.Unchecked:
            device = selected_item.text().split('_')[0]
            if device == 'sync':
                self.session_info.pop(device, None)
            else:
                self.session_info['devices'].pop(device, None)
        else:

            self.deselect_same_device(selected_item)

            items = self.get_selected_items(self.deviceList)
            for item in items:
                device = item.split('_')[0]

                with open(self.device_stub_files[item], 'r') as fp:
                    stub = yaml.safe_load(fp)

                if device == 'sync':
                    self.session_info[device] = stub[device]
                else:
                    if not self.session_info.get('devices', False):
                        self.session_info['devices'] = {}
                    self.session_info['devices'][device] = stub[device]

        self.validate_yaml(data=self.session_info)

    def deselect_same_device(self, selected_item):
        """Makes sure only one checkbox for each device can be selected"""
        device = selected_item.text().split('_')[0]
        n_items = self.deviceList.count()

        # temporarily block signals to device list widget
        self.deviceList.blockSignals(True)
        for i in range(n_items):
            item = self.deviceList.item(i)
            if item == selected_item:
                continue
            elif device in item.text() and item.checkState() == Qt.Checked:
                item.setCheckState(Qt.Unchecked)
        # unblock signals
        self.deviceList.blockSignals(False)

    def on_table_changed(self):
        """Callback for when remote devices table is edited"""
        remote_devices = self._get_state_map()
        # Initialize devices in experiment description if empty
        if any(remote_devices.values()) and not self.session_info.get('devices', False):
            self.session_info['devices'] = {}
        for device in remote_devices:
            if remote_devices[device] is True:
                uri = next(
                    self.remoteDeviceTable.item(i, 1).text()
                    for i in range(self.remoteDeviceTable.rowCount())
                    if self.remoteDeviceTable.item(i, 0).text() == device
                )
                device_dict = self.session_info['devices'].get(device, {})
                device_dict['URI'] = uri
                self.session_info['devices'][device] = device_dict
            elif device in self.session_info.get('devices', []):
                self.session_info['devices'][device].pop('URI', None)  # Remove remote device flag
                if not self.session_info['devices'][device]:
                    self.session_info['devices'].pop(device)
        self.validate_yaml(data=self.session_info)

    @staticmethod
    def get_selected_items(list_view):
        """Return the data of all ticked list items"""
        n_items = list_view.count()
        items = map(list_view.item, range(n_items))
        ticked = filter(lambda x: x.checkState() == Qt.Checked, items)
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
        self.subject_settings.setValue('selected_devices', self.get_selected_items(self.deviceList))
        # TODO save updated URIs to remote file?

        # Should this be the case?
        # self.close()

    def validate_description_path(self):
        """Callback for when experiment description path has been edited"""
        self.previously_selected_description_path = self.descriptionPath.text()

    def on_clear_button_pressed(self):
        self.session_info = {}
        self.validate_yaml(data=self.session_info)
        self.clear_list(self.projectList)
        self.clear_list(self.procedureList)
        self.clear_list(self.deviceList)

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
                item.setCheckState(Qt.Unchecked)
            # Then update all the necessary items
            for value in map(str.strip, values):
                items = list_view.findItems(value, Qt.MatchExactly)
                if not items:  # add to list
                    item = QtWidgets.QListWidgetItem()
                    item.setText(value)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked)
                    list_view.addItem(item)
                else:  # Tick item in list
                    assert len(items) == 1
                    items[0].setCheckState(Qt.Checked)

        # Update remote devices table
        remote_devices = [self.remoteDeviceTable.item(i, 0).text() for i in range(self.remoteDeviceTable.rowCount())]
        for device in self.session_info.get('devices', []):
            if device in remote_devices:
                uri = self.session_info['devices'][device].get('URI', False)
                state = Qt.Checked if uri else Qt.Unchecked
                i = remote_devices.index(device)
                self.remoteDeviceTable.item(i, 2).setCheckState(state)

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
        self.clear_list(self.deviceList)
        self.subject_settings.setValue('selected_description_path', str(file_path))
        self.subject_settings.setValue('selected_description', self.session_info)
        self.subject_settings.setValue('selected_devices', [])

    def validate_yaml(self, *_, data=None):
        """Validate and update yaml data"""
        if data is None:
            data = self.plainTextEdit.toPlainText()
        if isinstance(data, dict):
            self.session_info.update(data)
            default = {'protocols': None, 'procedures': None}
            self.plainTextEdit.setPlainText(yaml.dump(self.session_info or default))
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
        print('Saving experiment description file')
        session_params.prepare_experiment(Path(get_alf_path(self.session_path)),
                                          acquisition_description=self.session_info, local=Path(self.local_data_path),
                                          remote=Path(self.remote_data_path))


if __name__ == '__main__':
    """Experimental session parameter GUI.
    A GUI for managing the devices, procedures and projects associated with a given session.

    python experiment_form.py <subject> <user> --session-path <session path>

    Examples:
      python experiment_form.py SW_023 Nate
    """
    # Parse parameters
    parser = argparse.ArgumentParser(description='Integration tests for ibllib.')
    parser.add_argument('subject', help='subject name')
    parser.add_argument('--user', '--u', help='an Alyx username')
    parser.add_argument('--computer', '--c', help='Computer e.g behavior, video, ephys')
    parser.add_argument('--session-path', '-p',
                        help='a session path in which to save the experiment description file')
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = MainWindow(args.subject, user=args.user, session_path=args.session_path, computer=args.computer)
    mainapp.show()
    app.exec_()
