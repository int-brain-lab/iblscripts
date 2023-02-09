"""Integration tests iblscripts deploy.experiment_form GUI."""

import sys
import unittest
import unittest.mock
from pathlib import Path
import tempfile
import json
import yaml

from one.api import ONE
from ibllib import __file__ as ibllib_init  # for use of the experiment description fixture
from ci.tests.base import IntegrationTest, TEST_DB
from deploy.project_procedure_gui.experiment_form import MainWindow

raise unittest.SkipTest('Import / Instantiation of PyQt5 app failed')

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa
from PyQt5.QtTest import QTest  # noqa
from PyQt5.QtCore import Qt, QSettings  # noqa

app = QApplication(sys.argv)


class TestMainForm(IntegrationTest):

    def setUp(self) -> None:
        self.one = ONE(**TEST_DB)
        self.subject = 'ZM_1743'
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.session_path = Path(self.tmpdir.name, self.subject, '2022-01-01', '001')

        # A path to a YAML file to use in tests
        self.yaml_file = Path(ibllib_init).parent.joinpath('tests', 'fixtures', 'io', '_ibl_experiment.description.yaml')
        assert self.yaml_file.exists(), f'missing ibllib test fixture: {self.yaml_file.relative_to(Path(ibllib_init).parent)}'

        # Clear settings
        QSettings('int-brain-lab', 'project_protocol_form').clear()
        QSettings('int-brain-lab', f'project_protocol_form_{self.subject}').clear()

        # Set up remote data params
        param_file = Path(self.tmpdir.name, '.transfer_params')
        with open(param_file, 'w') as fp:
            json.dump({'REMOTE_DATA_FOLDER_PATH': self.tmpdir.name}, fp)
        mock_loc = 'iblutil.io.params.getfile'
        with unittest.mock.patch(mock_loc, return_value=param_file):
            self.form = MainWindow('ZM_1743', 'test_user', session_path=self.session_path, one=self.one)

    def test_user_filter(self):
        """Test that the project filter radio button toggles the project list"""
        self.assertEqual(3, self.form.projectList.count())
        self.form.filterprojectButton.click()
        self.assertEqual(0, self.form.projectList.count())
        self.form.filterprojectButton.click()
        self.assertEqual(3, self.form.projectList.count())

    def test_check_box_updates_yaml_form(self):
        """Test that (de)selecting a project adds or removes the entry from the YAML form"""
        item = self.form.projectList.item(0)
        project = item.text()
        item.setCheckState(Qt.Checked)
        self.assertIn(project, self.form.plainTextEdit.toPlainText(), 'failed to add project to text form')
        self.assertIn(project, self.form.session_info.get('projects', []), 'failed to add to session info')

        # Uncheck
        item.setCheckState(Qt.Unchecked)
        self.assertNotIn(project, self.form.plainTextEdit.toPlainText(), 'failed to add project to text form')
        self.assertNotIn(project, self.form.session_info.get('projects', []), 'failed to add to session info')

        # Also check procedure list
        item = self.form.procedureList.item(0)
        procedure = item.text()
        item.setCheckState(Qt.Checked)
        self.assertIn(procedure, self.form.plainTextEdit.toPlainText(), 'failed to add procedure to text form')
        self.assertIn(procedure, self.form.session_info.get('procedures', []), 'failed to add to session info')

    def test_validate_yaml(self):
        """Test that validate button prompts user on syntax errors"""
        assert QMessageBox not in map(type, app.allWidgets())
        text = self.form.plainTextEdit.toPlainText()
        text_broken = text + '\nfoobar'  # Add invalid YAML syntax
        self.form.plainTextEdit.setPlainText(text_broken)  # Set textbox text
        # Use mock to capture expected call to critical message box
        with unittest.mock.patch(
                'deploy.project_procedure_gui.experiment_form.QtWidgets.QMessageBox.critical'
        ) as message_box:
            QTest.mouseClick(self.form.validateButton, Qt.LeftButton)
            message_box.assert_called()
            (_, title, message), *_ = message_box.call_args
            self.assertEqual('YAML Parse Error', title)
            self.assertIn('foobar', message)

        self.form.plainTextEdit.setPlainText(text)
        QTest.mouseClick(self.form.validateButton, Qt.LeftButton)

    def test_remote_devices(self):
        """Test the population and validation of the remote devices table"""
        # Disconnect callback for now
        self.form.remoteDeviceTable.itemChanged.disconnect()

        # First test behaviour when params are missing
        with self.assertWarns(Warning):
            self.form.populate_table(None)
            self.assertEqual(0, self.form.remoteDeviceTable.rowCount())

        # Check behaviour when remote devices file doesn't exist
        self.form.populate_table(self.tmpdir.name)
        self.assertEqual(0, self.form.remoteDeviceTable.rowCount())

        with open(Path(self.tmpdir.name, 'remote_devices.yaml'), 'w') as fp:
            yaml.dump({'cameras': '192.168.0.1', 'mesoscope': 'ws://86.167.0.224:8888'}, fp)
        self.form.populate_table(self.tmpdir.name)
        self.assertEqual(2, self.form.remoteDeviceTable.rowCount())
        self.assertEqual(3, self.form.remoteDeviceTable.columnCount())

        # Check update of device states
        self.form.remoteDeviceTable.itemChanged.connect(self.form.on_table_changed)
        self.form.remoteDeviceTable.item(0, 2).setCheckState(Qt.Checked)
        uri = self.form.session_info.get('devices', {}).get('cameras', {}).get('URI', None)
        self.assertEqual('192.168.0.1', uri)

    def test_load_yaml(self):
        """Test the loading of a YAML file"""
        self.assertNotIn('sync_label', self.form.plainTextEdit.toPlainText(), 'expected empty YAML form')
        self.form.descriptionPath.setText(str(self.yaml_file))  # Enter path to YAML file
        # The following is set when user switches focus from the path input box
        self.form.previously_selected_description_path = str(self.yaml_file)

        QTest.mouseClick(self.form.loadButton, Qt.LeftButton)
        self.assertIn('sync_label', self.form.plainTextEdit.toPlainText(), 'expected empty YAML form')
        loaded_projects = self.form.session_info.get('projects', [])
        self.assertCountEqual(['ibl_neuropixel_brainwide_01'], loaded_projects)

        # Check saved loaded path in settings
        settings = QSettings('int-brain-lab', f'project_protocol_form_{self.subject}')
        self.assertEqual(self.form.session_info, settings.value('selected_description'))
        self.assertEqual(str(self.yaml_file), settings.value('selected_description_path'))

        # Check that the projects list was updated based on loaded file
        for item in map(self.form.projectList.item, range(self.form.projectList.count())):
            with self.subTest(project=item.text()):
                state = Qt.Checked if item.text() in loaded_projects else Qt.Unchecked
                self.assertEqual(state, item.checkState(), f'incorrect tick box state for project "{item.text()}"')

    def test_save_yaml(self):
        """Test the save button"""
        self.form.descriptionPath.setText(str(self.yaml_file))  # Enter path to YAML file
        # The following is set when user switches focus from the path input box
        self.form.previously_selected_description_path = str(self.yaml_file)
        self.form.on_load_button_pressed()  # Load a file to save
        self.form.on_save_button_pressed()
        # Check yaml file was saved
        file = self.session_path.joinpath('_ibl_experiment.description.yaml')
        self.assertTrue(file.exists())
        # Check saved to settings
        settings = QSettings('int-brain-lab', f'project_protocol_form_{self.subject}')
        self.assertIsNotNone(settings.value('selected_description'))


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
