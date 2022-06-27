"""Tests for the iblscripts/deploy/widefieldpc/transfer_widefield.py script"""
import logging
import shutil
import tempfile
from pathlib import Path
from unittest import mock
from ci.tests import base

from ibllib.pipes.misc import flags
import ibllib.tests.fixtures.utils as fu
from deploy.widefieldpc.transfer_widefield import main as transfer_widefield


class TestTransferWidefieldSession(base.IntegrationTest):
    def setUp(self):
        # Data emulating local rig data
        self.root_test_folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.root_test_folder.cleanup)

        self.remote_repo = Path(self.root_test_folder.name).joinpath("remote_repo")
        self.remote_repo.joinpath("fakelab/Subjects").mkdir(parents=True)

        self.local_repo = Path(self.root_test_folder.name).joinpath("local_repo")
        self.local_repo.mkdir()

        self.local_session_path = fu.create_fake_session_folder(self.local_repo)
        test_data = self.data_path.joinpath(
            'widefield', 'widefieldChoiceWorld', 'CSK-im-011', '2021-07-21', '001', 'orig')
        shutil.copytree(test_data, self.local_session_path.joinpath('raw_widefield_data'))

    @mock.patch('ibllib.pipes.misc.create_basic_transfer_params')
    def test_transfer_widefield(self, mock_params):
        paths = {'DATA_FOLDER_PATH': str(self.local_repo), 'REMOTE_DATA_FOLDER_PATH': str(self.remote_repo)}
        mock_params.return_value = paths

        # Create 'remote' behaviour folder
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        with mock.patch('builtins.input') as mock_in:
            transfer_widefield(self.local_repo, self.remote_repo)
            mock_in.assert_not_called()  # Expect no need for user input

        remote_data = remote_session.joinpath('raw_widefield_data')
        self.assertTrue(remote_data.exists() and any(remote_data.glob('*.*')))
        local_flag = self.local_session_path.joinpath('raw_widefield_data', 'transferred.flag')
        self.assertTrue(local_flag.exists())

        # Check whether all files in the transferred flag file are present in the remote location
        local_exists_remote = map(
            lambda x: remote_data.joinpath(Path(x).relative_to(local_flag.parent)).exists(),
            flags.read_flag_file(local_flag))
        self.assertTrue(all(local_exists_remote))

        # Check ignores sessions with flag file
        with self.assertLogs('ibllib.pipes.misc', logging.INFO) as log:
            transfer_widefield(self.local_repo, self.remote_repo)
            self.assertIn('No outstanding local sessions', log.records[-1].message)
