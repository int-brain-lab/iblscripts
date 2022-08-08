import tempfile
import logging
import unittest
from unittest import mock
from pathlib import Path
import json
import shutil

import ibllib.tests.fixtures.utils as fu
import ibllib.io.flags as flags
from ibllib.pipes import transfer_rig_data

from deploy.videopc.transfer_video_session import main as transfer_video_session
from deploy.widefieldpc.transfer_widefield import main as transfer_widefield
from deploy.transfer_data_folder import main as transfer_data_folder

from .base import IntegrationTest


def make_session(session_path, stype='training'):
    flags.write_flag_file(session_path.joinpath("transfer_me.flag"))
    session_path.joinpath("raw_behavior_data").mkdir()
    session_path.joinpath("raw_video_data").mkdir()
    session_path.joinpath("raw_behavior_data", "_iblrig_micData.raw.wav").touch()
    session_path.joinpath("raw_video_data", '_iblrig_leftCamera.raw.avi').touch()
    sf = session_path.joinpath('raw_behavior_data', '_iblrig_taskSettings.raw.json')
    if stype == 'training':
        pybpod_protcol = 'json_trainingChoiceWorld'
        pybpod_board = '_iblrig_somelab_behavior_0'
    elif stype == 'ephys':
        pybpod_protcol = 'json_ephysChoiceWorld'
        pybpod_board = '_iblrig_somelab_ephys_0'
        session_path.joinpath("raw_video_data", '_iblrig_rightCamera.raw.avi').touch()
        session_path.joinpath("raw_video_data", '_iblrig_bodyCamera.raw.avi').touch()
    elif stype == 'mixed':
        pybpod_protcol = 'json_trainingChoiceWorld'
        pybpod_board = '_iblrig_somelab_ephys_0'
    with open(sf, 'w+') as fid:
        fid.write(json.dumps({'PYBPOD_PROTOCOL': pybpod_protcol,
                              'PYBPOD_BOARD': pybpod_board}))


class TestTransferRigDataEphys(IntegrationTest):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_data_folder = Path(self.tmp_dir.name)
        self.session_path = self.root_data_folder.joinpath('src', 'algernon', '2019-01-21', '001')
        self.session_path.mkdir(parents=True, exist_ok=True)
        make_session(self.session_path, stype='ephys')

    def test_transfer_training(self):
        src_subjects_path = self.root_data_folder / "src"
        dst_subjects_path = self.root_data_folder / "dst"
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        gsrc = [x.name for x in list(src_subjects_path.rglob('*.*'))]
        gdst = [x.name for x in list(dst_subjects_path.rglob('*.*'))]
        self.assertTrue(set(gsrc).issubset(set(gdst)))
        dst_session_path = dst_subjects_path.joinpath(
            self.session_path.relative_to(src_subjects_path))
        flag_files = [dst_session_path.joinpath('raw_session.flag')]
        # only when all of the transfers did complete should the flag exist
        for fl in flag_files:
            self.assertFalse(fl.exists())


class TestTransferRigDataTraining(IntegrationTest):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_data_folder = Path(self.tmp_dir.name)
        self.session_path = self.root_data_folder.joinpath('src', 'algernon', '2019-01-21', '001')
        self.session_path.mkdir(parents=True, exist_ok=True)
        make_session(self.session_path, stype='training')

    def test_transfer_training(self):
        src_subjects_path = self.root_data_folder / "src"
        dst_subjects_path = self.root_data_folder / "dst"
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        gsrc = [x.name for x in list(src_subjects_path.rglob('*.*'))]
        gdst = [x.name for x in list(dst_subjects_path.rglob('*.*'))]
        self.assertTrue(set(gsrc).issubset(set(gdst)))
        dst_session_path = dst_subjects_path.joinpath(
            self.session_path.relative_to(src_subjects_path))
        flag_files = [dst_session_path.joinpath('raw_session.flag')]
        for fl in flag_files:
            self.assertTrue(fl.exists())

        # Test if folder exists not copy because no flag
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        # Test if flag exists and folder exists in dst
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data" / '_iblrig_micData.raw.wav').touch()
        (self.session_path / "raw_video_data" /
         '_iblrig_leftCamera.raw.avi').touch()
        transfer_rig_data.main(src_subjects_path, dst_subjects_path, force=True)
        # Test transfer w/o video and audios
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data" / "random.data1.ext").touch()
        transfer_rig_data.main(src_subjects_path, dst_subjects_path, force=True)

    def tearDown(self):
        self.tmp_dir.cleanup()


class TestTransferRigDataTrainingOnEphys(IntegrationTest):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_data_folder = Path(self.tmp_dir.name)
        self.session_path = self.root_data_folder.joinpath('src', 'algernon', '2019-01-21', '001')
        self.session_path.mkdir(parents=True, exist_ok=True)
        make_session(self.session_path, stype='mixed')

    def test_transfer_training(self):
        src_subjects_path = self.root_data_folder / "src"
        dst_subjects_path = self.root_data_folder / "dst"
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        gsrc = [x.name for x in list(src_subjects_path.rglob('*.*'))]
        gdst = [x.name for x in list(dst_subjects_path.rglob('*.*'))]
        self.assertTrue(set(gsrc).issubset(set(gdst)))
        dst_session_path = dst_subjects_path.joinpath(
            self.session_path.relative_to(src_subjects_path))
        flag_files = [dst_session_path.joinpath('raw_session.flag')]
        for fl in flag_files:
            self.assertFalse(fl.exists())

        # Test if folder exists not copy because no flag
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        # Test if flag exists and folder exists in dst
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data" / '_iblrig_micData.raw.wav').touch()
        (self.session_path / "raw_video_data" /
         '_iblrig_leftCamera.raw.avi').touch()
        transfer_rig_data.main(src_subjects_path, dst_subjects_path, force=True)
        # Test transfer w/o video and audios
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data" / "random.data1.ext").touch()
        transfer_rig_data.main(src_subjects_path, dst_subjects_path, force=True)

    def tearDown(self):
        self.tmp_dir.cleanup()


class TestTransferVideoSession(IntegrationTest):
    def setUp(self):
        # Data emulating local rig data
        self.root_test_folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.root_test_folder.cleanup)

        # Change location of transfer list
        # par_file = Path(self.root_test_folder.name).joinpath(".ibl_local_transfers").as_posix()
        # self.patch = unittest.mock.patch("iblutil.io.params.getfile", return_value=par_file)
        # self.patch.start()
        # self.addCleanup(self.patch.stop)

        self.remote_repo = Path(self.root_test_folder.name).joinpath("remote_repo")
        self.remote_repo.joinpath("fakelab/Subjects").mkdir(parents=True)

        self.local_repo = Path(self.root_test_folder.name).joinpath("local_repo")
        self.local_repo.mkdir()

        self.local_session_path = fu.create_fake_session_folder(self.local_repo)
        fu.create_fake_raw_video_data_folder(self.local_session_path)

    def test_transfer_video_session(self):
        # # --- Mock Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/ raw_behavior_data 1900-01-01
        self.local_session_path.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
            transfer_video_session(self.local_repo, self.remote_repo)
        shutil.rmtree(self.remote_repo)

        # --- Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/ raw_behavior_data 1900-01-01
        self.local_session_path.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
            transfer_video_session(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(self.remote_repo)

        # --- Test - 1 local session w/o transfer_me.flag 1900-01-01, 1 remote session w/ raw_behavior_data 1900-01-01
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
            with self.assertRaises(SystemExit) as cm:
                transfer_video_session(self.local_repo, self.remote_repo)
        self.assertEqual(cm.exception.code, 0)
        # --- Test clean up
        shutil.rmtree(self.remote_repo)

        # # --- Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/o behavior folder
        self.local_session_path.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        shutil.rmtree(self.remote_repo / "fakelab" / "Subjects" / "fakemouse" / "1900-01-01" / "001" / "raw_behavior_data")
        with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
            transfer_video_session(self.local_repo, self.remote_repo)
        # --- Test clean up
        self.local_session_path.joinpath("transfer_me.flag").unlink()
        shutil.rmtree(self.remote_repo)

        # --- Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/o date folder
        self.local_session_path.joinpath("transfer_me.flag").touch()
        fu.create_fake_raw_behavior_data_folder(remote_session)
        shutil.rmtree(self.remote_repo / "fakelab" / "Subjects" / "fakemouse")
        with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
            transfer_video_session(self.local_repo, self.remote_repo)
        # --- Test clean up
        self.local_session_path.joinpath("transfer_me.flag").unlink()
        shutil.rmtree(self.remote_repo)

        # --- Test - 1 local sessions w/ transfer_me.flag 1900-01-01, 2 remote sessions w/ raw_behavior_data 1900-01-01
        self.local_session_path.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        remote_session002 = fu.create_fake_session_folder(self.remote_repo, date="1900-01-01")
        fu.create_fake_raw_behavior_data_folder(remote_session002)
        with mock.patch("builtins.input", side_effect=["002"]):
            with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
                transfer_video_session(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(self.remote_repo)

        # # Test - 2 local sessions w/ transfer_me.flag 1900-01-01, 2 remote sessions w/ raw_behavior_data 1900-01-01
        self.local_session_path.joinpath("transfer_me.flag").touch()
        local_session002 = fu.create_fake_session_folder(self.local_repo, date="1900-01-01")
        fu.create_fake_raw_video_data_folder(local_session002)
        local_session002.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        remote_session002 = fu.create_fake_session_folder(self.remote_repo, date="1900-01-01")
        fu.create_fake_raw_behavior_data_folder(remote_session002)
        with mock.patch("builtins.input", side_effect=["001", "002"]):
            with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
                transfer_video_session(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(local_session002)
        shutil.rmtree(self.remote_repo)

        # Test - 2 local sessions w/ transfer_me.flag 1900-01-01, 1 remote sessions w/ raw_behavior_data 1900-01-01
        self.local_session_path.joinpath("transfer_me.flag").touch()
        local_session002 = fu.create_fake_session_folder(self.local_repo, date="1900-01-01")
        fu.create_fake_raw_video_data_folder(local_session002)
        local_session002.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        with mock.patch("builtins.input", side_effect=["002"]):
            with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
                transfer_video_session(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(local_session002)
        shutil.rmtree(self.remote_repo)


class TestTransferWidefieldSession(IntegrationTest):
    """Tests for the iblscripts/deploy/widefieldpc/transfer_widefield.py script"""
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


@unittest.skip('TODO Finish test')
class TestTransferMesoscopeSession(IntegrationTest):
    """Tests for the iblscripts/deploy/mesoscope/transfer_mesoscope.py script"""
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
    def test_transfer_mesoscope(self, mock_params):
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


class TestTransferRawDataSession(IntegrationTest):
    """Tests for the iblscripts/deploy/transfer_data_folder.py script"""
    def setUp(self):
        # Data emulating local rig data
        self.root_test_folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.root_test_folder.cleanup)

        self.remote_repo = Path(self.root_test_folder.name).joinpath("remote_repo")
        self.remote_repo.joinpath("fakelab/Subjects").mkdir(parents=True)

        self.local_repo = Path(self.root_test_folder.name).joinpath("local_repo")
        self.local_repo.mkdir()

        self.local_session_path = fu.create_fake_session_folder(self.local_repo)

    @mock.patch('ibllib.pipes.misc.create_basic_transfer_params')
    def test_transfer_sync(self, mock_params):
        data_folder = 'raw_sync_data'
        local_data_folder = self.local_session_path.joinpath(data_folder)
        local_data_folder.mkdir()
        local_data_folder.joinpath('foo.bar').touch()

        paths = {'DATA_FOLDER_PATH': str(self.local_repo), 'REMOTE_DATA_FOLDER_PATH': str(self.remote_repo)}
        mock_params.return_value = paths

        # Create 'remote' behaviour folder
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        with mock.patch('builtins.input') as mock_in:
            transfer_data_folder(data_folder, self.local_repo, self.remote_repo)
            mock_in.assert_not_called()  # Expect no need for user input

        remote_data = remote_session.joinpath(data_folder)
        self.assertTrue(remote_data.exists() and any(remote_data.glob('*.*')))
        local_flag = local_data_folder.joinpath('transferred.flag')
        self.assertTrue(local_flag.exists())

        # Check whether all files in the transferred flag file are present in the remote location
        local_exists_remote = map(
            lambda x: remote_data.joinpath(Path(x).relative_to(local_flag.parent)).exists(),
            flags.read_flag_file(local_flag))
        self.assertTrue(all(local_exists_remote))

        # Check ignores sessions with flag file
        with self.assertLogs('ibllib.pipes.misc', logging.INFO) as log:
            transfer_data_folder(data_folder, self.local_repo, self.remote_repo)
            self.assertIn('No outstanding local sessions', log.records[-1].message)
