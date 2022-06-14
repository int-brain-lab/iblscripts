import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ibllib.tests.fixtures.utils as fu
from ibllib.pipes import misc
from deploy.videopc.transfer_video_session import main as transfer_video_session


class TestTransferVideoSession(unittest.TestCase):
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

    # @mock.patch("ibllib.pipes.misc.check_create_raw_session_flag")
    # @mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None)
    # def test_transfer_video_session(self, chk_fcn):
    def test_transfer_video_session(self):
        # # --- Mock Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/ raw_behavior_data 1900-01-01
        # self.local_session_path.joinpath("transfer_me.flag").touch()
        # remote_session = fu.create_fake_session_folder(self.remote_repo)
        # fu.create_fake_raw_behavior_data_folder(remote_session)
        # with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
        #     transfer_video_session(self.local_repo, self.remote_repo)
        # shutil.rmtree(self.remote_repo)
        #
        #
        # # --- Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/ raw_behavior_data 1900-01-01
        # self.local_session_path.joinpath("transfer_me.flag").touch()
        # remote_session = fu.create_fake_session_folder(self.remote_repo)
        # fu.create_fake_raw_behavior_data_folder(remote_session)
        # with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
        #     transfer_video_session(self.local_repo, self.remote_repo)
        # # --- Test clean up
        # shutil.rmtree(self.remote_repo)
        #
        # # --- Test - 1 local session w/ transfer_me.flag and transfer_complete.flag
        # self.local_session_path.joinpath("transfer_me.flag").touch()
        # self.local_session_path.joinpath(Path("raw_video_data") / "transfer_complete.flag").touch()
        # remote_session = fu.create_fake_session_folder(self.remote_repo)
        # fu.create_fake_raw_behavior_data_folder(remote_session)
        # with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
        #     transfer_video_session(self.local_repo, self.remote_repo)
        # # --- Test clean up
        # self.local_session_path.joinpath("transfer_me.flag").unlink()
        # shutil.rmtree(self.remote_repo)
        # self.local_session_path.joinpath(Path("raw_video_data") / "transfer_complete.flag").unlink()
        #
        # # --- Test - 1 local session w/o transfer_me.flag 1900-01-01, 1 remote session w/ raw_behavior_data 1900-01-01
        # remote_session = fu.create_fake_session_folder(self.remote_repo)
        # fu.create_fake_raw_behavior_data_folder(remote_session)
        # with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
        #     misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # # --- Test clean up
        # shutil.rmtree(self.remote_repo)
        #
        # # --- Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/o behavior folder
        # self.local_session_path.joinpath("transfer_me.flag").touch()
        # remote_session = fu.create_fake_session_folder(self.remote_repo)
        # fu.create_fake_raw_behavior_data_folder(remote_session)
        # shutil.rmtree(self.remote_repo / "fakelab" / "Subjects" / "fakemouse" / "1900-01-01" / "001" / "raw_behavior_data")
        # with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
        #     misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # # --- Test clean up
        # self.local_session_path.joinpath("transfer_me.flag").unlink()
        # shutil.rmtree(self.remote_repo)
        #
        # # --- Test - 1 local session w/ transfer_me.flag 1900-01-01, 1 remote session w/o date folder
        # self.local_session_path.joinpath("transfer_me.flag").touch()
        # fu.create_fake_raw_behavior_data_folder(remote_session)
        # shutil.rmtree(self.remote_repo / "fakelab" / "Subjects" / "fakemouse")
        # with mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
        #     misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # # --- Test clean up
        # self.local_session_path.joinpath("transfer_me.flag").unlink()
        # shutil.rmtree(self.remote_repo)

        # --- Test - 1 local sessions w/ transfer_me.flag 1900-01-01, 2 remote sessions w/ raw_behavior_data 1900-01-01
        self.local_session_path.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        remote_session002 = fu.create_fake_session_folder(self.remote_repo, date="1900-01-01")
        fu.create_fake_raw_behavior_data_folder(remote_session002)
        with mock.patch("builtins.input", side_effect=["002"]), \
                mock.patch("deploy.videopc.transfer_video_session.check_create_raw_session_flag", return_value=None):
            misc.transfer_video_folders(self.local_repo, self.remote_repo)
            # with mock.patch("builtins.input", side_effect=["002"]):
            #     pass
                # chk_fcn.assert_called()  # ensure check_create_raw_session_flag function called
        # --- Test clean up
        shutil.rmtree(self.remote_repo)

        # Test - 2 local sessions w/ transfer_me.flag 1900-01-01, 2 remote sessions w/ raw_behavior_data 1900-01-01
        self.local_session_path.joinpath("transfer_me.flag").touch()
        local_session002 = fu.create_fake_session_folder(self.local_repo, date="1900-01-01")
        fu.create_fake_raw_video_data_folder(local_session002)
        local_session002.joinpath("transfer_me.flag").touch()
        remote_session = fu.create_fake_session_folder(self.remote_repo)
        fu.create_fake_raw_behavior_data_folder(remote_session)
        remote_session002 = fu.create_fake_session_folder(self.remote_repo, date="1900-01-01")
        fu.create_fake_raw_behavior_data_folder(remote_session002)
        with mock.patch("builtins.input", side_effect=["001", "002"]):
            misc.transfer_video_folders(self.local_repo, self.remote_repo)
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
            misc.transfer_video_folders(self.local_repo, self.remote_repo)
        # --- Test clean up
        shutil.rmtree(local_session002)
        shutil.rmtree(self.remote_repo)
