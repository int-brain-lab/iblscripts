import tempfile
import unittest
import json
from pathlib import Path

import ibllib.io.flags as flags
from ibllib.pipes import transfer_rig_data


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


class TestTransferRigDataEphys(unittest.TestCase):
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


class TestTransferRigDataTraining(unittest.TestCase):
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


class TestTransferRigDataTrainingOnEphys(unittest.TestCase):
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
