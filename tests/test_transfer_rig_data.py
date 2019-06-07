import tempfile
import unittest
from pathlib import Path

import ibllib.io.flags as flags


class TestTransferRigData(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_data_folder = Path(self.tmp_dir.name)
        self.session_path = self.root_data_folder / \
            "src" / 'algernon' / '2019-01-21' / '001'
        self.session_path.mkdir(parents=True, exist_ok=True)
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data").mkdir()
        (self.session_path / "raw_video_data").mkdir()
        (self.session_path / "raw_behavior_data" / '_iblrig_micData.raw.wav').touch()
        (self.session_path / "raw_video_data" /
         '_iblrig_leftCamera.raw.avi').touch()

    def test_transfer(self):
        src_subjects_path = self.root_data_folder / "src"
        dst_subjects_path = self.root_data_folder / "dst"
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        gsrc = [x.name for x in list(src_subjects_path.rglob('*.*'))]
        gdst = [x.name for x in list(dst_subjects_path.rglob('*.*'))]
        self.assertTrue('extract_me.flag' in gdst)
        gdst = [x for x in gdst if x != 'extract_me.flag']
        self.assertTrue('compress_video.flag' in gdst)
        gdst = [x for x in gdst if x != 'compress_video.flag']
        self.assertTrue('_iblrig_micData.raw.wav' in gdst)
        gdst = [x for x in gdst if x != '_iblrig_micData.raw.wav']
        self.assertTrue('_iblrig_leftCamera.raw.avi' in gdst)
        gdst = [x for x in gdst if x != '_iblrig_leftCamera.raw.avi']

        self.assertEqual(gsrc, gdst)
        # Test if folder exists not copy because no flag
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        # Test if flag exists and folder exists in dst
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data" / '_iblrig_micData.raw.wav').touch()
        (self.session_path / "raw_video_data" /
         '_iblrig_leftCamera.raw.avi').touch()
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)
        # Test transfer w/o video and audio
        flags.write_flag_file(self.session_path.joinpath("transfer_me.flag"))
        (self.session_path / "raw_behavior_data" / "random.data1.ext").touch()
        transfer_rig_data.main(src_subjects_path, dst_subjects_path)

    def tearDown(self):
        self.tmp_dir.cleanup()
