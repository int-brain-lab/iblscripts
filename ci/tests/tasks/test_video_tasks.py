import logging
import shutil
import tempfile
from pathlib import Path
import unittest.mock

from ibllib.pipes.video_tasks import VideoRegisterRaw, VideoCompress, VideoSyncQc

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestVideoRegisterRaw(base.IntegrationTest):
    def setUp(self) -> None:

        pass

    def test_register(self):
        pass


class TestVideoEphysCompress(base.IntegrationTest):
    def setUp(self) -> None:
        self.folder_path = self.data_path.joinpath('ephys', 'ephys_video_init', 'ZM_1735', '2019-08-01', '001', 'raw_video_data')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('ZM_1735', '2019-08-01', '001')
        shutil.copytree(self.folder_path, self.session_path.joinpath('raw_video_data'))
        # files in this folder are not named correctly so rename
        avi_files = self.session_path.joinpath('raw_video_data').glob('*.avi')
        for file in avi_files:
            new_file = self.session_path.joinpath('raw_video_data', file.name.replace('.avi', '.raw.avi'))
            file.replace(new_file)

    def test_compress(self):
        task = VideoCompress(self.session_path, device_collection='raw_video_data', cameras=['left', 'right', 'body'])
        status = task.run()
        assert status == 0
        task.assert_expected_outputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestVideoCompress(base.IntegrationTest):
    def setUp(self) -> None:
        self.folder_path = self.data_path.joinpath('Subjects_init', 'ZM_1085', '2019-02-12', '002', 'raw_video_data')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('ZM_1085', '2019-02-12', '002')
        shutil.copytree(self.folder_path, self.session_path.joinpath('raw_video_data'))

    def test_compress(self):
        task = VideoCompress(self.session_path, device_collection='raw_video_data', cameras=['left'])
        status = task.run()
        assert status == 0
        task.assert_expected_outputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestVideoSyncQCBpod(base.IntegrationTest):
    def setUp(self) -> None:
        self.folder_path = self.data_path.joinpath('Subjects_init', 'ZM_1085', '2019-02-12', '002')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('ZM_1085', '2019-02-12', '002')
        shutil.copytree(self.folder_path, self.session_path)
        task = VideoCompress(self.session_path, device_collection='raw_video_data', cameras=['left'])
        task.run()

    @unittest.mock.patch('ibllib.qc.camera.CameraQC')
    def test_videosync(self, mock_qc):
        task = VideoSyncQc(self.session_path, device_collection='raw_video_data', cameras=['left'], sync='bpod',
                           main_collection='raw_behavior_data')
        status = task.run()
        self.assertEqual(mock_qc.call_count, 1)
        assert status == 0
        task.assert_expected_outputs()


# TODO this hasn't been tested on my computer
class TestVideoSyncQCNidq(base.IntegrationTest):
    def setUp(self) -> None:

        self.folder_path = self.data_path.joinpath('ephys', 'choice_world_init', 'KS022', '2019-12-10', '001')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('KS022', '2019-12-10', '001')

        for ff in self.folder_path.rglob('*.*'):
            link = self.session_path.joinpath(ff.relative_to(self.folder_path))
            if 'alf' in link.parts:
                if 'dlc' in link.name or 'ROIMotionEnergy' in link.name:
                    link.parent.mkdir(exist_ok=True, parents=True)
                    link.symlink_to(ff)
                continue
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)

    @unittest.mock.patch('ibllib.qc.camera.CameraQC')
    def test_videosync(self, mock_qc):
        task = VideoSyncQc(self.session_path, device_collection='raw_video_data', sync='nidq', cameras=['left', 'right', 'body'],
                           main_collection='raw_behavior_data')
        status = task.run()
        self.assertEqual(mock_qc.call_count, 3)
        assert status == 0
        task.assert_expected_outputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
