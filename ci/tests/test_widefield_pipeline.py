import logging
import shutil
import numpy as np
import unittest

from iblutil.util import Bunch
from ibllib.pipes.widefield import WidefieldPreprocess, WidefieldCompress, WidefieldSync

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestWidefieldPreprocessAndCompress(base.IntegrationTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.session_path = cls.default_data_root().joinpath('widefield', 'widefieldChoiceWorld', 'CSK-im-011',
                                                            '2021-07-21', '001')
        if not cls.session_path.exists():
            return
        # Move the data into the correct folder
        cls.data_folder = cls.session_path.joinpath('orig')
        cls.widefield_folder = cls.session_path.joinpath('raw_widefield_data')
        cls.widefield_folder.mkdir(parents=True, exist_ok=True)
        cls.alf_folder = cls.session_path.joinpath('alf')

        # Symlink data from original folder to the new folder
        orig_cam_file = next(cls.data_folder.glob('*.camlog'))
        new_cam_file = cls.widefield_folder.joinpath('widefieldEvents.raw.camlog')
        new_cam_file.symlink_to(orig_cam_file)

        orig_data_file = next(cls.data_folder.glob('*.dat'))
        new_data_file = cls.widefield_folder.joinpath(orig_data_file.name)
        new_data_file.symlink_to(orig_data_file)

    def test_preprocess(self):
        wf = WidefieldPreprocess(self.session_path)
        status = wf.run()
        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

        # Test content of files
        # U
        assert np.array_equal(np.load(self.data_folder.joinpath('U.npy')),
                              np.load(self.alf_folder.joinpath('widefieldU.images.npy')))
        # SVT
        assert np.array_equal(np.load(self.data_folder.joinpath('SVT.npy')),
                              np.load(self.alf_folder.joinpath('widefieldSVT.uncorrected.npy')))

        # Haemo corrected SVT
        assert np.array_equal(np.load(self.data_folder.joinpath('SVTcorr.npy')),
                              np.load(self.alf_folder.joinpath('widefieldSVT.haemoCorrected.npy')))
        # Frame average
        assert np.array_equal(np.load(self.data_folder.joinpath('frames_average.npy')),
                              np.load(self.alf_folder.joinpath('widefieldChannels.frameAverage.npy')))

    def test_compress(self):
        wf = WidefieldCompress(self.session_path)
        status = wf.run()

        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.widefield_folder)
        shutil.rmtree(cls.alf_folder)


class TestWidefieldSync(base.IntegrationTest):

    def setUp(self):
        self.session_path = self.default_data_root().joinpath('widefield', 'widefieldChoiceWorld', 'CSK-im-011',
                                                              '2021-07-29', '001')
        if not self.session_path.exists():
            return
        # Move the data into the correct folder
        self.widefield_folder = self.session_path.joinpath('raw_widefield_data')
        self.widefield_folder.mkdir(parents=True, exist_ok=True)
        self.alf_folder = self.session_path.joinpath('alf')

        self.widefield_folder.joinpath('widefield.raw.mov').touch()

    @unittest.mock.patch('ibllib.io.extractors.widefield.get_video_meta')
    def test_sync(self, mock_meta):

        video_meta = Bunch()
        video_meta.length = 183340
        mock_meta.return_value = video_meta

        wf = WidefieldSync(self.session_path)
        status = wf.run()
        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

    @unittest.mock.patch('ibllib.io.extractors.widefield.get_video_meta')
    def test_sync_not_enough(self, mock_meta):

        # Mock video file with more frames than led timestamps
        video_meta = Bunch()
        video_meta.length = 183345
        mock_meta.return_value = video_meta

        wf = WidefieldSync(self.session_path)
        status = wf.run()
        assert status == -1
        assert 'ValueError: More frames than timestamps detected' in wf.log

    @unittest.mock.patch('ibllib.io.extractors.widefield.get_video_meta')
    def test_sync_too_many(self, mock_meta):

        # Mock video file with more that two extra led timestamps
        video_meta = Bunch()
        video_meta.length = 183338
        mock_meta.return_value = video_meta

        wf = WidefieldSync(self.session_path)
        status = wf.run()
        assert status == -1
        assert 'ValueError: Timestamps and frames differ by more than 2' in wf.log

    def tearDown(self):
        if self.alf_folder.exists():
            shutil.rmtree(self.alf_folder)
