import logging
import shutil
import unittest.mock

import numpy as np
from numpy.testing import assert_array_almost_equal
from iblutil.util import Bunch
from ibllib.pipes.widefield_tasks import WidefieldPreprocess, WidefieldCompress, WidefieldSync, WidefieldRegisterRaw

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestWidefieldRename(base.IntegrationTest):
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
        cls.alf_folder = cls.session_path.joinpath('alf', 'widefield')

        # Symlink data from original folder to the new folder
        orig_cam_file = next(cls.data_folder.glob('*.camlog'))
        new_cam_file = cls.widefield_folder.joinpath(orig_cam_file.name)
        new_cam_file.symlink_to(orig_cam_file)

        orig_data_file = next(cls.data_folder.glob('dorsal_cortex*'))
        new_data_file = cls.widefield_folder.joinpath(orig_data_file.name)
        new_data_file.symlink_to(orig_data_file)

        orig_led_wiring_file = next(cls.data_folder.glob('*widefield_wiring*'))
        new_led_wiring_file = cls.widefield_folder.joinpath(orig_led_wiring_file.name)
        new_led_wiring_file.symlink_to(orig_led_wiring_file)

        orig_wiring_file = next(cls.data_folder.glob('*configuration.json'))  # note this might change
        new_wiring_file = cls.widefield_folder.joinpath(orig_wiring_file.name)
        new_wiring_file.symlink_to(orig_wiring_file)

    def test_rename(self):
        wf = WidefieldRegisterRaw(self.session_path)
        wf.get_signatures()
        wf.rename_files(symlink_old=False)
        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.widefield_folder)
        shutil.rmtree(cls.alf_folder.parent)


class TestWidefieldPreprocessAndCompress(base.IntegrationTest):
    session_path = None
    widefield_folder = None
    data_folder = None
    alf_folder = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.session_path = cls.default_data_root().joinpath(
            'widefield', 'widefieldChoiceWorld', 'CSK-im-011', '2021-07-21', '001')
        if not cls.session_path.exists():
            return
        # Move the data into the correct folder
        cls.data_folder = cls.session_path.joinpath('orig')
        cls.widefield_folder = cls.session_path.joinpath('raw_widefield_data')
        cls.widefield_folder.mkdir(parents=True, exist_ok=True)
        cls.alf_folder = cls.session_path.joinpath('alf', 'widefield')

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
        PRECISION = 4  # Desired decimal precision
        # U
        assert_array_almost_equal(np.load(self.data_folder.joinpath('U.npy')),
                                  np.load(self.alf_folder.joinpath('widefieldU.images.npy')),
                                  decimal=PRECISION)
        # SVT
        assert_array_almost_equal(
            np.load(self.data_folder.joinpath('SVT.npy')),
            np.load(self.alf_folder.joinpath('widefieldSVT.uncorrected.npy')),
            decimal=PRECISION)

        # Haemo corrected SVT
        assert_array_almost_equal(
            np.load(self.data_folder.joinpath('SVTcorr.npy')),
            np.load(self.alf_folder.joinpath('widefieldSVT.haemoCorrected.npy')),
            decimal=PRECISION)

        # Frame average
        assert_array_almost_equal(
            np.load(self.data_folder.joinpath('frames_average.npy')),
            np.load(self.alf_folder.joinpath('widefieldChannels.frameAverage.npy')),
            decimal=PRECISION)

        wf.wf.remove_files()

        assert len(list(self.widefield_folder.glob('motion*'))) == 0

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
        shutil.rmtree(cls.alf_folder.parent)


class TestWidefieldSync(base.IntegrationTest):
    patch = None  # A mock of get_video_meta
    video_meta = Bunch()

    def setUp(self):
        self.session_path = self.default_data_root().joinpath(
            'widefield', 'widefieldChoiceWorld', 'JC076', '2022-02-04', '002')
        if not self.session_path.exists():
            return
        self.alf_folder = self.session_path.joinpath('alf', 'widefield')
        self.video_file = self.session_path.joinpath('raw_widefield_data', 'widefield.raw.mov')
        self.video_file.touch()
        self.video_meta.length = 2032
        self.patch = unittest.mock.patch('ibllib.io.extractors.widefield.get_video_meta',
                                         return_value=self.video_meta)
        self.patch.start()

    def test_sync(self):
        wf = WidefieldSync(self.session_path)
        status = wf.run()
        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

        # Check integrity of outputs
        times = np.load(self.alf_folder.joinpath('widefield.times.npy'))
        assert len(times) == self.video_meta['length']
        assert np.all(np.diff(times) > 0)
        leds = np.load(self.alf_folder.joinpath('widefield.widefieldLightSource.npy'))
        assert leds[0] == 2
        assert np.array_equal(np.unique(leds), np.array([1, 2]))

    def test_video_led_sync_not_enough(self):
        # Mock video file with more frames than led timestamps
        self.video_meta.length = 2035
        wf = WidefieldSync(self.session_path)
        status = wf.run()
        assert status == -1
        assert 'ValueError: More video frames than led frames detected' in wf.log

    def test_video_led_sync_too_many(self):
        # Mock video file with more that two extra led timestamps
        self.video_meta.length = 2029

        wf = WidefieldSync(self.session_path)
        status = wf.run()
        assert status == -1
        assert 'ValueError: Led frames and video frames differ by more than 2' in wf.log

    def tearDown(self):
        self.video_file.unlink()
        if self.alf_folder.exists():
            shutil.rmtree(self.alf_folder.parent)
        self.patch.stop()
