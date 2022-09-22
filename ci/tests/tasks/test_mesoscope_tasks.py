import logging
import shutil
import unittest.mock

import numpy as np
from numpy.testing import assert_array_almost_equal
from iblutil.util import Bunch
import one.alf.io as alfio

# from ibllib.pipes.mesoscope_tasks import MesoscopeSync, MesoscopeRegisterRaw
from ibllib.io.extractors.ephys_fpga import get_wheel_positions
from ibllib.io.extractors import mesoscope

from ci.tests import base

_logger = logging.getLogger('ibllib')


# class TestWidefieldRegisterRaw(base.IntegrationTest):
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.session_path = cls.default_data_root().joinpath('mesoscope', 'widefieldChoiceWorld', 'CSK-im-011',
#                                                             '2021-07-21', '001')
#         if not cls.session_path.exists():
#             return
#         # Move the data into the correct folder
#         cls.data_folder = cls.session_path.joinpath('orig')
#         cls.widefield_folder = cls.session_path.joinpath('raw_widefield_data')
#         cls.widefield_folder.mkdir(parents=True, exist_ok=True)
#         cls.alf_folder = cls.session_path.joinpath('alf', 'widefield')
#
#         # Symlink data from original folder to the new folder
#         orig_cam_file = next(cls.data_folder.glob('*.camlog'))
#         new_cam_file = cls.widefield_folder.joinpath(orig_cam_file.name)
#         new_cam_file.symlink_to(orig_cam_file)
#
#         orig_data_file = next(cls.data_folder.glob('dorsal_cortex*'))
#         new_data_file = cls.widefield_folder.joinpath(orig_data_file.name)
#         new_data_file.symlink_to(orig_data_file)
#
#         orig_led_wiring_file = next(cls.data_folder.glob('*widefield_wiring*'))
#         new_led_wiring_file = cls.widefield_folder.joinpath(orig_led_wiring_file.name)
#         new_led_wiring_file.symlink_to(orig_led_wiring_file)
#
#         orig_wiring_file = next(cls.data_folder.glob('*configuration.json'))  # note this might change
#         new_wiring_file = cls.widefield_folder.joinpath(orig_wiring_file.name)
#         new_wiring_file.symlink_to(orig_wiring_file)
#
#     def test_rename(self):
#         task = WidefieldRegisterRaw(self.session_path)
#         status = task.run()
#         assert status == 0
#
#         for exp_files in task.signature['output_files']:
#             file = self.session_path.joinpath(exp_files[1], exp_files[0])
#             assert file.exists()
#
#     @classmethod
#     def tearDownClass(cls) -> None:
#         shutil.rmtree(cls.widefield_folder)
#         shutil.rmtree(cls.alf_folder.parent)


class TesMesoscopeSync(base.IntegrationTest):
    session_path = None
    widefield_folder = None
    data_folder = None
    alf_folder = None

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('mesoscope', 'SP026', '2022-06-29', '001')

    def test_sync(self):
        # # NB: For now we're testing individual functions before we have complete data
        # task = MesoscopeSync(self.session_path, sync_collection='raw_mesoscope_data', sync_namespace='timeline')
        # status = task.run()
        # assert status == 0

        # Check timeline2sync
        sync, chmap = mesoscope._timeline2sync(self.session_path)
        self.assertCountEqual(('times', 'channels', 'polarities'), sync.keys())
        expected = ('left_camera', 'right_camera', 'belly_camera', 'frame2ttl', 'audio', 'bpod', 'rotary_encoder')
        self.assertCountEqual(expected, chmap.keys())

        # Check that we can extract the wheel as it's from a counter channel, instead of raw analogue input
        ts, pos = get_wheel_positions(sync, chmap)

