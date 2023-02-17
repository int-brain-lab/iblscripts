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


class TesMesoscopeSync(base.IntegrationTest):
    session_path = None

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('mesoscope', 'SP026', '2022-06-29', '001')
        # self.session_path = self.default_data_root().joinpath('mesoscope', 'test', '2023-01-31', '003')

    def test_sync(self):
        # task = MesoscopeSync(self.session_path, sync_collection='raw_mesoscope_data', sync_namespace='timeline')
        # status = task.run()
        # assert status == 0
        # from ibllib.pipes.dynamic_pipeline import make_pipeline
        # pipe = make_pipeline(self.session_path)
        # TODO Rename raw_timeline_data -> raw_sync_data
        # TODO rename TimelineHW.json -> _timeline_DAQdata.meta.json
        # status = pipe.tasks['ChoiceWorldTrialsTimeline_00'].run()

        # # NB: For now we're testing individual functions before we have complete data
        timeline_trials = mesoscope.TimelineTrials(self.session_path, sync_collection='raw_mesoscope_data')
        # Check that we can extract the wheel as it's from a counter channel, instead of raw analogue input
        wheel, moves = timeline_trials.get_wheel_positions()
        self.assertCountEqual(['timestamps', 'position'], wheel.keys())
        self.assertCountEqual(['intervals', 'peakAmplitude', 'peakVelocity_times'], moves.keys())
        self.assertEqual(7867, len(wheel['timestamps']))
        np.testing.assert_array_almost_equal([70.004, 70.007, 70.01, 70.013, 70.015], wheel['timestamps'][:5])
        np.testing.assert_array_almost_equal([0., -0.00153398, -0.00306796, -0.00460194, -0.00613592], wheel['position'][:5])
        expected = [[70.151, 71.612], [72.664, 73.158], [74.09, 74.377], [74.704, 75.13], [78.096, 79.692]]
        np.testing.assert_array_almost_equal(expected, moves['intervals'][:5, :])

