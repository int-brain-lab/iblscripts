import logging
import shutil
import unittest.mock
from unittest.mock import MagicMock, ANY

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
        self.session_path_0 = self.default_data_root().joinpath('mesoscope', 'SP026', '2022-06-29', '001')
        self.session_path_1 = self.default_data_root().joinpath('mesoscope', 'test', '2023-01-31', '003')
        # A new test session with Bpod channel fix'd in timeline
        self.session_path_2 = self.default_data_root().joinpath('mesoscope', 'test', '2023-02-17', '002')

    def test_sync(self):
        # task = MesoscopeSync(self.session_path, sync_collection='raw_mesoscope_data', sync_namespace='timeline')
        # status = task.run()
        # assert status == 0
        from ibllib.pipes.dynamic_pipeline import make_pipeline
        pipe = make_pipeline(self.session_path_2)
        # TODO Rename raw_timeline_data -> raw_sync_data
        # TODO rename TimelineHW.json -> _timeline_DAQdata.meta.json
        status = pipe.tasks['ChoiceWorldTrialsTimeline_00'].run()
        self.assertFalse(status)

    def test_get_wheel_positions(self):
        """Test for TimelineTrials.get_wheel_positions in ibllib.io.extractors.mesoscope."""
        # # NB: For now we're testing individual functions before we have complete data
        timeline_trials = mesoscope.TimelineTrials(self.session_path_0, sync_collection='raw_sync_data')
        # Check that we can extract the wheel as it's from a counter channel, instead of raw analogue input
        wheel, moves = timeline_trials.get_wheel_positions()
        self.assertCountEqual(['timestamps', 'position'], wheel.keys())
        self.assertCountEqual(['intervals', 'peakAmplitude', 'peakVelocity_times'], moves.keys())
        self.assertEqual(7867, len(wheel['timestamps']))
        np.testing.assert_array_almost_equal([70.004, 70.007, 70.01, 70.013, 70.015], wheel['timestamps'][:5])
        np.testing.assert_array_almost_equal([0., -0.00153398, -0.00306796, -0.00460194, -0.00613592], wheel['position'][:5])
        expected = [[70.151, 71.612], [72.664, 73.158], [74.09, 74.377], [74.704, 75.13], [78.096, 79.692]]
        np.testing.assert_array_almost_equal(expected, moves['intervals'][:5, :])
        # Check input validation
        self.assertRaises(ValueError, timeline_trials.get_wheel_positions, coding='x3')

    @unittest.mock.patch('ibllib.io.extractors.mesoscope.plt')
    def test_get_valve_open_times(self, plt_mock):
        """Test for TimelineTrials.get_valve_open_times in ibllib.io.extractors.mesoscope."""
        timeline_trials = mesoscope.TimelineTrials(self.session_path_2, sync_collection='raw_sync_data')
        expected = [26.053, 30.844, 34.821, 44.15, 53.244, 66.295]
        np.testing.assert_array_almost_equal(expected, timeline_trials.get_valve_open_times())
        # Test display
        plt_mock.subplots.return_value = (MagicMock(), MagicMock())
        open_times = timeline_trials.get_valve_open_times(display=True)
        plt_mock.subplots.assert_called()
        ax = plt_mock.subplots.return_value[1]
        ax.plot.assert_called()
        np.testing.assert_array_equal(ax.plot.call_args_list[1].args[0], open_times)

    @unittest.mock.patch('ibllib.io.extractors.mesoscope.plt')
    def test_plot_timeline(self, plt_mock):
        """Test for ibllib.io.extractors.mesoscope.plot_timeline."""
        ax = MagicMock()
        plt_mock.subplots.return_value = (MagicMock(), [ax] * 19)
        timeline = alfio.load_object(self.session_path_2 / 'raw_sync_data', 'DAQdata')
        fig, axes = mesoscope.plot_timeline(timeline)
        plt_mock.subplots.assert_called_with(19, 1)
        self.assertIs(ax, axes[0], 'failed to return figure axes')
        axes[0].set_ylabel.assert_called_with('syncEcho', rotation=ANY, fontsize=ANY)
        self.assertEqual(19, axes[0].set_ylabel.call_count)
        (x, y), _ = axes[0].plot.call_args
        np.testing.assert_array_equal(timeline['timestamps'], x)
        np.testing.assert_array_equal(timeline['raw'][:, 18], y)

        # Test with raw=False and channels
        ax.reset_mock(), plt_mock.reset_mock()
        channels = ['audio', 'bpod']
        fig, axes = mesoscope.plot_timeline(timeline, channels=channels, raw=False)
        self.assertEqual(2, axes[0].set_ylabel.call_count)
        axes[0].set_ylabel.assert_called_with('bpod', rotation=ANY, fontsize=ANY)
        ylabels = [x[0][0] for x in axes[0].set_ylabel.call_args_list]
        self.assertCountEqual(channels, ylabels)

        (x, y), _ = axes[0].plot.call_args
        self.assertEqual(56, len(x))
        self.assertCountEqual({-1, 1}, np.unique(y))

    def test_timeline2sync(self):
        """Test for ibllib.io.extractors.mesoscope._timeline2sync."""
        timeline = alfio.load_object(self.session_path_2 / 'raw_sync_data', 'DAQdata')
        sync, chmap = mesoscope._timeline2sync(timeline)
        self.assertIsInstance(sync, dict)
        self.assertCountEqual(('times', 'channels', 'polarities'), sync.keys())
        expected = {
            'bpod': 10,
            'frame2ttl': 12,
            'left_camera': 13,
            'right_camera': 14,
            'belly_camera': 15,
            'audio': 16}
        self.assertDictEqual(expected, chmap)