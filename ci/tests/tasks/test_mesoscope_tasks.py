import logging
import shutil
import unittest.mock
from unittest.mock import MagicMock, ANY

import numpy as np
from numpy.testing import assert_array_almost_equal
from iblutil.util import Bunch
import one.alf.io as alfio
from one.api import ONE

from ibllib.pipes.mesoscope_tasks import MesoscopeSync, MesoscopeFOV
from ibllib.io.extractors.ephys_fpga import get_wheel_positions
from ibllib.io.extractors import mesoscope

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestTimelineTrials(base.IntegrationTest):
    session_path = None

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        self.session_path_0 = self.default_data_root().joinpath('mesoscope', 'SP026', '2022-06-29', '001')
        self.session_path_1 = self.default_data_root().joinpath('mesoscope', 'test', '2023-01-31', '003')
        # A new test session with Bpod channel fix'd in timeline
        self.session_path_2 = self.default_data_root().joinpath('mesoscope', 'test', '2023-02-17', '002')

    def test_sync(self):
        # task = ChoiceWorldTrialsTimeline(self.session_path, sync_collection='raw_mesoscope_data', sync_namespace='timeline')
        # status = task.run()
        # assert status == 0
        session_path = self.default_data_root().joinpath('mesoscope', 'SP037', '2023-02-20', '001')
        from ibllib.pipes.dynamic_pipeline import make_pipeline
        pipe = make_pipeline(session_path, one=self.one)
        # TODO Rename raw_timeline_data -> raw_sync_data
        # TODO rename TimelineHW.json -> _timeline_DAQdata.meta.json
        # TODO Assert somewhere that the protocol matches the exp description field
        # from ibllib.io.extractors.mesoscope import plot_timeline
        # timeline = alfio.load_object(session_path / 'raw_sync_data', 'DAQData')
        # channels = [k for k in mesoscope.DEFAULT_MAPS['mesoscope']['timeline']
        #             if k not in ('neural_frames',)]
        # plot_timeline(timeline, channels, raw=False)
        #bpod_t[1987]
        # Out[20]: 2764.5935 (< 0.0005)
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
        timeline = alfio.load_object(self.session_path_2 / 'raw_sync_data', 'DAQData')
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
        timeline = alfio.load_object(self.session_path_2 / 'raw_sync_data', 'DAQData')
        sync, chmap = mesoscope.timeline2sync(timeline)
        self.assertIsInstance(sync, dict)
        self.assertCountEqual(('times', 'channels', 'polarities'), sync.keys())
        expected = {
            'neural_frames': 3,
            'bpod': 10,
            'frame2ttl': 12,
            'left_camera': 13,
            'right_camera': 14,
            'belly_camera': 15,
            'audio': 16}
        self.assertDictEqual(expected, chmap)


class TestMesoscopeFOV(base.IntegrationTest):
    session_path = None

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        self.session_path = self.default_data_root().joinpath('mesoscope', 'test', '2023-02-17', '002')

    def test_mesoscope_fov(self):
        task = MesoscopeFOV(self.session_path, device_collection='raw_imaging_data', one=self.one)
        status = task.run()
        assert status == 0


class TestMesoscopeSync(base.IntegrationTest):
    session_path = None

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        self.session_path = self.default_data_root().joinpath('mesoscope', 'test', '2023-02-17', '002')

    def test_mesoscope_sync(self):
        task = MesoscopeSync(self.session_path, device_collection='raw_imaging_data', one=self.one)
        status = task.run()
        assert status == 0

        # Check output
        nROIs = 9
        alf_path = self.session_path.joinpath('alf')
        ROI_folders = list(alf_path.rglob('ROI*'))
        self.assertEqual(nROIs, len(ROI_folders))
        ROI_times = list(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nROIs, len(ROI_times))
        expected = [1.106, 1.304, 1.503, 1.701, 1.899]
        np.testing.assert_array_almost_equal(np.load(ROI_times[0])[:5], expected)
        ROI_shifts = list(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nROIs, len(ROI_shifts))
        expected = [0., 4.157940e-05, 8.315880e-05, 1.247382e-04, 1.663176e-04]
        np.testing.assert_array_almost_equal(np.load(ROI_shifts[0])[:5], expected)
