import logging
import shutil
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, ANY

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from iblutil.util import Bunch
import one.alf.io as alfio
from one.alf.files import get_session_path
from one.api import ONE

from ibllib.pipes.mesoscope_tasks import \
    MesoscopeSync, MesoscopeFOV, MesoscopeRegisterSnapshots, MesoscopePreprocess, MesoscopeCompress
from ibllib.io.extractors.ephys_fpga import get_wheel_positions
from ibllib.io.extractors import mesoscope
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap

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
        # session_path = self.default_data_root().joinpath('mesoscope', 'SP037', '2023-02-20', '001')
        session_path = self.default_data_root().joinpath('mesoscope', 'SP037', '2023-03-09', '001')
        session_path = self.default_data_root().parent.joinpath('resources', 'SP035', '2023-03-03', '001')
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
        pipe.tasks['ChoiceWorldTrialsTimeline_00'].one = pipe.one
        status = pipe.tasks['ChoiceWorldTrialsTimeline_00'].run()
        # status = pipe.tasks['PassiveTaskTimeline_01'].run()
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
        """Test for ibllib.io.raw_daq_loaders.load_timeline_sync_and_chmap."""
        sync, chmap = mesoscope.load_timeline_sync_and_chmap(
            self.session_path_2 / 'raw_sync_data', save=False)
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
        ROI_folders = list(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nROIs, len(ROI_folders))
        ROI_times = list(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nROIs, len(ROI_times))
        expected = [1.106, 1.304, 1.503, 1.701, 1.899]
        np.testing.assert_array_almost_equal(np.load(ROI_times[0])[:5], expected)
        ROI_shifts = list(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nROIs, len(ROI_shifts))
        expected = [0., 4.157940e-05, 8.315880e-05, 1.247382e-04, 1.663176e-04]
        np.testing.assert_array_almost_equal(np.load(ROI_shifts[0])[:5], expected)

    def test_mesoscope_sync_multiple(self):
        session_path = self.default_data_root().joinpath('mesoscope', 'SP037', '2023-03-09', '001')
        task = MesoscopeSync(session_path, device_collection='raw_imaging_data*', one=self.one)
        status = task.run()
        assert status == 0

        # Check output
        nROIs = 6
        alf_path = session_path.joinpath('alf')
        ROI_folders = list(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nROIs, len(ROI_folders))
        ROI_times = list(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nROIs, len(ROI_times))
        expected = [1.0075, 1.1535, 1.2995, 1.446, 1.592]
        np.testing.assert_array_almost_equal(np.load(ROI_times[0])[:5], expected)
        ROI_shifts = list(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nROIs, len(ROI_shifts))
        expected = [0., 4.157550e-05, 8.315100e-05, 1.247265e-04, 1.663020e-04]
        np.testing.assert_array_almost_equal(np.load(ROI_shifts[0])[:5], expected)


class TestMesoscopeRegisterSnapshots(base.IntegrationTest):
    session_path = None

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        self.session_path = self.default_data_root().joinpath('mesoscope', 'test', '2023-02-17', '002')
        self.session_path = Path(r'F:\FlatIron\resources\SP035\2023-03-01\001')

    def test_register_snapshots(self):
        task = MesoscopeRegisterSnapshots(self.session_path, one=self.one)
        eid, *_ = self.one.search()
        with unittest.mock.patch.object(self.one, 'path2eid', return_value=eid):
            status = task.run()
        self.assertEqual(0, status)
        notes = self.one.alyx.rest('notes', 'list', session=eid)  # filter doesn't work
        self.assertEqual(4, len([n for n in notes if n['image'] and n['object_id'] == eid]))

    def test_get_signature(self):
        for i in range(2):
            p = self.session_path.joinpath(
                f'raw_imaging_data_{i:02}', 'reference', '2023-03-01_1_SP035_2P_00001_00001.tif')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(), self.addCleanup(p.unlink)

        task = MesoscopeRegisterSnapshots(self.session_path, one=self.one)
        task.get_signatures()


class TestMesoscopePreprocess(base.IntegrationTest):
    """Test for MesoscopePreprocess task."""
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.session_path = self.default_data_root().joinpath('mesoscope', 'SP037', '2023-03-23', '002')
        self.alf_path = self.session_path.joinpath('alf', 'suite2p', 'plane2')
        self.rename_dict = {
            'F.npy': 'mpci.ROIActivityF.npy',
            'spks.npy': 'mpci.ROIActivityDeconvolved.npy',
            'Fneu.npy': 'mpci.ROINeuropilActivityF.npy'}
        # Copy files to temp dir
        self.suite2pdir = Path(self.tempdir.name).joinpath(*self.alf_path.parts[-6:])
        shutil.copytree(self.alf_path, self.suite2pdir)
        self.one = ONE(**base.TEST_DB)

    def test_rename_outputs(self):
        """Test MesoscopePreprocess._rename_outputs method."""
        session_path = get_session_path(self.suite2pdir)
        task = MesoscopePreprocess(session_path, one=self.one)
        files = task._rename_outputs(self.suite2pdir.parent, None, None)
        self.assertTrue(all(map(Path.exists, files)))
        self.assertFalse(self.suite2pdir.exists())
        self.assertTrue((compressed := files[0].with_name('_suite2p_ROIData.raw.zip')).exists())
        self.assertIn(compressed, files)
        # Check files saved transposed
        for old, new in self.rename_dict.items():
            expected = np.load(self.alf_path / old).T
            np.testing.assert_array_equal(expected, np.load(files[0].with_name(new)))
        # Check frame QC not saved
        self.assertFalse(any('mpciFrameQC' in f.name for f in files))

    def test_rename_with_qc(self):
        """Test MesoscopePreprocess._rename_outputs method with frame QC input."""
        session_path = get_session_path(self.suite2pdir)
        task = MesoscopePreprocess(session_path, one=self.one)
        # Check frameQC is saved
        frameQC_names = pd.DataFrame([(0, 'ok'), (1, 'foo')], columns=['qc_values', 'qc_labels'])
        files = task._rename_outputs(self.suite2pdir.parent, frameQC_names, np.zeros(15))
        self.assertIn(files[0].with_name('mpciFrameQC.names.tsv'), files)
        self.assertIn(files[0].with_name('mpci.mpciFrameQC.npy'), files)


class TestMesoscopeCompress(base.IntegrationTest):
    """Test for MesoscopeCompress task."""
    def setUp(self) -> None:
        self.session_path = self.default_data_root() / 'mesoscope' / 'test' / '2023-03-03' / '002'
        self.session_path = self.default_data_root() / 'mesoscope' / 'test' / '2023-02-17' / '002'  # Small
        self.one = ONE(**base.TEST_DB)

    def test_compress(self):
        task = MesoscopeCompress(self.session_path, one=self.one)
        files = task.run()
