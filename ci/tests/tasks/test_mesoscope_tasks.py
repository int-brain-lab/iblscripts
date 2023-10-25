"""Tests for ibllib.pipes.mesoscope_tasks module."""
import logging
import shutil
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, ANY
import tarfile
from itertools import chain
from uuid import UUID

import numpy as np
import pandas as pd
import sparse

import one.alf.io as alfio
from one.alf.files import get_session_path
from one.api import ONE

from ibllib.pipes.mesoscope_tasks import (
    MesoscopeSync, MesoscopeFOV, MesoscopeRegisterSnapshots,
    MesoscopePreprocess, MesoscopeCompress, Provenance
)
from iblatlas.atlas import AllenAtlas
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsTimeline
from ibllib.io.extractors import mesoscope
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap

from ci.tests import base

_logger = logging.getLogger('ibllib')


def _delete_sync(*session_paths):
    """
    Delete the _timeline_sync.*.npy files that are created during testing.

    Parameters
    ----------
    *session_paths : pathlib.Path
        One or more session paths containing sync files to remove.

    Returns
    -------
    list of pathlib.Path
        A list of deleted files.
    """
    deleted = []
    for file in chain(*map(lambda x: x.glob('raw_sync_data/_timeline_sync.*.npy'), session_paths)):
        deleted.append(file)
        _logger.debug('Deleting %s', file.relative_to(base.IntegrationTest.default_data_root()))
        file.unlink()


class TestTimelineTrials(base.IntegrationTest):
    session_path = None

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        # A new test session with Bpod channel fix'd in timeline
        self.session_path = self.default_data_root().joinpath('mesoscope', 'test', '2023-02-17', '002')
        self.addCleanup(shutil.rmtree, self.session_path / 'alf', ignore_errors=True)
        self.addClassCleanup(_delete_sync, self.session_path)

    def test_sync(self):
        # Mocking training wheel extractor as session doesn't have Bpod rotary encoder data
        with unittest.mock.patch('ibllib.io.extractors.training_wheel.Wheel._extract') as mock:
            mock().__getitem__.return_value = np.zeros(7)  # n trials = 7
            task = ChoiceWorldTrialsTimeline(self.session_path, sync_collection='raw_sync_data',
                                             sync_namespace='timeline', collection='raw_task_data_00')
            task.one = ONE(**base.TEST_DB, mode='local')  # Don't try updating behaviour criterion
            self.assertFalse(task.run(), 'extraction task failed')

        # Check ALF trials
        trials = alfio.load_object(self.session_path / 'alf', 'trials')
        self.assertEqual(17, len(trials.keys()))
        expected = [[9.97294005, 24.00193085],
                    [24.52629002, 28.1602763],
                    [28.6754851, 32.9438336],
                    [33.46808532, 36.930929]]
        np.testing.assert_array_almost_equal(expected, trials['intervals'][:4, :])
        expected = [20.903, 26.053, 30.844, 34.821, 39.257, 44.15, 53.244]
        np.testing.assert_array_almost_equal(expected, trials['feedback_times'])
        expected = [np.nan, 25.892, 30.742, 34.731, 39.091, 43.992, 53.125]
        np.testing.assert_array_almost_equal(expected, trials['firstMovement_times'])

        # Check ALF wheel
        wheel = alfio.load_object(self.session_path / 'alf', 'wheel')
        expected = [0., 0.00153398, 0.00306796, 0.00460194, 0.00613592]
        np.testing.assert_array_almost_equal(expected, wheel['position'][:5])
        expected = [20.809, 20.811, 20.812, 20.813, 20.814]
        np.testing.assert_array_almost_equal(expected, wheel['timestamps'][:5])

    @unittest.mock.patch('ibllib.io.extractors.mesoscope.plt')
    def test_get_wheel_positions(self, plt_mock):
        """Test for TimelineTrials.get_wheel_positions in ibllib.io.extractors.mesoscope."""
        # # NB: For now we're testing individual functions before we have complete data
        timeline_trials = mesoscope.TimelineTrials(self.session_path, sync_collection='raw_sync_data')
        # Check that we can extract the wheel as it's from a counter channel, instead of raw analogue input
        wheel, moves = timeline_trials.get_wheel_positions()
        self.assertCountEqual(['timestamps', 'position'], wheel.keys())
        self.assertCountEqual(['intervals', 'peakAmplitude', 'peakVelocity_times'], moves.keys())
        self.assertEqual(4090, len(wheel['timestamps']))
        np.testing.assert_array_almost_equal([20.809, 20.811, 20.812, 20.813, 20.814], wheel['timestamps'][:5])
        np.testing.assert_array_almost_equal([0., 0.00153398, 0.00306796, 0.00460194, 0.00613592], wheel['position'][:5])
        expected = [[20.811, 21.216], [25.892, 26.251], [30.742, 31.173], [32.161, 33.208], [34.731, 36.756]]
        np.testing.assert_array_almost_equal(expected, moves['intervals'][:5, :])
        # Check input validation
        self.assertRaises(ValueError, timeline_trials.get_wheel_positions, coding='x3')
        # Test display
        plt_mock.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
        timeline_trials.bpod_trials = {'wheel_position': np.zeros_like(wheel['position']),
                                       'wheel_timestamps': wheel['timestamps']}
        timeline_trials.bpod2fpga = lambda x: x
        timeline_trials.get_wheel_positions(display=True)
        plt_mock.subplots.assert_called()
        # The second axes should be a plot of extracted wheel positions
        ax0, ax1 = plt_mock.subplots.return_value[1]
        ax1.plot.assert_called()
        np.testing.assert_array_equal(ax1.plot.call_args_list[0].args[0], wheel['timestamps'])

    @unittest.mock.patch('ibllib.io.extractors.mesoscope.plt')
    def test_get_valve_open_times(self, plt_mock):
        """Test for TimelineTrials.get_valve_open_times in ibllib.io.extractors.mesoscope."""
        timeline_trials = mesoscope.TimelineTrials(self.session_path, sync_collection='raw_sync_data')
        expected = [26.053, 30.844, 34.821, 44.15, 53.244, 66.295]
        np.testing.assert_array_almost_equal(expected, timeline_trials.get_valve_open_times())
        # Test display
        plt_mock.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
        open_times = timeline_trials.get_valve_open_times(display=True)
        plt_mock.subplots.assert_called()
        # The second axes should be a plot of expected valve open times
        ax0, ax1 = plt_mock.subplots.return_value[1]
        ax1.plot.assert_called()
        np.testing.assert_array_equal(ax1.plot.call_args_list[1].args[0], open_times)

    @unittest.mock.patch('ibllib.io.extractors.mesoscope.plt')
    def test_plot_timeline(self, plt_mock):
        """Test for ibllib.io.extractors.mesoscope.plot_timeline."""
        ax = MagicMock()
        plt_mock.subplots.return_value = (MagicMock(), [ax] * 19)
        timeline = alfio.load_object(self.session_path / 'raw_sync_data', 'DAQdata')
        fig, axes = mesoscope.plot_timeline(timeline)
        plt_mock.subplots.assert_called_with(19, 1, sharex=True)
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
        sync, chmap = mesoscope.load_timeline_sync_and_chmap(self.session_path / 'raw_sync_data', save=False)
        self.assertIsInstance(sync, dict)
        self.assertCountEqual(('times', 'channels', 'polarities'), sync.keys())
        expected = {
            'neural_frames': 3,
            'bpod': 10,
            'frame2ttl': 12,
            'left_camera': 13,
            'right_camera': 14,
            'belly_camera': 15,
            'audio': 16,
            'rotary_encoder': 17}
        self.assertDictEqual(expected, chmap)


class TestMesoscopeFOV(base.IntegrationTest):
    session_path = None

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.session_path = Path(tmpdir.name, 'subject', '2020-01-01', '001')
        self.session_path.joinpath('alf').mkdir(parents=True)
        # Make some toy datasets
        self.n_pixels = 512  # Number of pixels xy pixels in each FOV
        self.n_fov = 2  # Number of fields of view
        self.n_roi = 128  # Number of ROIs (will be multiplied by FOV number)
        self.expected_roi_mlapdv = {}  # Save the expected extracted ROI MLAPDV coordinates
        self.offset = 5.  # Offset between pixel number and MLAPDV coordinate
        self.mean_img_mlapdv = dict.fromkeys(range(self.n_fov))
        self.mean_img_ids = dict.fromkeys(range(self.n_fov))
        for i in range(self.n_fov):
            (alf_path := self.session_path.joinpath('alf', f'FOV_{i:02}')).mkdir()
            # Mean image MLAPDV coordinates
            ml = np.tile(np.arange(self.n_pixels), (self.n_pixels, 1)).astype(float) + self.offset
            self.mean_img_mlapdv[i] = np.dstack([ml, ml.T, np.zeros_like(ml)])

            # Mean image brain location IDs (a grid of 32x32 brain locations)
            n_tiles = 32
            tile_sz = int(self.n_pixels / n_tiles)
            x = np.repeat(np.arange(tile_sz), n_tiles)
            y = np.repeat(np.r_[0, (2 ** np.arange(tile_sz) * tile_sz)[:-1]], n_tiles)
            self.mean_img_ids[i] = x + y[..., None]

            # mpciROIs.stackPos (evenly spaced along the diagonal)
            n_roi = self.n_roi * (i + 1)  # 2nd FOV has twice as many as first
            v = np.linspace(0, self.n_pixels - 1, n_roi).astype(int)
            roi_mlapdv = np.vstack([v, v, np.zeros_like(v)]).T
            self.expected_roi_mlapdv[i] = np.c_[roi_mlapdv[:, :2] + self.offset, roi_mlapdv[:, 2]]
            np.save(alf_path / 'mpciROIs.stackPos.npy', roi_mlapdv)
        # For now the meta only contains number of FOVs
        alf_path = self.session_path.joinpath('raw_imaging_data')
        alf_path.mkdir()
        with open(alf_path / '_ibl_rawImagingData.meta.json', 'w') as fp:
            fp.write('{"FOV":[%s]}' % ','.join(['{}'] * self.n_fov))

    def test_mesoscope_fov(self):
        """Test for MesoscopeFOV._run and MesoscopeFOV.roi_mlapdv methods.

        This stubs both register_fov and project_mlapdv, which are tested separately.
        """
        # Test generation of mpciROI datasets
        task = MesoscopeFOV(self.session_path, device_collection='raw_imaging_data', one=self.one)
        mean_img_map = (self.mean_img_mlapdv, self.mean_img_ids)
        with unittest.mock.patch.object(task, 'register_fov') as mock_obj, \
                unittest.mock.patch.object(task, 'project_mlapdv', return_value=mean_img_map):
            self.assertEqual(0, task.run())
            mock_obj.assert_called_once_with(unittest.mock.ANY, 'estimate')
        self.assertEqual(self.n_fov * 4 + 1, len(task.outputs))  # + 1 for modified meta file
        # Mean image brain locations should be int
        file = next(f for f in task.outputs if 'mpciMeanImage.brainLocationIds_ccf_2017_estimate' in f.name)
        self.assertIs(np.load(file).dtype, np.dtype('int'))
        # Check ROI MLAPDV and brain locations
        rois = alfio.load_object(self.session_path / 'alf' / 'FOV_00', 'mpciROIs')
        expected = {'brainLocationIds_ccf_2017_estimate', 'mlapdv_estimate', 'stackPos'}
        self.assertCountEqual(expected, rois.keys())
        expected = self.expected_roi_mlapdv[0]
        np.testing.assert_array_equal(expected, rois['mlapdv_estimate'])
        expected = np.repeat(np.array([0, 17, 34, 67]), 8)
        self.assertIs(rois['brainLocationIds_ccf_2017_estimate'].dtype, np.dtype(int))
        np.testing.assert_array_equal(expected, rois['brainLocationIds_ccf_2017_estimate'][:32])

        # Test that we preferentially use the final coordinates
        # Copy data from another FOV and use as final
        for file in self.session_path.joinpath('alf', 'FOV_01').glob('mpciMeanImage.*'):
            file = file.replace(file.with_name(file.name.replace('_estimate', '')))
            shutil.copy(file, self.session_path.joinpath('alf', 'FOV_00', file.name))

        task = MesoscopeFOV(self.session_path, device_collection='raw_imaging_data', one=self.one)
        with unittest.mock.patch.object(task, 'register_fov') as mock_obj, \
                unittest.mock.patch.object(task, 'project_mlapdv', return_value=mean_img_map):
            self.assertEqual(0, task.run(provenance=Provenance.HISTOLOGY))
            mock_obj.assert_called_once_with(unittest.mock.ANY, None)
        self.assertEqual((self.n_fov * 4) + 1, len(task.outputs))  # + 1 for modified meta file
        self.assertFalse(any('_estimate' in x.name for x in task.outputs))
        rois = alfio.load_object(self.session_path / 'alf' / 'FOV_00', 'mpciROIs')
        expected = {'brainLocationIds_ccf_2017', 'mlapdv', 'stackPos'}
        self.assertTrue(expected <= set(rois.keys()))

        # Check behaviour when there are incomplete datasets
        self.session_path.joinpath('alf', 'FOV_00', 'mpciROIs.stackPos.npy').unlink()
        self.assertRaises(FileNotFoundError, task.roi_mlapdv, self.n_fov)


class TestProjectFOV(base.IntegrationTest):
    """Test MesoscopeFOV.project_mlapdv method."""
    session_path = None

    def setUp(self) -> None:
        # Load fixtures and create simple meta map
        self.session_path = Path('subject', '2020-01-01', '001')
        self.n_pixels = 64  # Number of pixels xy pixels in each FOV
        self.n_fov = 2  # Number of fields of view

        self.atlas = AllenAtlas(res_um=50)  # Use low res atlas for speed
        self.one = ONE(**base.TEST_DB, mode='local')

        # Create a toy meta file
        self.meta = {'centerMM': {'ML': 2.6, 'AP': -1.9}}
        MM = {'topLeft': [2.307, -1.607], 'topRight': [2.892, -1.607],
              'bottomLeft': [2.30, -2.193], 'bottomRight': [2.893, -2.193]}
        self.meta['FOV'] = [{'nXnYnZ': [self.n_pixels, self.n_pixels, 1], 'MM': MM}] * self.n_fov

    def test_project_mlapdv(self):
        """Test the full MesoscopeFOV.project_mlapdv method."""
        # Test generation of mpciROI datasets
        task = MesoscopeFOV(self.session_path, device_collection='raw_imaging_data', one=self.one)
        mlapdv, ids = task.project_mlapdv(self.meta, self.atlas)

        # Check MLAPDV coordinates
        self.assertCountEqual(mlapdv.keys(), range(self.n_fov))
        self.assertEqual(mlapdv[0].shape, (self.n_pixels, self.n_pixels, 3))
        # NB: Both FOVs will have the same values as the corner coords were duplicated
        expected = [
            [[2309.19916861, -1601.44040887, -231.35034825],
             [2317.83114255, -1601.89273938, -234.74282709],
             [2326.4631165, -1602.34506989, -238.13530593]],
            [[2309.09588003, -1610.65498922, -230.0804221],
             [2317.72972769, -1611.10741792, -233.47363734],
             [2326.36357535, -1611.55984662, -236.86685258]],
            [[2308.99259145, -1619.86956957, -228.81049596],
             [2317.62831283, -1620.32209646, -232.20444759],
             [2326.26403421, -1620.77462334, -235.59839922]]
        ]
        np.testing.assert_array_almost_equal(mlapdv[0][:3, :3, :], expected)

        # Check brain location IDs
        expected = [[1006, 981, 981],
                    [312782550, 981, 981],
                    [312782550, 981, 981]]
        np.testing.assert_array_almost_equal(ids[0][:3, 49:52], expected)
        self.assertCountEqual(ids.keys(), range(self.n_fov))
        self.assertEqual(ids[0].shape, (self.n_pixels, self.n_pixels))

        # Check meta map was modified
        FOV_00 = self.meta['FOV'][0]
        self.assertTrue(set(FOV_00.keys()) >= {'MLAPDV', 'brainLocationIds'})
        expected = {'topLeft': 312782550, 'topRight': 981, 'bottomLeft': 312782550,
                    'bottomRight': 312782604, 'center': 312782550}
        self.assertDictEqual(FOV_00['brainLocationIds'], expected)
        expected = [2575.3890558071657, -1901.209002390902, -297.8571573244117]
        np.testing.assert_array_almost_equal(FOV_00['MLAPDV']['center'], expected)

        # Test behaviour when outside of the brain (also remove one of the FOVs for speed)
        FOV_00 = self.meta['FOV'].pop()
        for k in FOV_00['MM']:
            FOV_00['MM'][k] = np.array(FOV_00['MM'][k]) + 10
        with self.assertLogs('ibllib.pipes.mesoscope_tasks', 'WARNING'):
            mlapdv, ids = task.project_mlapdv(self.meta, self.atlas)
        self.assertTrue(np.all(np.isnan(mlapdv[0])))
        np.testing.assert_array_equal(ids[0], np.zeros((self.n_pixels, self.n_pixels), dtype=int))


class TestMesoscopeSync(base.IntegrationTest):
    # session_path_0 = None  # A single imaging bout
    # session_path_1 = None  # Multiple imaging bouts

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        self.session_path_0 = self.default_data_root().joinpath('mesoscope', 'test', '2023-02-17', '002')
        self.session_path_1 = self.default_data_root().joinpath('mesoscope', 'test', '2023-03-03', '002')
        self.addClassCleanup(_delete_sync, self.session_path_0, self.session_path_1)
        self.addCleanup(shutil.rmtree, self.session_path_1 / 'alf', ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.session_path_0 / 'alf', ignore_errors=True)

    def test_mesoscope_sync(self):
        task = MesoscopeSync(self.session_path_0, device_collection='raw_imaging_data', one=self.one)
        status = task.run()
        assert status == 0

        # Check output
        nFOVs = 9
        alf_path = self.session_path_0.joinpath('alf')
        FOV_folders = sorted(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nFOVs, len(FOV_folders))
        FOV_times = sorted(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nFOVs, len(FOV_times))
        expected = [1.106, 1.304, 1.503, 1.701, 1.899]
        np.testing.assert_array_almost_equal(np.load(FOV_times[0])[:5], expected)
        FOV_shifts = sorted(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nFOVs, len(FOV_shifts))
        expected = [0., 4.157940e-05, 8.315880e-05, 1.247382e-04, 1.663176e-04]
        np.testing.assert_array_almost_equal(np.load(FOV_shifts[0])[:5], expected)

        # Test what happens when there are more frame TTLs than timestamps in the header file
        extractor = mesoscope.MesoscopeSyncTimeline(self.session_path_0, nFOVs)
        n_frames = 336
        sync = {'times': np.arange(n_frames + 5), 'channels': np.zeros(n_frames + 5)}
        chmap = {'neural_frames': 0}
        with self.assertLogs(mesoscope.__name__) as log:
            out, _ = extractor.extract(sync=sync, chmap=chmap)
            self.assertEqual('WARNING', log.records[0].levelname, 'failed to log warning')
            self.assertIn('Dropping last 5 frame times', log.output[-1])
        self.assertEqual({n_frames}, set(map(len, out[:nFOVs])), 'failed to drop timestamps')

    def test_mesoscope_sync_multiple(self):
        task = MesoscopeSync(self.session_path_1, device_collection='raw_imaging_data*', one=self.one)
        status = task.run()
        assert status == 0

        # Check output
        nROIs = 6
        alf_path = self.session_path_1.joinpath('alf')
        ROI_folders = list(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nROIs, len(ROI_folders))
        ROI_times = sorted(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nROIs, len(ROI_times))
        expected = [1.0075, 1.154, 1.3, 1.446, 1.5925]
        np.testing.assert_array_almost_equal(np.load(ROI_times[0])[:5], expected)
        ROI_shifts = sorted(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nROIs, len(ROI_shifts))
        expected = [0., 4.157550e-05, 8.315100e-05, 1.247265e-04, 1.663020e-04]
        np.testing.assert_array_almost_equal(np.load(ROI_shifts[0])[:5], expected)

    @unittest.mock.patch('ibllib.io.extractors.mesoscope.plt')
    def test_get_bout_edges(self, plt_mock):
        """Test for ibllib.io.extractors.mesoscope.MesoscopeSyncTimeline.get_bout_edges.

        This tests detection with and without the _ibl_softwareEvents.log.htsv file.
        """
        sync, chmap = load_timeline_sync_and_chmap(self.session_path_1 / 'raw_sync_data')
        extractor = mesoscope.MesoscopeSyncTimeline(self.session_path_1, 6)
        frame_times = sync['times'][sync['channels'] == chmap['neural_frames']]
        udp_events = self.session_path_1.joinpath('raw_sync_data', '_ibl_softwareEvents.log.htsv')
        events = pd.read_csv(udp_events, delimiter='\t')
        collections = ['raw_imaging_data_00', 'raw_imaging_data_01']
        bouts = extractor.get_bout_edges(frame_times, collections, events)
        np.testing.assert_array_equal(bouts, [[1.0075, 57.7175], [89.142, 132.5525]])

        # Test works with no end times
        bouts2 = extractor.get_bout_edges(frame_times, collections, events.drop([2, 4, 5]))
        np.testing.assert_array_equal(bouts, bouts2)

        # Test works with no events
        np.testing.assert_array_equal(bouts, extractor.get_bout_edges(frame_times, collections))

        # Test display
        plt_mock.subplots.return_value = (MagicMock(), MagicMock())
        extractor.get_bout_edges(frame_times, collections, events.drop([2, 4, 5]), display=True)
        plt_mock.subplots.assert_called()
        # Check plotted bout starts equal returned values
        ax = plt_mock.subplots.return_value[1]
        ax.plot.assert_called()
        plot_args = ax.plot.call_args_list[0]
        self.assertEqual('bout start', plot_args.kwargs['label'])
        bout_starts = np.unique(plot_args.args[0])
        np.testing.assert_array_equal(bout_starts[~np.isnan(bout_starts)], bouts2[:, 0])

        # Check validation
        collections.append('raw_imaging_data_02')
        self.assertRaises(ValueError, extractor.get_bout_edges, frame_times, collections, events)


class TestMesoscopeRegisterSnapshots(base.IntegrationTest):
    session_path = None
    one = None
    reference_files = ['referenceImage.raw.tif', 'referenceImage.stack.tif', 'referenceImage.meta.json']

    @classmethod
    def setUpClass(cls) -> None:
        cls.one = ONE(**base.TEST_DB)
        cls.session_path = cls.default_data_root().joinpath('mesoscope', 'test', '2023-03-03', '002')
        # Create some reference images to register
        for i in range(2):
            for file in cls.reference_files:
                p = cls.session_path.joinpath(f'raw_imaging_data_{i:02}', 'reference', file)
                if p.parents[1].exists():
                    cls.addClassCleanup(p.unlink)
                else:
                    # For now these raw_imaging_data_0* folders are created new
                    cls.addClassCleanup(shutil.rmtree, p.parents[1])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()

    def test_register_snapshots(self):
        """Test for MesoscopeRegisterSnapshots.

        NB: More thorough tests of register_snapshots exist in
          ibllib.tests.test_base_tasks.TestRegisterRawDataTask.test_register_snapshots
          ibllib.tests.test_pipes.TestRegisterRawDataTask.test_rename_files
        """
        task = MesoscopeRegisterSnapshots(self.session_path, one=self.one)
        eid, *_ = self.one.search()
        with unittest.mock.patch.object(self.one, 'path2eid', return_value=eid), \
                unittest.mock.patch.object(task, 'register_snapshots') as reg_mock:
            status = task.run()
            reg_mock.assert_called_once_with(collection=['raw_imaging_data_*', ''])
        self.assertEqual(0, status)

    def test_get_signature(self):
        task = MesoscopeRegisterSnapshots(self.session_path, one=self.one)
        task.get_signatures()
        N = 2  # Number of raw_imaging_data collections
        self.assertEqual(len(task.signature['input_files']) * N, len(task.input_files))
        self.assertEqual(len(task.signature['output_files']) * N, len(task.output_files))


class TestMesoscopePreprocess(base.IntegrationTest):
    session_path = None

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
        # Check sparse mask files
        sparse_files = sorted(f for f in files if f.suffix == '.sparse_npz')
        self.assertEqual(2, len(sparse_files))
        arr = sparse.load_npz(sparse_files[0])
        self.assertEqual((222, 512, 512), arr.shape)
        # Check first 10 non-zero elements of the first ROI
        mask = arr[0].todense()
        expected = [1.9042398, 2.0305383, 3.5443015, 4.247522, 3.14291, 2.286991,
                    3.8462281, 3.553623, 2.456772, 3.4159436]
        np.testing.assert_array_almost_equal(expected, mask[np.nonzero(mask)][:10])
        # Check ROI UUIDs were generated
        self.assertTrue((uuids_file := files[0].with_name('mpciROIs.uuids.csv')).exists())
        try:
            uuids = pd.read_csv(uuids_file).squeeze().apply(UUID)
        except ValueError as ex:
            self.assertFalse(True, f'failed to load and parse mpciROIs.uuids.csv: {ex}')
        expected_rois = 222
        self.assertEqual(uuids.size, expected_rois)
        self.assertEqual(uuids.nunique(), expected_rois)

    def test_rename_with_qc(self):
        """Test MesoscopePreprocess._rename_outputs method with frame QC input."""
        session_path = get_session_path(self.suite2pdir)
        task = MesoscopePreprocess(session_path, one=self.one)
        # Check frameQC is saved
        frameQC_names = pd.DataFrame([(0, 'ok'), (1, 'foo')], columns=['qc_values', 'qc_labels'])
        files = task._rename_outputs(self.suite2pdir.parent, frameQC_names, np.zeros(15))
        self.assertIn(files[0].with_name('mpciFrameQC.names.tsv'), files)
        self.assertIn(files[0].with_name('mpci.mpciFrameQC.npy'), files)

    @classmethod
    def tearDownClass(cls) -> None:
        if not cls.session_path:
            return
        for file in cls.session_path.joinpath('raw_sync_data').glob('_timeline_sync.*.npy'):
            file.unlink()


class TestMesoscopeCompress(base.IntegrationTest):
    """Test for MesoscopeCompress task."""
    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)

        self.alf_path = Path(tempdir.name, 'test', '2023-03-03', '002', 'raw_imaging_data_00')
        self.alf_path.mkdir(parents=True)
        for i in range(2):
            with open(str(self.alf_path / f'2023-03-03_2_test_2P_00001_{i:05}.tif'), 'wb') as fp:
                np.save(fp, np.zeros((512, 512, 2), dtype=np.int16))

        # Touch some unnecessary files
        for name in ('_ibl_rawImagingData.meta.json', '2023-03-03_2_test_2P_00001_00001.mat'):
            self.alf_path.joinpath(name).touch()

        self.one = ONE(**base.TEST_DB)

    def test_compress(self):
        task = MesoscopeCompress(self.alf_path.parent, one=self.one)

        # Check fails if compressed file too small
        self.assertEqual(-1, task.run(remove_uncompressed=True))
        self.assertIn('Compressed file < 1KB', task.log)

        # Shouldn't unlink files if compression failed
        tif_files = list(self.alf_path.glob('*.tif'))
        self.assertEqual(2, len(tif_files), 'deleted tif files after failed compression')

        self.alf_path.joinpath('imaging.frames.tar.bz2').unlink()
        # With a mocked file size the task should complete
        status = task.run(verify_min_size=False, remove_uncompressed=True)
        self.assertFalse(status, 'compression task failed')

        self.assertTrue(self.alf_path.joinpath('imaging.frames.tar.bz2').exists())
        # Should delete the tifs after compression
        self.assertFalse(any(x.exists() for x in tif_files), 'failed to remove tifs')
        tfile = tarfile.open(self.alf_path.joinpath('imaging.frames.tar.bz2'))
        self.assertEqual(set(tfile.getnames()), set(x.name for x in tif_files))
