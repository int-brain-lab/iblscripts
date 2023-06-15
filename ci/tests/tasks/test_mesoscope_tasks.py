import logging
import shutil
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, ANY
import tarfile
from itertools import chain

import numpy as np
import pandas as pd
import sparse

import one.alf.io as alfio
from one.alf.files import get_session_path
from one.api import ONE

from ibllib.pipes.mesoscope_tasks import \
    MesoscopeSync, MesoscopeFOV, MesoscopeRegisterSnapshots, MesoscopePreprocess, MesoscopeCompress
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
        self.assertEqual(16, len(trials.keys()))
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

    def test_get_wheel_positions(self):
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


@unittest.skip('TODO')
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
        nROIs = 9
        alf_path = self.session_path_0.joinpath('alf')
        ROI_folders = list(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nROIs, len(ROI_folders))
        ROI_times = sorted(list(alf_path.rglob('mpci.times.npy')))
        self.assertEqual(nROIs, len(ROI_times))
        expected = [1.106, 1.304, 1.503, 1.701, 1.899]
        np.testing.assert_array_almost_equal(np.load(ROI_times[0])[:5], expected)
        ROI_shifts = list(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nROIs, len(ROI_shifts))
        expected = [0., 4.157940e-05, 8.315880e-05, 1.247382e-04, 1.663176e-04]
        np.testing.assert_array_almost_equal(np.load(ROI_shifts[0])[:5], expected)

    def test_mesoscope_sync_multiple(self):
        task = MesoscopeSync(self.session_path_1, device_collection='raw_imaging_data*', one=self.one)
        status = task.run()
        assert status == 0

        # Check output
        nROIs = 6
        alf_path = self.session_path_1.joinpath('alf')
        ROI_folders = list(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nROIs, len(ROI_folders))
        ROI_times = list(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nROIs, len(ROI_times))
        expected = [1.0075, 1.154, 1.3, 1.446, 1.5925]
        np.testing.assert_array_almost_equal(np.load(ROI_times[0])[:5], expected)
        ROI_shifts = list(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nROIs, len(ROI_shifts))
        expected = [0., 4.157550e-05, 8.315100e-05, 1.247265e-04, 1.663020e-04]
        np.testing.assert_array_almost_equal(np.load(ROI_shifts[0])[:5], expected)

    def test_get_bout_edges(self):
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

        # Check validation
        collections.append('raw_imaging_data_02')
        self.assertRaises(ValueError, extractor.get_bout_edges, frame_times, collections, events)


class TestMesoscopeRegisterSnapshots(base.IntegrationTest):
    session_path = None
    one = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.one = ONE(**base.TEST_DB)
        cls.session_path = cls.default_data_root().joinpath('mesoscope', 'test', '2023-02-17', '002')
        # Create some reference images to register
        for i in range(2):
            p = cls.session_path.joinpath(
                f'raw_imaging_data_{i:02}', 'reference', '2023-02-17_2_test_2P_00001_00001.tif')
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
        # Check that the reference images were renamed and registered
        expected = ['raw_imaging_data_00/reference/reference.image.tif',
                    'raw_imaging_data_01/reference/reference.image.tif']
        outputs = [o.relative_to(self.session_path).as_posix() for o in task.outputs]
        self.assertCountEqual(expected, outputs)

    def test_get_signature(self):
        task = MesoscopeRegisterSnapshots(self.session_path, one=self.one)
        task.get_signatures()
        expected = [('*.tif', 'raw_imaging_data_00/reference', False),
                    ('*.tif', 'raw_imaging_data_01/reference', False)]
        self.assertCountEqual(expected, task.input_files)
        expected = [('reference.image.tif', 'raw_imaging_data_00/reference', False),
                    ('reference.image.tif', 'raw_imaging_data_01/reference', False)]
        self.assertCountEqual(expected, task.output_files)


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
        with self.assertLogs('ibllib.pipes.mesoscope_tasks', logging.ERROR):
            self.assertEqual(-1, task.run())
            self.assertIn('Compressed file < 1KB', task.log)
        # self.assertRaises(AssertionError, task.run)

        # Shouldn't unlink files if compression failed
        tif_files = list(self.alf_path.glob('*.tif'))
        self.assertEqual(2, len(tif_files), 'deleted tif files after failed compression')

        self.alf_path.joinpath('imaging.frames.tar.bz2').unlink()
        # With a mocked file size the task should complete
        status = task.run(verify_min_size=False)
        self.assertFalse(status, 'compression task failed')

        self.assertTrue(self.alf_path.joinpath('imaging.frames.tar.bz2').exists())
        # Should delete the tifs after compression
        self.assertFalse(any(x.exists() for x in tif_files), 'failed to remove tifs')
        tfile = tarfile.open(self.alf_path.joinpath('imaging.frames.tar.bz2'))
        self.assertEqual(set(tfile.getnames()), set(x.name for x in tif_files))
