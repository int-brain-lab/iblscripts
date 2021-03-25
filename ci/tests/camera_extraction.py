"""Integration tests for the camera extraction and QC
Folders used:
    - camera/
    - ephys/ephys_choice_world_task/ibl_witten_27/2021-01-21/001/
    - Subjects_init/ZM_1098/2019-01-25/001/

NB: These tests hit the main Alyx database.  This is required for full coverage of the QC (in
particular the CameraQC._ensure_required_data method and MotionAlign using eid2ref).
"""
import unittest
from unittest import mock
from pathlib import Path
from collections import OrderedDict
import tempfile
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ibllib.io.extractors.video_motion import MotionAlignment
from ibllib.io.extractors.ephys_fpga import get_main_probe_sync
import ibllib.io.extractors.camera as camio
import ibllib.io.raw_data_loaders as raw
import ibllib.qc.camera as camQC
from ibllib.qc.camera import CameraQC
from ibllib.qc.base import CRITERIA
import ibllib.io.video as vidio
from ibllib.pipes.training_preprocessing import TrainingVideoCompress
from ibllib.pipes.ephys_preprocessing import EphysVideoCompress
from brainbox.core import Bunch
import alf.io as alfio
import oneibl.params
from oneibl.one import ONE

from ci.tests import base


def _get_video_lengths(eid):
    urls = vidio.url_from_eid(eid)
    return {k: camio.get_video_length(v) for k, v in urls.items()}


def _save_qc_frames(qc, **kwargs):
    """
    Given a QC object, save the frames required for wheel alignment, etc.
    This may then be used as a test fixture.
    :param qc:
    :return:
    """
    if all(x is None for x in qc.data.values()):
        # Get wheel period for alignment frame indices
        qc.load_data(load_video=False, **kwargs)
    length = camio.get_video_length(qc.video_path)
    indices = np.linspace(100, length - 100, qc.n_samples).astype(int)
    frame_ids = np.insert(indices, 0, 0)  # First read is not saved and may be
    # re-read
    wheel_present = camQC.data_for_keys(('position', 'timestamps', 'period'), qc.data['wheel'])
    if wheel_present and qc.side != 'body':
        a, b = qc.data.wheel.period
        mask = np.logical_and(qc.data.timestamps >= a, qc.data.timestamps <= b)
        wheel_align_frames, = np.where(mask)
        # Again, the first read is not saved and may be re-read, so repeat the first index
        wheel_align_frames = np.insert(wheel_align_frames, 0, wheel_align_frames[0])
        frame_ids = np.r_[frame_ids, wheel_align_frames]

    # load and save the frames to file
    frames = vidio.get_video_frames_preload(qc.video_path, frame_ids)
    file = base.IntegrationTest.default_data_root() / 'camera' / (qc.eid + '_frame_samples.npy')
    if not file.parent.exists():
        file.parent.mkdir()
    np.save(file, frames)
    assert file.exists()


class TestTrainingCameraExtractor(base.IntegrationTest):

    def setUp(self) -> None:
        self.training_eid = '7082d576-4eb4-41dc-a16e-8a742829a83a'  # For reference
        self.session_path = self.data_path / 'camera' / 'FMR007' / '2021-02-25' / '001'
        self.n_frames = 107913  # Number of frames in video
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Use non-interactive backend
        self.addCleanup(matplotlib.use, backend)
        self.addCleanup(plt.close, 'all')

    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_extract_all(self, mock_vc):
        mock_vc().get.return_value = self.n_frames

        out, fil = camio.extract_all(self.session_path, save=False)
        self.assertTrue(len(out), 3)
        self.assertTrue(all(x is None for x in fil))

    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_extraction(self, mock_vc):
        """
        Mock the VideoCapture class of cv2 so that we can control the number of frames
        :param mock_vc:
        :return:
        """
        mock_vc().get.return_value = self.n_frames
        mock_vc().isOpened.return_value = True

        ext = camio.CameraTimestampsBpod(self.session_path)
        ts, _ = ext.extract(save=False)
        self.assertEqual(ts.size, self.n_frames, 'unexpected size')
        self.assertTrue(not np.isnan(ts).any(), 'nans in timestamps')
        self.assertTrue(np.all(np.diff(ts) > 0), 'timestamps not strictly increasing')
        expected = np.array([1.27701818, 1.33762424, 1.36792727, 1.3982303, 1.42853333,
                             1.45883636, 1.48913939, 1.51944242, 1.54974545, 1.58004848])
        np.testing.assert_array_almost_equal(ts[:10], expected)

        # Test extraction parameters
        ts, _ = ext.extract(save=False, display=True, extrapolate_missing=False)
        self.assertEqual(ts.size, self.n_frames, 'unexpected size')
        self.assertEqual(np.isnan(ts).sum(), 388, 'unexpected number of nans')
        # Verify plots
        figs = [plt.figure(i) for i in plt.get_fignums()]
        lines = figs[0].axes[0].lines
        actual = {ln._label: len(ln._xy) for ln in lines}
        expected = {
            'assigned GPIO up state': 93,
            'unassigned GPIO up state': 1422,
            'audio onset': 2391,
            'assigned audio onset': 1515,
            'audio TTLs': 6624,
            'GPIO': 126,
            'cam times': 107913,
            'assigned audio TTL': 124
        }
        self.assertEqual(actual, expected, 'unexpected plot')

        lines = figs[1].axes[0].lines
        actual = {ln._label: len(ln._xy) for ln in lines}
        expected = {'GPIO': 107913, 'FPGA timestamps': 107913, 'audio TTL': 186}
        self.assertEqual(actual, expected, 'unexpected plot')

        # Test behaviour when some Bpod input values are empty
        """
        I haven't yet seen this behaviour in the wild although 
        CameraTimestampsBpod._times_from_bpod looks for this.  Hence I don't expect the 
        extraction to be successful, but we may need to if we discover such sessions.
        """
        trials = raw.load_data(self.session_path)
        for i in range(5):
            trials[i]['behavior_data']['Events timestamps']['Port1In'] = None
        with self.assertRaises(AssertionError):
            ts, _ = ext.extract(save=False, bpod_trials=trials)

        # Test behaviour when frame count array longer than number of frames
        mock_vc().get.return_value = self.n_frames - 400
        ts, _ = ext.extract(save=False)

    @mock.patch('ibllib.io.extractors.camera.raw.load_embedded_frame_data')
    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_basic_extraction(self, mock_vc, mock_aux):
        """
        Tests extraction of a session without pin state and GPIO files, etc.
        :param mock_vc: A mock OpenCV VideoCapture object for stubbing the video length
        :param mock_aux: A mock object for stubbing the load_embedded_frame_data function
        :return:
        """
        mock_vc().get.return_value = self.n_frames
        mock_vc().isOpened.return_value = True

        # Act as though the embedded frame data files don't exist
        mock_aux.return_value = (None, [None] * 4)
        ext = camio.CameraTimestampsBpod(self.session_path)
        ts, _ = ext.extract(save=False)

        # Verify behaviour when no frame data and fewer timestamps than frames
        self.assertEqual(ts.size, self.n_frames)
        expected = np.array([15901.65804537, 15934.55278222,
                             15967.44751906, 16000.3422559, 16033.23699274])
        np.testing.assert_array_almost_equal(ts[-5:], expected)

        # Verify behaviour when no frame data and frames than timestamps
        # Expect raw Bpod times to be returned
        mock_vc().get.return_value = self.n_frames - 400
        ts, _ = ext.extract(save=False)
        self.assertEqual(ts.size, 107525)  # NB: This behaviour will change in the future


class TestEphysCameraExtractor(base.IntegrationTest):

    def setUp(self) -> None:
        self.ephys_eid = '6c6983ef-7383-4989-9183-32b1a300d17a'
        self.session_path = self.data_path / 'camera' / 'SWC_054' / '2020-10-07' / '001'
        self.n_frames = OrderedDict(left=255617, right=641484, body=128035)
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Use non-interactive backend
        self.addCleanup(matplotlib.use, backend)
        self.addCleanup(plt.close, 'all')

    # @unittest.skip
    # def test_load_embedded_frame_data(self):
    #     # eid = 'c7832bca-5cfb-4676-a1ec-f87cd7640ae5'  # messed up pin state
    #     pass

    def test_groom_pin_state(self):
        # ibl_witten_27\2021-01-14\001  # Can't assign a pin state
        # CSK-im-002\2021-01-16\001  # Another example
        root = self.data_path
        session_path = root.joinpath('ephys', 'ephys_choice_world_task',
                                     'ibl_witten_27', '2021-01-21', '001')
        _, ts = raw.load_camera_ssv_times(session_path, 'left')
        _, (*_, gpio) = raw.load_embedded_frame_data(session_path, 'left')
        bpod_trials = raw.load_data(session_path)
        _, audio = raw.load_bpod_fronts(session_path, bpod_trials)
        # NB: syncing the timestamps to the audio doesn't work very well but we don't need it to
        # for the extraction, so long as the audio and GPIO fronts match.
        gpio, audio, _ = camio.groom_pin_state(gpio, audio, ts)
        # Do some checks
        self.assertEqual(gpio['indices'].size, audio['times'].size)
        expected = np.array([164179, 164391, 164397, 164900, 164906], dtype=int)
        np.testing.assert_array_equal(gpio['indices'][-5:], expected)
        expected = np.array([2734.4496, 2737.9659, 2738.0659, 2746.4488, 2746.5488])
        np.testing.assert_array_almost_equal(audio['times'][-5:], expected)

        # Verify behaviour when audio and GPIO match in size
        _, audio_, _ = camio.groom_pin_state(gpio, audio, ts, take='nearest', tolerance=.5)
        self.assertEqual(audio, audio_)

        # Verify behaviour when there are GPIO fronts beyond number of video frames
        ts_short = ts[:gpio['indices'].max() - 10]
        gpio_, *_ = camio.groom_pin_state(gpio, audio, ts_short)
        self.assertFalse(np.any(gpio_['indices'] >= ts.size))

    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_extract_all(self, mock_vc):
        mock_vc().get.side_effect = self.n_frames.values()

        out, fil = camio.extract_all(self.session_path, save=False)
        self.assertTrue(len(out), 3)
        self.assertFalse(fil)

        with self.assertRaises(ValueError):
            camio.extract_all(self.session_path, save=False, session_type='fake')
        with self.assertRaises(ValueError):
            camio.extract_all(self.session_path, save=False, labels=('head', 'tail', 'front'))

    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_extraction(self, mock_vc):
        """
        Mock the VideoCapture class of cv2 so that we can control the number of frames
        :param mock_vc:
        :return:
        """
        side = 'left'
        n_frames = self.n_frames[side]  # Number of frames in video
        mock_vc().get.return_value = n_frames
        mock_vc().isOpened.return_value = True

        # out = camio.extract_all(session_path, save=False)
        ext = camio.CameraTimestampsFPGA(side, self.session_path)
        sync, chmap = get_main_probe_sync(self.session_path)

        ts, _ = ext.extract(save=False, sync=sync, chmap=chmap)
        self.assertEqual(ts.size, n_frames, 'unexpected size')
        self.assertTrue(not np.isnan(ts).any(), 'nans in timestamps')
        self.assertTrue(np.all(np.diff(ts) > 0), 'timestamps not strictly increasing')
        expected = np.array([197.76558813, 197.79905145, 197.81578311, 197.83251477,
                             197.84924643, 197.86597809, 197.88270975, 197.89944141,
                             197.91617307, 197.93290473])
        np.testing.assert_array_almost_equal(ts[:10], expected)

        # Test extraction parameters
        ts, _ = ext.extract(save=False, sync=sync, chmap=chmap,
                            display=True, extrapolate_missing=False)
        self.assertEqual(ts.size, n_frames, 'unexpected size')
        self.assertEqual(np.isnan(ts).sum(), 499, 'unexpected number of nans')
        # Verify plots
        figs = [plt.figure(i) for i in plt.get_fignums()]
        lines = figs[0].axes[0].lines
        actual = {ln._label: len(ln._xy) for ln in lines}
        expected = {
            'audio TTLs': 3400,
            'GPIO': 3394,
            'cam times': 255617,
            'assigned audio TTL': 3392
        }
        self.assertEqual(actual, expected, 'unexpected plot')

        lines = figs[1].axes[0].lines
        actual = {ln._label: len(ln._xy) for ln in lines}
        expected = {'GPIO': 255617, 'FPGA timestamps': 255617, 'audio TTL': 5088}
        self.assertEqual(actual, expected, 'unexpected plot')

    @mock.patch('ibllib.io.extractors.camera.raw.load_embedded_frame_data')
    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_basic_extraction(self, mock_vc, mock_aux):
        """
        Tests extraction of a session without pin state and GPIO files, etc.
        :param mock_vc: A mock OpenCV VideoCapture object for stubbing the video length
        :param mock_aux: A mock object for stubbing the load_embedded_frame_data function
        :return:
        """
        side = 'left'
        mock_vc().get.return_value = self.n_frames[side]
        mock_vc().isOpened.return_value = True

        # Act as though the embedded frame data files don't exist
        mock_aux.return_value = (None, [None] * 4)
        ext = camio.CameraTimestampsFPGA(side, self.session_path)
        sync, chmap = get_main_probe_sync(self.session_path)
        ts, _ = ext.extract(save=False, sync=sync, chmap=chmap)

        # Verify returns unaltered FPGA times.  This behaviour will change in the future
        self.assertEqual(ts.size, 255505)
        expected = np.array([0.01363197, 0.03036363, 0.04709529, 0.06382695, 0.08055861])
        np.testing.assert_array_almost_equal(ts[:5], expected)

    def test_get_video_length(self):
        # Verify using URL
        url = (oneibl.params.get().HTTP_DATA_SERVER +
               '/mainenlab/Subjects/ZM_1743/2019-06-14/001/raw_video_data/'
               '_iblrig_leftCamera.raw.71cfeef2-2aa5-46b5-b88f-ca07e3d92474.mp4')
        length = camio.get_video_length(url)
        self.assertEqual(length, 144120)

        # Verify using local path
        video_path = next(
            self.data_path.joinpath('Subjects_init/ZM_1098/2019-01-25/001').rglob('*.mp4')
        )
        length = camio.get_video_length(video_path)
        self.assertEqual(length, 34442)


class TestVideoQC(base.IntegrationTest):
    """Test the video QC on some failure cases"""

    @classmethod
    def setUpClass(cls) -> None:
        """Load a few 10 second videos for testing the various video QC checks"""
        data_path = base.IntegrationTest.default_data_root()
        video_path = data_path.joinpath('camera')
        videos = video_path.rglob('*.mp4')
        # dummy = data_path.joinpath('Subjects_init', 'ZM_1085', '2019-06-24', '001')
        dummy_id = '2da56f6b-e04f-4847-9b44-686abc1e57d2'
        qc = CameraQC(dummy_id, 'left',
                      n_samples=10, stream=False, download_data=False, one=ONE(offline=True))
        qc.one = None
        qc._type = 'ephys'  # All videos come from ephys sessions
        qcs = {}
        for video in videos:
            qc.video_path = video
            qc.side = vidio.label_from_path(video)
            qc.n_samples = 10
            qc.load_video_data()
            qcs[video] = qc.data.copy()
        cls.qc = qc
        cls.data = qcs

    def tearDown(self) -> None:
        plt.close('all')

    def test_video_checks(self, display=False):
        # A tuple of QC checks and the expected outcome for each 10 second video
        video_checks = (
            (self.qc.check_position, (3, 3, 3, 3, 3, 1, 3, 1, 2, 3, 3, 3, 3, 3, 1, 1, 3, 1)),
            (self.qc.check_focus, (3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
            (self.qc.check_brightness, (3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 3, 3)),
            (self.qc.check_file_headers, [1] * 18),
            (self.qc.check_resolution, (3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 1, 3, 1))
        )

        # For each check get the outcome and determine whether it matches our expected outcome
        # for each video
        for (check, expected) in video_checks:
            check_name = check.__name__
            outcomes = []
            frame_samples = []
            for path, data in self.data.items():
                self.qc.data = data
                self.qc.side = vidio.label_from_path(path)
                outcomes.append(check())
                frame_samples.append(data.frame_samples[0])

            # If display if True, plot 1 frame per video along with its outcome
            # This is purely for manual inspection
            if display:  # Check outcomes look reasonable by eye
                fig, axes = plt.subplots(int(len(self.data) / 4), 4)
                [self.qc.imshow(frm, ax=ax, title=o)
                 for frm, ax, o in zip(frame_samples, axes.flatten(), outcomes)]
                fig.suptitle(check_name)
                plt.show()

            # Verify the outcome for each video matches what we expect
            [self.assertTrue(x == CRITERIA[y], f'Unexpected outcome for {check_name} video {i}')
             for i, (x, y) in enumerate(zip(expected, outcomes))]


class TestCameraQC(base.IntegrationTest):
    """Test the video QC on some failure cases"""

    def setUp(self) -> None:
        self.incomplete = self.data_path.joinpath('Subjects_init', 'ZM_1098', '2019-01-25', '001')
        self.ephys = (
            '6c6983ef-7383-4989-9183-32b1a300d17a',
            self.data_path.joinpath('camera', 'SWC_054', '2020-10-07', '001')
        )
        self.training = (
            '7082d576-4eb4-41dc-a16e-8a742829a83a',
            self.data_path.joinpath('camera', 'FMR007', '2021-02-25', '001')
        )
        self._call_count = -1
        self.frames = np.array([])

    def test_incomplete_session(self):
        # Verify using local path
        session_path = self.incomplete

        qc = CameraQC(session_path, 'left',
                      stream=False, download_data=False, one=ONE(offline=True), n_samples=20)
        outcome, extended = qc.run(update=False)
        self.assertEqual('FAIL', outcome)
        expected = {
            '_videoLeft_brightness': 'FAIL',
            '_videoLeft_camera_times': ('PASS', 0),
            '_videoLeft_dropped_frames': 'NOT_SET',
            '_videoLeft_file_headers': 'PASS',
            '_videoLeft_focus': 'PASS',
            '_videoLeft_framerate': 'NOT_SET',
            '_videoLeft_pin_state': 'NOT_SET',
            '_videoLeft_position': 'WARNING',
            '_videoLeft_resolution': 'PASS',
            '_videoLeft_timestamps': 'NOT_SET',
            '_videoLeft_wheel_alignment': 'NOT_SET'
        }
        self.assertEqual(expected, extended)

    @mock.patch('ibllib.qc.camera.get_video_meta')
    @mock.patch('ibllib.io.video.cv2.VideoCapture')
    def test_ephys_session(self, mock_ext, mock_meta):
        """
        Tests the full QC process for an ephys session.  Mock a load of things so we don't need
        the ful video file.
        :param mock_ext: mock cv.VideoCapture in camera extractor module
        :param mock_meta: mock get_video_meta
        :return:
        """
        n_samples = 100
        length = 255617
        eid, session_path = self.ephys
        self.frames = np.load(self.data_path / 'camera' / f'{eid}_frame_samples.npy')
        mock_meta.return_value = \
            Bunch({'length': length, **CameraQC.video_meta['ephys']['left']})
        mock_ext().get.return_value = length
        mock_ext().read.side_effect = self.side_effect()

        # Run QC for the left side
        qc = camQC.run_all_qc(session_path, cameras=('left',), stream=False, update=False,
                              n_samples=n_samples, download_data=False, extract_times=True)
        self.assertIsInstance(qc, dict)
        self.assertFalse(qc['left'].download_data)
        self.assertEqual(qc['left'].type, 'ephys')
        expected = {
            '_videoLeft_brightness': 'PASS',
            '_videoLeft_camera_times': ('PASS', 0),
            '_videoLeft_dropped_frames': ('WARNING', 1, 1),
            '_videoLeft_file_headers': 'PASS',
            '_videoLeft_focus': 'PASS',
            '_videoLeft_framerate': ('PASS', 59.767),
            '_videoLeft_pin_state': ('WARNING', 2, 1),
            '_videoLeft_position': 'FAIL',
            '_videoLeft_resolution': 'PASS',
            '_videoLeft_timestamps': 'PASS',
            '_videoLeft_wheel_alignment': ('PASS', 0)
        }
        self.assertEqual(expected, qc['left'].metrics)

    @mock.patch('ibllib.qc.camera.get_video_meta')
    @mock.patch('ibllib.io.video.cv2.VideoCapture')
    def test_training_session(self, mock_ext, mock_meta):
        """
        Tests the full QC process for a training session.  Mock a load of things so we don't need
        the ful video file.
        :param mock_ext: mock cv.VideoCapture in camera extractor module
        :param mock_meta: mock get_video_meta
        :return:
        """
        n_samples = 100
        length = 107913
        eid, session_path = self.training
        self.frames = np.load(self.data_path / 'camera' / f'{eid}_frame_samples.npy')
        mock_meta.return_value = \
            Bunch({'length': length, **CameraQC.video_meta['training']['left']})
        mock_ext().get.return_value = length
        mock_ext().read.side_effect = self.side_effect()

        qc = CameraQC(session_path, 'left', stream=False, n_samples=n_samples)
        qc.load_data(download_data=False, extract_times=True)
        outcome, extended = qc.run(update=False)
        self.assertEqual('FAIL', outcome)
        expected = {
            '_videoLeft_brightness': 'FAIL',
            '_videoLeft_camera_times': ('PASS', 0),
            '_videoLeft_dropped_frames': ('PASS', 1, 0),
            '_videoLeft_file_headers': 'PASS',
            '_videoLeft_focus': 'FAIL',
            '_videoLeft_framerate': ('FAIL', 32.895),
            '_videoLeft_pin_state': ('WARNING', 1151, 0),
            '_videoLeft_position': 'FAIL',
            '_videoLeft_resolution': 'PASS',
            '_videoLeft_timestamps': 'PASS',
            '_videoLeft_wheel_alignment': ('FAIL', -95)
        }
        self.assertEqual(expected, extended)

    def side_effect(self):
        for frame in self.frames:
            yield True, frame


class TestCameraPipeline(base.IntegrationTest):

    def setUp(self) -> None:
        self.training_folder = self.data_path.joinpath('training', 'CSHL_003', '2019-04-05', '001')
        self.ephys_folder = self.data_path.joinpath('ephys', 'choice_world_init',
                                                    'KS022', '2019-12-10', '001')
        if not self.ephys_folder.exists():
            raise FileNotFoundError(f'Fixture {self.ephys_folder} does not exist')
        if not self.training_folder.exists():
            raise FileNotFoundError(f'Fixture {self.training_folder} does not exist')
        self.one = ONE(offline=True)

    def test_training(self):
        with tempfile.TemporaryDirectory() as tdir:
            subjects_path = Path(tdir).joinpath('Subjects', *self.training_folder.parts[-3:])
            session_path = shutil.copytree(self.training_folder, subjects_path)
            # task running part - there are no videos, should exit gracefully
            job = TrainingVideoCompress(session_path, one=self.one)
            with self.assertLogs('ibllib', level='INFO'):
                job.run()

            self.assertEqual(job.status, 0)
            self.assertTrue('skipping' in job.log)

            # Now mock the video data so that extraction and QC succeed
            video_path = session_path.joinpath('raw_video_data')
            video_path.mkdir(parents=True)
            video_path.joinpath('_iblrig_leftCamera.raw.mp4').touch()
            with mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture') as mock_vc, \
                    mock.patch('ibllib.pipes.training_preprocessing.CameraQC') as mock_qc:
                def side_effect():
                    return True, np.random.randint(0, 255, size=(1024, 1280, 3))
                mock_vc().read.side_effect = side_effect
                length = 68453
                mock_vc().get.return_value = length
                job.run()
                self.assertEqual(job.status, 0)
                self.assertEqual(mock_qc.call_args.args, (session_path, 'left'))
                mock_qc().run.assert_called_once_with(update=True)
                self.assertTrue(len(job.outputs) > 0)

            # check the file objects
            ts = alfio.load_object(session_path / 'alf', 'leftCamera')
            self.assertTrue(ts['times'].size == length)

    @mock.patch('ibllib.qc.camera.CameraQC')
    def test_ephys(self, mock_qc):
        # task running part
        job = EphysVideoCompress(self.ephys_folder, one=self.one)
        job.run()

        self.assertEqual(job.status, 0)
        self.assertEqual(len(mock_qc.call_args_list), 3)  # Once per camera
        labels = ('left', 'right', 'body')
        self.assertCountEqual(labels, [arg.kwargs['side'] for arg in mock_qc.call_args_list])

        [self.assertEqual(call[0][0], self.ephys_folder) for call in mock_qc.call_args_list]
        mock_qc().run.assert_called_with(update=True)
        self.assertEqual(len(job.outputs), 6)

        # check the file objects
        for label in labels:
            ts = alfio.load_object(self.ephys_folder / 'alf', f'{label}Camera')
            self.assertTrue(ts['times'].size > 0)


class TestWheelMotionNRG(base.IntegrationTest):

    def setUp(self) -> None:
        self.eid = '6c6983ef-7383-4989-9183-32b1a300d17a'
        self.frames = np.load(self.data_path / 'camera' / f'{self.eid}_frame_samples.npy')

    def side_effect(self):
        for frame in self.frames:
            yield True, frame

    @mock.patch('ibllib.io.video.cv2.VideoCapture')
    def test_wheel_motion(self, mock_cv):
        side = 'left'
        period = np.array([1730.3513333, 1734.1743333])
        mock_cv().read.side_effect = self.side_effect()
        aln = MotionAlignment(self.eid)
        one = aln.one
        cam = alfio.load_object(aln.session_path / 'alf', f'{side}Camera')
        aln.data.camera_times = {side: cam['times']}
        aln.video_paths = {
            side: one.path_from_eid(aln.eid) / 'raw_video_data' / f'_iblrig_{side}Camera.raw.mp4'
        }
        aln.data.wheel = alfio.load_object(aln.session_path / 'alf', 'wheel')

        dt_i, c, df = aln.align_motion(period=period, side=side)
        expected = np.array([0.90278801, 0.68067675, 0.73734772, 0.82648895, 0.80950881,
                             0.88054471, 0.84264046, 0.302118, 0.94302567, 0.86188695])
        np.testing.assert_array_almost_equal(expected, df[:10])
        self.assertEqual(dt_i, 0)
        self.assertEqual(c, 19.48841741869628)

        # Test saving alignment video
        with tempfile.TemporaryDirectory() as tdir:
            aln.plot_alignment(save=tdir)
            vid = next(Path(tdir).glob('*.mp4'))
            self.assertTrue(vid.name.endswith('2020-10-07_1_SWC_054_l.mp4'))
            self.assertEqual(vid.stat().st_size, 1799319)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
