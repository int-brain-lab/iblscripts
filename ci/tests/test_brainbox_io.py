import logging
import hashlib

import numpy as np

from one.api import One
from one.alf.io import load_object
import brainbox.io.one as bbone
from neuropixel import trace_header
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader, SessionLoader

from ci.tests.base import IntegrationTest

_logger = logging.getLogger('ibllib')
_logger.setLevel(10)
br = BrainRegions()


def _check(times, spike_sorter='pykilosort'):
    if spike_sorter == 'pykilosort':
        hash = 'f66c53aec01333245acc2fd658339ee9fceda5b9'
    elif spike_sorter == 'ks2_preproc_tests':
        hash = 'e833cd9df46aa791fcec48b52ddff7f96f98a6ab'
    elif spike_sorter == '':
        hash = '5628850285f44fba66ea241cd106d8c7c1871754'
    assert hashlib.sha1(times.tobytes()).hexdigest() == hash


class TestReadSpikeSorting(IntegrationTest):

    def setUp(self) -> None:
        self.root_path = self.data_path.joinpath('brainbox/io/spike_sorting')
        self.session_path = self.root_path.joinpath('SWC_054/2020-10-05/001')
        print('Building ONE cache from filesystem...')
        self.one = One.setup(self.root_path, silent=True)

    def tearDown(self) -> None:
        for file in self.root_path.glob('*.pqt'):
            file.unlink()

    def test_channel_conversion_interpolation(self):
        BUNCH_KEYS = {'x', 'y', 'z', 'acronym', 'atlas_id', 'axial_um', 'lateral_um'}
        ALF_KEYS = {'localCoordinates', 'mlapdv', 'brainLocationIds_ccf_2017'}
        pname = 'probe01'
        alf_channels = load_object(self.session_path.joinpath('alf', pname), 'channels')
        channels = bbone._channels_alf2bunch(alf_channels)
        assert set(channels.keys()) == BUNCH_KEYS

        h = trace_header(1)
        raw_channels = bbone.channel_locations_interpolation(
            alf_channels, {'localCoordinates': np.c_[h['x'], h['y']]})
        assert set(raw_channels.keys()) == ALF_KEYS
        channels = bbone.channel_locations_interpolation(
            alf_channels, {'localCoordinates': np.c_[h['x'], h['y']]}, brain_regions=br)
        assert set(channels.keys()) == BUNCH_KEYS

        # this function should also be able to take a bunch formatted dict as input
        channels = bbone.channel_locations_interpolation(
            channels, {'localCoordinates': np.c_[h['x'], h['y']]}, brain_regions=br)
        assert set(channels.keys()) == BUNCH_KEYS

    def test_read_spike_sorting(self):
        pname = 'probe01'
        one = self.one
        eid = one.path2eid(self.session_path)

        sl = SpikeSortingLoader(eid=eid, pname=pname, one=one)
        _logger.setLevel(0)
        spikes, clusters, channels = sl.load_spike_sorting(spike_sorter='')
        _check(spikes['times'], spike_sorter='')
        clusters = sl.merge_clusters(spikes, clusters, channels)
        assert 'acronym' in clusters.keys()

        # load spike sorting for a non default sorter
        spikes, clusters, channels = sl.load_spike_sorting(spike_sorter='ks2_preproc_tests')
        _check(spikes['times'], spike_sorter='ks2_preproc_tests')

        # load spike sorting using collection
        spikes, clusters, channels = sl.load_spike_sorting(
            collection=f'alf/{pname}/ks2_preproc_tests')
        _check(spikes['times'], spike_sorter='ks2_preproc_tests')

        # makes sure this is the pykilosort that is returned by default1
        spikes, clusters, channels = bbone._load_spike_sorting(
            eid=eid, one=one, collection=f'alf/*{pname}/*', return_channels=True)
        _check(spikes[pname]['times'])

    def test_samples2times(self):
        pname = 'probe01'
        one = self.one
        eid = one.path2eid(self.session_path)
        sl = SpikeSortingLoader(eid=eid, pname=pname, one=one)
        _logger.setLevel(0)
        spikes, _, _ = sl.load_spike_sorting(spike_sorter='', dataset_types=['spikes.samples'])
        # TODO: add the sync files
        assert(np.all(np.abs(sl.samples2times(spikes.samples) - spikes.times) < 1e11))


class TestSessionLoader(IntegrationTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.root_path = cls.default_data_root().joinpath('ephys', 'choice_world_init')
        if not cls.root_path.exists():
            return
        cls.session_path = cls.root_path.joinpath('KS022', '2019-12-10', '001')
        print('Building ONE cache from filesystem...')
        cls.one = One.setup(cls.root_path, silent=True)
        cls.sess_loader = SessionLoader(cls.one, cls.session_path)

    @classmethod
    def tearDownClass(cls) -> None:
        for file in cls.root_path.glob('*.pqt'):
            file.unlink()

    def test_load_trials_data(self):
        expected = [
            'stimOff_times', 'goCueTrigger_times', 'intervals_bpod_0', 'intervals_bpod_1',
            'probabilityLeft', 'contrastRight', 'firstMovement_times', 'goCue_times', 'feedbackType', 'choice',
            'contrastLeft', 'stimOn_times', 'rewardVolume', 'feedback_times', 'response_times',
            'intervals_0', 'intervals_1'
        ]
        self.sess_loader.load_trials()
        self.assertCountEqual(expected, self.sess_loader.trials.columns)
        self.assertEqual((626, 17), self.sess_loader.trials.shape)

    def test_load_wheel(self):
        self.sess_loader.load_wheel(sampling_rate=100, smooth_size=0.05)
        self.assertCountEqual(['times', 'position', 'velocity', 'acceleration'], self.sess_loader.wheel.columns)

    def test_load_pose(self):
        self.sess_loader.load_pose(likelihood_thr=0.9, views=['left', 'body'])
        self.assertIsInstance(self.sess_loader.pose, dict)
        self.assertCountEqual(['leftCamera', 'bodyCamera'], self.sess_loader.pose.keys())
        self.assertIn('times', self.sess_loader.pose['leftCamera'].columns)
        self.assertCountEqual(['times', 'tail_start_x', 'tail_start_y', 'tail_start_likelihood'],
                              self.sess_loader.pose['bodyCamera'].columns)
        self.assertEqual((4000, 4), self.sess_loader.pose['bodyCamera'].shape)
        self.assertEqual((4000, 34), self.sess_loader.pose['leftCamera'].shape)

        all_nan = [c for c in self.sess_loader.pose['leftCamera'].columns if
                   all(np.isnan(self.sess_loader.pose['leftCamera'][c]))]
        self.assertCountEqual(['pupil_bottom_r_x', 'pupil_bottom_r_y'], all_nan)

        self.sess_loader.load_pose(likelihood_thr=0.5, views=['left'])
        all_nan = [c for c in self.sess_loader.pose['leftCamera'].columns if
                   all(np.isnan(self.sess_loader.pose['leftCamera'][c]))]
        self.assertTrue(len(all_nan) == 0)

    def test_load_motion_energy(self):
        self.sess_loader.load_motion_energy()
        self.assertIsInstance(self.sess_loader.motion_energy, dict)
        self.assertCountEqual(['leftCamera', 'rightCamera', 'bodyCamera'], self.sess_loader.motion_energy.keys())
        self.assertCountEqual(['times', 'whiskerMotionEnergy'], self.sess_loader.motion_energy['leftCamera'].columns)
        self.assertCountEqual(['times', 'whiskerMotionEnergy'], self.sess_loader.motion_energy['rightCamera'].columns)
        self.assertCountEqual(['times', 'bodyMotionEnergy'], self.sess_loader.motion_energy['bodyCamera'].columns)
        self.assertCountEqual([158377, 396504, 79095], [df.shape[0] for df in self.sess_loader.motion_energy.values()])

        self.sess_loader.load_motion_energy(views=['left'])
        self.assertCountEqual(['leftCamera'], self.sess_loader.motion_energy.keys())

    def test_load_pupil(self):
        self.sess_loader.load_pupil()
        self.assertCountEqual(['pupilDiameter_raw', 'pupilDiameter_smooth'], self.sess_loader.pupil.columns)
        self.assertEqual(4000, self.sess_loader.pupil.shape[0])

        with self.assertLogs(_logger, level='ERROR') as cm:
            self.sess_loader.load_pupil(snr_thresh=20)
            self.assertEqual(['ERROR:ibllib:Pupil diameter SNR (12.07) below threshold SNR (20), removing data.'],
                             cm.output)
            self.assertTrue(self.sess_loader.pupil.empty)

    def test_load_session_data(self):
        # Instantiate new session loader
        self.sess_loader = SessionLoader(self.one, self.session_path)
        self.sess_loader.load_session_data()
        self.assertTrue(all(self.sess_loader.data_info['is_loaded']))

        # Make sure data is not reloaded
        with self.assertLogs(_logger, level='DEBUG') as cm:
            self.sess_loader.load_session_data()
            self.assertEqual([
                'DEBUG:ibllib:Not loading trials data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading wheel data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading pose data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading motion_energy data, is already loaded and reload=False.',
                'DEBUG:ibllib:Not loading pupil data, is already loaded and reload=False.'
            ], cm.output)
        # Make sure data IS reloaded
        with self.assertLogs(_logger, level='INFO') as cm:
            self.sess_loader.load_session_data(reload=True)
            self.assertEqual([
                'INFO:ibllib:Loading trials data',
                'INFO:ibllib:Loading wheel data',
                'INFO:ibllib:Loading pose data',
                'INFO:ibllib:Loading motion_energy data',
                'INFO:ibllib:Loading pupil data',
                'INFO:ibllib:Pupil diameter not available, trying to compute on the fly.'
            ], cm.output)


class TestLoadTrials(IntegrationTest):

    def setUp(self) -> None:
        self.root_path = self.data_path.joinpath('ephys', 'choice_world_init')
        self.session_path = self.root_path.joinpath('KS022', '2019-12-10', '001')
        print('Building ONE cache from filesystem...')
        self.one = One.setup(self.root_path, silent=True)

    def test_load_trials_df(self):
        eid = self.one.to_eid(self.session_path)
        trials = bbone.load_trials_df(eid, one=self.one)
        expected = [
            'choice', 'probabilityLeft', 'feedbackType', 'feedback_times', 'contrastLeft',
            'contrastRight', 'goCue_times', 'stimOn_times', 'trial_start', 'trial_end'
        ]
        self.assertCountEqual(trials.columns, expected)
        trials = bbone.load_trials_df(eid, one=self.one, ret_wheel=True)
        self.assertIn('wheel_velocity', trials)

    def tearDown(self) -> None:
        for file in self.root_path.glob('*.pqt'):
            file.unlink()
