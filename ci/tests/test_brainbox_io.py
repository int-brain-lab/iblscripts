import logging
import hashlib

import numpy as np

from one.api import One
from one.alf.io import load_object
import brainbox.io.one as bbone
from neuropixel import trace_header
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader

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
