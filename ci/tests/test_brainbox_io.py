import logging
import hashlib

from one.api import One

from ci.tests.base import IntegrationTest
import brainbox.io.one as bbone
from ibllib.atlas.regions import BrainRegions

_logger = logging.getLogger('ibllib')
_logger.setLevel(10)
br = BrainRegions()


def _check(times, spike_sorter='pykilosort'):
    if spike_sorter == 'pykilosort':
        hash = '973e6ecc820dd18cae3098ff12e45bddf47e4c08'
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

    def test_read_spike_sorting(self):
        pname = 'probe01'
        one = self.one
        eid = one.path2eid(self.session_path)

        spikes, clusters, channels = bbone.load_spike_sorting_fast(
            eid, one=one, probe=pname, spike_sorter=None, revision=None)
        _check(spikes[pname]['times'])
        assert channels[pname]['acronym'] is None
        spikes, clusters, channels = bbone.load_spike_sorting_fast(
            eid, one=one, probe=pname, spike_sorter=None, revision=None, brain_regions=br)
        _check(spikes[pname]['times'])
        assert len(channels[pname]['acronym']) == 384
        assert 'acronym' in clusters[pname].keys()

        # try loading data that doesn't exist
        spikes, clusters, channels = bbone.load_spike_sorting_fast(
            eid, one=one, probe=pname, spike_sorter='not_on_list', revision=None)
        assert spikes == {}
        spikes, clusters, channels = bbone.load_spike_sorting_fast(
            eid, one=one, probe='not_on_list', spike_sorter=None, revision=None)
        assert spikes == {}

        # makes sure this is the pykilosort that is returned by default1
        spikes, clusters, channels = bbone._load_spike_sorting(
            eid=eid, one=one, collection=f'alf/*{pname}/*', return_channels=True)
        _check(spikes[pname]['times'])

        # load spike sorting for a non default sorter
        spikes, clusters, channels = bbone.load_spike_sorting_fast(
            eid, one=one, probe=pname, spike_sorter='ks2_preproc_tests', revision=None)
        _check(spikes[pname]['times'], spike_sorter='ks2_preproc_tests')

        # load spike sorting for a non default sorter at the folder root
        spikes, clusters, channels = bbone.load_spike_sorting_fast(
            eid, one=one, probe=pname, spike_sorter='', revision=None)
        _check(spikes[pname]['times'], spike_sorter='')


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
