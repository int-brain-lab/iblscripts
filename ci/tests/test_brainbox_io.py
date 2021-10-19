import logging
import hashlib

from one.api import One

from ci.tests.base import IntegrationTest
from brainbox.io.one import _load_spike_sorting, load_spike_sorting_fast

_logger = logging.getLogger('ibllib')
_logger.setLevel(10)


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

        spikes, clusters, channels = load_spike_sorting_fast(eid, one=one, probe=pname, spike_sorter=None, revision=None)
        _check(spikes[pname]['times'])

        # try loading data that doesn't exist
        spikes, clusters, channels = load_spike_sorting_fast(eid, one=one, probe=pname, spike_sorter='not_on_list', revision=None)
        assert spikes == {}
        spikes, clusters, channels = load_spike_sorting_fast(eid, one=one, probe='not_on_list', spike_sorter=None, revision=None)
        assert spikes == {}

        # makes sure this is the pykilosort that is returned by default1
        spikes, clusters, channels = _load_spike_sorting(eid=eid, one=one, collection=f'alf/*{pname}/*', return_channels=True)
        _check(spikes[pname]['times'])

        # load spike sorting for a non default sorter
        spikes, clusters, channels = load_spike_sorting_fast(eid, one=one, probe=pname, spike_sorter='ks2_preproc_tests', revision=None)
        _check(spikes[pname]['times'], spike_sorter='ks2_preproc_tests')

        # load spike sorting for a non default sorter at the folder root
        spikes, clusters, channels = load_spike_sorting_fast(eid, one=one, probe=pname, spike_sorter='', revision=None)
        _check(spikes[pname]['times'], spike_sorter='')
