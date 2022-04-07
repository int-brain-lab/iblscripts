from one.api import ONE
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader

import unittest
import logging

_logger = logging.getLogger('ibllib')
_logger.setLevel(10)

ba = AllenAtlas()
one = ONE(base_url='https://openalyx.internationalbrainlab.org')


class TestReadSpikeSorting(unittest.TestCase):

    def test_spike_sorting_loader(self):
        # insertions = one.alyx.rest('insertions', 'list')
        pid = 'da8dfec1-d265-44e8-84ce-6ae9c109b8bd'
        self = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = self.load_spike_sorting()
        SpikeSortingLoader.merge_clusters(spikes, clusters, channels)

        assert set(['depths', 'clusters', 'amps', 'times']) == set(spikes.keys())
        assert str(self.spike_sorting_path.relative_to(self.session_path)) == self.collection
        assert self.histology == 'alf'
        assert len(self.collections) == 1
