import tempfile
import unittest
import logging

from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from brainbox.io.spikeglx import Streamer


_logger = logging.getLogger('ibllib')
_logger.setLevel(10)

ba = AllenAtlas()
one = ONE(
    base_url='https://openalyx.internationalbrainlab.org',
    silent=True,
    password='international'
)


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
        assert len(self.collections) == 2


class TestStreamData(unittest.TestCase):

    def test_streamer_object(self):
        pid = '675952a4-e8b3-4e82-a179-cc970d5a8b01'
        t0 = 50
        self.td = tempfile.TemporaryDirectory()
        tmp_one = ONE(
            base_url='https://openalyx.internationalbrainlab.org',
            password='international',
            silent=True,
            cache_dir=self.td.name)

        sr = Streamer(pid=pid, one=tmp_one, typ='lf')
        # read once to download the data
        raw_ = sr[int(t0 * 2500):int((t0 + 1) * 2500), :]  # noqa
        # second read to use the local cache
        raw_ = sr[int(t0 * 2500):int((t0 + 1) * 2500), :]
        sl_raw = sr[int(t0 * 2500):int((t0 + 1) * 2500), :-1]
        assert sl_raw.shape == (2500, 384)
        assert sr.nc == 385
        assert sr.nsync == 1
        assert sr.rl == 6085.7024
        assert (raw_.shape == (2500, 385))
        assert sr.target_dir.exists()
        assert sr.geometry.keys()

    def tearDown(self) -> None:
        self.td.cleanup()
