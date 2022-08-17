import tempfile
import unittest
import logging

import numpy as np

from one.api import ONE
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from brainbox.io.spikeglx import Streamer, stream


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
        with tempfile.TemporaryDirectory() as td:
            tmp_one = ONE(
                base_url='https://openalyx.internationalbrainlab.org',
                password='international',
                silent=True,
                cache_dir=td)
            sr = Streamer(pid=pid, one=tmp_one)
            # read once to download the data
            raw_ = sr[int(t0 * 30000):int((t0 + 1) * 30000), :]  # noqa
            # second read to use the local cache
            raw_ = sr[int(t0 * 30000):int((t0 + 1) * 30000), :]
            sl_raw = sr[int(t0 * 30000):int((t0 + 1) * 30000), :-1]
            assert sl_raw.shape == (30000, 384)
            assert sr.nc == 385
            assert sr.nsync == 1
            assert sr.rl == 6085.7024
            assert (raw_.shape == (30000, 385))
            assert sr.target_dir.exists()
            assert sr.geometry.keys()
            """
            Test deprecation if this fails here it means the grace period expired.
            -   remove all the lines below
            -   remove the function brainbox.io.spikeglx.stream
            """
            # ########################## delete from here
            import datetime
            if datetime.datetime.now() > datetime.datetime(2022, 11, 26):
                raise NotImplementedError
            raw, t0out = stream(pid, t0, nsecs=1, one=tmp_one, remove_cached=False, typ='ap')
            assert (raw.shape == (30000, 385))
            assert np.all(raw[:, :] == raw_)
            # ########################## end delete from here
