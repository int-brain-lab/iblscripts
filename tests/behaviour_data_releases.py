import unittest
from pathlib import Path

import logging
import oneibl.onelight

_logger = logging.getLogger('ibllib')

TEST_PATH = Path('/mnt/s0/Data/IntegrationTests')

datadir = TEST_PATH.joinpath('data-releases', 'ibl-behavioral-data-Dec2019')
one = oneibl.onelight.LocalOne(root_dir=str(datadir))


class Test(unittest.TestCase):

    def test_behaviour_paper_release_2020_01(self):
        """ This mimics the script provided in the fig share"""
        # Search all sessions that have these dataset types.
        eids = one.search(['_ibl_trials.*'])
        # Select the first session.
        eid = eids[0]

        # List all dataset types available in that session.
        dset_types = one.list(eid)
        assert(len(dset_types) == 13)
        # Loading a single dataset.
        one.load_dataset(eid, dset_types[0])
        obj = one.load_object(eid, "_ibl_trials")
        assert(len(obj.keys()) == 13)
