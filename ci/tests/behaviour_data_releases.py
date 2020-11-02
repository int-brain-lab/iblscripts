import logging
import oneibl.onelight

from . import base

_logger = logging.getLogger('ibllib')


class Test(base.IntegrationTest):

    def setUp(self) -> None:
        datadir = self.data_path.joinpath('data-releases', 'ibl-behavioral-data-Dec2019')
        self.one = oneibl.onelight.LocalOne(root_dir=str(datadir))

    def test_behaviour_paper_release_2020_01(self):
        """ This mimics the script provided in the fig share"""
        one = self.one
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
