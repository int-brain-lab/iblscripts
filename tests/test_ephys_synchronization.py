import unittest
from pathlib import Path

import ibllib.ephys.sync_probes as sync_probes


class TestEphysCheckList(unittest.TestCase):
    def setUp(self):
        self.folder3a = Path('/mnt/s0/Data/IntegrationTests/ephys/sync/sync_3A')
        self.folder3b = Path('/mnt/s0/Data/IntegrationTests/ephys/sync/sync_3B')

    def test_sync_3A(self):
        if not self.folder3a.exists():
            return
        # the assertion is already in the files
        # test both residual smoothed and linear
        for ses_path in self.folder3a.rglob('raw_ephys_data'):
            self.assertTrue(sync_probes.version3A(ses_path.parent))
            self.assertTrue(sync_probes.version3A(ses_path.parent, linear=True, tol=2))

    def test_sync_3B(self):
        # the assertion is already in the files
        if not self.folder3b.exists():
            return
        for ses_path in self.folder3b.rglob('raw_ephys_data'):
            self.assertTrue(sync_probes.version3B(ses_path.parent))
            self.assertTrue(sync_probes.version3B(ses_path.parent, linear=True, tol=10))
