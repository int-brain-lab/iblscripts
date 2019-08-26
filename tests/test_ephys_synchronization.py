import unittest
from pathlib import Path

from ibllib.ephys.sync3A import sync_probe_folders_3A


class TestEphysCheckList(unittest.TestCase):
    def setUp(self):
        self.folder3a = Path('/mnt/s0/Data/IntegrationTests/ephys/sync3a')
        if not self.init_folder.exists():
            return

    def test_sync_3A(self):
        # the assertion is already in the files in this case
        for ses_path in (self.folder3a('raw_ephys_data')):
            sync_probe_folders_3A(ses_path)
