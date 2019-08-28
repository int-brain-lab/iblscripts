"""
The ttl_check pipeline is to validate a setup before doing recordings,
on the ephysChoiceWorld task, doing a dummy without animal and check that all syncs are there
"""
import unittest
from pathlib import Path

from ibllib.io.extractors import ephys_fpga


class TestEphysCheckList(unittest.TestCase):
    def setUp(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/ttl_check')
        if not self.init_folder.exists():
            return

    def test_checklist_mock_3B_single(self):
        ses_path = self.init_folder / 'ttl_3B_single'
        self.assertTrue(ephys_fpga.validate_ttl_test(ses_path))

    def test_checklist_mock_3A_single(self):
        ses_path = self.init_folder / 'ttl_3A_single'
        self.assertTrue(ephys_fpga.validate_ttl_test(ses_path))
