"""
The ttl_check pipeline is to validate a setup before doing recordings,
on the ephysChoiceWorld task, doing a dummy without animal and check that all syncs are there

This test is also a synchronization extraction test, as it checks the ouput. The tear down function
removes all _spikeglx_ files so they are regenerated from the small files accessible
"""

import unittest
from pathlib import Path

import ibllib.ephys.ephysqc


class TestEphysCheckList(unittest.TestCase):
    def setUp(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/ttl_check')
        if not self.init_folder.exists():
            return

    def test_checklist_mock_3B_single(self):
        ses_path = self.init_folder / 'ttl_3B_single'
        self.assertTrue(ibllib.ephys.ephysqc.validate_ttl_test(ses_path))

    def test_checklist_mock_3A_single(self):
        ses_path = self.init_folder / 'ttl_3A_single'
        self.assertTrue(ibllib.ephys.ephysqc.validate_ttl_test(ses_path))

    def test_checklist_mock_3B_dual(self):
        ses_path = self.init_folder / 'ttl_3B_dual'
        self.assertTrue(ibllib.ephys.ephysqc.validate_ttl_test(ses_path))

    def tearDown(self) -> None:
        for sf in self.init_folder.rglob('_spikeglx_sync.*.npy'):
            sf.unlink()
