import unittest
from pathlib import Path

from ibllib.io.extractors import ephys_fpga


class TestEphysCheckList(unittest.TestCase):
    def setUp(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/checklist_choice_world')
        if not self.init_folder.exists():
            return

    def test_checklist_mock_3b(self):
        ses_path = self.init_folder / 'mock_session_3B'
        self.assertTrue(ephys_fpga.validate_mock_recording(ses_path))
