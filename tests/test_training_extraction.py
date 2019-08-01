import unittest
from pathlib import Path
import shutil

import numpy as np

from ibllib.pipes import extract_session
import ibllib.io.flags
import alf.io


class TestSessions(unittest.TestCase):

    def setUp(self):
        self.subjects_path = Path('/mnt/s0/Data/IntegrationTests/training')
        if not self.subjects_path.exists():
            return

    def test_only_BNC2_low_for_a_trial(self):
        """
        For this session only the downgoing front of a trial was detected, resulting in an error
        for the gocuetime. The fix was to extract the downgoing front and subtract 100ms.
        """
        session_path = self.subjects_path / "CSHL_007/2019-07-31/001"
        ibllib.io.flags.create_extract_flags(session_path)
        extract_session.from_path(session_path, force=True)
        trials = alf.io.load_object(session_path / 'alf', '_ibl_trials')
        self.assertTrue(np.all(np.logical_not(np.isnan(trials.goCue_times))))
        shutil.rmtree(session_path.joinpath('alf'), ignore_errors=True)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main(exit=False)
