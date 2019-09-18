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
        # extract all sessions
        for fil in self.subjects_path.rglob('_iblrig_taskData.raw*.jsonable'):
            session_path = fil.parents[1]
            ibllib.io.flags.create_extract_flags(session_path)
            extract_session.from_path(session_path, force=True)

    def test_extractors(self):
        """
        Test the full exctraction for all the session in the folder.
        """
        for fil in self.subjects_path.rglob('_iblrig_taskData.raw*.jsonable'):
            session_path = fil.parents[1]
            lc = alf.io.load_object(session_path / 'alf', '_ibl_leftCamera')
            self.assertTrue(np.all(np.diff(lc.times) > 0))

        """
        For this session only the downgoing front of a trial was detected, resulting in an error
        for the gocuetime. The fix was to extract the downgoing front and subtract 100ms.
        """
        session_path = self.subjects_path / "CSHL_007/2019-07-31/001"
        trials = alf.io.load_object(session_path / 'alf', '_ibl_trials')
        self.assertTrue(np.all(np.logical_not(np.isnan(trials.goCue_times))))

    def tearDown(self):
        for fil in list(self.subjects_path.rglob('_iblrig_taskData.raw*.jsonable')):
            alf_path = fil.parents[1].joinpath('alf')
            shutil.rmtree(alf_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(exit=False)
