import unittest

import pandas as pd
import numpy as np
from one.api import ONE

from ibllib.pipes.ephys_preprocessing import EphysPostDLC
from ci.tests import base


class TestEphysPostDLC(base.IntegrationTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.root_folder = cls.default_data_root().joinpath('dlc')
        if not cls.root_folder.exists():
            raise unittest.SkipTest(f'File not found: {cls.root_folder}')
        # Run the task, without qc as we don't use a real session here and qc requires the database
        cls.task = EphysPostDLC(cls.root_folder, one=ONE(**base.TEST_DB, mode='local'))
        cls.task.signature['input_files'] = []  # To prevent task from trying to load inputs
        cls.task.run(overwrite=True, run_qc=False, plot_qc=False)

    @classmethod
    def tearDownClass(cls) -> None:
        for out in cls.task.outputs:
            out.unlink()

    def test_pupil_diameter(self):
        # Load integration test target data, replace NaNs with 0 to enable comparison
        for cam in ['left', 'right']:
            target_features = pd.read_parquet(self.root_folder.joinpath('targets', f'_ibl_{cam}Camera.features.pqt')).fillna(0)
            fname = [f for f in self.task.outputs if f'_ibl_{cam}Camera.features.pqt' in f.parts][0]
            features = pd.read_parquet(fname).fillna(0)
            for field in ('pupilDiameter_raw', 'pupilDiameter_smooth'):
                with self.subTest(field=field):
                    np.testing.assert_array_equal(features[field], target_features[field])

    def test_licks_times(self):
        # Load integration test target data
        target_licks = np.load(self.root_folder.joinpath('targets', 'licks.times.npy'))
        fname = [f for f in self.task.outputs if 'licks.times.npy' in f.parts][0]
        licks = np.load(fname)
        np.testing.assert_array_equal(licks, target_licks)


if __name__ == '__main__':
    unittest.main(exit=False)
