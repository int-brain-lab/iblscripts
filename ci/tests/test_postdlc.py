import pandas as pd
import numpy as np

from ci.tests import base
from ibllib.pipes.ephys_preprocessing import EphysPostDLC


class TestEphysPostDLC(base.IntegrationTest):

	def setUp(self) -> None:
		self.root_folder = self.data_path.joinpath('dlc', 'test_data')
		if not self.root_folder.exists():
			return
		# Run the task, without qc as we don't use a real session here and qc requires the database
		self.task = EphysPostDLC(self.root_folder)
		self.task.run(run_qc=False)
		# Load integration test target data, replace NaNs with 0 to enable comparison
		self.target_data = dict()
		self.target_data['licks'] = np.load(self.root_folder.joinpath('targets', 'licks.times.npy'))
		for cam in ['left', 'right']:
			features = pd.read_parquet(self.root_folder.joinpath('targets', f'_ibl_{cam}Camera.features.pqt')).fillna(0)
			self.target_data[f'pupil_raw_{cam}'] = features['pupilDiameter_raw']
			self.target_data[f'pupil_smooth_{cam}'] = features['pupilDiameter_smooth']

	def tearDown(self) -> None:
		for out in self.task.outputs:
			out.unlink()

	def test_pupil_diameter(self):
		for cam in ['left', 'right']:
			fname = [f for f in self.task.outputs if f'_ibl_{cam}Camera.features.pqt' in f.parts][0]
			features = pd.read_parquet(fname).fillna(0)
			self.assertTrue(all(features['pupilDiameter_raw'] == self.target_data[f'pupil_raw_{cam}']))
			self.assertTrue(all(features['pupilDiameter_smooth'] == self.target_data[f'pupil_smooth_{cam}']))

	def test_licks_times(self):
		fname = [f for f in self.task.outputs if 'licks.times.npy' in f.parts][0]
		licks = np.load(fname)
		self.assertTrue(all(licks == self.target_data['licks']))



