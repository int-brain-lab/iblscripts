import shutil

import pandas as pd


import ibllib.io.session_params
import ibllib.pipes.neurophotometrics as photometry_tasks

from ci.tests import base


class TestTaskFibrePhotometryPreprocess(base.IntegrationTest):

    def setUp(self) -> None:
        test_dir = self.data_path.joinpath('dynamic_pipeline', 'neurophotometrics')
        self.session_path = test_dir.joinpath('cortexlab', 'Subjects', 'CQ001', '2024-11-07', '001')

    def test_sync_fp_data(self):
        sess_params = ibllib.io.session_params.read_params(self.session_path)
        stub = sess_params['devices']['neurophotometrics']
        task = photometry_tasks.FibrePhotometrySync(self.session_path, one=None, location='local', **stub)
        status = task.run()
        self.assertEqual(0, status)
        task.run()
        task.assert_expected_outputs()
        task.register_datasets()
        fp_table = pd.read_parquet(task.outputs[-2])
        fp_locations = pd.read_parquet(task.outputs[-1])
        self.assertEqual(5, len(task.outputs))
        self.assertTrue(all(fp_locations.columns == ['fiber', 'brain_region']))
        self.assertEqual(0, len(set(fp_locations.index).difference(set(fp_table.columns))))

    def tearDown(self) -> None:
        if self.session_path.joinpath('alf').exists():
            shutil.rmtree(self.session_path.joinpath('alf'))
