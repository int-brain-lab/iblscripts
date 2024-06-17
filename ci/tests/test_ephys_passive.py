"""Test passiveChoiceWorld extraction.

Basic passive extraction is tested using ephys/passive_extraction/SWC_054/2020-10-10/001.
Chained passive protocol extraction is tested with ephys/passive_extraction/ZFM-05496/2022-12-08/001.
Currently ZFM-05496/2022-12-08/001/raw_task_data_00 is not used.
"""
import unittest
import logging
import shutil

import pandas as pd
import numpy as np

from ibllib.io.extractors import ephys_passive
from ibllib.pipes.behavior_tasks import PassiveTaskNidq
from ci.tests import base


log = logging.getLogger('ibllib')


class TestEphysPassiveExtraction(base.IntegrationTest):
    def setUp(self) -> None:
        self.root_folder = self.data_path.joinpath('ephys', 'passive_extraction')
        self.session_path = self.root_folder.joinpath('SWC_054', '2020-10-10', '001')
        if not self.root_folder.exists():
            log.error(f'{self.root_folder} does not exist')

    def test_task_extraction(self):
        ext = ephys_passive.PassiveChoiceWorld(self.session_path)
        data, paths = ext.extract()
        self.assertTrue(len(data) == 4)
        self.assertTrue(paths is None)
        # data tests

    def test_task_extraction_files(self):
        ext = ephys_passive.PassiveChoiceWorld(self.session_path)
        data, paths = ext.extract(save=True)
        path_names = [x.name for x in paths]
        expected = [
            '_ibl_passivePeriods.intervalsTable.csv',
            '_ibl_passiveRFM.times.npy',
            '_ibl_passiveGabor.table.csv',
            '_ibl_passiveStims.table.csv',
        ]
        self.assertTrue(all([x in path_names for x in expected]))

        # data tests
        # paths test

    def tearDown(self):
        # remove alf folder
        shutil.rmtree(self.session_path.joinpath('alf'), ignore_errors=True)


class TestChainedPassiveExtraction(base.IntegrationTest):
    """Test for the dynamic pipeline extraction of two chained passive protocols.

     Employs the ibllib.pipes.behavior_tasks.PassiveTaskNidq class.
     """
    def setUp(self) -> None:
        self.root_folder = self.data_path.joinpath('ephys', 'passive_extraction')
        self.session_path = self.root_folder.joinpath('ZFM-05496', '2022-12-08', '001')
        if not self.session_path.exists():
            log.error(f'{self.root_folder} does not exist')

    def test_chained_passive_task_extraction(self):
        kwargs = {
            'collection': 'raw_task_data_01',
            'protocol': 'passiveChoiceWorld',
            'protocol_number': 1,
            'sync': 'nidq',
            'sync_collection': 'raw_ephys_data',
            'sync_ext': 'bin',
            'sync_namespace': 'spikeglx'
        }
        task = PassiveTaskNidq(self.session_path, location='local', **kwargs)
        self.assertEqual(0, task.run())
        self.assertEqual('alf/task_01', task.output_collection)
        self.assertEqual(4, len(task.output_files))
        out_path = self.session_path.joinpath(task.output_collection)
        for file, *_ in task.output_files:
            with self.subTest(file):
                self.assertTrue((out_path / file).exists())
        df = pd.read_csv(out_path / '_ibl_passivePeriods.intervalsTable.csv', index_col=0)
        expected = [3119.84190444, 4100.975761]
        np.testing.assert_array_almost_equal(df['passiveProtocol'].values, expected)

    def tearDown(self):
        # remove alf folder
        shutil.rmtree(self.session_path.joinpath('alf'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(exit=False)
