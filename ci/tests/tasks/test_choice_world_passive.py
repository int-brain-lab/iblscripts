import logging
import shutil
import unittest.mock

import pandas as pd
import numpy as np

from one.api import ONE
from ibllib.pipes.behavior_tasks import PassiveRegisterRaw, PassiveTaskNidq

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestPassiveRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        self.raw_session_path = next(self.default_data_root().joinpath(
            'tasks', 'choice_world_ephys').rglob('raw_behavior_data')).parent
        self.session_path, self.extraction_path = base.make_sym_links(self.raw_session_path)
        self.one = ONE(**base.TEST_DB, mode='local')

    def test_register(self):
        task = PassiveRegisterRaw(self.session_path, one=self.one, collection='raw_passive_data')
        status = task.run()

        assert status == 0
        task.assert_expected_outputs()


class TestPassiveTrials(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('ephys', 'passive_extraction', 'SWC_054', '2020-10-10', '001')
        self.alf_path = self.session_path.joinpath('alf')
        self.one = ONE(**base.TEST_DB, mode='local')

    def test_passive_extract(self):
        task = PassiveTaskNidq(self.session_path, collection='raw_passive_data', sync_collection='raw_ephys_data',
                               sync_namespace='spikeglx', one=self.one)
        status = task.run()

        assert status == 0
        task.assert_expected_outputs()

        assert len(task.outputs) == 4

        passive_intervals = pd.read_csv(next(o for o in task.outputs
                                             if '_ibl_passivePeriods.intervalsTable.csv' in o.name))
        assert not np.all(np.isnan(passive_intervals.taskReplay.values))

    @unittest.mock.patch("ibllib.io.extractors.ephys_passive.skip_task_replay", return_value=True)
    def test_passive_extract_no_task_replay(self, mock_qc):
        task = PassiveTaskNidq(self.session_path, collection='raw_passive_data', sync_collection='raw_ephys_data',
                               sync_namespace='spikeglx', one=self.one)
        status = task.run()

        assert status == 0
        task.assert_expected_outputs()

        assert len(task.outputs) == 2

        passive_intervals = pd.read_csv(next(o for o in task.outputs
                                             if '_ibl_passivePeriods.intervalsTable.csv' in o.name))
        assert np.all(np.isnan(passive_intervals.taskReplay.values))

    def tearDown(self) -> None:
        shutil.rmtree(self.alf_path)
