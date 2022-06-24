from pathlib import Path
import logging
import shutil
import tempfile
import ibllib.pipes.behavior_tasks as btasks

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestTrialsRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('ephys', 'passive_extraction', 'SWC_054', '2020-10-10', '001')

    def test_register(self):
        wf = btasks.TrialRegisterRaw(self.session_path, task_collection='raw_behavior_data')
        status = wf.run()

        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs


class TestPassiveRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('ephys', 'passive_extraction', 'SWC_054', '2020-10-10', '001')

    def test_register(self):
        wf = btasks.PassiveRegisterRaw(self.session_path, task_collection='raw_passive_data')
        status = wf.run()

        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs


class TestPassiveTrials(base.IntegrationTest):
    pass
    # TODO test with normal passive and standalone passive (need to get the dataset on integration server)
