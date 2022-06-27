import logging

from ibllib.pipes.behavior_tasks import PassiveRegisterRaw, PassiveTask

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestPassiveRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('ephys', 'passive_extraction', 'SWC_054', '2020-10-10', '001')

    def test_register(self):
        task = PassiveRegisterRaw(self.session_path, task_collection='raw_passive_data')
        status = task.run()

        assert status == 0
        task.assert_expected_outputs()


class TestPassiveTrials(base.IntegrationTest):
    pass
    # TODO test with normal passive and standalone passive (need to get the dataset on integration server)
