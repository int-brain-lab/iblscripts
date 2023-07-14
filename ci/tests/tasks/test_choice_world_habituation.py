import logging
import shutil
import ibllib.pipes.behavior_tasks as btasks
from one.api import ONE

from ci.tests import base

_logger = logging.getLogger('ibllib')


class HabituationTemplate(base.IntegrationTest):

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB, mode='local')
        self.raw_session_path = next(self.default_data_root().joinpath(
            'tasks', 'choice_world_habituation').rglob('raw_behavior_data')).parent
        self.session_path, self.extraction_path = base.make_sym_links(self.raw_session_path)

    def tearDown(self):
        shutil.rmtree(self.extraction_path)


class TestHabituationRegisterRaw(HabituationTemplate):

    def test_task(self):
        wf = btasks.HabituationRegisterRaw(self.session_path, one=self.one, collection='raw_behavior_data')
        status = wf.run()
        assert status == 0
        wf.assert_expected_outputs()
        wf.assert_expected_inputs()


class TestHabituationTrialsBpod(HabituationTemplate):

    def test_task(self):
        wf = btasks.HabituationTrialsBpod(self.session_path, collection='raw_behavior_data')
        status = wf.run(update=False)
        assert status == 0
        wf.assert_expected_outputs()
