import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import requests.exceptions
from one.api import ONE
from ibllib.tests.fixtures import utils
from ibllib.tests import TEST_DB

from deploy.serverpc.crontab.report_create_jobs import run_job_creator_task


class TestJobCreator(unittest.TestCase):
    """Test deploy.serverpc.crontab.report_create_jobs.run_job_creator_task function."""

    session = None
    """dict: a fake Alyx session record."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Create a local server session with an experiment description and raw session flag
        self.root = Path(tmp.name, 'Data')
        self.session_path = utils.create_fake_session_folder(self.root, lab='')
        utils.create_fake_raw_behavior_data_folder(
            self.session_path, task='training', folder='raw_task_data_00', write_pars_stub=True)
        self.session_path.joinpath('raw_session.flag').touch()
        # Create some junk sessions to check that they are skipped
        for subj in ('test', 'test_subject'):
            (p := self.session_path.parents[2].joinpath(subj, '2020-01-01', '001')).mkdir(parents=True)
            p.joinpath('raw_session.flag').touch()

        # Create a fake mouse on the database if one doesn't already exist
        self.one = ONE(**TEST_DB, mode='remote', cache_rest=None)
        try:
            r = self.one.alyx.rest('subjects', 'read', id='fakemouse')
        except requests.exceptions.HTTPError as ex:
            if ex.errno == 404:
                r = self.one.alyx.rest('subjects', 'create', data={'nickname': 'fakemouse', 'lab': 'mainenlab'})
            else:
                raise ex
        self.addCleanup(self.one.alyx.rest, 'subjects', 'delete', id=r['nickname'])

    @patch('deploy.serverpc.crontab.report_create_jobs.get_local_endpoint_id')
    @patch('ibllib.pipes.local_server.IBLRegistrationClient.register_session')
    @patch('deploy.serverpc.crontab.report_create_jobs.report_health', side_effect=ValueError('foo'))
    def test_job_creator(self, report_health_mock, register_session_mock, get_local_endpoint_id_mock):
        """Test job creator registers a test session and creates preprocessing tasks."""
        register_session_mock.side_effect = self._register_session  # register the session w/o task settings
        get_local_endpoint_id_mock.return_value = '2dc8ccc6-2f8e-11e9-9351-0e3d676669f4'  # corresponds to mainen_lab_SR
        subjects_dir = self.root / 'Subjects'
        t = run_job_creator_task(one=self.one, root_path=subjects_dir)
        # OneIBL client's session registration method should have been called
        register_session_mock.assert_called_once_with(self.session_path, file_list=False)
        # get_local_endpoint_id should have been called to get the repo name
        get_local_endpoint_id_mock.assert_called_once()
        report_health_mock.assert_called_once()
        self.assertIsInstance(t, dict, 'failed to return the task dict')
        self.assertIn('ValueError: foo', t['log'], 'report_health error not logged')
        # Despite the report_health exception, we expect registration to continue
        self.assertIn(f'creating session for {self.session_path}', t['log'], 'failed to log session creation')
        test_session = self.session_path.parents[2].joinpath('test_subject', '2020-01-01', '001')
        self.assertNotIn(f'creating session for {test_session}', t['log'], 'failed to skip test subjects')
        assert self.session, 'failed to create test session'
        tasks = self.one.alyx.rest('tasks', 'list', session=self.session['id'])
        self.assertEqual(5, len(tasks), 'unexpected number of pipeline tasks registered')
        task = next((t for t in tasks if t['name'].startswith('Trials_')), None)
        self.assertTrue(task, 'failed to create trials task')
        self.assertEqual('ibllib.pipes.behavior_tasks.ChoiceWorldTrialsBpod', task['executable'])
        expected_args = {
            'sync': 'bpod', 'protocol': 'training', 'sync_ext': 'jsonable', 'collection': 'raw_task_data_00',
            'sync_namespace': None, 'protocol_number': 0, 'sync_collection': 'raw_behavior_data'
        }
        self.assertDictEqual(expected_args, task['arguments'])

    def _register_session(self, session_path, **_):
        """Create a fake session for tasks to be registered to."""
        self.session = self.one.alyx.rest('sessions', 'create', data={
            'subject': session_path.parts[-3],
            'users': [self.one.alyx.user],
            'location': '_iblrig_mainenlab_behavior_2',
            'lab': 'mainenlab',
            'type': 'Experiment',
            'number': session_path.parts[-1],
            'start_time': f'{session_path.parts[-2]}T00:00:00',
            'end_time': None,
        })

    def tearDown(self):
        tasks = self.one.alyx.rest(
            'tasks', 'list', name='JobCreator', django='data_repository__name,mainen_lab_SR', no_cache=True)
        for t in tasks:
            self.one.alyx.rest('tasks', 'delete', id=t['id'])


if __name__ == '__main__':
    unittest.main()
