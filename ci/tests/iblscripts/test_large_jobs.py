"""Tests for deploy/serverpc/crontab/large_jobs.py."""
import unittest
from unittest import mock

try:
    from deploy.serverpc.crontab import large_jobs
    large_jobs_missing = False
except ModuleNotFoundError:
    large_jobs_missing = True


@unittest.skipIf(large_jobs_missing, 'iblscripts/serverpc/crontab/large_jobs.py not in python path')
class TestLargeJobs(unittest.TestCase):

    def setUp(self):
        self.tasks = [
            {'executable': 'mpci.suite2p.task.MesoscopePreprocess', 'priority': 100, 'name': 'MesoscopePreprocess'},
            {'executable': 'ibllib.pipes.video_tasks.VideoCompress', 'priority': 90, 'name': 'VideoCompress'},  # large
            {'executable': 'ibllib.pipes.video_tasks.DLC', 'priority': 80, 'name': 'DLC'},
            {'executable': 'mpci.sync.task.MesoscopeSync', 'priority': 40, 'name': 'MesoscopeSync'},  # small
            {'executable': 'ibllib.pipes.video_tasks.VideoSyncQcNidq', 'priority': 40, 'name': 'VideoSyncQC'},  # small
            {'executable': 'projects.not_installed.Task', 'priority': 100, 'name': 'Foo'},
        ]
        for t in self.tasks:
            t['session'] = 'eid'

    @mock.patch('deploy.serverpc.crontab.large_jobs.installed_envs', return_value=[None, 'dlc'])
    @mock.patch('ibllib.pipes.local_server._waiting_tasks')
    def test_list_queued_envs(self, waiting_tasks_mock, _):
        """Test list_queued_envs function returns the installed envs of waiting tasks."""
        one = mock.MagicMock()
        waiting_tasks_mock.return_value = [{'executable': 'ibllib.pipes.behavior_tasks.HabituationRegisterRaw'}]
        self.assertEqual({None}, large_jobs.list_queued_envs(one))
        waiting_tasks_mock.return_value = self.tasks
        self.assertEqual({None, 'dlc'}, large_jobs.list_queued_envs(one))

    def test_is_large_job(self):
        """Test is_large_job function."""
        self.assertEqual([True, True, True, True, False, False], list(map(large_jobs.is_large_job, self.tasks)))

    @mock.patch('deploy.serverpc.crontab.large_jobs.run_alyx_task', return_value=({}, []))
    @mock.patch('deploy.serverpc.crontab.large_jobs.installed_envs', return_value=[None, 'dlc', 'mpci'])
    @mock.patch('deploy.serverpc.crontab.large_jobs.task_queue')
    def test_process_next_large_job(self, task_queue_mock, _, run_mock):
        """Test process_next_large_job only runs the next task if it belongs to the given env."""
        one = mock.MagicMock()
        one.alyx.rest.return_value = [{'subject': 'foo', 'start_time': '2020-01-01T00:00:00', 'number': 1}]
        task_queue_mock.side_effect = lambda **kwargs: [t for t in self.tasks if t['name'] != 'Foo']
        # The highest priority task is in the mpci env
        for env in (None, 'dlc'):
            with self.subTest(env=env):
                task, _ = large_jobs.process_next_large_job('/mnt/s0', env=env, one=one)
                self.assertEqual('MesoscopePreprocess', task['name'])
                run_mock.assert_not_called()
        large_jobs.process_next_large_job('/mnt/s0', env='mpci', one=one)
        run_mock.assert_called_once()
        self.assertEqual(self.tasks[0], run_mock.call_args.kwargs['tdict'])
        self.assertEqual('/mnt/s0/foo/2020-01-01/001', run_mock.call_args.kwargs['session_path'].as_posix())
        self.assertEqual([None, 'dlc', 'mpci'], task_queue_mock.call_args.kwargs['env'])
        self.assertEqual('all', task_queue_mock.call_args.kwargs['mode'])
        # Small tasks in the base env are not run
        run_mock.reset_mock()
        task_queue_mock.side_effect = lambda **kwargs: [self.tasks[4]]
        self.assertEqual((None, []), large_jobs.process_next_large_job('/mnt/s0', one=one))
        # Small tasks in other envs are run
        task_queue_mock.side_effect = lambda **kwargs: [self.tasks[3]]
        large_jobs.process_next_large_job('/mnt/s0', env='mpci', one=one)
        self.assertEqual(self.tasks[3], run_mock.call_args.kwargs['tdict'])


if __name__ == '__main__':
    unittest.main()
