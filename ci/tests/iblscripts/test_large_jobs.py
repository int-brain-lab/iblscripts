"""Tests for deploy/serverpc/crontab/large_jobs.py."""
import unittest
from unittest import mock
import tempfile
from pathlib import Path

try:
    from deploy.serverpc.crontab import large_jobs
    large_jobs_missing = False
except ModuleNotFoundError:
    large_jobs_missing = True


@unittest.skipIf(large_jobs_missing, 'iblscripts/serverpc/crontab/large_jobs.py not in python path')
class TestLargeJobs(unittest.TestCase):

    def test_list_available_envs(self):
        """Test list_available_envs function."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            self.assertEqual([None], large_jobs.list_available_envs(tmpdir))
            tmpdir.joinpath('foo.bar').touch()
            tmpdir.joinpath('bazenv').mkdir()
            self.assertCountEqual([None, 'bazenv'], large_jobs.list_available_envs(tmpdir))
            self.assertEqual([None], large_jobs.list_available_envs(tmpdir / 'does_not_exist'))

    @mock.patch('deploy.serverpc.crontab.large_jobs.task_queue')
    def test_list_queued_envs(self, task_queue_mock):
        one = mock.MagicMock()
        task_queue_mock.return_value = [{'executable': 'ibllib.pipes.behavior_tasks.HabituationRegisterRaw'}]
        self.assertEqual({None}, large_jobs.list_queued_envs(one))
        task_queue_mock.return_value.append({'executable': 'ibllib.pipes.mesoscope_tasks.MesoscopePreprocess'})
        self.assertCountEqual({None, 'suite2p'}, large_jobs.list_queued_envs(one))


if __name__ == '__main__':
    unittest.main()
