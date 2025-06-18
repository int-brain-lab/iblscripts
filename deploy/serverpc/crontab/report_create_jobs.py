"""Pipeline entry script.

This script searches for new sessions (those with a raw_session.flag file), and creates the pipeline
preprocessing tasks in Alyx. No file registration takes place in this script.
"""
import logging
import traceback
from pathlib import Path
from one.api import ONE
from one.webclient import AlyxClient
from one.remote.globus import get_local_endpoint_id

from ibllib.pipes.local_server import report_health
from ibllib.pipes.local_server import job_creator
from ibllib.pipes.base_tasks import Task

_logger = logging.getLogger('ibllib')

subjects_path = Path('/mnt/s0/Data/Subjects/')


class JobCreator(Task):
    """A task for creating session preprocessing tasks."""
    level = 0
    priority = 100
    io_charge = 20

    def __init__(self, subjects_path, **kwargs):
        """A task for creating session preprocessing tasks.

        Parameters
        ----------
        subjects_path : pathlib.Path
            The root path containing the sessions to register.
        """
        self.subjects_path = subjects_path
        self.pipes = []
        super().__init__(None, **kwargs)

    def _run(self):
        # Label the lab endpoint json field with health indicators
        try:
            report_health(self.one.alyx)
            _logger.info('Reported health of local server')
        except BaseException:
            _logger.error(f'Error in report_health\n {traceback.format_exc()}')

        #  Create sessions: for this server, finds the extract_me flags, identify the session type,
        #  create the session on Alyx if it doesn't already exist, register the raw data and create
        #  the tasks backlog
        pipes, _ = job_creator(self.subjects_path, one=self.one, dry=False)
        self.pipes.extend(pipes)
        _logger.info('Ran job creator.')


def get_repo_from_endpoint_id(endpoint=None, alyx=None):
    """
    Extracts data repository name associated with a given a Globus endpoint UUID.

    Parameters
    ----------
    endpoint : uuid.UUID, str
        Endpoint UUID, optional if not given will get attempt to find local endpoint UUID.
    alyx : one.webclient.AlyxClient
        An instance of AlyxClient to use.

    Returns
    -------
    str
        The data repository name associated with the endpoint UUID.
    """
    alyx = alyx or AlyxClient(silent=True)
    if not endpoint:
        endpoint = get_local_endpoint_id()
    repo = alyx.rest('data-repository', 'list', globus_endpoint_id=endpoint)
    if any(repo):
        return repo[0]['name']


def run_job_creator_task(one=None, data_repository_name=None, root_path=subjects_path):
    """Run the JobCreator task.

    Parameters
    ----------
    one : one.api.OneAlyx
        An instance of ONE to use.
    data_repository_name : str
        The associated data repository name. If None, this is determined from the local Globus
        endpoint ID.
    root_path : pathlib.Path
        The root path containing the sessions to register.

    Returns
    -------
    JobCreator
        The run JobCreator task.
    """
    one = one or ONE(cache_rest=None, mode='remote')
    data_repository_name = data_repository_name or get_repo_from_endpoint_id(alyx=one.alyx)
    tasks = one.alyx.rest(
        'tasks', 'list', name='JobCreator', django=f'data_repository__name,{data_repository_name}', no_cache=True)
    assert len(tasks) < 2
    if not any(tasks):
        t = JobCreator(root_path, one=one, clobber=True)
        task_dict = {
            'executable': 'deploy.serverpc.crontab.report_create_jobs.JobCreator',
            'priority': t.priority, 'io_charge': t.io_charge, 'gpu': t.gpu, 'cpu': t.cpu,
            'ram': t.ram, 'module': 'deploy.serverpc.crontab.report_create_jobs',
            'parents': [], 'level': t.level, 'status': 'Empty', 'name': t.name, 'session': None,
            'graph': 'Base', 'arguments': {}, 'data_repository': data_repository_name}
        talyx = one.alyx.rest('tasks', 'create', data=task_dict)
    else:
        talyx = tasks[0]
        tkwargs = talyx.get('arguments') or {}  # if the db field is null it returns None
        t = JobCreator(root_path, one=one, taskid=talyx['id'], clobber=True, **tkwargs)

    one.alyx.rest('tasks', 'partial_update', id=talyx['id'], data={'status': 'Started'})
    status = t.run()
    patch_data = {
        'time_elapsed_secs': t.time_elapsed_secs, 'log': t.log, 'version': t.version,
        'status': 'Empty' if status == 0 else 'Errored'}
    t = one.alyx.rest('tasks', 'partial_update', id=talyx['id'], data=patch_data)
    return t


if __name__ == '__main__':
    run_job_creator_task()
