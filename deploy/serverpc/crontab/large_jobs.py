"""Run the next large pipeline task.

This script is called by 01_large_jobs.sh once per task environment (see
ibllib.pipes.routing.ROUTES), within that environment. The environment of each queued task is
determined from its executable so that the task classes of other environments are never imported.

Tasks in the base environment (env=None) are split into small and large jobs, with the small jobs
run by a separate service (see small_jobs.py). All tasks in other environments are run here,
regardless of job size.

If the highest priority task belongs to another installed environment, this script does nothing
so that the other environment runs it next.
"""
import traceback
import logging
from pathlib import Path
import argparse

from one.api import ONE
from ibllib.pipes.local_server import task_queue, list_queued_envs as _list_queued_envs
from ibllib.pipes.routing import task_env, installed_envs
from ibllib.pipes.tasks import run_alyx_task, str2class

_logger = logging.getLogger('ibllib')
_logger.setLevel(logging.DEBUG)


def list_queued_envs(one=None):
    """
    The set of installed envs in the list of waiting tasks.

    Returns
    -------
    set
        All installed environments required to process waiting tasks.
    """
    one = one or ONE(mode='remote', cache_rest=None)
    return _list_queued_envs(one=one) & set(installed_envs())


def is_large_job(task):
    """
    Whether a task should be run by this service, i.e. not by the small jobs service.

    Only the job size of tasks in the base environment is checked (by importing the task class).

    Parameters
    ----------
    task : dict
        An Alyx task dictionary.

    Returns
    -------
    bool
        True if the task is in a non-base environment, or is a large base environment task.
    """
    if task_env(task['executable']) is not None:
        return True
    try:
        return str2class(task['executable']).job_size == 'large'
    except (ImportError, AttributeError):  # e.g. a personal projects task not installed in this env
        _logger.debug('Task %s not found in this env', task['executable'])
        return False


def process_next_large_job(subjects_path, env=None, one=None):
    """
    Process the next large job.

    Parameters
    ----------
    subjects_path : pathlib.Path
        The location of the session paths.
    env : str
        Whether to run only tasks with a specific environment label (assumes this function is
        called within said env).  If None, only large tasks in the base environment are run.

    Returns
    -------
    Task | None
        The highest priority task dict.
    list of pathlib.Path
        A list of registered datasets.
    """
    one = one or ONE(mode='remote', cache_rest=None)
    envs = installed_envs()
    _logger.info(f'Installed environments: {envs}')
    waiting_tasks = list(filter(is_large_job, task_queue(mode='all', alyx=one.alyx, env=envs) or []))

    if len(waiting_tasks) == 0:
        _logger.info('No large tasks in the queue')
        return None, []
    _logger.info(f'Found {len(waiting_tasks)} tasks in the queue, logging first 10')
    for tdict in waiting_tasks[:10]:
        _logger.info(f"priority {tdict['priority']}, {tdict['name']}, env {task_env(tdict['executable'])}")
    tdict = waiting_tasks[0]
    if task_env(tdict['executable']) != env:
        _logger.debug('Higher priority task should be run in another env; will not run')
        return tdict, []
    _logger.info(f"Running task {tdict['name']} for session {tdict['session']}")
    ses = one.alyx.rest('sessions', 'list', django=f"pk,{tdict['session']}")[0]
    session_path = Path(subjects_path).joinpath(
        Path(ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3)))
    return run_alyx_task(tdict=tdict, session_path=session_path, one=one)


if __name__ == '__main__':
    """Run large pipeline tasks.

    Examples
    --------
    >>> python large_jobs.py
    >>> python large_jobs.py --subjects-path /mnt/s0/Data/Subjects --env dlc
    """
    # Parse parameters
    parser = argparse.ArgumentParser(description='Run large pipeline tasks.')
    parser.add_argument('--env', type=str, help='Specify the environment label (only compatible tasks are run)')
    parser.add_argument('--subjects-path', type=Path, default='/mnt/s0/Data/Subjects/', help='Specify the location of the data.')
    args = parser.parse_args()  # returns data from the options specified (echo)
    try:
        _logger.info(f'Running large task queue with environment {args.env}')
        task, _ = process_next_large_job(args.subjects_path, env=args.env)
    except Exception:
        _logger.error(f'Error running large task queue \n {traceback.format_exc()}')
