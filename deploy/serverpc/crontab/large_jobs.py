import traceback
import logging
from pathlib import Path
import argparse

import numpy as np

from one.api import ONE
from ibllib.pipes.local_server import task_queue, list_available_envs
from ibllib.pipes.tasks import run_alyx_task, str2class

_logger = logging.getLogger('ibllib')
_logger.setLevel(logging.DEBUG)


def process_next_large_job(subjects_path, env=None, one=None):
    """
    Process the next large job.

    Parameters
    ----------
    subjects_path : pathlib.Path
        The location of the session paths.
    env : str
        Whether to run only tasks with a specific environment name (assumes this function is called
        within said env).  If None, only tasks with no env name specified are run.

    Returns
    -------
    Task | None
        The highest priority task dict.
    list of pathlib.Path
        A list of registered datasets.
    """
    one = one or ONE(mode='remote', cache_rest=None)
    envs = list_available_envs()
    _logger.info(f'Available environments: {envs}')
    waiting_tasks = task_queue(mode='large', alyx=one.alyx, env=envs)

    if len(waiting_tasks) == 0:
        _logger.info('No large tasks in the queue')
        return None, []
    else:
        _logger.info(f'Found {len(waiting_tasks)} tasks in the queue, logging first 10')
        for i in np.arange(np.minimum(10, len(waiting_tasks))):
            _logger.info(f"priority {waiting_tasks[i]['priority']}, {waiting_tasks[i]['name']}"
                         f", env {str2class(waiting_tasks[i]['executable']).env}")
        tdict = waiting_tasks[0]
        if str2class(tdict['executable']).env != env:
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
    >>> python large_jobs.py --subjects-path /mnt/s0/Data/Subjects --env dlcenv
    """
    # Parse parameters
    parser = argparse.ArgumentParser(description='Run large pipeline tasks.')
    parser.add_argument('--env', type=str, help='Specify the environment (only compatible tasks are run)')
    parser.add_argument('--subjects-path', type=Path, default='/mnt/s0/Data/Subjects/', help='Specify the location of the data.')
    args = parser.parse_args()  # returns data from the options specified (echo)
    try:
        _logger.info(f'Running large task queue with environment {args.env}')
        task, _ = process_next_large_job(args.subjects_path, env=args.env)
    except Exception:
        _logger.error(f'Error running large task queue \n {traceback.format_exc()}')
