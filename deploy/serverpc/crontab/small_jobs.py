import traceback
import time
import logging
import argparse
from pathlib import Path

from one.api import ONE
from ibllib.pipes.local_server import task_queue, list_available_envs
from ibllib.pipes.tasks import run_alyx_task, str2class

_logger = logging.getLogger('ibllib')
sleep_time = 3600  # How long to sleep if task queue is empty, before re-querying the database
count = 20  # How many tasks to run at a time (max) before re-querying the database

envs = list_available_envs()
parser = argparse.ArgumentParser(description='Run large pipeline tasks.')
parser.add_argument('--subjects-path', type=Path, default='/mnt/s0/Data/Subjects/', help='Specify the location of the data.')
parser.add_argument('--env', type=str, help='Specify the environment (only compatible tasks are run)')
args = parser.parse_args()


try:
    one = ONE(cache_rest=None)
    waiting_tasks = task_queue(mode='small', lab=None, alyx=one.alyx, env=envs)

    if len(waiting_tasks) == 0:
        _logger.info(f'No small tasks in the queue, retrying in {int(sleep_time / 60)} min')
        # Sleep for 60min if queue is empty
        time.sleep(sleep_time)
    else:
        # In the case of small tasks we run a set of them at a time before re-querying
        # Often they are from the same session, so we cache the session path between tasks
        last_session = None
        c = 0
        for tdict in waiting_tasks:
            if c >= count:
                break
            env = str2class(tdict['executable']).env
            if env != args.env:
                _logger.debug(f"Skipping task {tdict['name']} for session {tdict['session']}, "
                              f"env {env} does not match requested env {args.env}")
                continue
            _logger.info(f"Running task {tdict['name']} for session {tdict['session']}")
            if last_session != tdict['session']:
                ses = one.alyx.rest('sessions', 'list', django=f"pk,{tdict['session']}")[0]
                session_path = Path(args.subjects_path).joinpath(
                    ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3))
                last_session = tdict['session']
            task, dsets = run_alyx_task(tdict=tdict, session_path=session_path, one=one)
            if dsets:
                c += 1  # i.e. only tasks that output datasets are counted towards count
except Exception:
    _logger.error(f'Error running small task queue \n {traceback.format_exc()}')
    time.sleep(int(sleep_time / 2))
