import traceback
import time
import logging
from pathlib import Path

from one.api import ONE
from ibllib.pipes.local_server import task_queue
from ibllib.pipes.tasks import run_alyx_task

_logger = logging.getLogger('ibllib')
subjects_path = Path('/mnt/s0/Data/Subjects/')
sleep_time = 3600

try:
    one = ONE(cache_rest=None)
    waiting_tasks = task_queue(mode='large', lab=None, alyx=one.alyx)

    if len(waiting_tasks) == 0:
        _logger.info(f'No large tasks in the queue, retrying in {int(sleep_time / 60)} min')
        # Query again only in 60min if queue is empty
        time.sleep(sleep_time)
    else:
        tdict = waiting_tasks[0]
        _logger.info(f"Running task {tdict['name']} for session {tdict['session']}")
        ses = one.alyx.rest('sessions', 'list', django=f"pk,{tdict['session']}")[0]
        session_path = Path(subjects_path).joinpath(
            Path(ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3)))
        run_alyx_task(tdict=tdict, session_path=session_path, one=one)
except Exception:
    _logger.error(f'Error running large task queue \n {traceback.format_exc()}')
    time.sleep(int(sleep_time / 2))
