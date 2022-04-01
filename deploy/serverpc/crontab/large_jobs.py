import traceback
import time

from pathlib import Path

from one.api import ONE
from ibllib.pipes.local_server import task_queue
from ibllib.pipes.tasks import run_alyx_task

subjects_path = Path('/mnt/s0/Data/Subjects/')

try:
    one = ONE(cache_rest=None)
    waiting_tasks = task_queue(mode='large', lab=None, one=one)

    if len(waiting_tasks) == 0:
        # TODO: proper logging
        print("No large tasks in the queue")
        # Query again only in 60 min if queue is empty
        time.sleep(3600)
    else:
        tdict = waiting_tasks[0]
        # TODO: proper logging
        print(f"Running task {tdict['name']} for session {tdict['session']}")
        ses = one.alyx.rest('sessions', 'list', django=f"pk,{tdict['session']}")[0]
        session_path = Path(subjects_path).joinpath(
            Path(ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3)))
        run_alyx_task(tdict=tdict, session_path=session_path, one=one)
except BaseException:
    # TODO: proper logging
    print(traceback.format_exc())
