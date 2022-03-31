import traceback
import time

from one.api import ONE
from ibllib.pipes.local_server import task_queue
from ibllib.pipes.tasks import run_alyx_task

try:
    one = ONE(cache_rest=None)
    waiting_tasks = task_queue(mode='large', lab=None, one=one)

    if len(waiting_tasks) == 0:
        # TODO: proper logging
        print("No large tasks in the queue")
        # Query again only in 10 min if queue is empty
        time.sleep(600)
    else:
        tdict = waiting_tasks[0]
        session_path = one.eid2path(tdict['session'])
        run_alyx_task(tdict=tdict, session_path=session_path, one=one)
except BaseException:
    # TODO: proper logging
    print(traceback.format_exc())
