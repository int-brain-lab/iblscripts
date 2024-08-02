
from ibllib.pipes.local_server import task_queue
from ibllib.pipes.tasks import run_alyx_task
from one.api import ONE
from pathlib import Path

if __name__ == "__main__":
    one = ONE()
    tasks = task_queue(mode='large', alyx=one.alyx, env=["iblsorter"])
    if len(tasks) == 0:
        pass
    tdict = tasks[0]
    ses = one.alyx.rest('sessions', 'list', django=f"pk,{tdict['session']}")[0]
    subjects_path = "/mnt/s0/Data/Subjects/"
    session_path = Path(subjects_path).joinpath(
        Path(ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3))
    )
    task, _ = run_alyx_task(tdict=tdict, session_path=session_path, one=one)