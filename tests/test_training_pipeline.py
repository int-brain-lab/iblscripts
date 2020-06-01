import shutil
import tempfile
import unittest
from pathlib import Path

from oneibl.one import ONE
from oneibl.registration import RegistrationClient
from ibllib.pipes.training_preprocessing import TrainingExtractionPipeline
from ibllib.pipes.tasks import _run_alyx_task
# one = ONE(base_url='https://testdev.alyx.internationalbrainlab.org')
one = ONE(base_url='http://localhost:8000')

PATH_TESTS = Path('/mnt/s0/Data/IntegrationTests')
INIT_FOLDER = PATH_TESTS.joinpath('Subjects_init')


class TestPipeline(unittest.TestCase):

    def test_full_pipeline(self):

        if not INIT_FOLDER.exists():
            return

        with tempfile.TemporaryDirectory() as tdir:
            # create symlinks in a temporary directory
            subjects_path = Path(tdir).joinpath('Subjects')
            subjects_path.mkdir(exist_ok=True)
            for ff in INIT_FOLDER.rglob('*.*'):
                link = subjects_path.joinpath(ff.relative_to(INIT_FOLDER))
                if 'alf' in link.parts:
                    continue
                link.parent.mkdir(exist_ok=True, parents=True)
                link.symlink_to(ff)

            # register jobs in alyx for all the sessions
            for fil in subjects_path.rglob('_iblrig_taskData.raw*.jsonable'):
                session_path = fil.parents[1]
                register_pipeline(session_path)

            # TODO execute the list of jobs with the simplest scheduler possible
            #
            # training_jobs = one.alyx.rest('jobs', 'list', status='Waiting', ready=True,
            #                               pipeline='ibllib.pipes.training_preprocessing')
            # one.alyx.rest('jobs', 'read', id='32c83da4-8a2f-465e-8227-c3b540e61142')

            # for tdict in training_jobs:
            #     ses = one.alyx.rest('sessions', 'list', django=f"pk,{tdict['session']}")[0]
            #     session_path = subjects_path.joinpath(Path(ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3)))
            #     job, dsets = _run_alyx_task(tdict=tdict, session_path=session_path, one=one)


def register_pipeline(session_path):
    # creates the session if necessary
    print(session_path)
    # creates the session if it doesn't exist
    if one.eid_from_path(session_path) is None:
        RegistrationClient(one=one).register_session(session_path, file_list=False)
    pipe = TrainingExtractionPipeline(session_path, one=one)
    # this is boilerplate code just for the test
    tasks = one.alyx.rest('tasks', 'list', pipeline=pipe.label)  # todo get the right rest filter
    jobs = pipe.create_alyx_tasks()
    assert (len(tasks) == len(jobs) == len(pipe.jobs))
