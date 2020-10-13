import shutil
import tempfile
import unittest
from pathlib import Path

from ibllib.io import raw_data_loaders as rawio
from oneibl.one import ONE
from ibllib.pipes import local_server

one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
          username='test_user', password='TapetesBloc18')

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
            nses = 0
            for fil in subjects_path.rglob('_iblrig_taskData.raw*.jsonable'):
                session_path = fil.parents[1]
                create_pipeline(session_path)
                nses += 1

            # execute the list of jobs with the simplest scheduler possible
            training_jobs = one.alyx.rest(
                'tasks', 'list', status='Waiting', graph='TrainingExtractionPipeline')
            self.assertEqual(nses * 5, len(training_jobs))
            # one.alyx.rest('jobs', 'read', id='32c83da4-8a2f-465e-8227-c3b540e61142')

            local_server.tasks_runner(subjects_path, training_jobs, one=one, dry=True,
                                      count=nses * 10)
            local_server.tasks_runner(subjects_path, training_jobs, one=one, count=nses * 10,
                                      dry=False, max_md5_size=1024 * 1024 * 20)
            tasks = one.alyx.rest('tasks', 'list', status='Errored',
                                  graph='TrainingExtractionPipeline',
                                  django="~session__task_protocol__icontains,habituation")
            # FIXME: remove django line above when habituation extraction gets fixed
            assert(len(tasks) == 0)
            eids = list(set([t['session'] for t in training_jobs]))
            session_dict = one.alyx.rest('sessions', 'read', id=eids[1])
            self.assertTrue(len(session_dict['extended_qc'].keys()) > 4)


def create_pipeline(session_path):
    # creates the session if necessary
    task_type = rawio.get_session_extractor_type(session_path)
    print(session_path, task_type)
    session_path.joinpath('extract_me.flag').touch()
    # delete the session if it exists
    eid = one.eid_from_path(session_path)
    if eid is not None:
        one.alyx.rest('sessions', 'delete', id=eid)
    local_server.job_creator(session_path, one=one, max_md5_size=1024 * 1024 * 20)
    eid = one.eid_from_path(session_path)
    assert(eid)
    alyx_tasks = one.alyx.rest('tasks', 'list', session=eid, graph='TrainingExtractionPipeline')
    assert(len(alyx_tasks) == 5)
