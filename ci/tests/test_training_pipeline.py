import tempfile
from pathlib import Path

from ibllib.io.extractors.base import get_session_extractor_type
from one.api import ONE
from ibllib.pipes import local_server

from ci.tests import base


class TestPipeline(base.IntegrationTest):
    def test_full_pipeline(self):
        """
        Test the full Training extraction pipeline.
        :return:
        """
        INIT_FOLDER = self.data_path.joinpath('Subjects_init')
        self.assertTrue(INIT_FOLDER.exists())

        with tempfile.TemporaryDirectory() as tdir:
            one = ONE(**base.TEST_DB, cache_dir=tdir, cache_rest=None)
            # create symlinks in a temporary directory
            subjects_path = Path(tdir).joinpath('mainenlab', 'Subjects')
            subjects_path.mkdir(exist_ok=True, parents=True)
            for ff in INIT_FOLDER.rglob('*.*'):
                link = subjects_path.joinpath(ff.relative_to(INIT_FOLDER))
                if 'alf' in link.parts:
                    continue
                link.parent.mkdir(exist_ok=True, parents=True)
                link.symlink_to(ff)

            # register jobs in alyx for all the sessions
            nses = 0
            session_stubs = [
                './IBL_46/2019-02-19/001',  # time stamps million years in future
                './ZM_335/2018-12-13/001',  # rotary encoder ms instead of us
                './ZM_1085/2019-02-12/002',  # rotary encoder corrupt
                './ZM_1085/2019-07-01/001'  # habituation session rig version 5.0.0
            ]
            for stub in session_stubs:
                session_path = subjects_path.joinpath(stub)
                create_pipeline(session_path, one)
                nses += 1

            # execute the list of jobs with the simplest scheduler possible
            training_jobs = one.alyx.rest(
                'tasks', 'list', status='Waiting',
                graph='TrainingExtractionPipeline', no_cache=True)
            self.assertEqual(nses * 6, len(training_jobs))
            # one.alyx.rest('jobs', 'read', id='32c83da4-8a2f-465e-8227-c3b540e61142')

            local_server.tasks_runner(subjects_path, training_jobs, one=one, dry=True,
                                      count=nses * 10)
            local_server.tasks_runner(subjects_path, training_jobs, one=one, count=nses * 10,
                                      dry=False, max_md5_size=1024 * 1024 * 20)
            errored_tasks = one.alyx.rest('tasks', 'list', status='Errored',
                                          graph='TrainingExtractionPipeline', no_cache=True)
            self.assertTrue(len(errored_tasks) == 0)
            session_dict = one.alyx.rest('sessions', 'list',
                                         django='extended_qc__isnull,False', no_cache=True)
            self.assertTrue(len(session_dict) > 0)


def create_pipeline(session_path, one):
    # creates the session if necessary
    task_type = get_session_extractor_type(session_path)
    print(session_path, task_type)
    session_path.joinpath('raw_session.flag').touch()
    # delete the session if it exists
    eid = one.path2eid(session_path, query_type='remote')
    if eid is not None:
        one.alyx.rest('sessions', 'delete', id=eid)
    local_server.job_creator(session_path, one=one, max_md5_size=1024 * 1024 * 20)
    eid = one.path2eid(session_path, query_type='remote')
    assert eid
    alyx_tasks = one.alyx.rest('tasks', 'list',
                               session=eid, graph='TrainingExtractionPipeline', no_cache=True)
    assert(len(alyx_tasks) == 6)
