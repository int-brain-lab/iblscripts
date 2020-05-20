import unittest
from pathlib import Path
import shutil

from ibllib.pipes import ephys_preprocessing
from ibllib.pipes.jobs import run_alyx_job
from oneibl.one import ONE


PATH_TESTS = Path('/mnt/s0/Data/IntegrationTests')
SESSION_PATH = PATH_TESTS.joinpath("ephys/choice_world/KS022/2019-12-10/001")
one = ONE(base_url='http://localhost:8000')


class TestEphysPipeline(unittest.TestCase):

    def setUp(self) -> None:
        self.init_folder = PATH_TESTS.joinpath('ephys', 'choice_world_init')
        if not self.init_folder.exists():
            return
        self.main_folder = PATH_TESTS.joinpath('ephys', 'choice_world')
        if self.main_folder.exists():
            shutil.rmtree(self.main_folder)
        self.main_folder.mkdir(exist_ok=True)
        for ff in self.init_folder.rglob('*.*'):
            link = self.main_folder.joinpath(ff.relative_to(self.init_folder))
            if 'alf' in link.parts:
                continue
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)

    def test_pipeline_with_alyx(self):
        eid = one.eid_from_path(SESSION_PATH)

        # prepare by deleting all jobs/tasks related
        jobs = one.alyx.rest('jobs', 'list', session=eid)
        tasks = list(set([j['task'] for j in jobs]))
        [one.alyx.rest('tasks', 'delete', id=task) for task in tasks]

        # create jobs from scratch
        NJOBS = 4
        ephys_pipe = ephys_preprocessing.EphysExtractionPipeline(SESSION_PATH, one=one)
        ephys_pipe.make_graph(show=False)
        alyx_tasks = ephys_pipe.init_alyx_tasks()
        self.assertTrue(len(alyx_tasks) == NJOBS)

        alyx_jobs = ephys_pipe.register_alyx_jobs()
        self.assertTrue(len(alyx_jobs) == NJOBS)

        # get the pending jobs from alyx
        jobs = one.alyx.rest('jobs', 'list', session=eid, status='Waiting')
        self.assertTrue(len(jobs) == NJOBS)

        # run them and make sure their statuses got updated
        all_datasets = []
        for jdict in jobs:
            status, dsets = run_alyx_job(jdict=jdict, session_path=SESSION_PATH, one=one)
            if dsets is not None:
                all_datasets.extend(dsets)

        for dset in all_datasets:
            print(dset['name'])