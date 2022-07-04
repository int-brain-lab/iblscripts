import logging
import shutil
import tempfile
from pathlib import Path
from one.registration import RegistrationClient
from one.api import ONE
from ibllib.pipes.local_server import job_creator, tasks_runner
import ibllib.pipes.dynamic_pipeline as dynamic

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestDynamicPipeline(base.IntegrationTest):

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        path, self.eid = RegistrationClient(self.one).create_new_session('ZM_1743')
        # need to create a session here
        session_path = Path(r'C:\Users\Mayo\Downloads\FlatIron\mrsicflogellab\Subjects\dynamic_ephys_SWC\2022-06-28\001')
        self.pipeline = dynamic.make_pipeline(session_path, one=self.one, eid=str(self.eid))
        self.expected_pipeline = dynamic.load_pipeline_dict(session_path)

    def test_alyx_task_dicts(self):

        pipeline_list = self.pipeline.create_tasks_list_from_pipeline()

        self.compare_dicts(pipeline_list, self.expected_pipeline, id=False)

    def test_alyx_task_creation_pipeline(self):

        alyx_tasks_from_pipe = self.pipeline.create_alyx_tasks()
        alyx_tasks_from_dict = self.pipeline.create_alyx_tasks(self.pipeline.create_tasks_list_from_pipeline())

        self.compare_dicts(alyx_tasks_from_pipe, alyx_tasks_from_dict)

    def test_alyx_task_creation_task_dict(self):
        # Now do the other way around to the tasks are made from the task_list first
        alyx_tasks_from_dict = self.pipeline.create_alyx_tasks(self.pipeline.create_tasks_list_from_pipeline())
        alyx_tasks_from_pipe = self.pipeline.create_alyx_tasks()

        self.compare_dicts(alyx_tasks_from_dict, alyx_tasks_from_pipe)

    def compare_dicts(self, dict1, dict2, id=True):
        assert len(dict1) == len(dict2)
        for d1, d2 in zip(dict1, dict2):
            if id:
                assert d1['id'] == d2['id']
            assert d1['executable'] == d2['executable']
            assert d1['parents'] == d2['parents']
            assert d1['level'] == d2['level']
            assert d1['name'] == d2['name']
            assert d1['graph'] == d2['graph']
            assert d1['arguments'] == d1['arguments'] # comment out until alyx updated

    def tearDown(self) -> None:
        self.one.alyx.rest('sessions', 'delete', id=self.eid)


class TestStandardPipelines(base.IntegrationTest):
    def setUp(self) -> None:

        self.folder_path = self.data_path.joinpath('dynamic_pipeline')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('mars', '2054-07-13', '001')
        # self.session_path.mkdir(parents=True)

    def test_ephys_3B(self):
        shutil.copytree(self.folder_path.joinpath('ephys_NP3B'), self.session_path)
        self.check_pipeline()

    def test_ephys_3A(self):
        shutil.copytree(self.folder_path.joinpath('ephys_NP3A'), self.session_path)
        self.check_pipeline()

    def test_ephys_NP24(self):
        shutil.copytree(self.folder_path.joinpath('ephys_NP24'), self.session_path)
        self.check_pipeline()

    def test_training(self):
        shutil.copytree(self.folder_path.joinpath('training'), self.session_path)
        self.check_pipeline()

    def test_habituation(self):
        shutil.copytree(self.folder_path.joinpath('habituation'), self.session_path)
        self.check_pipeline()

    def test_widefield(self):
        shutil.copytree(self.folder_path.joinpath('widefield'), self.session_path)
        self.check_pipeline()

    def check_pipeline(self):
        pipe = dynamic.make_pipeline(self.session_path)
        dy_pipe = dynamic.make_pipeline_dict(pipe, save=False)
        expected_pipe = dynamic.load_pipeline_dict(self.session_path)
        self.compare_dicts(dy_pipe, expected_pipe)

    def compare_dicts(self, dict1, dict2):
        assert len(dict1) == len(dict2)
        for d1, d2 in zip(dict1, dict2):

            assert d1['executable'] == d2['executable']
            assert d1['parents'] == d2['parents']
            assert d1['name'] == d2['name']

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestDynamicPipelineWithAlyx(base.IntegrationTest):
    def setUp(self) -> None:

        DB = {
            'base_url': 'https://dev.alyx.internationalbrainlab.org',
            'username': 'test_user',
            'password': 'TapetesBloc18',
            'silent': True
        }
        self.one = ONE(**DB)
        self.folder_path = self.data_path.joinpath('Subjects_init', 'ZM_1085', '2019-02-12', '002')

        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        path, self.eid = RegistrationClient(self.one).create_new_session('ZM_1085')
        self.session_path = self.temp_dir.joinpath(path.relative_to(self.one.cache_dir))
        self.session_path.mkdir(exist_ok=True, parents=True)

        for ff in self.folder_path.rglob('*.*'):
            link = self.session_path.joinpath(ff.relative_to(self.folder_path))
            if 'alf' in link.parts:
                continue
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)

        self.session_path.joinpath('raw_session.flag').touch()
        shutil.copy(self.data_path.joinpath('dynamic_pipeline', 'training', 'experiment_description.yaml'),
                    self.session_path.joinpath('experiment_description.yaml'))
        # also need to make an experiment description file

    def test_job_creator(self):
        dsets = job_creator(self.session_path, one=self.one)
        assert len(dsets) == 0

        tasks = self.one.alyx.rest('tasks', 'list', session=self.eid, no_cache=True)
        assert len(tasks) == 7

        all_dsets = tasks_runner(self.temp_dir, tasks, one=self.one, count=10, max_md5_size=1024 * 1024 * 20)
        print(len(all_dsets))

        complete_tasks = self.one.alyx.rest('tasks', 'list', status='Complete', session=self.eid, no_cache=True)
        assert len(complete_tasks) == 7

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
        self.one.alyx.rest('sessions', 'delete', id=self.eid)
