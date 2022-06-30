import logging
import shutil
import tempfile
from pathlib import Path
from one.api import ONE
import ibllib.pipes.dynamic_pipeline as dynamic

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestDynamicPipeline():

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        from one.registration import RegistrationClient
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
            # assert d1['arguments'] == d1['arguments'] # comment out until alyx updated

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
