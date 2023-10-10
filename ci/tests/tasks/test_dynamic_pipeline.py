import logging
import shutil
import tempfile
from pathlib import Path
from one.registration import RegistrationClient
from one.api import ONE
from ibllib.pipes.local_server import job_creator, tasks_runner
import ibllib.pipes.dynamic_pipeline as dynamic
import ibllib.io.session_params as sess_params
from ibllib.io.raw_data_loaders import patch_settings

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestDynamicPipeline(base.IntegrationTest):

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        path, self.eid = RegistrationClient(self.one).create_new_session('ZM_1743')
        # need to create a session here
        session_path = self.data_path.joinpath('dynamic_pipeline', 'ephys_NP3B')
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
        self.assertEqual(len(dict2), len(dict1))
        for d1, d2 in zip(dict1, dict2):
            if id:
                self.assertEqual(d2['id'], d1['id'])
            for k in ('executable', 'parents', 'name', 'level', 'graph', 'arguments'):
                with self.subTest(key=k):
                    if d2[k] is list:
                        self.assertCountEqual(d2[k], d1[k])
                    else:
                        self.assertEqual(d2[k], d1[k])

    def tearDown(self) -> None:
        self.one.alyx.rest('sessions', 'delete', id=self.eid)


class TestStandardPipelines(base.IntegrationTest):
    def setUp(self) -> None:
        self.folder_path = self.data_path.joinpath('dynamic_pipeline')
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.session_path = Path(self.temp_dir.name).joinpath('mars', '2054-07-13', '001')

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

    def test_photometry(self):
        src = self.folder_path.joinpath('photometry', 'server_data', 'ZFM-03448', '2022-09-06', '001')
        shutil.copytree(src, self.session_path)
        self.check_pipeline()

    def test_mesoscope(self):
        shutil.copytree(self.folder_path.joinpath('mesoscope'), self.session_path)
        filepath = Path(base.__file__).parents[2].joinpath('deploy', 'mesoscope', '_ibl_experiment.description.yaml')
        shutil.copy(filepath, self.session_path)
        self.check_pipeline()

    def test_chained(self):
        """Test pipeline creation when there are multiple task protocols run within a session"""
        shutil.copytree(self.folder_path.joinpath('chained'), self.session_path)
        shutil.copytree(self.folder_path.joinpath('ephys_NP3B', 'raw_ephys_data'),
                        self.session_path.joinpath('raw_ephys_data'))
        self.check_pipeline()

    def test_extractors(self):
        """
        Test pipeline creation when the tesk extractors are defined within the
        experiment.description file.
        """
        shutil.copytree(self.folder_path.joinpath('extractors'), self.session_path)
        shutil.copytree(self.folder_path.joinpath('ephys_NP24', 'raw_ephys_data'),
                        self.session_path.joinpath('raw_ephys_data'))
        self.check_pipeline()
        # Tests that an error is raised if sync and extractor aren't matching
        exp_desc = sess_params.read_params(self.session_path)
        exp_desc['sync'] = {'bpod': exp_desc['sync']['nidq']}
        sess_params.write_params(self.session_path, exp_desc)
        self.assertRaises(ValueError, self.check_pipeline)
        # Modify the experiment description to include an novel task
        exp_desc['tasks'] = [
            {'nouveauChoiceWorld':
                {'collection': 'raw_task_data_00',
                 'extractors': ['TrialRegisterRaw', 'ChoiceWorldTrialsBpod'],
                 'sync_label': 'bpod'}}
        ]
        sess_params.write_params(self.session_path, exp_desc)
        pipe = dynamic.make_pipeline(self.session_path)
        dy_pipe = dynamic.make_pipeline_dict(pipe, save=False)
        task = next((x for x in dy_pipe if x['name'] == 'ChoiceWorldTrialsBpod_00'), None)
        self.assertIsNotNone(task, 'failed to create specified extractor task')
        self.assertEqual('ibllib.pipes.behavior_tasks.ChoiceWorldTrialsBpod', task['executable'])
        self.assertEqual(['TrialRegisterRaw_00'], task['parents'])
        self.assertEqual('nouveauChoiceWorld', task['arguments'].get('protocol'))
        # Finally, check raises not implemented error when extractor not found
        exp_desc['tasks'][0]['nouveauChoiceWorld']['extractors'].append('FooBarBpod')
        sess_params.write_params(self.session_path, exp_desc)
        self.assertRaises(NotImplementedError, self.check_pipeline)

    def check_pipeline(self):
        pipe = dynamic.make_pipeline(self.session_path)
        dy_pipe = dynamic.make_pipeline_dict(pipe, save=False)
        expected_pipe = dynamic.load_pipeline_dict(self.session_path)
        self.compare_dicts(dy_pipe, expected_pipe)

    def compare_dicts(self, dict1, dict2):
        self.assertEqual(len(dict1), len(dict2))
        for d1, d2 in zip(dict1, dict2):
            for k in ('executable', 'parents', 'name', 'arguments'):
                with self.subTest(key=k, name_1=d1.get('name'), name_2=d2.get('name')):
                    self.assertEqual(d2[k], d1[k])


class TestDynamicPipelineWithAlyx(base.IntegrationTest):
    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        self.folder_path = self.data_path.joinpath('Subjects_init', 'ZM_1085', '2019-02-12', '002')

        self.temp_dir = tempfile.TemporaryDirectory()
        path, self.eid = RegistrationClient(self.one).create_new_session('ZM_1085')
        self.session_path = Path(self.temp_dir.name).joinpath(path.relative_to(self.one.cache_dir))
        self.session_path.mkdir(exist_ok=True, parents=True)

        for ff in self.folder_path.rglob('*.*'):
            link = self.session_path.joinpath(ff.relative_to(self.folder_path))
            if 'alf' in link.parts:
                continue
            if link.name == '_iblrig_taskSettings.raw.json':
                shutil.copy(ff, link)  # Copy settings as we'll modify them
            else:
                link.parent.mkdir(exist_ok=True, parents=True)
                link.symlink_to(ff)

        self.session_path.joinpath('raw_session.flag').touch()
        shutil.copy(
            self.data_path.joinpath('dynamic_pipeline', 'training', '_ibl_experiment.description.yaml'),
            self.session_path.joinpath('_ibl_experiment.description.yaml'),
        )
        # Patch the settings file
        subject, date, number = self.session_path.parts[-3:]
        patch_settings(self.session_path, subject=subject, date=date, number=path.parts[-1])

    def test_run_dynamic_pipeline_full(self):
        """
        This runs the full suite of tasks on a TrainingChoiceWorld task
        """
        pipes, dsets = job_creator(self.session_path, one=self.one)
        self.assertEqual(0, len(dsets))

        tasks = self.one.alyx.rest('tasks', 'list', session=self.eid, no_cache=True)
        self.assertEqual(8, len(tasks))

        all_dsets = tasks_runner(self.temp_dir.name, tasks, one=self.one, count=10, max_md5_size=1024 * 1024 * 20)

        for t in self.one.alyx.rest('tasks', 'list', session=self.eid, no_cache=True):
            with self.subTest(name=t['name']):
                self.assertEqual(t['status'], 'Complete')

        self.assertEqual(len(all_dsets), 20)
        self.assertIn('_ibl_experiment.description.yaml', [d['name'] for d in all_dsets])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        self.one.alyx.rest('sessions', 'delete', id=self.eid)


class TestExperimentDescription(base.IntegrationTest):
    def setUp(self) -> None:
        file = self.data_path.joinpath('dynamic_pipeline', 'ephys_NP3B', '_ibl_experiment.description.yaml')
        self.experiment_description = sess_params.read_params(file)

    def test_params_reading(self):
        self.assertEqual(sess_params.get_sync_label(self.experiment_description), 'nidq')
        self.assertEqual(sess_params.get_sync_extension(self.experiment_description), 'bin')
        self.assertEqual(sess_params.get_sync_namespace(self.experiment_description), 'spikeglx')
        self.assertEqual(sess_params.get_sync_collection(self.experiment_description), 'raw_ephys_data')
        self.assertEqual(sess_params.get_cameras(self.experiment_description), ['body', 'left', 'right'])
        self.assertEqual(sess_params.get_task_collection(self.experiment_description, 'ephysChoiceWorld'), 'raw_behavior_data')
        self.assertEqual(sess_params.get_task_protocol(self.experiment_description, 'raw_behavior_data'), 'ephysChoiceWorld')
        self.assertEqual(sess_params.get_task_protocol(self.experiment_description, 'raw_passive_data'), 'passiveChoiceWorld')

        collections = sess_params.get_task_collection(self.experiment_description)
        self.assertCountEqual({'raw_behavior_data', 'raw_passive_data'}, collections)
        protocols = sess_params.get_task_protocol(self.experiment_description)
        self.assertCountEqual({'ephysChoiceWorld', 'passiveChoiceWorld'}, protocols)

    def test_compatibility(self):
        """Test for ibllib.io.session_params._patch_file.

        This checks whether a description file is old and modified the dict to be compatible with
        the most recent spec.
        """
        files = sorted(self.data_path.joinpath('dynamic_pipeline', 'old').glob('_ibl_experiment.description*.yaml'))
        for file in files:
            with self.subTest(file.stem.rsplit('_')[-1]):
                exp_dec = sess_params.read_params(file)
                self.assertIsInstance(exp_dec['tasks'], list, 'failed to convert tasks key to list')
                expected = ('passiveChoiceWorld', 'ephysChoiceWorld')
                self.assertCountEqual(expected, (next(iter(x.keys())) for x in exp_dec['tasks']))
