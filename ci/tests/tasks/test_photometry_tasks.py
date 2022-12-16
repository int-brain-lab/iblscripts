import shutil

import pandas as pd

from one.api import ONE
from one.registration import RegistrationClient
from ibllib.io.session_params import read_params
from ibllib.io.extractors import fibrephotometry
import ibllib.pipes.photometry_tasks as photometry_tasks

from ci.tests import base


class TestCopy2Server(base.IntegrationTest):

    def test_check_timestamps(self):
        # LOAD THE CSV FILES
        FOLDER_RAW_PHOTOMETRY = self.data_path.joinpath('dynamic_pipeline', 'photometry', 'rigs_data', 'photometry')
        daily_folders = [f for f in FOLDER_RAW_PHOTOMETRY.glob('20*') if f.is_dir()]

        for daily_folder in daily_folders:
            daq_files = list(daily_folder.glob("sync_*.tdms"))
            photometry_files = list(daily_folder.glob("raw_photometry*.csv"))
            fp_config_files = list(daily_folder.glob("FP3002Config*.xml"))
            daq_files.sort()
            photometry_files.sort()
            fp_config_files.sort()
            assert len(daq_files) == len(photometry_files) == len(fp_config_files)
            n_run = len(daq_files)
            for n in range(n_run):
                daq_file = daq_files[n]
                photometry_file = photometry_files[n]
                fibrephotometry.check_timestamps(daq_file, photometry_file)


class BasePhotometryTaskTest(base.IntegrationTest):

    def setUp(self) -> None:
        cache_dir = self.data_path.joinpath('dynamic_pipeline', 'photometry', 'server_data')
        self.session_path = cache_dir.joinpath('ZFM-03448', '2022-09-06', '001')
        self.one = ONE(**base.TEST_DB, cache_dir=cache_dir, cache_rest=None)
        try:
            self.one.alyx.rest('subjects', 'delete', id='ZFM-03448')
        except BaseException:
            pass
        self.one.alyx.rest('subjects', 'create', data={
            'nickname': 'ZFM-03448', 'responsible_user': 'root', 'birth_date': '2022-02-02', 'lab': 'mainenlab'})
        self.acquisition_description = read_params(self.session_path)
        sdict = RegistrationClient(self.one).create_session(self.session_path)
        self.kwargs = self.acquisition_description['devices']['photometry']
        self.eid = sdict['id']

    def tearDown(self) -> None:
        self.one.alyx.rest('subjects', 'delete', id='ZFM-03448')


class TestTaskPhotometryRegisterRaw(BasePhotometryTaskTest):

    def test_register_raw(self):
        task = photometry_tasks.TaskFibrePhotometryRegisterRaw(self.session_path, one=self.one, **self.kwargs)
        status = task.run()
        assert status == 0
        # Even if we run the task again we should get the same output
        task.run()


class TestTaskFibrePhotometryPreprocess(BasePhotometryTaskTest):

    def test_extract_fp_data(self):
        task = photometry_tasks.TaskFibrePhotometryPreprocess(self.session_path, one=self.one, **self.kwargs)
        status = task.run()
        assert status == 0
        # Even if we run the task again we should get the same output
        task.run()
        fp_table = pd.read_parquet(task.outputs)
        self.assertEqual(set(fp_table.keys()), set(list(['Region1G', 'Region3G', 'color', 'name', 'times', 'wavelength'])))

    def tearDown(self) -> None:
        if self.session_path.joinpath('alf').exists():
            shutil.rmtree(self.session_path.joinpath('alf'))
