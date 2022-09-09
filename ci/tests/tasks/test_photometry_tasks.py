from one.api import ONE
from ci.tests import base

from ibllib.io.extractors import fibrephotometry
import ibllib.pipes.photometry_tasks as photometry_tasks


class TestCopy2Server(base.IntegrationTest):

    def test_check_timestamps(self):
        # LOAD THE CSV FILES
        FOLDER_RAW_PHOTOMETRY = self.data_path.joinpath('dynamic_pipeline', 'photometry', 'rigs_data')
        daily_folders = [f for f in FOLDER_RAW_PHOTOMETRY.glob('20*') if f.is_dir()]

        for daily_folder in daily_folders:
            daq_files = list(daily_folder.glob("sync_*.tdms"))
            photometry_files = list(daily_folder.glob("raw_photometry*.csv"))
            daq_files.sort()
            photometry_files.sort()
            assert len(daq_files) == len(photometry_files)
            n_run = len(daq_files)
            for n in range(n_run):
                daq_file = daq_files[n]
                photometry_file = photometry_files[n]
                fibrephotometry.check_timestamps(daq_file, photometry_file)
                fibrephotometry.sync_photometry_to_daq(daq_file, photometry_file)


class TestPhotometryRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        from one.registration import RegistrationClient
        cache_dir = self.data_path.joinpath('dynamic_pipeline', 'photometry', 'server_data')
        self.session_path = cache_dir.joinpath('ZFM-03448', '2022-09-06', '001')
        self.one = ONE(**base.TEST_DB, cache_dir=cache_dir, cache_rest=None)
        self.one.alyx.rest('subjects', 'create', data={
            'nickname': 'ZFM-03448', 'responsible_user': 'root', 'birth_date': '2022-02-02', 'lab': 'mainenlab'})
        path, self.eid = RegistrationClient(self.one).create_new_session('ZFM-03448')

    def test_register_raw(self):
        task = photometry_tasks.FibrePhotometryRegisterRaw(self.session_path, one=self.one, collection='raw_photometry_data')
        status = task.run()
        assert status == 0
        # Even if we run the task again we should get the same output
        task.run()

    def tearDown(self) -> None:
        self.one.alyx.rest('subjects', 'delete', id='ZFM-03448')
