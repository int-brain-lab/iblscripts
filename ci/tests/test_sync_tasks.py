import logging
import shutil

from ibllib.pipes.sync_tasks import SyncRegisterRaw, SyncMtscomp, SyncPulses

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestSyncRegisterRaw(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('widefield', 'widefieldChoiceWorld', 'JC076',
                                                              'test_date', 'test_sess')
        self.sync_collection = 'raw_device_collection'
        self.sync = 'random'
        self.sync_ext = 'tdms'

        self.sync_path = self.session_path.joinpath(self.sync_collection)
        self.sync_path.mkdir(exist_ok=True, parents=True)
        self.daq_file = self.sync_path.joinpath(f'daq.raw.{self.sync}.{self.sync_ext}')
        self.wiring_file = self.sync_path.joinpath(f'daq.raw.{self.sync}.wiring.json')
        self.daq_file.touch()
        self.wiring_file.touch()

    def test_register(self):
        task = SyncRegisterRaw(self.session_path, sync_collection=self.sync_collection, sync=self.sync, sync_ext=self.sync_ext)
        status = task.run()

        assert status == 0

        for exp_files in task.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in task.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.session_path)


class TestSyncMtscomp(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('widefield', 'widefieldChoiceWorld', 'JC076',
                                                              '2022-02-04', '001')
        self.widefield_path = self.session_path.joinpath('raw_widefield_data')

    def test_rename_and_compress(self):
        shutil.copytree(self.session_path.joinpath('rename_compress'), self.widefield_path)
        task = SyncMtscomp(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        status = task.run()
        assert status == 0
        self.check_files(task)

    def test_rename(self):
        shutil.copytree(self.session_path.joinpath('rename'), self.widefield_path)
        task = SyncMtscomp(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        status = task.run()
        assert status == 0
        self.check_files(task)

    def test_compress(self):
        shutil.copytree(self.session_path.joinpath('compress'), self.widefield_path)
        task = SyncMtscomp(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        status = task.run()
        assert status == 0
        self.check_files(task)

    def test_register(self):
        shutil.copytree(self.session_path.joinpath('register'), self.widefield_path)
        task = SyncMtscomp(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        status = task.run()
        assert status == 0
        # Here we don't expect the .cbin file
        self.check_files(task, ignore_ext='.cbin')

    def check_files(self, task, ignore_ext=None):
        for exp_files in task.signature['output_files']:
            if ignore_ext not in exp_files[0]:
                file = next(self.session_path.joinpath(exp_files[1]).glob(exp_files[0]), None)
                assert file.exists()
                assert file in task.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.widefield_path)


class TestSyncPulses(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('widefield', 'widefieldChoiceWorld', 'JC076',
                                                              '2022-02-04', '001')
        self.widefield_path = self.session_path.joinpath('raw_widefield_data')
        shutil.copytree(self.session_path.joinpath('compress'), self.widefield_path)

    def test_extract_pulses_bin(self):
        task = SyncPulses(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        status = task.run()
        assert status == 0
        for exp_files in task.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in task.outputs

    def test_extract_pulses_cbin(self):
        task = SyncMtscomp(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        task.run()

        task = SyncPulses(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        status = task.run()
        assert status == 0
        for exp_files in task.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in task.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.widefield_path)


