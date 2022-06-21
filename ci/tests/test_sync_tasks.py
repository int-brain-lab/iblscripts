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
        wf = SyncRegisterRaw(self.session_path, sync_collection=self.sync_collection, sync=self.sync, sync_ext=self.sync_ext)
        status = wf.run()

        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.session_path)


class TestSyncMtscomp(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('widefield', 'widefieldChoiceWorld', 'JC076',
                                                              '2022-02-04', '001')
        self.widefield_path = self.session_path.joinpath('raw_widefield_data')

    def test_rename_and_compress(self):
        shutil.copytree(self.session_path.joinpath('rename_compress'), self.widefield_path)
        wf = SyncMtscomp(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        wf.run()

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

    def test_rename(self):
        shutil.copytree(self.session_path.joinpath('rename'), self.widefield_path)
        wf = SyncMtscomp(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        wf.run()

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

    def test_compress(self):
        shutil.copytree(self.session_path.joinpath('compress'), self.widefield_path)
        wf = SyncMtscomp(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        wf.run()

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.widefield_path)


class TestSyncPulses(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('widefield', 'widefieldChoiceWorld', 'JC076',
                                                              '2022-02-04', '001')
        self.widefield_path = self.session_path.joinpath('raw_widefield_data')
        shutil.copytree(self.session_path.joinpath('compress'), self.widefield_path)

    def test_extract_pulses_bin(self):
        wf = SyncPulses(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        wf.run()

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

    def test_extract_pulses_cbin(self):
        wf = SyncMtscomp(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        wf.run()

        wf = SyncPulses(self.session_path, sync_collection='raw_widefield_data', sync='nidq')
        wf.run()

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.widefield_path)


