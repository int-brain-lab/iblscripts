import logging
import random
import string
import shutil
import tempfile
from pathlib import Path

from one.api import ONE
from ibllib.pipes.ephys_tasks import EphysRegisterRaw, EphysCompressNP1, EphysCompressNP21, EphysCompressNP24

from ci.tests import base

_logger = logging.getLogger('ibllib')

class TestEphysRegisterRaw(base.IntegrationTest):

    # TODO need to ask about how to add the extra probe models to test alyx

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB, cache_dir=self.data_path / 'ephys', cache_rest=None)
        # make a random session path and move the meta files into random probe names
        self.session_path = Path(tempfile.TemporaryDirectory().name).joinpath('flowers', '2018-07-13', '001')
        # make random probe names and the directories

        self.probe_NP1, self.probe_NP21, self.probe_NP24 = [''.join(random.choices(string.ascii_letters, k=5)) for i in range(3)]

        self.expected_probes = [self.probe_NP24 + ext for ext in ['a', 'b', 'c', 'd']] + [self.probe_NP21, self.probe_NP24]
        self.expected_models = ['NP2_4'] * 4 + ['NP2_1', '3B2']

        meta_path = self.data_path.joinpath('ephys', 'ephys_np2', 'raw_ephys_data', 'probe00')
        shutil.copytree(meta_path.joinpath('NP1_meta'), self.session_path.joinpath(self.probe_NP1))
        shutil.copytree(meta_path.joinpath('NP21_meta'), self.session_path.joinpath(self.probe_NP21))
        shutil.copytree(meta_path.joinpath('NP24_meta'), self.session_path.joinpath(self.probe_NP24))

    def test_register_raw(self):
        register_task = EphysRegisterRaw(self.session_path, one=self.one)
        status = register_task.run()

        assert status == 0

        # check all the probes have been created
        for probe, model in zip(self.expected_probes, self.expected_models):
            pid = self.one.alyx.rest('insertions', 'list', session=self.eid, name=probe)
            assert len(pid) == 1
            assert pid['model'] == model

    def tearDown(self) -> None:
        shutil.rmtree(self.session_path)
        for probe in self.expected_probes:
            pid = self.one.alyx.rest('insertions', 'list', session=self.eid, name=probe)
            if len(pid) > 0:
                self.one.alyx.rest('insertions', 'delete', pid['id'])


class TestEphysSyncRegisterRaw(base.IntegrationTest):
    pass


class TestEphysCompressNP1(base.IntegrationTest):
    def setUp(self) -> None:
        self.data_path = self.data_path.joinpath('ephys', 'ephys_np2', 'raw_ephys_data', 'probe00')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('jupiter', '2054-07-13', '001')
        self.probe = 'probe00'
        self.probe_path = self.session_path.joinpath('raw_ephys_data', self.probe)
        self.probe_path.mkdir(parents=True)

        self.probe_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.wiring.json').touch()
        self.meta_path = self.data_path.joinpath('NP1_meta', '_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        ap_meta_path = self.probe_path.joinpath(self.meta_path.name)
        lf_meta_path = self.probe_path.joinpath(self.meta_path.name.replace('ap', 'lf'))

        shutil.copy(self.meta_path, ap_meta_path)
        shutil.copy(self.meta_path, lf_meta_path)

        self.file_path = self.data_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.bin')

    def test_compress(self):
        ap_file_path = self.probe_path.joinpath(self.file_path.name)
        lf_file_path = self.probe_path.joinpath(self.file_path.name.replace('ap', 'lf'))
        shutil.copy(self.file_path, ap_file_path)
        shutil.copy(self.file_path, lf_file_path)

        task = EphysCompressNP1(self.session_path, sync_collection='raw_ephys_data', pname=self.probe)
        status = task.run()
        assert status == 0
        self.check_files(task)

    def test_already_compress(self):
        ap_file_path = self.probe_path.joinpath(self.file_path.name).with_suffix('.cbin')
        lf_file_path = self.probe_path.joinpath(self.file_path.name.replace('ap', 'lf')).with_suffix('.cbin')
        ap_ch_path = self.probe_path.joinpath(self.file_path.name).with_suffix('.ch')
        lf_ch_path = self.probe_path.joinpath(self.file_path.name.replace('ap', 'lf')).with_suffix('.ch')
        shutil.copy(self.file_path, ap_file_path)
        shutil.copy(self.file_path, lf_file_path)
        shutil.copy(self.meta_path, ap_ch_path)
        shutil.copy(self.meta_path, lf_ch_path)

        task = EphysCompressNP1(self.session_path, sync_collection='raw_ephys_data', pname=self.probe)
        status = task.run()
        assert status == 0
        self.check_files(task)

    def test_register(self):
        ap_ch_path = self.probe_path.joinpath(self.file_path.name).with_suffix('.ch')
        lf_ch_path = self.probe_path.joinpath(self.file_path.name.replace('ap', 'lf')).with_suffix('.ch')
        shutil.copy(self.meta_path, ap_ch_path)
        shutil.copy(self.meta_path, lf_ch_path)

        task = EphysCompressNP1(self.session_path, device_collection='raw_ephys_data', pname=self.probe)
        status = task.run()
        assert status == 0

        self.check_files(task, ignore_ext='.cbin')

    def check_files(self, task, ignore_ext='blablabla'):
        for exp_files in task.signature['output_files']:
            if ignore_ext not in exp_files[0]:
                file = next(self.session_path.joinpath(exp_files[1]).glob(exp_files[0]), None)
                assert file.exists()
                assert file in task.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestEphysCompressNP21(base.IntegrationTest):

    def setUp(self) -> None:
        self.data_path = self.data_path.joinpath('ephys', 'ephys_np2', 'raw_ephys_data', 'probe00')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('jupiter', '2054-07-13', '001')
        self.probe = 'probe00'
        self.probe_path = self.session_path.joinpath('raw_ephys_data', self.probe)
        self.probe_path.mkdir(parents=True)

        self.probe_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.wiring.json').touch()
        self.meta_path = self.data_path.joinpath('NP21_meta', '_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        ap_meta_path = self.probe_path.joinpath(self.meta_path.name)
        lf_meta_path = self.probe_path.joinpath(self.meta_path.name.replace('ap', 'lf'))

        shutil.copy(self.meta_path, ap_meta_path)
        shutil.copy(self.meta_path, lf_meta_path)

        self.file_path = self.data_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.bin')
        self.cfile_path = self.data_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.cbin')
        self.ch_path = self.data_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.ch')

    def test_process(self):
        ap_file_path = self.probe_path.joinpath(self.file_path.name)
        shutil.copy(self.file_path, ap_file_path)

        task = EphysCompressNP21(self.session_path, sync_collection='raw_ephys_data', pname=self.probe)
        status = task.run()
        assert status == 0

        self.check_files(task)

    def test_already_processed(self):
        ap_file_path = self.probe_path.joinpath(self.cfile_path.name)
        lf_file_path = self.probe_path.joinpath(self.cfile_path.name.replace('ap', 'lf'))
        ap_ch_path = self.probe_path.joinpath(self.ch_path.name)
        lf_ch_path = self.probe_path.joinpath(self.ch_path.name.replace('ap', 'lf'))
        shutil.copy(self.cfile_path, ap_file_path)
        shutil.copy(self.cfile_path, lf_file_path)
        shutil.copy(self.ch_path, ap_ch_path)
        shutil.copy(self.ch_path, lf_ch_path)

        task = EphysCompressNP21(self.session_path, sync_collection='raw_ephys_data', pname=self.probe)
        status = task.run()
        print(task.outputs)
        assert status == 0
        self.check_files(task)

    def test_register(self):
        ap_ch_path = self.probe_path.joinpath(self.ch_path.name)
        lf_ch_path = self.probe_path.joinpath(self.ch_path.name.replace('ap', 'lf'))
        shutil.copy(self.meta_path, ap_ch_path)
        shutil.copy(self.meta_path, lf_ch_path)

        task = EphysCompressNP21(self.session_path, device_collection='raw_ephys_data', pname=self.probe)
        status = task.run()
        assert status == 0
        self.check_files(task, ignore_ext='.cbin')

    def check_files(self, task, ignore_ext='blablabla'):
        for exp_files in task.signature['output_files']:
            if ignore_ext not in exp_files[0]:
                file = next(self.session_path.joinpath(exp_files[1]).glob(exp_files[0]), None)
                assert file.exists()
                assert file in task.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestEphysCompressNP24(base.IntegrationTest):

    def setUp(self) -> None:
        self.data_path = self.data_path.joinpath('ephys', 'ephys_np2', 'raw_ephys_data', 'probe00')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('jupiter', '2054-07-13', '001')
        self.probe = 'probe00'
        self.probe_path = self.session_path.joinpath('raw_ephys_data', self.probe)
        self.probe_path.mkdir(parents=True)

        self.probe_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.wiring.json').touch()
        self.meta_path = self.data_path.joinpath('NP24_meta', '_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        ap_meta_path = self.probe_path.joinpath(self.meta_path.name)
        lf_meta_path = self.probe_path.joinpath(self.meta_path.name.replace('ap', 'lf'))

        shutil.copy(self.meta_path, ap_meta_path)
        shutil.copy(self.meta_path, lf_meta_path)

        self.file_path = self.data_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.bin')
        self.cfile_path = self.data_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.cbin')
        self.ch_path = self.data_path.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.ch')

    def test_process(self):
        ap_file_path = self.probe_path.joinpath(self.file_path.name)
        shutil.copy(self.file_path, ap_file_path)

        task = EphysCompressNP24(self.session_path, sync_collection='raw_ephys_data', pname=self.probe)
        status = task.run()
        assert status == 0
        self.check_files(task)

        # This checks the already processed
        status = task.run()
        assert status == 0
        assert
        self.check_files(task)


    def check_files(self, task, ignore_ext='blablalba'):
        for exp_files in task.signature['output_files']:
            if ignore_ext not in exp_files[0]:
                file = next(self.session_path.joinpath(exp_files[1]).glob(exp_files[0]), None)
                assert file.exists()
                assert file in task.outputs

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)




