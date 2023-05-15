import shutil

import numpy as np
import numpy.testing
import one.alf.io as alfio
from one.api import ONE
from ibllib.io.extractors import ephys_fpga

from ci.tests import base

BPOD_FILES = [
    '_ibl_trials.table.pqt',
    '_ibl_trials.goCueTrigger_times.npy',
    '_ibl_trials.intervals_bpod.npy',
    '_ibl_trials.quiescencePeriod.npy'
]

ALIGN_BPOD_FPGA_FILES = [
    '_ibl_trials.goCueTrigger_times.npy',
    '_ibl_trials.response_times.npy'
]


class TestEphysTaskExtraction(base.IntegrationTest):

    def setUp(self) -> None:
        self.root_folder = self.data_path.joinpath('ephys')
        if not self.root_folder.exists():
            return

    def test_task_extraction_output(self):
        init_folder = self.root_folder.joinpath('choice_world_init')
        self.sessions = [f.parent for f in init_folder.rglob('raw_ephys_data')]
        for session_path in self.sessions:
            self._task_extraction_assertions(session_path)

    def test_task_extraction_problems(self):
        init_folder = self.root_folder.joinpath('ephys_choice_world_task')
        self.sessions = [
            init_folder.joinpath('CSP004/2019-11-27/001'),  # normal session
            init_folder.joinpath('ibl_witten_13/2019-11-25/001'),  # FPGA stops before bpod, custom sync
            # init_folder.joinpath('ibl_witten_27/2021-01-21/001'),  # frame2ttl flicker
        ]
        for session_path in self.sessions:
            with self.subTest(msg=session_path):
                self._task_extraction_assertions(session_path)

    def _task_extraction_assertions(self, session_path):
        alf_path = session_path.joinpath('alf')
        bk_path = alf_path.parent / 'alf.bk'
        if alf_path.exists():
            # Back-up alf files and restore on teardown
            shutil.move(alf_path, bk_path)
            self.addCleanup(shutil.move, str(bk_path), alf_path)
        # this gets the full output
        ephys_fpga.extract_all(session_path, save=True)
        # check that the output is complete
        for f in BPOD_FILES:
            self.assertTrue(alf_path.joinpath(f).exists())
        # check dimensions after alf load
        alf_trials = alfio.load_object(alf_path, 'trials')
        self.assertTrue(alfio.check_dimensions(alf_trials) == 0)
        # check new trials table the same as old individual datasets
        # in the future if extraction changes this test can be removed
        if any(bk_path.glob('*trials*')):
            alf_trials_old = alfio.load_object(alf_path.parent / 'alf.bk', 'trials')
            for k, v in alf_trials.items():
                numpy.testing.assert_array_almost_equal(v, alf_trials_old[k])
        # go deeper and check the internal fpga trials structure consistency
        sync, chmap = ephys_fpga.get_main_probe_sync(session_path)
        fpga_trials = ephys_fpga.extract_behaviour_sync(sync, chmap)
        # check dimensions
        self.assertEqual(alfio.check_dimensions(fpga_trials), 0)
        # check that the stimOn < stimFreeze < stimOff
        self.assertTrue(
            np.all(fpga_trials['stimOn_times'][:-1] < fpga_trials['stimOff_times'][:-1]))
        self.assertTrue(
            np.all(fpga_trials['stimFreeze_times'][:-1] < fpga_trials['stimOff_times'][:-1]))
        # a trial is either an error-nogo or a reward
        self.assertTrue(np.all(np.isnan(fpga_trials['valveOpen_times'][:-1] *
                                        fpga_trials['errorCue_times'][:-1])))
        self.assertTrue(np.all(np.logical_xor(np.isnan(fpga_trials['valveOpen_times'][:-1]),
                                              np.isnan(fpga_trials['errorCue_times'][:-1]))))

        # do the task qc
        # tqc_ephys.extractor.settings['PYBPOD_PROTOCOL']
        from ibllib.qc.task_extractors import TaskQCExtractor
        ex = TaskQCExtractor(session_path, lazy=True, one=None, bpod_only=False)
        ex.data = fpga_trials
        ex.extract_data()

        from ibllib.qc.task_metrics import TaskQC
        # '/mnt/s0/Data/IntegrationTests/ephys/ephys_choice_world_task/CSP004/2019-11-27/001'
        tqc_ephys = TaskQC(session_path, one=ONE(mode='local'))
        tqc_ephys.extractor = ex
        _, res_ephys = tqc_ephys.run(bpod_only=False, download_data=False)

        tqc_bpod = TaskQC(session_path, one=ONE(mode='local'))
        _, res_bpod = tqc_bpod.run(bpod_only=True, download_data=False)

        # for a swift comparison using variable explorer
        # import pandas as pd
        # df = pd.DataFrame([[res_bpod[k], res_ephys[k]] for k in res_ephys], index=res_ephys.keys())

        ok = True
        for k in res_ephys:
            if k == '_task_response_feedback_delays':
                continue
            if np.abs(res_bpod[k] - res_ephys[k]) > .2:
                ok = False
                print(f'{k} bpod: {res_bpod[k]}, ephys: {res_ephys[k]}')
        assert ok
        shutil.rmtree(alf_path, ignore_errors=True)


class TestEphysTrialsFPGA(base.IntegrationTest):

    def test_frame2ttl_flicker(self):
        from ibllib.io.extractors import ephys_fpga
        from ibllib.qc.task_metrics import TaskQC
        from ibllib.qc.task_extractors import TaskQCExtractor
        init_path = self.data_path.joinpath('ephys', 'ephys_choice_world_task')
        session_path = init_path.joinpath('ibl_witten_27/2021-01-21/001')
        dsets, out_files = ephys_fpga.extract_all(session_path, save=True)
        # Run the task QC
        qc = TaskQC(session_path, one=ONE(mode='local'))
        qc.extractor = TaskQCExtractor(session_path, lazy=True, one=ONE(mode='local'))
        # Extr+act extra datasets required for QC
        qc.extractor.data = dsets
        qc.extractor.extract_data()
        # Aggregate and update Alyx QC fields
        _, myqc = qc.run(update=False)
        # from ibllib.misc import pprint
        # pprint(myqc)
        assert myqc['_task_stimOn_delays'] > 0.9  # 0.6176
        assert myqc['_task_stimFreeze_delays'] > 0.9
        assert myqc['_task_stimOff_delays'] > 0.9
        assert myqc['_task_stimOff_itiIn_delays'] > 0.9
        assert myqc['_task_stimOn_delays'] > 0.9
        assert myqc['_task_stimOn_goCue_delays'] > 0.9
