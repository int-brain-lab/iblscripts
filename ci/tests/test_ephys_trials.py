import shutil
import warnings

import numpy as np
import numpy.testing
import one.alf.io as alfio
from one.api import ONE

from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.task_metrics import TaskQC
from ibllib.io.extractors import ephys_fpga

from ci.tests import base

BPOD_FILES = [
    '_ibl_trials.table.pqt',
    '_ibl_trials.goCueTrigger_times.npy',
    '_ibl_trials.quiescencePeriod.npy'
]

ALIGN_BPOD_FPGA_FILES = [
    '_ibl_trials.goCueTrigger_times.npy',
    '_ibl_trials.response_times.npy'
]


class TestEphysTaskExtraction(base.IntegrationTest):
    """Test the FpgaTrials extractor."""

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

    @staticmethod
    def _restore_alf(session_path):
        """Restore backup alf folder."""
        alf_path = session_path.joinpath('alf')
        bk_path = alf_path.parent / 'alf.bk'
        if alf_path.exists() and bk_path.exists():
            shutil.rmtree(alf_path, ignore_errors=True)
            shutil.move(str(bk_path), str(alf_path))

    def _task_extraction_assertions(self, session_path):
        """Compare task extraction with expected ALF trials output."""
        alf_path = session_path.joinpath('alf')
        bk_path = alf_path.parent / 'alf.bk'
        if alf_path.exists():
            # Back-up alf files and restore on teardown
            if bk_path.exists():  # if last cleanup failed
                warnings.warn(f'{bk_path} already exists; removing alf path')
                # assume backup is correct validation data and delete the alf folder
                shutil.rmtree(alf_path, ignore_errors=True)
            else:
                shutil.move(alf_path, bk_path)
            self.addCleanup(self._restore_alf, session_path)
        elif not bk_path.exists():
            raise ValueError(f'alf folder missing for session {session_path}')

        from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsNidq
        task = ChoiceWorldTrialsNidq(session_path,
                                     one=ONE(mode='local'), collection='raw_behavior_data',
                                     sync_collection='raw_ephys_data')
        fpga_trials, _ = task._extract_behaviour(save=True)
        tqc_ephys = task._run_qc(fpga_trials.copy(), update=False, plot_qc=False)

        # check that the output is complete
        for f in BPOD_FILES:
            with self.subTest(file=f):
                self.assertTrue(alf_path.joinpath(f).exists())
        # check dimensions after alf load
        alf_trials = alfio.load_object(alf_path, 'trials')
        self.assertTrue(alfio.check_dimensions(alf_trials) == 0)
        # check new trials table the same as old individual datasets
        # in the future if extraction changes this test can be removed
        if any(bk_path.glob('*trials*')):
            alf_trials_old = alfio.load_object(alf_path.parent / 'alf.bk', 'trials')
            for k, v in alf_trials.items():
                if k in alf_trials_old:  # added quiescencePeriod from the old dataset
                    with self.subTest(attribute=k):
                        numpy.testing.assert_array_almost_equal(v, alf_trials_old[k])
        # go deeper and check the internal fpga trials structure consistency
        fpga_trials = {k: v for k, v in fpga_trials.items() if 'wheel' not in k}
        # check dimensions
        self.assertEqual(alfio.check_dimensions(fpga_trials), 0)
        # check that the stimOn < stimFreeze < stimOff
        self.assertTrue(
            np.all(fpga_trials['table']['stimOn_times'][:-1] < fpga_trials['stimOff_times'][:-1]))
        self.assertTrue(
            np.all(fpga_trials['stimFreeze_times'][:-1] < fpga_trials['stimOff_times'][:-1]))
        # a trial is either an error-nogo or a reward
        self.assertTrue(np.all(np.isnan(fpga_trials['valveOpen_times'][:-1] *
                                        fpga_trials['errorCue_times'][:-1])))
        self.assertTrue(np.all(np.logical_xor(np.isnan(fpga_trials['valveOpen_times'][:-1]),
                                              np.isnan(fpga_trials['errorCue_times'][:-1]))))

        # do the task qc
        _, res_ephys = tqc_ephys.run(bpod_only=False, download_data=False)

        tqc_bpod = TaskQC(session_path, one=ONE(mode='local'))
        tqc_bpod.extractor = TaskQCExtractor(session_path, lazy=True, one=None, bpod_only=True)
        tqc_bpod.extractor.settings = task.extractor.settings
        tqc_bpod.extractor.data = tqc_bpod.extractor.rename_data(task.extractor.bpod_trials.copy())
        # tqc_bpod.extractor.frame_ttls = task.extractor.bpod_extractor.frame2ttl  # used in iblapps QC viewer
        # tqc_bpod.extractor.audio_ttls = task.extractor.bpod_extractor.audio  # used in iblapps QC viewer
        from ibllib.io.raw_data_loaders import load_bpod_fronts  # FIXME Remove these two lines and uncomment the above
        tqc_bpod.extractor.frame_ttls, tqc_bpod.extractor.audio_ttls = (
            load_bpod_fronts(session_path, data=task.extractor.bpod_extractor.bpod_trials, task_collection='raw_behavior_data'))

        _, res_bpod = tqc_bpod.run(bpod_only=True, download_data=False)

        # for a swift comparison using variable explorer
        # import pandas as pd
        # df = pd.DataFrame([[res_bpod[k], res_ephys[k]] for k in res_ephys], index=res_ephys.keys())

        for k in res_ephys:
            if k == '_task_response_feedback_delays':
                continue
            with self.subTest(check=k):
                self.assertFalse(
                    np.abs(res_bpod[k] - res_ephys[k]) > .2, f'{k} bpod: {res_bpod[k]}, ephys: {res_ephys[k]}')


class TestEphysTrialsFPGA(base.IntegrationTest):

    def test_frame2ttl_flicker(self):
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
