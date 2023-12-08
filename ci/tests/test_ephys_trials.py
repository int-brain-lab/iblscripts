"""Test NI DAQ trials extraction."""
import numpy as np
import numpy.testing
import one.alf.io as alfio
from one.alf.files import get_session_path
from one.api import ONE

from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsNidq, HabituationTrialsNidq

from ci.tests import base


class TestEphysTaskExtraction(base.IntegrationTest):
    """Test the FpgaTrials extractor."""

    alf_files = ['_ibl_trials.table.pqt', '_ibl_trials.goCueTrigger_times.npy', '_ibl_trials.quiescencePeriod.npy']
    """A subset of expected output datasets."""

    trials_task = ChoiceWorldTrialsNidq
    """The task to use for trials extraction (may depend on task protocol)."""

    # Expect 'alf', 'raw_ephys_data', 'raw_behavior_data' collections in each.
    required_files = [
        'ephys/ephys_choice_world_task/CSP004/2019-11-27/001/*',  # normal session
        'ephys/ephys_choice_world_task/ibl_witten_13/2019-11-25/001/*',  # FPGA stops before bpod, custom sync
        # 'ephys/ephys_choice_world_task/ibl_witten_27/2021-01-21/001/*'  # frame2ttl flicker
    ]

    # def test_task_extraction_output(self):
    # # TODO This was removed because it appears to be a redundant test;
    #     should check if alf folder still required or can be deleted.
    #     init_folder = self.data_path.joinpath('ephys', 'choice_world_init')
    #     self.sessions = [f.parent for f in init_folder.rglob('raw_ephys_data')]
    #     for session_path in self.sessions:
    #         self._task_extraction_assertions(session_path)

    def test_task_extraction(self):
        """Test ephysChoiceWorld task extraction with NI DAQ.

        Test each session in `required_files` list.
        """
        folders = (next(f, None) for f in map(self.data_path.glob, self.required_files))
        self.sessions = set(filter(None, map(get_session_path, folders)))
        for session_path in self.sessions:
            with self.subTest(msg=session_path):
                self._task_extraction_assertions(session_path)

    def _task_extraction_assertions(self, session_path, trials_task=None):
        """Compare task extraction with expected ALF trials output."""
        self.backup_alf(session_path)
        alf_path = session_path.joinpath('alf')
        bk_path = alf_path.parent / 'alf.bk'

        # Run extractor
        TrialsTask = trials_task or self.trials_task
        task = TrialsTask(session_path,
                          one=ONE(mode='local'), collection='raw_behavior_data',
                          sync_collection='raw_ephys_data')
        fpga_trials, _ = task._extract_behaviour(save=True)
        tqc_ephys = task._run_qc(fpga_trials.copy(), update=False, plot_qc=False)

        # check that the output is complete
        for f in self.alf_files:
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
        # check some task-specific trial events
        self._check_task_trial_events(fpga_trials)

        # compute the task qc
        _, res_ephys = tqc_ephys.run(bpod_only=False, download_data=False)

        TaskQC = type(tqc_ephys)  # Use the same TaskQC class as the task for the Bpod only QC
        tqc_bpod = TaskQC(session_path, one=ONE(mode='local'))
        tqc_bpod.extractor = TaskQCExtractor(session_path, lazy=True, one=None, bpod_only=True)
        tqc_bpod.extractor.settings = task.extractor.settings
        tqc_bpod.extractor.data = tqc_bpod.extractor.rename_data(task.extractor.bpod_trials.copy())
        tqc_bpod.extractor.frame_ttls = task.extractor.bpod_extractor.frame2ttl  # used in iblapps QC viewer
        tqc_bpod.extractor.audio_ttls = task.extractor.bpod_extractor.audio  # used in iblapps QC viewer

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

    def _check_task_trial_events(self, trials):
        """Check task-specific trial events."""
        # check that the stimOn < stimFreeze < stimOff
        self.assertTrue(np.less(trials['table']['stimOn_times'], trials['stimOff_times']).all())
        self.assertTrue(np.less(trials['stimFreeze_times'], trials['stimOff_times']).all())
        # a trial is either an error-nogo or a reward
        self.assertTrue(np.all(np.isnan(trials['valveOpen_times'] * trials['errorCue_times'])))
        self.assertTrue(np.all(np.logical_xor(np.isnan(trials['valveOpen_times']),
                                              np.isnan(trials['errorCue_times']))))


class TestEphysHabituationTaskExtraction(TestEphysTaskExtraction):
    """Test the FpgaTrialsHabituation extractor."""

    # Expect 'alf', 'raw_ephys_data', 'raw_behavior_data' collections in each.
    required_files = [
        'ephys/habituation_choice_world_task/MM015/2023-10-05/002/*',  # normal session
    ]

    trials_task = HabituationTrialsNidq

    alf_files = ['_ibl_trials.intervals.npy',
                 '_ibl_trials.stimCenter_times.npy',
                 '_ibl_trials.goCue_times.npy',
                 '_ibl_trials.feedback_times.npy',
                 '_ibl_trials.rewardVolume.npy']
    """A subset of expected output datasets."""

    def _check_task_trial_events(self, trials):
        """Check habituationChoiceWorld specific trial events.

        Check that the stimOn < stimCenter < stimOff.
        Note that we don't test the first trial as the stimulus on the first trial is messed up!
        """
        self.assertTrue(
            np.less(trials['stimOn_times'][1:], trials['stimOff_times'][1:]).all())
        self.assertTrue(
            np.less(trials['stimCenter_times'][1:], trials['stimOff_times'][1:]).all())


class TestEphysTrialsFPGA(base.IntegrationTest):

    required_files = ['ephys/ephys_choice_world_task/ibl_witten_27/2021-01-21/001/*']

    def test_frame2ttl_flicker(self):
        session_path = get_session_path(next(self.data_path.glob(self.required_files[0])))
        # dsets, out_files = ephys_fpga.extract_all(session_path, save=True)
        task = ChoiceWorldTrialsNidq(session_path,
                                     one=ONE(mode='local'), collection='raw_behavior_data',
                                     sync_collection='raw_ephys_data')
        fpga_trials, _ = task._extract_behaviour(save=True)
        # Run the task QC
        qc = task._run_qc(fpga_trials, update=False, plot_qc=False)
        # Aggregate and update Alyx QC fields
        _, myqc, _ = qc.compute_session_status()

        # from ibllib.misc import pprint
        # pprint(myqc)
        assert myqc['_task_stimOn_delays'] > 0.9  # 0.6176
        assert myqc['_task_stimFreeze_delays'] > 0.9
        assert myqc['_task_stimOff_delays'] > 0.9
        assert myqc['_task_stimOff_itiIn_delays'] > 0.9
        assert myqc['_task_stimOn_delays'] > 0.9
        assert myqc['_task_stimOn_goCue_delays'] > 0.9
