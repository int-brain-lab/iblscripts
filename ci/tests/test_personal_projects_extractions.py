import logging
from pathlib import Path
import shutil

import numpy as np

import alf.io
from ibllib.pipes.ephys_preprocessing import EphysTrials
from ibllib.pipes.training_preprocessing import TrainingTrials
from oneibl.one import ONE

from ci.tests import base


_logger = logging.getLogger('ibllib')

TRAINING_TRIALS_SIGNATURE = ('_ibl_trials.feedbackType.npy',
                             '_ibl_trials.contrastLeft.npy',
                             '_ibl_trials.contrastRight.npy',
                             '_ibl_trials.probabilityLeft.npy',
                             '_ibl_trials.choice.npy',
                             '_ibl_trials.rewardVolume.npy',
                             '_ibl_trials.feedback_times.npy',
                             '_ibl_trials.intervals.npy',
                             '_ibl_trials.response_times.npy',
                             '_ibl_trials.goCueTrigger_times.npy',
                             '_ibl_trials.goCue_times.npy',
                             '_ibl_trials.stimOnTrigger_times.npy',
                             '_ibl_trials.included.npy',
                             '_ibl_trials.stimOn_times.npy',
                             '_ibl_wheel.timestamps.npy',
                             '_ibl_wheel.position.npy',
                             '_ibl_wheelMoves.intervals.npy',
                             '_ibl_wheelMoves.peakAmplitude.npy',
                             '_ibl_trials.firstMovement_times.npy')

EPHYS_TRIALS_SIGNATURE = ('_ibl_trials.feedbackType.npy',
                          '_ibl_trials.choice.npy',
                          '_ibl_trials.rewardVolume.npy',
                          '_ibl_trials.intervals_bpod.npy',
                          '_ibl_trials.intervals.npy',
                          '_ibl_trials.response_times.npy',
                          '_ibl_trials.goCueTrigger_times.npy',
                          '_ibl_trials.feedback_times.npy',
                          '_ibl_trials.goCue_times.npy',
                          '_ibl_trials.stimOff_times.npy',
                          '_ibl_trials.stimOn_times.npy',
                          '_ibl_trials.firstMovement_times.npy',
                          '_ibl_wheel.timestamps.npy',
                          '_ibl_wheel.position.npy',
                          '_ibl_wheelMoves.intervals.npy',
                          '_ibl_wheelMoves.peakAmplitude.npy',
                          '_ibl_trials.probabilityLeft.npy',
                          '_ibl_trials.contrastLeft.npy',
                          '_ibl_trials.contrastRight.npy')


class TestEphysTaskExtraction(base.IntegrationTest):

    def setUp(self) -> None:
        self.one_offline = ONE(offline=True)

    def test_ephys_biased_opto(self):
        """Guido's task"""
        desired_output = list(EPHYS_TRIALS_SIGNATURE) + ['_ibl_trials.laser_probability.npy', '_ibl_trials.laser_stimulation.npy']
        session_path = self.data_path.joinpath("personal_projects/ephys_biased_opto/ZFM-01802/2021-03-10/001")
        shutil.rmtree(session_path.joinpath('alf'), ignore_errors=True)
        _logger.info(f"{session_path}")
        task = EphysTrials(session_path, one=self.one_offline)
        task.run()
        assert task.status == 0
        assert set([p.name for p in task.outputs]) == set(desired_output)
        check_trials_and_clean_up(session_path)


class TestTrainingTaskExtraction(base.IntegrationTest):
    """
    The bpod jsonable can optionally contains 'laser_stimulation' and 'laser_probability' fields
    Sometimes only the former. THe normal biased extractor detects them automatically
    """

    def setUp(self) -> None:
        self.one_offline = ONE(offline=True)

    def test_biased_opto(self):
        desired_output = list(TRAINING_TRIALS_SIGNATURE) + ['_ibl_trials.laser_stimulation.npy']
        # this session has only laser stimulation labeled
        session_path = self.data_path.joinpath("personal_projects/biased_opto/ZFM-01804/2021-01-15/001")
        shutil.rmtree(session_path.joinpath('alf'), ignore_errors=True)
        task = TrainingTrials(session_path, one=self.one_offline)
        task.run()
        assert task.status == 0
        assert set([p.name for p in task.outputs]) == set(desired_output)
        trials = check_trials_and_clean_up(session_path)
        assert (np.all(np.unique(trials.laser_stimulation) == np.array([0, 1])))
        assert set([p.name for p in task.outputs]) == set(desired_output)

        # this session has both laser probability and laser stimulation fields labeled
        desired_output = list(TRAINING_TRIALS_SIGNATURE) + ['_ibl_trials.laser_probability.npy', '_ibl_trials.laser_stimulation.npy']
        session_path = self.data_path.joinpath("personal_projects/biased_opto/ZFM-01802/2021-02-08/001")
        shutil.rmtree(session_path.joinpath('alf'), ignore_errors=True)
        task = TrainingTrials(session_path, one=self.one_offline)
        task.run()
        assert task.status == 0
        assert set([p.name for p in task.outputs]) == set(desired_output)
        trials = check_trials_and_clean_up(session_path)
        assert(np.all(np.unique(trials.laser_stimulation) == np.array([0, 1])))
        assert(np.all(np.logical_and(trials.laser_probability >= 0, trials.laser_probability <= 1)))


def check_trials_and_clean_up(session_path):
    trials = alf.io.load_object(session_path.joinpath('alf'), 'trials')
    assert (alf.io.check_dimensions(trials) == 0)
    shutil.rmtree(session_path.joinpath('alf'), ignore_errors=True)
    return trials
