import shutil

import numpy as np
import one.alf.io as alfio
from one.api import ONE

from ibllib.pipes.ephys_preprocessing import LaserTrialsLegacy
from ibllib.pipes.training_preprocessing import TrainingTrials

from ci.tests import base


TRAINING_TRIALS_SIGNATURE = ('_ibl_trials.goCueTrigger_times.npy',
                             '_ibl_trials.included.npy',
                             '_ibl_trials.laserStimulation.npy',
                             '_ibl_trials.stimOnTrigger_times.npy',
                             '_ibl_trials.quiescencePeriod.npy',
                             '_ibl_trials.table.pqt',
                             '_ibl_wheel.position.npy',
                             '_ibl_wheel.timestamps.npy',
                             '_ibl_wheelMoves.intervals.npy',
                             '_ibl_wheelMoves.peakAmplitude.npy')

EPHYS_TRIALS_SIGNATURE = ('_ibl_trials.intervals_bpod.npy',
                          '_ibl_trials.goCueTrigger_times.npy',
                          '_ibl_trials.stimOff_times.npy',
                          '_ibl_trials.quiescencePeriod.npy',
                          '_ibl_wheel.timestamps.npy',
                          '_ibl_wheel.position.npy',
                          '_ibl_wheelMoves.intervals.npy',
                          '_ibl_wheelMoves.peakAmplitude.npy',
                          '_ibl_trials.table.pqt')


class TestEphysTaskExtraction(base.IntegrationTest):

    def setUp(self) -> None:
        self.one_offline = ONE(mode='local')
        self.session_path = self.data_path.joinpath('personal_projects/ephys_biased_opto/ZFM-01802/2021-03-10/001')

    def test_ephys_biased_opto(self):
        """Guido's task

        NB: This way of extracting personal projects is deprecated. Instead, a new extractor task
        should be defined in the experiment.description file, which has its own _extract_behavior method.
        The EphysTrials
        """
        desired_output = list(EPHYS_TRIALS_SIGNATURE) + ['_ibl_trials.laserProbability.npy', '_ibl_trials.laserStimulation.npy']
        shutil.move(self.session_path.joinpath('alf'), self.session_path.joinpath('alf.bk'))
        task = LaserTrialsLegacy(self.session_path, one=self.one_offline)
        task.run()
        self.assertEqual(0, task.status)
        self.assertCountEqual([p.name for p in task.outputs], desired_output)
        check_trials(self.session_path)

    def tearDown(self) -> None:
        cleanup(self.session_path)


class TestTrainingTaskExtraction(base.IntegrationTest):
    """
    The bpod jsonable can optionally contains 'laserStimulation' and 'laserProbability' fields
    Sometimes only the former. THe normal biased extractor detects them automatically
    """
    session_path = None

    def setUp(self) -> None:
        self.one_offline = ONE(mode='local')

    def test_biased_opto_stim(self):
        # this session has only laser stimulation labeled
        self.session_path = self.data_path.joinpath('personal_projects/biased_opto/ZFM-01804/2021-01-15/001')
        desired_output = list(TRAINING_TRIALS_SIGNATURE) + ['_ibl_trials.laserStimulation.npy']
        if self.session_path.joinpath('alf').exists():
            shutil.move(self.session_path.joinpath('alf'), self.session_path.joinpath('alf.bk'))
        task = TrainingTrials(self.session_path, one=self.one_offline)
        task.run()
        self.assertEqual(0, task.status)
        self.assertEqual(set(p.name for p in task.outputs), set(desired_output))
        trials = check_trials(self.session_path)
        self.assertTrue(np.all(np.unique(trials.laserStimulation) == np.array([0, 1])))
        self.assertEqual(set(p.name for p in task.outputs), set(desired_output))
        cleanup(self.session_path)

    def test_biased_opto_prob(self):
        # this session has both laser probability and laser stimulation fields labeled
        extra_outputs = ['_ibl_trials.laserProbability.npy', '_ibl_trials.laserStimulation.npy']
        desired_output = list(TRAINING_TRIALS_SIGNATURE) + extra_outputs

        self.session_path = self.data_path.joinpath('personal_projects/biased_opto/ZFM-01802/2021-02-08/001')
        if self.session_path.joinpath('alf').exists():
            shutil.move(self.session_path.joinpath('alf'), self.session_path.joinpath('alf.bk'))
        task = TrainingTrials(self.session_path, one=self.one_offline)
        task.run()
        self.assertEqual(0, task.status)
        self.assertEqual(set(p.name for p in task.outputs), set(desired_output))
        trials = check_trials(self.session_path)
        self.assertTrue(np.all(np.unique(trials.laserStimulation) == np.array([0, 1])))
        self.assertTrue(
            np.all(np.logical_and(trials.laserProbability >= 0, trials.laserProbability <= 1))
        )

    def tearDown(self) -> None:
        cleanup(self.session_path)


def check_trials(session_path):
    trials = alfio.load_object(session_path.joinpath('alf'), 'trials')
    assert (alfio.check_dimensions(trials) == 0)
    return trials


def cleanup(session_path):
    shutil.rmtree(session_path.joinpath('alf'), ignore_errors=True)
    if session_path.joinpath('alf.bk').exists():
        shutil.move(session_path.joinpath('alf.bk'), session_path.joinpath('alf'))
