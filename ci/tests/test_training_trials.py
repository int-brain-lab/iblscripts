import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ibllib.misc import version
from ibllib.pipes.training_preprocessing import TrainingTrials
import ibllib.io.raw_data_loaders as rawio
from ibllib.io.extractors import bpod_trials
from oneibl.one import OneOffline, ONE
import alf.io

from ci.tests import base

TRIAL_KEYS_ge5 = ['goCue_times', 'probabilityLeft', 'intervals', 'stimOnTrigger_times',
                  'goCueTrigger_times', 'response_times', 'feedbackType', 'contrastLeft',
                  'feedback_times', 'rewardVolume', 'included', 'choice', 'contrastRight',
                  'stimOn_times', 'firstMovement_times']
TRIAL_KEYS_lt5 = ['goCue_times', 'probabilityLeft', 'intervals', 'itiDuration',
                  'goCueTrigger_times', 'response_times', 'feedbackType', 'contrastLeft',
                  'feedback_times', 'rewardVolume', 'choice', 'contrastRight', 'stimOn_times',
                  'firstMovement_times']
WHEEL_KEYS = ['position', 'timestamps']


class TestLaserBpod(base.IntegrationTest):
    """
    The bpod jsonable can optionally contains 'laser_stimulation' and 'laser_probability' fields
    Sometimes only the former. THe normal biased extractor detects them automatically
    """
    def test_single_session(self):
        # this session has both laser probability and laser stimulation fields labeled
        session_path = self.data_path.joinpath("Subjects_init/ZFM-01802/2021-02-08/001")
        _, _, _ = bpod_trials.extract_all(session_path, save=True)
        trials = alf.io.load_object(session_path.joinpath('alf'), 'trials')
        assert(alf.io.check_dimensions(trials) == 0)
        assert(np.all(np.unique(trials.laser_stimulation) == np.array([0, 1])))
        assert(np.all(np.logical_and(trials.laser_probability >= 0, trials.laser_probability <= 1)))
        # this session has only laser stimulation labeled
        session_path = self.data_path.joinpath("Subjects_init/ZFM-01804/2021-01-15/001")
        _, _, _ = bpod_trials.extract_all(session_path, save=True)
        trials = alf.io.load_object(session_path.joinpath('alf'), 'trials')
        assert(alf.io.check_dimensions(trials) == 0)
        assert(np.all(np.unique(trials.laser_stimulation) == np.array([0, 1])))
        assert('laser_probability' not in trials)


class TestHabituation(base.IntegrationTest):

    def test_legacy_habituation_session(self):
        session_path = self.data_path.joinpath("Subjects_init/ZM_1098/2019-01-25/001")
        job = TrainingTrials(session_path)
        status = job.run()
        assert status == 0
        assert "No extraction of legacy habituation sessions" in job.log


class TestSessions(base.IntegrationTest):

    def setUp(self):
        self.INIT_FOLDER = self.data_path.joinpath('training')
        if not self.INIT_FOLDER.exists():
            raise FileNotFoundError(f'Fixture {self.INIT_FOLDER.absolute()} does not exist')
        self.one = OneOffline()

    def test_trials_extraction(self):
        # extract all sessions
        with tempfile.TemporaryDirectory() as tdir:
            subjects_path = Path(tdir).joinpath('Subjects')
            shutil.copytree(self.INIT_FOLDER, subjects_path)
            for fil in subjects_path.rglob('_iblrig_taskData.raw*.jsonable'):
                session_path = fil.parents[1]
                # task running part
                job = TrainingTrials(session_path, one=self.one)
                job.run()
                # load camera timestamps
                lc = alf.io.load_object(session_path / 'alf', 'leftCamera')
                self.assertTrue(np.all(np.diff(lc.times) > 0))
                # check the trials objects
                trials = alf.io.load_object(session_path / 'alf', 'trials')
                self.assertTrue(alf.io.check_dimensions(trials) == 0)
                settings = rawio.load_settings(session_path)
                if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
                    tkeys = TRIAL_KEYS_ge5
                else:
                    tkeys = TRIAL_KEYS_lt5
                self.assertTrue(set(trials.keys()) == set(tkeys))
                # check the wheel object if the extraction didn't fail
                if job.status != -1:
                    wheel = alf.io.load_object(session_path / 'alf', 'wheel')
                    self.assertTrue(alf.io.check_dimensions(wheel) == 0)
                    self.assertTrue(set(wheel.keys()) == set(WHEEL_KEYS))
                    self.assertTrue(np.all(np.diff(lc.times) > 0))
            """
            For this session only the downgoing front of a trial was detected, resulting in
             an error for the gocuetime. The fix was to extract the downgoing front and
             subtract 100ms.
            """
            session_path = subjects_path / "CSHL_007/2019-07-31/001"
            trials = alf.io.load_object(session_path / 'alf', 'trials')
            self.assertTrue(np.all(np.logical_not(np.isnan(trials.goCue_times))))


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
