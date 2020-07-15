import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ibllib.misc import version
from ibllib.pipes.training_preprocessing import TrainingTrials
import ibllib.io.raw_data_loaders as rawio
import alf.io

PATH_TESTS = Path('/mnt/s0/Data/IntegrationTests')
INIT_FOLDER = PATH_TESTS.joinpath('training')
TRIAL_KEYS_ge5 = ['goCue_times', 'probabilityLeft', 'intervals', 'stimOnTrigger_times',
                  'goCueTrigger_times', 'response_times', 'feedbackType', 'contrastLeft',
                  'feedback_times', 'rewardVolume', 'included', 'choice', 'contrastRight',
                  'stimOn_times', 'firstMovement_times']
TRIAL_KEYS_lt5 = ['goCue_times', 'probabilityLeft', 'intervals', 'itiDuration',
                  'goCueTrigger_times', 'response_times', 'feedbackType', 'contrastLeft',
                  'feedback_times', 'rewardVolume', 'choice', 'contrastRight', 'stimOn_times',
                  'firstMovement_times']
WHEEL_KEYS = ['position', 'timestamps']


class TestSessions(unittest.TestCase):

    def test_trials_extraction(self):
        if not INIT_FOLDER.exists():
            return
        # extract all sessions
        with tempfile.TemporaryDirectory() as tdir:
            subjects_path = Path(tdir).joinpath('Subjects')
            shutil.copytree(INIT_FOLDER, subjects_path)
            for fil in subjects_path.rglob('_iblrig_taskData.raw*.jsonable'):
                session_path = fil.parents[1]
                # task running part
                job = TrainingTrials(session_path)
                job.run()
                # load camera timestamps
                lc = alf.io.load_object(session_path / 'alf', '_ibl_leftCamera')
                self.assertTrue(np.all(np.diff(lc.times) > 0))
                # check the trials objects
                trials = alf.io.load_object(session_path / 'alf', '_ibl_trials')
                self.assertTrue(alf.io.check_dimensions(trials) == 0)
                settings = rawio.load_settings(session_path)
                if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
                    tkeys = TRIAL_KEYS_ge5
                else:
                    tkeys = TRIAL_KEYS_lt5
                self.assertTrue(set(trials.keys()) == set(tkeys))
                # check the wheel object if the extraction didn't fail
                if job.status != -1:
                    wheel = alf.io.load_object(session_path / 'alf', '_ibl_wheel')
                    self.assertTrue(alf.io.check_dimensions(wheel) == 0)
                    self.assertTrue(set(wheel.keys()) == set(WHEEL_KEYS))
                    self.assertTrue(np.all(np.diff(lc.times) > 0))
            """
            For this session only the downgoing front of a trial was detected, resulting in
             an error for the gocuetime. The fix was to extract the downgoing front and
             subtract 100ms.
            """
            session_path = subjects_path / "CSHL_007/2019-07-31/001"
            trials = alf.io.load_object(session_path / 'alf', '_ibl_trials')
            self.assertTrue(np.all(np.logical_not(np.isnan(trials.goCue_times))))
