import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np

from packaging import version
from ibllib.pipes.training_preprocessing import TrainingTrials
import ibllib.io.raw_data_loaders as rawio
from one.api import One
import one.alf.io as alfio

from ci.tests import base

TRIAL_KEYS_lt5 = ['goCue_times', 'probabilityLeft', 'intervals', 'goCueTrigger_times', 'quiescencePeriod',
                  'response_times', 'feedbackType', 'contrastLeft', 'feedback_times',
                  'rewardVolume', 'choice', 'contrastRight', 'stimOn_times', 'firstMovement_times']

TRIAL_KEYS_ge5 = TRIAL_KEYS_lt5 + ['stimOnTrigger_times', 'included']

WHEEL_KEYS = ['position', 'timestamps']


class TestHabituation(base.IntegrationTest):

    def test_legacy_habituation_session(self):
        session_path = self.data_path.joinpath('Subjects_init/ZM_1098/2019-01-25/001')
        job = TrainingTrials(session_path, one=One(mode='local'))
        status = job.run()
        self.assertEqual(0, status)
        self.assertIn('No extraction of legacy habituation sessions', job.log)


class TestSessions(base.IntegrationTest):

    def setUp(self):
        self.INIT_FOLDER = self.data_path.joinpath('training')
        if not self.INIT_FOLDER.exists():
            raise FileNotFoundError(f'Fixture {self.INIT_FOLDER.absolute()} does not exist')
        self.one = One(mode='local')

    def test_trials_extraction(self):
        # extract all sessions
        with tempfile.TemporaryDirectory() as tdir:
            subjects_path = Path(tdir).joinpath('Subjects')
            shutil.copytree(self.INIT_FOLDER, subjects_path)
            for fil in subjects_path.rglob('_iblrig_taskData.raw*.jsonable'):
                # read task settings and determine iblrig version to throw into subtests
                session_path = fil.parents[1]
                settings = rawio.load_settings(session_path)
                iblrig_version = version.parse(settings['IBLRIG_VERSION'])
                with self.subTest(file=fil, iblrig_version=iblrig_version):
                    # task running part
                    job = TrainingTrials(session_path, one=self.one)
                    job.run()
                    # check the trials objects
                    trials = alfio.load_object(session_path / 'alf', 'trials')
                    self.assertTrue(alfio.check_dimensions(trials) == 0)
                    if iblrig_version >= version.parse('5.0.0'):
                        tkeys = TRIAL_KEYS_ge5
                    else:
                        tkeys = TRIAL_KEYS_lt5
                    self.assertEqual(set(trials.keys()), set(tkeys))
                    # check the wheel object if the extraction didn't fail
                    if job.status != -1:
                        wheel = alfio.load_object(session_path / 'alf', 'wheel')
                        self.assertTrue(alfio.check_dimensions(wheel) == 0)
                        self.assertEqual(set(wheel.keys()), set(WHEEL_KEYS))
            """
            For this session only the downgoing front of a trial was detected, resulting in
             an error for the gocuetime. The fix was to extract the downgoing front and
             subtract 100ms.
            """
            session_path = subjects_path / 'CSHL_007/2019-07-31/001'
            trials = alfio.load_object(session_path / 'alf', 'trials')
            self.assertTrue(np.all(np.logical_not(np.isnan(trials.goCue_times))))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
