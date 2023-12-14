"""Tests TaskQC object and Bpod extractors.

NB: FPGA TaskQC extractor is tested in test_ephys_extraction_choiceWorld.

This module uses ZM_1150/2019-05-07/001 ('b1c968ad-4874-468d-b2e4-5ffa9b9964e9').
"""
import unittest
import unittest.mock
import tempfile

from packaging import version
from ibllib.qc.task_metrics import TaskQC
from ibllib.qc.task_extractors import TaskQCExtractor
from one.api import ONE
from ci.tests import base

one = ONE(**base.TEST_DB)


class TestTaskQCObject(base.IntegrationTest):
    def setUp(self):
        self.one = one
        self.eid = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
        # Make sure the data exists locally
        self.session_path = self.data_path.joinpath('training', 'ZM_1150', '2019-05-07', '001')
        self.qc = TaskQC(self.session_path, one=one)
        self.qc.load_data(bpod_only=True, download_data=False)  # Test session has no raw FPGA data

    def test_compute(self):
        # Compute metrics
        self.assertTrue(self.qc.metrics is None)
        self.assertTrue(self.qc.passed is None)
        self.qc.compute()
        self.assertTrue(self.qc.metrics is not None)
        self.assertTrue(self.qc.passed is not None)

    def test_run(self):
        # Reset Alyx fields before test
        assert 'test' in self.qc.one.alyx.base_url
        reset = self.qc.update('NOT_SET', override=True)
        assert reset == 'NOT_SET', 'failed to reset QC field for test'
        extended = self.one.alyx.json_field_write('sessions', field_name='extended_qc',
                                                  uuid=self.eid, data={})
        assert not extended, 'failed to reset extended QC field for test'

        # Test update as False
        outcome, _ = self.qc.run(update=False)
        self.assertEqual('FAIL', outcome)
        extended = self.one.alyx.rest('sessions', 'read',
                                      id=self.eid, no_cache=True)['extended_qc']
        self.assertDictEqual({}, extended, 'unexpected update to extended qc')
        outcome = self.one.alyx.rest('sessions', 'read', id=self.eid, no_cache=True)['qc']
        self.assertEqual('NOT_SET', outcome, 'unexpected update to qc')

        # Test update as True
        outcome, results = self.qc.run(update=True)
        self.assertEqual('FAIL', outcome)
        extended = self.one.alyx.rest('sessions', 'read',
                                      id=self.eid, no_cache=True)['extended_qc']
        expected = list(results.keys()) + ['task']
        self.assertCountEqual(expected, extended.keys(), 'unexpected update to extended qc')
        qc_field = self.one.alyx.rest('sessions', 'read', id=self.eid, no_cache=True)['qc']
        self.assertEqual(outcome, qc_field, 'unexpected update to qc')

    def test_compute_session_status(self):
        with self.assertRaises(AttributeError):
            self.qc.compute_session_status()
        self.qc.compute()
        outcome, results, outcomes = self.qc.compute_session_status()
        self.assertEqual('FAIL', outcome)

        # Check each outcome matches...
        expected_outcomes = {
            '_task_audio_pre_trial': 'PASS',
            '_task_correct_trial_event_sequence': 'FAIL',
            '_task_detected_wheel_moves': 'WARNING',
            '_task_errorCue_delays': 'WARNING',
            '_task_error_trial_event_sequence': 'FAIL',
            '_task_goCue_delays': 'WARNING',
            '_task_iti_delays': 'NOT_SET',
            '_task_n_trial_events': 'FAIL',
            '_task_negative_feedback_stimOff_delays': 'WARNING',
            '_task_positive_feedback_stimOff_delays': 'WARNING',
            '_task_response_feedback_delays': 'FAIL',
            '_task_response_stimFreeze_delays': 'WARNING',
            '_task_reward_volume_set': 'PASS',
            '_task_reward_volumes': 'PASS',
            '_task_stimFreeze_delays': 'WARNING',
            '_task_stimOff_delays': 'WARNING',
            '_task_stimOff_itiIn_delays': 'WARNING',
            '_task_stimOn_delays': 'WARNING',
            '_task_stimOn_goCue_delays': 'FAIL',
            '_task_stimulus_move_before_goCue': 'NOT_SET',
            '_task_trial_length': 'WARNING',
            '_task_wheel_freeze_during_quiescence': 'PASS',
            '_task_wheel_integrity': 'PASS',
            '_task_wheel_move_before_feedback': 'PASS',
            '_task_wheel_move_during_closed_loop': 'PASS',
            '_task_wheel_move_during_closed_loop_bpod': 'PASS',
            '_task_passed_trial_checks': 'NOT_SET'
        }
        for k in outcomes:
            with self.subTest(check=k[6:].replace('_', ' ')):
                self.assertEqual(outcomes[k], expected_outcomes[k], f'{k} should be {expected_outcomes[k]}')


class TestBpodQCExtractors(base.IntegrationTest):

    def setUp(self):
        self.one = one
        # TODO: this is an old 4.3 iblrig session below, add a session ge 5.0.0
        self.eid = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
        self.eid_incomplete = '4e0b3320-47b7-416e-b842-c34dc9004cf8'  # Missing required datasets
        # Make sure the data exists locally
        self.session_path = self.data_path.joinpath('training', 'ZM_1150', '2019-05-07', '001')

    def test_lazy_extract(self):
        ex = TaskQCExtractor(self.session_path, lazy=True, one=self.one, download_data=False)
        self.assertIsNone(ex.data)

    def test_extraction(self):
        ex = TaskQCExtractor(self.session_path,
                             lazy=True, one=self.one, bpod_only=True, download_data=False)
        self.assertIsNone(ex.raw_data)

        # Test loading raw data
        ex.load_raw_data()
        self.assertIsNotNone(ex.raw_data, 'Failed to load raw data')
        self.assertIsNotNone(ex.settings, 'Failed to load task settings')
        self.assertIsNotNone(ex.frame_ttls, 'Failed to load BNC1')
        self.assertIsNotNone(ex.audio_ttls, 'Failed to load BNC2')

        # Test extraction
        ex.extract_data()
        expected = ['choice', 'contrastLeft', 'contrastRight', 'correct', 'errorCue_times',
                    'feedbackType', 'feedback_times', 'firstMovement_times', 'goCueTrigger_times',
                    'goCue_times', 'intervals', 'phase', 'position', 'probabilityLeft',
                    'quiescence', 'response_times', 'rewardVolume', 'stimOn_times',
                    'valveOpen_times', 'wheel_moves_intervals', 'wheel_moves_peak_amplitude',
                    'wheel_position', 'wheel_timestamps']
        expected_5up = ['errorCueTrigger_times', 'itiIn_times',
                        'stimFreezeTrigger_times', 'stimFreeze_times', 'stimOffTrigger_times',
                        'stimOff_times', 'stimOnTrigger_times']
        expected += expected_5up

        self.assertTrue(len(set(expected).difference(set(ex.data.keys()))) == 0)
        self.assertEqual('ephys', ex.type)
        self.assertEqual('X1', ex.wheel_encoding)

    def test_partial_extraction(self):
        ex = TaskQCExtractor(self.session_path,
                             lazy=True, one=self.one, bpod_only=True, download_data=False)
        ex.extract_data()

        expected = ['contrastLeft',
                    'contrastRight',
                    'phase',
                    'position',
                    'probabilityLeft',
                    'quiescence',
                    'stimOn_times']
        expected_5up = ['contrast',
                        'errorCueTrigger_times',
                        'itiIn_times',
                        'stimFreezeTrigger_times',
                        'stimFreeze_times',
                        'stimOffTrigger_times',
                        'stimOff_times',
                        'stimOnTrigger_times']
        if version.parse(ex.settings['IBLRIG_VERSION']) >= version.parse('5.0.0'):
            expected += expected_5up
        self.assertTrue(set(expected).issubset(set(ex.data.keys())))

    def test_download_data(self):
        """Test behavior when download_data flag is True."""
        path = one.eid2path(self.eid_incomplete)
        ex = TaskQCExtractor(path, lazy=True, one=self.one, download_data=True)
        self.assertTrue(ex.lazy, 'Failed to set lazy flag')

        # Swap cache dir for temporary directory.  This should trigger re-download of the data
        # without interfering with the integration data
        with tempfile.TemporaryDirectory() as tdir:
            _cache = self.one.cache_dir
            self.one.alyx._par = self.one.alyx._par.set('CACHE_DIR', tdir)
            try:
                with unittest.mock.patch.object(self.one, '_download_datasets') as download_method:
                    TaskQCExtractor(self.session_path, lazy=True, one=self.one, download_data=True)
                    download_method.assert_called()
            finally:
                self.one.alyx._par = self.one.alyx._par.set('CACHE_DIR', _cache)


if __name__ == '__main__':
    unittest.main(exit=False)
