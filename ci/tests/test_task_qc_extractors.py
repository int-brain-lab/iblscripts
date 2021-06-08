"""Tests TaskQC object and Bpod extractors
NB: FPGA TaskQC extractor is tested in test_ephys_extraction_choiceWorld
"""
import unittest
import shutil
from pathlib import Path
import tempfile

import numpy as np

from ibllib.misc import version
from ibllib.qc.task_metrics import TaskQC
from ibllib.qc.task_extractors import TaskQCExtractor
from one.api import ONE
from ci.tests import base

one = ONE(
    base_url="https://test.alyx.internationalbrainlab.org",
    username="test_user",
    password="TapetesBloc18",
)


class TestTaskQCObject(base.IntegrationTest):
    def setUp(self):
        self.one = one
        self.eid = "b1c968ad-4874-468d-b2e4-5ffa9b9964e9"
        # Make sure the data exists locally
        self.session_path = self.one.eid2path(self.eid)
        self.qc = TaskQC(self.eid, one=one)
        self.qc.load_data(bpod_only=True)  # Test session has no raw FPGA data

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
        # NOT_SET
        not_set = [k for k, v in results.items() if np.isnan(v)]
        self.assertTrue(all(outcomes[k] == 'NOT_SET' for k in not_set))
        # PASS
        passed = [k for k, v in results.items() if v >= self.qc.criteria['PASS']]
        self.assertTrue(all(outcomes[k] == 'PASS' for k in passed))
        # WARNING
        wrn = [k for k, v in results.items()
               if self.qc.criteria['WARNING'] <= v <= self.qc.criteria['PASS']]
        self.assertTrue(all(outcomes[k] == 'WARNING' for k in wrn))
        # FAIL
        fail = [k for k, v in results.items() if v <= self.qc.criteria['FAIL']
                and k not in TaskQC.fcns_value2status]
        self.assertTrue(all(outcomes[k] == 'FAIL' for k in fail))


class TestBpodQCExtractors(base.IntegrationTest):

    def setUp(self):
        self.one = one
        # TODO: this is an old 4.3 iblrig session below, add a session ge 5.0.0
        self.eid = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
        self.eid_incomplete = '4e0b3320-47b7-416e-b842-c34dc9004cf8'  # Missing required datasets
        # Make sure the data exists locally
        self.session_path = self.one.eid2path(self.eid)

    def test_lazy_extract(self):
        ex = TaskQCExtractor(self.session_path, lazy=True, one=self.one, download_data=True)
        self.assertIsNone(ex.data)

    def test_extraction(self):
        ex = TaskQCExtractor(self.session_path,
                             lazy=True, one=self.one, bpod_only=True, download_data=True)
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
                             lazy=True, one=self.one, bpod_only=True, download_data=True)
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
        if version.ge(ex.settings['IBLRIG_VERSION_TAG'], '5.0.0'):
            expected += expected_5up
        self.assertTrue(set(expected).issubset(set(ex.data.keys())))

    def test_download_data(self):
        """Test behavior when download_data flag is True
        """
        path = one.eid2path(self.eid_incomplete)
        ex = TaskQCExtractor(path, lazy=True, one=self.one, download_data=True)
        self.assertTrue(ex.lazy, 'Failed to set lazy flag')

        # Swap cache dir for temporary directory.  This should trigger re-download of the data
        # without interfering with the integration data
        with tempfile.TemporaryDirectory() as tdir:
            _cache = self.one._cache_dir
            self.one.alyx._par = self.one.alyx._par.set('CACHE_DIR', tdir)
            try:
                TaskQCExtractor(self.session_path, lazy=True, one=self.one, download_data=True)
                files = list(self.session_path.rglob('*.*'))
                expected = 6  # NB This session is missing raw ephys data and missing some datasets
                self.assertEqual(len(files), expected)
            finally:
                self.one.alyx._par = self.one.alyx._par.set('CACHE_DIR', _cache)


if __name__ == "__main__":
    unittest.main(exit=False)
