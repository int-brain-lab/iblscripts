"""Tests for ibllib.mpci.registration module."""
from unittest import mock
from pathlib import Path
import shutil

import numpy as np
from one.api import ONE
from iblatlas.atlas import MRITorontoAtlas

from ibllib.mpci.registration import MesoscopeFOVHistology
from ibllib.oneibl.data_handlers import ServerDataHandler

from ci.tests.base import IntegrationTest, TEST_DB


class TestUpdateCraniotomyCenter(IntegrationTest):
    """Tests for the update_craniotomy_center method."""

    required_files = ['mesoscope/SP037/2023-02-20/001']

    def setUp(self):
        self.session_path = self.data_path / self.required_files[0]
        ref_eid = '839bb5b1-120f-49d0-b7c9-5174c0c66b5a'
        self.task = MesoscopeFOVHistology(self.session_path, one=ONE(**TEST_DB), reference_session=ref_eid)
        self.task.atlas = MRITorontoAtlas(res_um=25)
        # A data handler is used for ensuring the reference image is present
        self.task.data_handler = ServerDataHandler(self.session_path, {'input_files': [], 'output_files': []}, one=self.task.one)
        with mock.patch.object(self.task.one, 'eid2path', return_value=self.task.session_path):
            self.referenceImage = self.task.load_reference_stack()
        # Backup the meta file and restore it at the end of the test
        meta_path = self.session_path / 'raw_imaging_data_00' / 'reference' / 'referenceImage.meta.json'
        shutil.copy(meta_path, meta_path.with_suffix('.json.bk'))
        self.addCleanup(shutil.move, meta_path.with_suffix('.json.bk'), meta_path)

    @mock.patch('ibllib.mpci.registration.json.dump')
    def test_update_craniotomy_center(self, mock_json_dump):
        """Test that the craniotomy center is updated correctly."""
        craniotomy_00 = {
            'center': [2.5, -2.3],
            'surface_normal_unit_vector': [0.31581724037833464, 0.05093826457715075, 0.947451721135004]
        }
        subject_json = {'json': {
            'history': {'cage': [{'value': 'None', 'date_time': '2022-09-06T11:10:54.464477+00:00'}]},
            'craniotomy_00': craniotomy_00}
        }

        with mock.patch.object(self.task.one.alyx, 'rest', return_value=subject_json) as rest_mock, \
                mock.patch.object(self.task.one.alyx, 'json_field_update') as put_mock:
            self.task.update_craniotomy_center(self.referenceImage)
        expected = {**craniotomy_00, 'center_resolved': [1.676, -2.397]}
        rest_mock.assert_called_once_with('subjects', 'read', id='SP037')
        put_mock.assert_called_once_with('subjects', 'SP037', data={'craniotomy_00': expected})
        mock_json_dump.assert_called_once()
        data, f = mock_json_dump.call_args[0]
        expected = 'raw_imaging_data_00/reference/referenceImage.meta.json'
        self.assertEqual(expected, Path(f.name).relative_to(self.session_path).as_posix())
        self.assertIn('AP_resolved', data['centerMM'])
        self.assertIn('ML_resolved', data['centerMM'])
        self.assertEqual(1.676472, data['centerMM']['ML_resolved'])
        self.assertEqual(-2.397074999999999, data['centerMM']['AP_resolved'])

    def test_get_brain_surface_plane_from_ref_points(self):
        """This tests that the output exactly matches Georg's original code for this session."""
        p_ref, n_ref, dv_avg = self.task.get_brain_surface_plane_from_ref_points(self.referenceImage)
        expected_p_ref = np.array([2866.472, -1056.775, -125.])
        expected_n_ref = np.array([7.72367e-04, 1.16962e-01, 9.93136e-01])
        expected_dv_avg = 150.0
        np.testing.assert_array_almost_equal(p_ref, expected_p_ref, decimal=5)
        np.testing.assert_array_almost_equal(n_ref, expected_n_ref, decimal=5)
        self.assertAlmostEqual(dv_avg, expected_dv_avg, delta=1e-2)
