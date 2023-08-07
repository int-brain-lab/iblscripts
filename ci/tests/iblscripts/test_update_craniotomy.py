import unittest

from one.webclient import AlyxClient

from deploy.mesoscope.update_craniotomy import update_craniotomy_coordinates
from ci.tests.base import IntegrationTest, TEST_DB


class TestUpdateCraniotomy(IntegrationTest):
    """Tests for iblscripts/deploy/mesoscope/update_craniotomy.py"""

    def setUp(self) -> None:
        self.alyx = AlyxClient(**TEST_DB)
        self.subject = 'algernon'
        # UUID of surgery that should be updated
        self.uuid = 'd46ceb42-ab57-4a5c-967e-feaf07a7d991'
        self.alyx.json_field_delete('surgeries', self.uuid, 'json')

    def test_update_craniotomy_coordinates(self):
        """Test update_craniotomy_coordinates function."""
        record = update_craniotomy_coordinates(self.subject, 2.7, 1, alyx=self.alyx)
        expected = {'craniotomy_00': {'center': [2.7, 1.]}}
        self.assertEqual(record['id'], self.uuid)
        self.assertEqual(expected, record['json'])

    def tearDown(self) -> None:
        self.alyx.json_field_delete('surgeries', self.uuid, 'json')


if __name__ == '__main__':
    unittest.main()
