import unittest

from one.webclient import AlyxClient

from deploy.mesoscope.update_craniotomy import update_craniotomy_coordinates
from ci.tests.base import IntegrationTest, TEST_DB


class TestUpdateCraniotomy(IntegrationTest):
    """Tests for iblscripts/deploy/mesoscope/update_craniotomy.py"""

    def setUp(self) -> None:
        self.alyx = AlyxClient(**TEST_DB)
        self.subject = 'algernon'
        self.surgeries = [
            self.alyx.rest('surgeries', 'create', data={'subject': self.subject})
        ]

    def test_update_craniotomy_coordinates(self):
        """Test update_craniotomy_coordinates function."""
        record = update_craniotomy_coordinates(self.subject, 2.7, 1, alyx=self.alyx)
        expected = {'craniotomy_00': [2.7, 1.]}
        self.assertEqual(expected, record)

    def tearDown(self) -> None:
        for s in self.surgeries or []:
            self.alyx.rest('surgeries', 'delete', id=s['id'])


if __name__ == '__main__':
    unittest.main()
