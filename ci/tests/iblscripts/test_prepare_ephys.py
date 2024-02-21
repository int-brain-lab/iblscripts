import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

from ibllib.pipes.misc import create_ephyspc_params
from deploy.ephyspc.prepare_ephys_session import main_v8, main, _v8_check

if not _v8_check():
    raise unittest.SkipTest('iblrigv8 not installed')


class TestPrepareEphys(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.local = self.tmp / 'local_repo'
        self.remote = self.tmp / 'remote_repo'
        with patch('builtins.input', side_effect=(str(self.local), str(self.remote), '2', '3B', '3B')):
            create_ephyspc_params(force=True)

    def test_v8_main(self):
        main('fakemouse')  # Baseline
        expected = list(self.local.rglob('*/001/**'))
        [x.rmdir() for x in reversed(expected)]
        with patch('builtins.input', return_value=''):
            main_v8('fakemouse')
        actual = list(self.local.rglob('*/001/**'))
        self.assertEqual(len(expected), len(actual))
        # Check folders for main and main_v8 identical
        # x = [x.relative_to(self.local).as_posix().replace('/001', '/00?') for x in expected]
        # y = [x.relative_to(self.local).as_posix().replace('/002', '/00?') for x in actual]
        self.assertCountEqual(expected, actual)
        expected = ['fakemouse/2024-02-16/001/transfer_me.flag',
                    'fakemouse/2024-02-16/001/_ibl_experiment.description_ephys.yaml']
        actual = [x.relative_to(self.local).as_posix() for x in filter(Path.is_file, self.local.rglob('*'))]
        self.assertCountEqual(expected, actual)

        with patch('builtins.input', side_effect=['abort', '']):
            main_v8('fakemouse')
        actual = list(self.local.rglob('*/002/*'))
        self.assertEqual(1, len(actual))
        self.assertIsNotNone(next(self.remote.rglob('002/_devices/*.yaml'), None))

        with patch('builtins.input', side_effect=['abort', 'y']):
            main_v8('fakemouse')
        actual = list(self.local.rglob('*/003/*'))
        self.assertEqual(0, len(actual))
        self.assertIsNone(next(self.remote.rglob('003/_devices/*.yaml'), None))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
