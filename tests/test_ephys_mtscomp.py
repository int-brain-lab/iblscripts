import unittest
from pathlib import Path
import shutil
import logging

from ibllib.io import spikeglx
from ibllib.pipes.ephys_preprocessing import EphysMtscomp

_logger = logging.getLogger('ibllib')

TEST_PATH = Path('/mnt/s0/Data/IntegrationTests')


class TestMtsCompRegistration(unittest.TestCase):
    """Makes sure the ch files are registered properly"""

    def test_single_run(self):
        SESSION_PATH = TEST_PATH.joinpath("ephys/choice_world/KS022/2019-12-10/001")
        task = EphysMtscomp(SESSION_PATH)
        task.run()
        self.assertTrue(sum(map(lambda x: x.suffix == '.cbin', task.outputs)) == 5)
        self.assertTrue(sum(map(lambda x: x.suffix == '.ch', task.outputs)) == 5)


class TestEphysCompression(unittest.TestCase):

    def setUp(self):
        """
        replicate the full folder architecture with symlinks from compress_init to compress
        """
        self.init_folder = TEST_PATH.joinpath('ephys', 'compression_init')
        if not self.init_folder.exists():
            return
        self.main_folder = TEST_PATH.joinpath('ephys', 'compression')
        if self.main_folder.exists():
            shutil.rmtree(self.main_folder)
        self.main_folder.mkdir(exist_ok=True)
        for ff in self.init_folder.rglob('*.*'):
            link = self.main_folder.joinpath(ff.relative_to(self.init_folder))
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)

    def test_compress_session(self):
        EphysMtscomp(self.main_folder).run()
        ephys_files = spikeglx.glob_ephys_files(self.main_folder)
        for ef in ephys_files:
            # there is only one compressed file afterwards
            self.assertTrue(ef.ap.suffix == '.cbin')
            self.assertFalse(ef.ap.with_suffix('.bin').exists())
            # the compressed file is readable
            sr = spikeglx.Reader(ef.ap)
            self.assertTrue(sr.is_mtscomp)

    def tearDown(self):
        shutil.rmtree(self.main_folder)


if __name__ == "__main__":
    unittest.main(exit=False)
