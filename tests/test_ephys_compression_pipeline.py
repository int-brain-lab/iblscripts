import unittest
from pathlib import Path
import shutil
import logging

from ibllib.io import spikeglx
import ibllib.pipes.experimental_data as iblrig_pipeline

_logger = logging.getLogger('ibllib')


class TestEphysCompression(unittest.TestCase):

    def setUp(self):
        """
        replicate the full folder architecture with symlinks from compress_init to compress
        """
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/compression_init')
        if not self.init_folder.exists():
            return
        self.main_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/compression')
        if self.main_folder.exists():
            shutil.rmtree(self.main_folder)
        self.main_folder.mkdir(exist_ok=True)
        for ff in self.init_folder.rglob('*.*'):
            link = self.main_folder.joinpath(ff.relative_to(self.init_folder))
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)

    def test_compress_session(self):
        iblrig_pipeline.compress_ephys(self.main_folder)
        ephys_files = spikeglx.glob_ephys_files(self.main_folder)
        for ef in ephys_files:
            # there is only one compressed file afterwards
            self.assertTrue(ef.ap.suffix == '.cbin')
            self.assertFalse(ef.ap.with_suffix('.bin').exists())
            # the compress flag disappeared
            self.assertFalse(ef.ap.parent.joinpath('compress_ephys.flag').exists())
            # the compressed file is readable
            sr = spikeglx.Reader(ef.ap)
            self.assertTrue(sr.is_mtscomp)

    def tearDown(self):
        shutil.rmtree(self.main_folder)


if __name__ == "__main__":
    unittest.main(exit=False)
