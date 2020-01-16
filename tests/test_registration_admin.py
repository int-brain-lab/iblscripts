import unittest
import uuid
import tempfile
from pathlib import Path

import numpy as np

from ibllib.io import hashfile
from oneibl.one import ONE
from oneibl.patcher import Patcher


_ONE = ONE(base_url='https://test.alyx.internationalbrainlab.org',
           username='test_user', password='TapetesBloc18')


class TestPatchDatasets(unittest.TestCase):

    def setUp(self):
        self.one = _ONE
        self.patcher = Patcher(one=self.one)

    def test_create_file(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td).joinpath('')

    def test_patch_file(self):
        """
        Downloads a file from the flatiron, modify it, patch it and download it again
        """
        dataset_id = '04abb580-e14b-4716-9ff2-f7b95740b99f'
        dataset = self.one.alyx.rest('datasets', 'read', id=dataset_id)
        # download
        local_file_path = self.one.load(dataset['session'],
                          dataset_types=dataset['dataset_type'],
                          download_only=True, clobber=True)[0]
        old_check_sum = hashfile.md5(local_file_path)
        # change it
        np.save(local_file_path, ~np.load(local_file_path))
        new_check_sum = hashfile.md5(local_file_path)
        # try once with dry
        self.patcher.patch_dataset(local_file_path, dset_id=dataset['url'][-36:], dry=True)
        self.patcher.patch_dataset(local_file_path, dset_id=dataset['url'][-36:], dry=False)
        # download again and check the hash
        local_file_path.unlink()
        local_file_path = self.one.load(dataset['session'],
                          dataset_types=dataset['dataset_type'],
                          download_only=True, clobber=True)[0]
        self.assertEqual(hashfile.md5(local_file_path), new_check_sum)
        # the dataset hash should have been updated too
        dataset = self.one.alyx.rest('datasets', 'read', id=dataset_id)
        self.assertEqual(uuid.UUID(dataset['md5']), uuid.UUID(new_check_sum))

    def test_patch_existing_file(self):
        pass


if __name__ == "__main__":
    unittest.main(exit=False)
