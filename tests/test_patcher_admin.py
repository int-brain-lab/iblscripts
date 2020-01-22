import unittest
import uuid
import tempfile
from pathlib import Path

import numpy as np
import globus_sdk

from ibllib.misc import version
from ibllib.io import hashfile, params
from oneibl.one import ONE
from oneibl.patcher import SSHPatcher, GlobusPatcher, FTPPatcher


_ONE = ONE(base_url='https://test.alyx.internationalbrainlab.org',
           username='test_user', password='TapetesBloc18')


def _test_create_and_delete_file(self):
    """
    Creates a file, upload it to Flatiron twice and then removes it
    """
    with tempfile.TemporaryDirectory() as td:
        # creates the local file
        session_path = Path(td).joinpath('flowers', '2018-07-13', '001')
        alf_path = session_path.joinpath('alf')
        alf_path.mkdir(parents=True)
        new_file = alf_path.joinpath('spikes.amps.npy')
        np.save(new_file, np.random.rand(500, 1))
        # try a dry run first
        self.patcher.create_dataset(new_file, dry=True)
        # creates it on the database
        self.patcher.create_dataset(new_file, server_repository='flatiron_zadorlab')
        # download through ONE and check hashes
        eid = self.one.search(subjects='flowers', dataset_types=['spikes.amps'])[0]
        download0 = self.one.load(eid, dataset_types=['spikes.amps'], download_only=True,
                                  dclass_output=True, clobber=True)[0]
        # creates it a second time an makes sure it's not duplicated (also test automatic repo)
        self.patcher.create_dataset(new_file)
        download = self.one.load(eid, dataset_types=['spikes.amps'], download_only=True,
                                 dclass_output=True, clobber=True)[0]
        self.assertEqual(download.dataset_id, download0.dataset_id)
        self.assertTrue(hashfile.md5(download.local_path) == hashfile.md5(new_file))
        # deletes the file
        self.patcher.delete_dataset(dset_id=download.dataset_id, dry=False)
        # makes sure it's not in the database anymore
        session = self.one.search(subjects='flowers', dataset_types=['spikes.amps'])
        self.assertEqual(len(session), 0)


def _patch_file(self, download=True):
    """
    Downloads a file from the flatiron, modify it locally, patch it and download it again
    """
    dataset_id = '04abb580-e14b-4716-9ff2-f7b95740b99f'
    dataset = self.one.alyx.rest('datasets', 'read', id=dataset_id)
    # download
    local_file_path = self.one.load(dataset['session'],
                                    dataset_types=dataset['dataset_type'],
                                    download_only=True, clobber=True)[0]
    # change it
    np.save(local_file_path, ~np.load(local_file_path))
    new_check_sum = hashfile.md5(local_file_path)
    # try once with dry
    self.patcher.patch_dataset(local_file_path, dset_id=dataset['url'][-36:], dry=True)
    self.patcher.patch_dataset(local_file_path, dset_id=dataset['url'][-36:], dry=False)
    # the dataset hash should have been updated
    dataset = self.one.alyx.rest('datasets', 'read', id=dataset_id)
    self.assertEqual(uuid.UUID(dataset['hash']), uuid.UUID(new_check_sum))
    self.assertEqual(dataset['version'], version.ibllib())
    if download:
        # download again and check the hash
        local_file_path.unlink()
        local_file_path = self.one.load(dataset['session'],
                                        dataset_types=dataset['dataset_type'],
                                        download_only=True, clobber=True)[0]
        self.assertEqual(hashfile.md5(local_file_path), new_check_sum)


class TestPatchDatasetsGlobus(unittest.TestCase):
    """This requires admin access to Globus"""

    def setUp(self):
        self.one = _ONE
        remote_repo = '15f76c0c-10ee-11e8-a7ed-0a448319c2f8'  # flatiron
        self.par = params.read('globus')
        label = 'test_patcher'
        authorizer = globus_sdk.AccessTokenAuthorizer(self.par.TRANSFER_TOKEN)
        self.gtc = globus_sdk.TransferClient(authorizer=authorizer)
        globus_transfer = globus_sdk.TransferData(
            self.gtc, self.par.LOCAL_REPO, remote_repo, verify_checksum=True,
            sync_level='checksum', label=label)
        globus_delete = globus_sdk.DeleteData(
            self.gtc, remote_repo, verify_checksum=True, sync_level='checksum', label=label)
        self.patcher = GlobusPatcher(one=self.one, globus_delete=globus_delete,
                                     globus_transfer=globus_transfer)

    def test_create_and_delete_file(self):
        session_path = Path(self.par.LOCAL_PATH).joinpath('flowers', '2018-07-13', '001')
        alf_path = session_path.joinpath('alf')
        alf_path.mkdir(parents=True, exist_ok=True)
        new_files = [alf_path.joinpath('spikes.amps.npy'),
                     alf_path.joinpath('spikes.times.npy')]
        for nf in new_files:
            np.save(nf, np.random.rand(500, 1))
        # try a dry run first
        self.patcher.create_dataset(new_files, dry=True)
        # creates it on the database
        self.patcher.create_dataset(new_files, server_repository='flatiron_zadorlab')
        self.patcher.launch_transfers(self.gtc, wait=True)
        # download through ONE and check hashes
        eid = self.one.search(subjects='flowers', dataset_types=['spikes.amps'])[0]
        download0 = self.one.load(eid, dataset_types=['spikes.amps'],
                                  download_only=True, dclass_output=True, clobber=True)[0]
        self.assertEqual(download0.dataset_id, download0.dataset_id)
        self.assertTrue(hashfile.md5(download0.local_path) == hashfile.md5(new_files[0]))
        # deletes the file
        self.patcher.delete_dataset(dset_id=download0.dataset_id, dry=False)
        self.patcher.launch_transfers(self.gtc, wait=True)
        # makes sure it's not in the database anymore
        session = self.one.search(subjects='flowers', dataset_types=[download0.dataset_type])
        self.assertEqual(len(session), 0)


class TestPatchDatasetsSSH(unittest.TestCase):
    def setUp(self):
        self.one = _ONE
        self.patcher = SSHPatcher(one=self.one)

    def test_create_and_delete_file(self):
        return
        _test_create_and_delete_file(self)

    def test_patch_file(self):
        return
        _patch_file(self)


class TestPatchDatasetsFTP(unittest.TestCase):
    def setUp(self):
        self.one = _ONE
        self.patcher = FTPPatcher(one=self.one)

    def test_create_and_delete_file(self):
        pass
        # _test_create_and_delete_file(self)

    def test_patch_file(self):
        # we can only test registration as the FlatIron upload is not immediate in this case
        _patch_file(self, download=False)


if __name__ == "__main__":
    unittest.main(exit=False)
