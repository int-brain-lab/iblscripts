import shutil
import tarfile
import tempfile
import numpy as np
from pathlib import Path
from ibllib.pipes.ephys_manual_curation import ALF_SS_FILES, ManualCuration, CLUSTER_FILES

from ci.tests import base


class TestManualCurationExtraction(base.IntegrationTest):

    @classmethod
    def setUpClass(cls) -> None:

        # Need to make this into a proper session path
        cls.session_path = cls.default_data_root().joinpath('ephys/phy_curation/wittenlab/Subjects/dop_48/2024-03-10/001')
        cls.tar_path = cls.session_path.joinpath('spike_sorters/pykilosort/probe00')
        cls.alf_path = cls.session_path.joinpath('alf/probe00/pykilosort')

        # cls.temp_dir = Path('/Users/admin/temp_merge')
        cls.temp_dir = Path(tempfile.TemporaryDirectory().name)
        cls.tar_temp_path = cls.temp_dir.joinpath('tar_ss')
        cls.tar_temp_path.mkdir(exist_ok=True, parents=True)
        cls.alf_temp_path = cls.temp_dir.joinpath('alf_ss')
        cls.alf_temp_path.mkdir(exist_ok=True, parents=True)

        cls.tar_out_path = cls.temp_dir.joinpath('tar_manual')
        cls.tar_out_path.mkdir(exist_ok=True, parents=True)
        cls.alf_out_path = cls.temp_dir.joinpath('alf_manual')
        cls.alf_out_path.mkdir(exist_ok=True, parents=True)

        # extract the tar file to the temp tar path
        with tarfile.open(cls.tar_path.joinpath('_kilosort_raw.output.tar'), 'r') as tar_dir:
            tar_dir.extractall(path=cls.tar_temp_path)

        # copy the relevant alf files to the temp alf path
        for file in ALF_SS_FILES:
            shutil.copy(cls.alf_path.joinpath(file), cls.alf_temp_path.joinpath(file))

        # mimic a manual creation process
        spike_clusters = np.load(cls.alf_path.joinpath('spikes.templates.npy'))
        cls.max_clus = np.max(spike_clusters)
        # Split cluster 0 into two new clusters
        idx_0 = np.where(spike_clusters == 0)[0]
        spike_clusters[idx_0[:int(idx_0.size / 2)]] = cls.max_clus + 1
        spike_clusters[idx_0[int(idx_0.size / 2):]] = cls.max_clus + 2

        # Merge cluster 15 and 16
        spike_clusters[np.bitwise_or(spike_clusters == 15, spike_clusters == 16)] = cls.max_clus + 3

        cls.cluster_file = Path(cls.temp_dir.joinpath('spikes.clusters.npy'))
        np.save(cls.cluster_file, spike_clusters)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir)

    def test_extraction_from_tar(self):
        # test the extraction from tar spikesorting output
        mc = ManualCuration(self.session_path, 'probe00', namespace='test', conversion_path=self.temp_dir,
                            out_path=self.tar_out_path)
        mc.cluster_file = self.cluster_file

        mc.extract_from_tar_data()
        self.compare_curated_vs_original(self.tar_out_path, tar=True)

        mc.compute_cluster_metrics('tar_ss')
        mc.rename_files()
        for f in CLUSTER_FILES:
            self.assertTrue(self.tar_out_path.joinpath('_test_' + f).exists())

        self.assertTrue(np.array_equal(np.load(self.tar_out_path.joinpath('_test_spikes.clusters.npy')),
                                       np.load(self.cluster_file)))

    def test_extraction_from_alf(self):
        # test the extraction from alf spikesorting output
        mc = ManualCuration(self.session_path, 'probe00', namespace='test', conversion_path=self.temp_dir,
                            out_path=self.alf_out_path)
        mc.cluster_file = self.cluster_file

        mc.extract_from_alf_data()
        self.compare_curated_vs_original(self.alf_out_path, tar=False)

        mc.compute_cluster_metrics('alf_ss')
        mc.rename_files()

        for f in CLUSTER_FILES:
            self.assertTrue(self.alf_out_path.joinpath('_test_' + f).exists())

        self.assertTrue(np.array_equal(np.load(self.alf_out_path.joinpath('_test_spikes.clusters.npy')),
                                       np.load(self.cluster_file)))

    def compare_curated_vs_original(self, curated_path, tar=True):

        files = ['clusters.amps.npy', 'clusters.channels.npy', 'clusters.depths.npy', 'clusters.peakToTrough.npy',
                 'clusters.waveforms.npy', 'clusters.waveformsChannels.npy']
        for f in files:
            orig_file = np.load(self.alf_path.joinpath(f))
            new_file = np.load(curated_path.joinpath(f))

            # Should have three new clusters, one from a merge and two from a split
            self.assertEqual(orig_file.shape[0] + 3, new_file.shape[0])

            # Check that the clusters that were not affected by the curation are unchanged
            merge_idx = [0, 15, 16]
            non_merge_idx = np.setdiff1d(np.arange(orig_file.shape[0]), np.array(merge_idx))
            self.assertTrue(np.allclose(orig_file[non_merge_idx], new_file[non_merge_idx]))

            # Check that the clusters that were affected have been replaced by the correct values
            for idx in merge_idx:
                if f == 'clusters.channels.npy':
                    self.assertEqual(new_file[idx], 0)
                elif f == 'clusters.waveformsChannels.npy':
                    self.assertTrue(np.array_equal(np.sort(new_file[idx]), np.arange(32, dtype=np.int32)))
                else:
                    if type(new_file[idx]) == np.float32:
                        self.assertTrue(np.array_equal(new_file[idx], np.nan, equal_nan=True))
                    else:
                        self.assertTrue(np.array_equal(new_file[idx], np.full_like(new_file[idx], np.nan), equal_nan=True))

            if tar:
                # Check that the split and merged cluster takes on correct value, we expect amps and waveforms to differ
                # as these are scaled by the weighted average amplitude of the split cluster
                if f in ['clusters.amps.npy', 'clusters.waveforms.npy']:
                    # Split clusters
                    self.assertFalse(np.allclose(orig_file[0], new_file[-3]))
                    self.assertFalse(np.allclose(orig_file[0], new_file[-2]))
                    # Merged clusters
                    self.assertFalse(np.allclose(orig_file[15], new_file[-1]))
                    self.assertFalse(np.allclose(orig_file[16], new_file[-1]))
                else:
                    # Split clusters
                    self.assertTrue(np.allclose(orig_file[0], new_file[-3]))
                    self.assertTrue(np.allclose(orig_file[0], new_file[-2]))
                    # Merged clusters
                    # The merged cluster takes on the value with the most spikes
                    self.assertTrue(np.allclose(orig_file[15], new_file[-1]))
            else:

                # Check for the split clusters we expect these to take on same values apart from amplitude
                # When splitting in the sparse case, waveforms are just copied
                if f in ['clusters.amps.npy']:
                    self.assertFalse(np.allclose(orig_file[0], new_file[-3]))
                    self.assertFalse(np.allclose(orig_file[0], new_file[-2]))
                else:
                    self.assertTrue(np.allclose(orig_file[0], new_file[-3]))
                    self.assertTrue(np.allclose(orig_file[0], new_file[-2]))

                # Check for the merge cluster, this is more complicated, the following
                # is true for this test example, but depending on the merge they could all differ.
                if f in ['clusters.amps.npy', 'clusters.waveforms.npy', 'clusters.waveformsChannels.npy']:
                    self.assertFalse(np.allclose(orig_file[15], new_file[-1]))
                    self.assertFalse(np.allclose(orig_file[16], new_file[-1]))
                elif f == 'clusters.peakToTrough.npy':
                    self.assertFalse(np.allclose(orig_file[15], new_file[-1]))
                    self.assertTrue(np.allclose(orig_file[16], new_file[-1]))
                else:
                    # The merged cluster takes on the value with the most spikes
                    self.assertTrue(np.allclose(orig_file[15], new_file[-1]))


if __name__ == "__main__":
    import unittest
    unittest.main(exit=False)
