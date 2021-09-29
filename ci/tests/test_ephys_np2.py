import numpy as np
import random
import shutil

from ibllib.ephys.np2_converter import NP2Converter
from ibllib.io import spikeglx

from ci.tests import base


class TestNeuropixel2Converter(base.IntegrationTest):
    def setUp(self) -> None:
        self.file_path = r'C:\Users\Mayo\Downloads\NP2\probe00\_spikeglx_ephysData_g0_t0.imec0.ap.bin'
        self.fs = 30000
        self.sglx_instances = []
        self.temp_directories = []

    def tearDown(self):
        _ = [sglx.close() for sglx in self.sglx_instances]
        _ = [shutil.rmtree(temp) for temp in self.temp_directories]

    def testDecimate(self):
        """
        Check integrity of windowing and downsampling by comparing results when using different
        window lengths for iterating through data
        :return:
        """

        FS = 30000
        NSAMPLES = int(2 * FS)
        np_a = NP2Converter(self.file_path, post_check=False)
        np_a.init_params(nsamples=NSAMPLES, nwindow=0.5 * FS, extra='_0_5s', nshank=[0])
        np_a.process()
        self.temp_directories.append(np_a.shank_info['shank0']['ap_file'].parent)

        np_b = NP2Converter(self.file_path, post_check=False)
        np_b.init_params(nsamples=NSAMPLES, nwindow=1 * FS, extra='_1s', nshank=[0])
        np_b.process()
        self.temp_directories.append(np_b.shank_info['shank0']['ap_file'].parent)

        np_c = NP2Converter(self.file_path, post_check=False)
        np_c.init_params(nsamples=NSAMPLES, nwindow=2 * FS, extra='_2s', nshank=[0])
        np_c.process()
        self.temp_directories.append(np_c.shank_info['shank0']['ap_file'].parent)

        sr = spikeglx.Reader(self.file_path)
        self.sglx_instances.append(sr)
        sr_a_ap = spikeglx.Reader(np_a.shank_info['shank0']['ap_file'])
        self.sglx_instances.append(sr_a_ap)
        sr_b_ap = spikeglx.Reader(np_b.shank_info['shank0']['ap_file'])
        self.sglx_instances.append(sr_b_ap)
        sr_c_ap = spikeglx.Reader(np_c.shank_info['shank0']['ap_file'])
        self.sglx_instances.append(sr_c_ap)

        # Make sure all the aps are the same regardless of window size we used
        assert np.array_equal(sr_a_ap[:, :], sr_b_ap[:, :])
        assert np.array_equal(sr_a_ap[:, :], sr_c_ap[:, :])
        assert np.array_equal(sr_b_ap[:, :], sr_c_ap[:, :])

        # For AP also check that all values are the same as the original file
        assert np.array_equal(sr_a_ap[:, :], sr[:NSAMPLES, np_a.shank_info['shank0']['chns']])
        assert np.array_equal(sr_b_ap[:, :], sr[:NSAMPLES, np_b.shank_info['shank0']['chns']])
        assert np.array_equal(sr_c_ap[:, :], sr[:NSAMPLES, np_c.shank_info['shank0']['chns']])

        sr_a_lf = spikeglx.Reader(np_a.shank_info['shank0']['lf_file'])
        self.sglx_instances.append(sr_a_lf)
        sr_b_lf = spikeglx.Reader(np_b.shank_info['shank0']['lf_file'])
        self.sglx_instances.append(sr_b_lf)
        sr_c_lf = spikeglx.Reader(np_c.shank_info['shank0']['lf_file'])
        self.sglx_instances.append(sr_c_lf)

        # Make sure all the lfps are the same regardless of window size we used
        assert np.array_equal(sr_a_lf[:, :], sr_b_lf[:, :])
        assert np.array_equal(sr_a_lf[:, :], sr_c_lf[:, :])
        assert np.array_equal(sr_b_lf[:, :], sr_c_lf[:, :])

    def testProcessNP1(self):
        # for NP1 needs to do nada
        pass

    def testProcessNP2_4(self):
        # make sure it runs without problems
        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(nsamples=int(5 * self.fs))
        status = np_conv.process()
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)

        for sh in np_conv.shank_info.keys():
            self.temp_directories.append(np_conv.shank_info[sh]['ap_file'].parent)

        # test a random ap metadata file and make sure it all makes sense
        shank_n = random.randint(0, 3)
        sr_ap = spikeglx.Reader(np_conv.shank_info[f'shank{shank_n}']['ap_file'])
        assert np.array_equal(sr_ap.meta['acqApLfSy'], [96, 0, 1])
        assert np.array_equal(sr_ap.meta['snsApLfSy'], [96, 0, 1])
        assert np.equal(sr_ap.meta['nSavedChans'], 97)
        assert (sr_ap.meta['snsSaveChanSubset'] == '0:96')
        assert np.equal(sr_ap.meta['NP2.4_shank'], shank_n)
        assert (sr_ap.meta['original_meta'] == 'False')
        sr_ap.close()

        # test a random lf metadata file and make sure it all makes sense
        shank_n = random.randint(0, 3)
        sr_lf = spikeglx.Reader(np_conv.shank_info[f'shank{shank_n}']['lf_file'])
        assert np.array_equal(sr_lf.meta['acqApLfSy'], [0, 96, 1])
        assert np.array_equal(sr_lf.meta['snsApLfSy'], [0, 96, 1])
        assert np.equal(sr_lf.meta['nSavedChans'], 97)
        assert (sr_lf.meta['snsSaveChanSubset'] == '0:96')
        assert np.equal(sr_lf.meta['NP2.4_shank'], shank_n)
        assert (sr_lf.meta['original_meta'] == 'False')
        assert np.equal(sr_lf.meta['imSampRate'], 2500)
        sr_lf.close()

        # Rerun again and make sure that nothing happens because it has already existed
        # Not sure how best to do this, currently just looking at the flag that should be created
        # I guess I could time it :/

        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(nsamples=int(5 * self.fs))
        status = np_conv.process()
        self.assertTrue(np_conv.already_exists)
        self.assertFalse(status)

        # Now try with the overwrite flag and make sure it runs
        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(nsamples=int(5 * self.fs))
        status = np_conv.process(overwrite=True)
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)

        # Change some of the data and make sure the test doesn't check out
        shank_n = random.randint(0, 3)
        ap_file = np_conv.shank_info[f'shank{shank_n}']['ap_file']
        with open(ap_file, "r+b") as f:
            f.write((chr(10) + chr(20) + chr(30) + chr(40)).encode())

        # Now that we have changed the file we expect an assertion error when we do the check
        with self.assertRaises(AssertionError) as context:
            np_conv.check()
        self.assertTrue('data in original file and split files do no match'
                        in str(context.exception))

        # Finally test that we cannot process a file that has already been split
        np_conv = NP2Converter(ap_file)
        status = np_conv.process()
        self.assertTrue(np_conv.already_processed)
        self.assertFalse(status)

    def testprocess_NP2_1(self):
        # need to get hands on NP2.1 meta file
        pass


if __name__ == "__main__":
    import unittest
    unittest.main(exit=False)
