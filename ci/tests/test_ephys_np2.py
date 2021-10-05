import numpy as np
import random
import shutil

from ibllib.ephys.np2_converter import NP2Converter
from ibllib.io import spikeglx

from ci.tests import base


class TestNeuropixel2ConverterNP24(base.IntegrationTest):
    """
    Check NP2 converter with NP2.4 type probes
    """
    def setUp(self) -> None:
        self.file_path = self.data_path.joinpath('ephys', 'ephys_np2', 'raw_ephys_data', 'probe00',
                                                 '_spikeglx_ephysData_g0_t0.imec0.ap.bin')
        meta_file = self.file_path.parent.joinpath('NP24_meta',
                                                   '_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        self.meta_file = self.file_path.parent.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        shutil.copy(meta_file, self.meta_file)
        self.sglx_instances = []
        self.temp_directories = []

    def tearDown(self):
        _ = [sglx.close() for sglx in self.sglx_instances]

        # here should look for any directories with test in it and delete
        test_dir = list(self.file_path.parent.parent.glob('*test*'))
        _ = [shutil.rmtree(test) for test in test_dir]
        self.meta_file.unlink()

    def testDecimate(self):
        """
        Check integrity of windowing and downsampling by comparing results when using different
        window lengths for iterating through data
        :return:
        """

        FS = 30000
        np_a = NP2Converter(self.file_path, post_check=False)
        np_a.init_params(nwindow=0.5 * FS, extra='_0_5s_test', nshank=[0])
        np_a.process()

        np_b = NP2Converter(self.file_path, post_check=False)
        np_b.init_params(nwindow=1 * FS, extra='_1s_test', nshank=[0])
        np_b.process()

        np_c = NP2Converter(self.file_path, post_check=False)
        np_c.init_params(nwindow=3 * FS, extra='_2s_test', nshank=[0])
        np_c.process()

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
        assert np.array_equal(sr_a_ap[:, :], sr[:, np_a.shank_info['shank0']['chns']])
        assert np.array_equal(sr_b_ap[:, :], sr[:, np_b.shank_info['shank0']['chns']])
        assert np.array_equal(sr_c_ap[:, :], sr[:, np_c.shank_info['shank0']['chns']])

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

    def testProcessNP24(self):
        """
        Check normal workflow of splittig data into individual shanks
        :return:
        """
        # Make sure normal workflow runs without problems
        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(extra='_test')
        status = np_conv.process()
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)

        # Test a random ap metadata file and make sure it all makes sense
        shank_n = random.randint(0, 3)
        sr_ap = spikeglx.Reader(np_conv.shank_info[f'shank{shank_n}']['ap_file'])
        assert np.array_equal(sr_ap.meta['acqApLfSy'], [96, 0, 1])
        assert np.array_equal(sr_ap.meta['snsApLfSy'], [96, 0, 1])
        assert np.equal(sr_ap.meta['nSavedChans'], 97)
        assert (sr_ap.meta['snsSaveChanSubset'] == '0:96')
        assert np.equal(sr_ap.meta['NP2.4_shank'], shank_n)
        assert (sr_ap.meta['original_meta'] == 'False')
        sr_ap.close()

        # Test a random lf metadata file and make sure it all makes sense
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

        # Rerun again and make sure that nothing happens because folders already exists
        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(extra='_test')
        status = np_conv.process()
        self.assertTrue(np_conv.already_exists)
        self.assertFalse(status)

        # But if we set the overwrite flag to True we force rerunning
        np_conv = NP2Converter(self.file_path)
        np_conv.init_params(extra='_test')
        status = np_conv.process(overwrite=True)
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)

        # Change some of the data and make sure the checking function is working as expected
        shank_n = random.randint(0, 3)
        ap_file = np_conv.shank_info[f'shank{shank_n}']['ap_file']
        with open(ap_file, "r+b") as f:
            f.write((chr(10) + chr(20) + chr(30) + chr(40)).encode())

        # Now that we have changed the file we expect an assertion error when we do the check
        with self.assertRaises(AssertionError) as context:
            np_conv.check_NP24()
        self.assertTrue('data in original file and split files do no match'
                        in str(context.exception))

        # Finally test that we cannot process a file that has already been split
        np_conv = NP2Converter(ap_file)
        status = np_conv.process()
        self.assertTrue(np_conv.already_processed)
        self.assertFalse(status)


class TestNeuropixel2ConverterNP21(base.IntegrationTest):
    """
    Check NP2 converter with NP2.1 type probes
    """
    def setUp(self) -> None:
        self.file_path = self.data_path.joinpath('ephys', 'ephys_np2', 'raw_ephys_data', 'probe00',
                                                 '_spikeglx_ephysData_g0_t0.imec0.ap.bin')
        meta_file = self.file_path.parent.joinpath('NP21_meta',
                                                   '_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        self.meta_file = self.file_path.parent.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        shutil.copy(meta_file, self.meta_file)
        self.sglx_instances = []

    def tearDown(self):
        _ = [sglx.close() for sglx in self.sglx_instances]
        # here should look for anything with test in it and delete
        lf_file = self.file_path.parent.joinpath(self.file_path.name.replace('ap', 'lf'))
        lf_meta = self.meta_file.parent.joinpath(self.meta_file.name.replace('ap', 'lf'))
        lf_file.unlink()
        lf_meta.unlink()
        self.meta_file.unlink()

    def testProcessNP21(self):
        """
        Check normal workflow of getting LFP data out and storing in main probe folder
        :return:
        """

        # make sure it runs smoothly
        np_conv = NP2Converter(self.file_path)
        status = np_conv.process()
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)

        # test the meta file
        sr_ap = spikeglx.Reader(np_conv.shank_info[f'shank0']['lf_file'])
        assert np.array_equal(sr_ap.meta['acqApLfSy'], [0, 384, 1])
        assert np.array_equal(sr_ap.meta['snsApLfSy'], [0, 384, 1])
        assert np.equal(sr_ap.meta['nSavedChans'], 385)
        assert (sr_ap.meta['snsSaveChanSubset'] == '0:384')
        assert np.equal(sr_ap.meta['NP2.1_shank'], 0)
        assert (sr_ap.meta['original_meta'] == 'False')
        sr_ap.close()

        # now run again and make sure that it doesn't run
        np_conv = NP2Converter(self.file_path)
        status = np_conv.process()
        self.assertTrue(np_conv.already_exists)
        self.assertFalse(status)

        # Now try with the overwrite flag and make sure it runs
        np_conv = NP2Converter(self.file_path)
        status = np_conv.process(overwrite=True)
        self.assertFalse(np_conv.already_exists)
        self.assertTrue(status)


class TestNeuropixel2ConverterNP1(base.IntegrationTest):
    """
    Check NP2 converter with NP1 type probes
    """
    def setUp(self) -> None:
        self.file_path = self.data_path.joinpath('ephys', 'ephys_np2', 'raw_ephys_data', 'probe00',
                                                 '_spikeglx_ephysData_g0_t0.imec0.ap.bin')
        meta_file = self.file_path.parent.joinpath('NP1_meta',
                                                   '_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        self.meta_file = self.file_path.parent.joinpath('_spikeglx_ephysData_g0_t0.imec0.ap.meta')
        shutil.copy(meta_file, self.meta_file)
        self.sglx_instances = []
        self.temp_directories = []

    def tearDown(self):
        # here should look for anything with test in it and delete
        self.meta_file.unlink()

    def testProcessNP1(self):
        """
        Check normal workflow -> nothing should happen!
        """
        np_conv = NP2Converter(self.file_path)
        status = np_conv.process()
        self.assertFalse(status)


if __name__ == "__main__":
    import unittest
    unittest.main(exit=False)
