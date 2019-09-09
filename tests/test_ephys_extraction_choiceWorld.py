import unittest
from pathlib import Path
import shutil

from ibllib.ephys import ephysqc
import ibllib.io.flags
import alf.io
import ibllib.pipes.experimental_data as iblrig_pipeline
from oneibl.one import ONE


class TestEphysQC(unittest.TestCase):

    def setUp(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/choice_world')
        if not self.init_folder.exists():
            return
        self.alf_folder = self.init_folder / 'alf'

    def test_qc_extract(self):
        # extract a short lf signal RMS
        for fbin in Path(self.init_folder).rglob('*.lf.bin'):
            ephysqc.extract_rmsmap(fbin, out_folder=self.alf_folder)
            rmsmap_lf = alf.io.load_object(self.alf_folder, '_spikeglx_ephysQcTimeLF')
            spec_lf = alf.io.load_object(self.alf_folder, '_spikeglx_ephysQcFreqLF')
            ntimes = rmsmap_lf['times'].shape[0]
            nchannels = rmsmap_lf['rms'].shape[1]
            nfreqs = spec_lf['freq'].shape[0]
            # makes sure the dimensions are consistend
            self.assertTrue(rmsmap_lf['rms'].shape == (ntimes, nchannels))
            self.assertTrue(spec_lf['power'].shape == (nfreqs, nchannels))

    def tearDown(self):
        if not self.init_folder.exists():
            return
        shutil.rmtree(self.alf_folder, ignore_errors=True)


class TestEphysExtraction(unittest.TestCase):

    def setUp(self):
        self.session_path = Path('/mnt/s0/Data/IntegrationTests/ephys/ZM_1150/2019-05-07/001')
        if not self.session_path.exists():
            return

    def test_sync_extract(self):
        pass
        # session_path = self.session_path
        # dry = False
        # ibllib.io.flags.create_other_flags(session_path, 'extract_ephys.flag', force=True)
        # iblrig_pipeline.extract_ephys(session_path, dry=dry)
        #
        # one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
        #           username='test_user', password='TapetesBloc18')
        # iblrig_pipeline.register(session_path, one=one)


if __name__ == "__main__":
    unittest.main(exit=False)
