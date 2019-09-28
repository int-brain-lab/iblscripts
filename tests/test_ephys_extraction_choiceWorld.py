import unittest
from pathlib import Path
import shutil

import numpy as np

from ibllib.ephys import ephysqc
import alf.io
import ibllib.pipes.experimental_data as iblrig_pipeline
from oneibl.one import ONE


class TestSpikeSortingOutput(unittest.TestCase):

    def setUp(self):
        """
        replicate the full folder architecture with symlinks from choice_world_init to
        choice_world
        """
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/choice_world_init')
        if not self.init_folder.exists():
            return
        self.main_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/choice_world')
        if self.main_folder.exists():
            shutil.rmtree(self.main_folder)
        self.main_folder.mkdir(exist_ok=True)
        for ff in self.init_folder.rglob('*.*'):
            link = self.main_folder.joinpath(ff.relative_to(self.init_folder))
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)
        # instantiate the one object for registration
        self.one = ONE(base_url='https://test.alyx.internationalbrainlab.org',  # testdev
                       username='test_user', password='TapetesBloc18')

    def testMergeSpikes(self):
        sessions_paths = [f.parent for f in self.main_folder.rglob('extract_ephys.flag')]

        """test extraction of behaviour first"""
        iblrig_pipeline.extract_ephys(self.main_folder)

        """ then sync/merge the spike sorting, making sure there are no flag files left
         and controlling that the output files dimensions make sense"""
        iblrig_pipeline.sync_merge_ephys(self.main_folder)
        self.assertIsNone(list(Path(self.main_folder).rglob('sync_merge_ephys.flag')))

        for session_path in sessions_paths:
            self.check_session_output(session_path)

        """ at last make sure all of those register properly on the test database"""
        iblrig_pipeline.register(self.main_folder, one=self.one)

    def check_session_output(self, session_path):
        from pathlib import Path
        import alf.io
        session_path = Path("/mnt/s0/Data/IntegrationTests/ephys/choice_world/KS005/2019-08-30/001")
        session_path = Path("/mnt/s0/Data/IntegrationTests/ephys/choice_world/ZM_1736/2019-08-09/004")

        """ Check the spikes object """
        spikes_attributes = ['depths', 'amps', 'clusters', 'times']  # todo: template
        spikes = alf.io.load_object(session_path.joinpath('alf'), 'spikes')
        self.assertTrue(alf.io.check_dimensions(spikes) == 0)
        # check that it contains the proper keys
        self.assertTrue(set(spikes.keys()).issubset(set(spikes_attributes)))
        self.assertTrue(np.min(spikes.depths) >= 0)
        self.assertTrue(np.max(spikes.depths) <= 3840)
        self.assertTrue(10 < np.median(spikes.amps) < 80)  # we expect microvolts

        """Check the clusters object"""
        clusters = alf.io.load_object(session_path.joinpath('alf'), 'clusters')
        clusters_attributes = ['depths', 'probes']
        self.assertTrue(np.unique([clusters[k].shape[0] for k in clusters]).size == 1)
        self.assertTrue(set(clusters_attributes).issubset(set(clusters.keys())))

        """Check the channels object"""
        channels = alf.io.load_object(session_path.joinpath('alf'), 'channels')
        channels_attributes = ['probes', 'rawInd']
        self.assertTrue(set(channels_attributes).issubset(set(channels.keys())))


    def tearDown(self):
        shutil.rmtree(self.main_folder)


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
