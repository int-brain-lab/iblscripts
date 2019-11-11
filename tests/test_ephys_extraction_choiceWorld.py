import unittest
from pathlib import Path
import shutil
import logging

import numpy as np

from ibllib.ephys import ephysqc
import alf.io
import ibllib.pipes.experimental_data as iblrig_pipeline
from oneibl.one import ONE

_logger = logging.getLogger('ibllib')


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
        self.one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
                       username='test_user', password='TapetesBloc18')

    def testMergeSpikes(self):
        sessions_paths = [f.parent for f in self.main_folder.rglob('extract_ephys.flag')]

        """test extraction of behaviour first"""
        iblrig_pipeline.extract_ephys(self.main_folder)

        """ then sync/merge the spike sorting, making sure there are no flag files left
         and controlling that the output files dimensions make sense"""
        iblrig_pipeline.sync_merge_ephys(self.main_folder)
        self.assertFalse(list(Path(self.main_folder).rglob('sync_merge_ephys.flag')))

        for session_path in sessions_paths:
            self.check_session_output(session_path)

        # for session_path in sessions_paths:
        #     session_path.joinpath('register_me.flag').touch()

        """ at last make sure we get all expected datasets after registration on the test db"""
        iblrig_pipeline.register(self.main_folder, one=self.one)
        # expected dataset types and min expected, max expected
        EXPECTED_DATASETS = [('_iblrig_Camera.raw', 3, 3),
                             ('_iblrig_Camera.timestamps', 3, 3),
                             ('_iblrig_ambientSensorData.raw', 1, 1),
                             ('_iblrig_codeFiles.raw', 1, 1),
                             ('_iblrig_encoderEvents.raw', 1, 1),
                             ('_iblrig_encoderPositions.raw', 1, 1),
                             ('_iblrig_encoderTrialInfo.raw', 1, 1),
                             ('_iblrig_taskData.raw', 1, 1),
                             ('_iblrig_taskSettings.raw', 1, 1),
                             ('_spikeglx_sync.channels', 2, 3),
                             ('_spikeglx_sync.polarities', 2, 3),
                             ('_spikeglx_sync.times', 2, 3),
                             ('camera.times', 3, 3),
                             ('channels._phy_ids', 3, 3),
                             ('channels.localCoordinates', 3, 3),
                             ('channels.probes', 1, 1),
                             ('channels.rawInd', 3, 3),
                             ('clusters.amps', 3, 3),
                             ('clusters.channels', 3, 3),
                             ('clusters.depths', 3, 3),
                             ('clusters.probes', 1, 1),
                             ('clusters.peakToTrough', 3, 3),
                             ('ephysData.raw.ap', 2, 2),
                             ('ephysData.raw.lf', 2, 2),
                             ('ephysData.raw.ch', 4, 5),
                             ('ephysData.raw.meta', 4, 5),
                             ('ephysData.raw.sync', 2, 2),
                             ('ephysData.raw.wiring', 2, 3),
                             ('spikes.amps', 3, 3),
                             ('spikes.clusters', 3, 3),
                             ('spikes.depths', 3, 3),
                             ('spikes.templates', 3, 3),
                             ('spikes.times', 3, 3),
                             ('trials.choice', 1, 1),
                             ('trials.contrastLeft', 1, 1),
                             ('trials.contrastRight', 1, 1),
                             ('trials.feedbackType', 1, 1),
                             ('trials.feedback_times', 1, 1),
                             ('trials.goCue_times', 1, 1),
                             ('trials.goCueTrigger_times', 1, 1),
                             ('trials.intervals', 1, 1),
                             ('trials.itiDuration', 1, 1),
                             ('trials.probabilityLeft', 1, 1),
                             ('trials.response_times', 1, 1),
                             ('trials.rewardVolume', 1, 1),
                             ('trials.stimOn_times', 1, 1),
                             ('wheel.position', 1, 1),
                             ('wheel.timestamps', 1, 1),
                             ('wheel.velocity', 1, 1),
                             ]
        # check that we indeed find expected number of datasets after registration
        success = True
        for session_path in sessions_paths:
            _logger.info(f'\n{session_path}')
            sub, date, number = session_path.parts[-3:]
            eid = self.one.search(task_protocol='ephys', subject=sub, date=date)[0]
            ses_details = self.one.alyx.rest('sessions', 'read', id=eid)
            dtypes = [ds['dataset_type'] for ds in ses_details['data_dataset_session_related']]
            dtypes.sort()
            for ed in EXPECTED_DATASETS:
                count = sum([1 if ed[0] == dt else 0 for dt in dtypes])
                if not ed[1] <= count <= ed[2]:
                    _logger.error(f'missing dataset types: {ed[0]} found {count}, '
                                  f'expected between [{ed[1]} and {ed[2]}]')
                    success = False
                else:
                    _logger.info(f'check dataset types registration OK: {ed[0]}')
        self.assertTrue(success)

    def check_session_output(self, session_path):
        """ Check the spikes object """
        spikes_attributes = ['depths', 'amps', 'clusters', 'times', 'templates', 'samples']
        spikes = alf.io.load_object(session_path.joinpath('alf'), 'spikes')
        self.assertTrue(np.max(spikes.times) > 1000)
        self.assertTrue(alf.io.check_dimensions(spikes) == 0)
        # check that it contains the proper keys
        self.assertTrue(set(spikes.keys()).issubset(set(spikes_attributes)))
        self.assertTrue(np.min(spikes.depths) >= 0)
        self.assertTrue(np.max(spikes.depths) <= 3840)
        self.assertTrue(10 < np.median(spikes.amps) < 80)  # we expect microvolts

        """Check the clusters object"""
        clusters = alf.io.load_object(session_path.joinpath('alf'), 'clusters')
        clusters_attributes = ['depths', 'probes', 'channels', 'peakToTrough', 'amps']
        self.assertTrue(np.unique([clusters[k].shape[0] for k in clusters]).size == 1)
        self.assertTrue(set(clusters_attributes) == set(clusters.keys()))
        self.assertTrue(10 < np.nanmedian(clusters.amps) < 80)  # we expect microvolts
        self.assertTrue(0 < np.median(np.abs(clusters.peakToTrough)) < 5)  # we expect ms

        """Check the channels object"""
        channels = alf.io.load_object(session_path.joinpath('alf'), 'channels')
        channels_attributes = ['probes', 'rawInd', 'localCoordinates', '_phy_ids']
        np.diff(channels.rawInd[channels.probes == 0])
        self.assertTrue(set(channels.keys()) == set(channels_attributes))

        """Check the template object"""
        templates = alf.io.load_object(session_path.joinpath('alf'), 'templates')
        templates_attributes = ['waveforms']
        self.assertTrue(set(templates.keys()) == set(templates_attributes))

        # """Check the probes object"""
        # probes = alf.io.load_object(session_path.joinpath('alf'), 'probes')

        """(basic) check cross-references"""
        nclusters = clusters.depths.size
        nchannels = channels.probes.size
        ntemplates = templates.waveforms.shape[0]
        self.assertTrue(np.all(0 <= spikes.clusters) and
                        np.all(spikes.clusters <= (nclusters - 1)))
        self.assertTrue(np.all(0 <= spikes.templates) and
                        np.all(spikes.templates <= (ntemplates - 1)))
        self.assertTrue(np.all(0 <= clusters.channels) and
                        np.all(clusters.channels <= (nchannels - 1)))
        # check that the site positions channels match the depth with indexing
        self.assertTrue(np.all(clusters.depths == channels.localCoordinates[clusters.channels, 1]))
        # check that the probe index from clusters and channels check out
        self.assertTrue(np.all(clusters.probes == channels.probes[clusters.channels]))

    def tearDown(self):
        shutil.rmtree(self.main_folder)


class TestEphysQC(unittest.TestCase):

    def setUp(self):
        self.init_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/ephys_qc')
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


if __name__ == "__main__":
    unittest.main(exit=False)
