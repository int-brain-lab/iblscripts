import logging
import unittest
from pathlib import Path
import shutil
from operator import itemgetter
import numpy as np

import alf.io
from ibllib.pipes import local_server, ephys_preprocessing
from oneibl.one import ONE

_logger = logging.getLogger('ibllib')

PATH_TESTS = Path('/mnt/s0/Data/IntegrationTests')
SESSION_PATH = PATH_TESTS.joinpath("ephys/choice_world/KS022/2019-12-10/001")
# one = ONE(base_url='http://localhost:8000')
one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
          username='test_user', password='TapetesBloc18')


class TestEphysPipeline(unittest.TestCase):

    def setUp(self) -> None:
        self.init_folder = PATH_TESTS.joinpath('ephys', 'choice_world_init')
        if not self.init_folder.exists():
            return
        self.main_folder = PATH_TESTS.joinpath('ephys', 'choice_world')
        if self.main_folder.exists():
            shutil.rmtree(self.main_folder)
        self.main_folder.mkdir(exist_ok=True)
        for ff in self.init_folder.rglob('*.*'):
            link = self.main_folder.joinpath(ff.relative_to(self.init_folder))
            if 'alf' in link.parts:
                continue
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)
        SESSION_PATH.joinpath('raw_session.flag').touch()

    def test_pipeline_with_alyx(self):
        """
        Test the ephys pipeline exactly as it is supposed to run on the local servers
        :return:
        """
        # first step is to remove the session and create it anew
        eid = one.eid_from_path(SESSION_PATH)
        if eid is not None:
            one.alyx.rest('sessions', 'delete', id=eid)

        # create the jobs and run them
        raw_ds = local_server.job_creator(SESSION_PATH, one=one, max_md5_size=1024 * 1024 * 20)
        eid = one.eid_from_path(SESSION_PATH)
        self.assertFalse(eid is None)  # the session is created on the database
        # the flag has been erased
        self.assertFalse(SESSION_PATH.joinpath('raw_session.flag').exists())

        subject_path = SESSION_PATH.parents[2]
        tasks_dict = one.alyx.rest('tasks', 'list', session=eid, status='Waiting')
        for td in tasks_dict:
            print(td['name'])
        all_datasets = local_server.tasks_runner(
            subject_path, tasks_dict, one=one, max_md5_size=1024 * 1024 * 20, count=20)

        # check the trajectories and probe info
        self.assertTrue(len(one.alyx.rest('insertions', 'list', session=eid)) == 2)
        self.assertTrue(len(one.alyx.rest(
            'trajectories', 'list', session=eid, provenance='Micro-manipulator')) == 2)

        # check the spike sorting output on disk
        self.check_spike_sorting_output(SESSION_PATH)

        # check the registration of datasets
        dsets = one.alyx.rest('datasets', 'list', session=eid)
        self.assertEqual(set([ds['url'][-36:] for ds in dsets]),
                         set([ds['id'] for ds in all_datasets + raw_ds]))

        nss = 2
        EXPECTED_DATASETS = [('_iblqc_ephysSpectralDensity.freqs', 4, 4),
                             ('_iblqc_ephysSpectralDensity.power', 4, 4),
                             ('_iblqc_ephysTimeRms.rms', 4, 4),
                             ('_iblqc_ephysTimeRms.timestamps', 4, 4),

                             ('_iblrig_micData.raw', 1, 1),

                             ('_spikeglx_sync.channels', 2, 3),
                             ('_spikeglx_sync.polarities', 2, 3),
                             ('_spikeglx_sync.times', 2, 3),

                             ('camera.times', 3, 3),

                             ('kilosort.whitening_matrix', nss, nss),
                             ('_phy_spikes_subset.channels', nss, nss),
                             ('_phy_spikes_subset.spikes', nss, nss),
                             ('_phy_spikes_subset.waveforms', nss, nss),

                             ('channels.localCoordinates', nss, nss),
                             ('channels.rawInd', nss, nss),
                             ('clusters.amps', nss, nss),
                             ('clusters.channels', nss, nss),
                             ('clusters.depths', nss, nss),
                             ('clusters.probes', 0, 0),
                             ('clusters.metrics', nss, nss),
                             ('clusters.peakToTrough', nss, nss),
                             ('clusters.uuids', nss, nss),
                             ('clusters.waveforms', nss, nss),
                             ('clusters.waveformsChannels', nss, nss),

                             # ('ephysData.raw.ap', 2, 2),
                             # ('ephysData.raw.lf', 2, 2),
                             # ('ephysData.raw.ch', 4, 5),
                             # ('ephysData.raw.meta', 4, 5),
                             ('ephysData.raw.sync', 2, 2),
                             ('ephysData.raw.timestamps', 2, 2),
                             # ('ephysData.raw.wiring', 2, 3),

                             ('probes.description', 1, 1),
                             ('probes.trajectory', 1, 1),
                             ('spikes.amps', nss, nss),
                             ('spikes.clusters', nss, nss),
                             ('spikes.depths', nss, nss),
                             ('spikes.templates', nss, nss),
                             ('spikes.times', nss, nss),
                             ('templates.waveforms', nss, nss),
                             ('templates.waveformsChannels', nss, nss),
                             ('templates.amps', nss, nss),

                             ('trials.choice', 1, 1),
                             ('trials.contrastLeft', 1, 1),
                             ('trials.contrastRight', 1, 1),
                             ('trials.feedback_times', 1, 1),
                             ('trials.feedbackType', 1, 1),
                             ('trials.firstMovement_times', 1, 1),
                             ('trials.goCue_times', 1, 1),
                             ('trials.goCueTrigger_times', 1, 1),
                             ('trials.intervals', 2, 2),
                             ('trials.probabilityLeft', 1, 1),
                             ('trials.response_times', 1, 1),
                             ('trials.rewardVolume', 1, 1),
                             ('trials.stimOff_times', 1, 1),
                             ('trials.stimOn_times', 1, 1),
                             ('wheel.position', 1, 1),
                             ('wheel.timestamps', 1, 1),
                             ('wheelMoves.intervals', 1, 1),
                             ('wheelMoves.peakAmplitude', 1, 1),
                             ]
        # check that we indeed find expected number of datasets after registration
        # for this we need to get the unique set of datasets
        dids = np.array([d['id'] for d in all_datasets])
        [_, iu] = np.unique(dids, return_index=True)
        dtypes = sorted([ds['dataset_type'] for ds in itemgetter(*iu)(all_datasets)])
        success = True
        for ed in EXPECTED_DATASETS:
            count = sum([1 if ed[0] == dt else 0 for dt in dtypes])
            if not ed[1] <= count <= ed[2]:
                _logger.error(f'missing dataset types: {ed[0]} found {count}, '
                              f'expected between [{ed[1]} and {ed[2]}]')
                success = False
            else:
                _logger.info(f'check dataset types registration OK: {ed[0]}')
        self.assertTrue(success)

    def check_spike_sorting_output(self, session_path):
        """ Check the spikes object """
        spikes_attributes = ['depths', 'amps', 'clusters', 'times', 'templates', 'samples']
        probe_folders = list(set([p.parent for p in session_path.joinpath(
            'alf').rglob('spikes.times.npy')]))
        for probe_folder in probe_folders:
            spikes = alf.io.load_object(probe_folder, 'spikes')
            self.assertTrue(np.max(spikes.times) > 1000)
            self.assertTrue(alf.io.check_dimensions(spikes) == 0)
            # check that it contains the proper keys
            self.assertTrue(set(spikes.keys()).issubset(set(spikes_attributes)))
            self.assertTrue(np.nanmin(spikes.depths) >= 0)
            self.assertTrue(np.nanmax(spikes.depths) <= 3840)
            self.assertTrue(80 < np.median(spikes.amps) * 1e6 < 200)  # we expect Volts

            """Check the clusters object"""
            clusters = alf.io.load_object(probe_folder, 'clusters')
            clusters_attributes = ['depths', 'channels', 'peakToTrough', 'amps', 'metrics',
                                   'uuids', 'waveforms', 'waveformsChannels']
            self.assertTrue(np.unique([clusters[k].shape[0] for k in clusters]).size == 1)
            self.assertTrue(set(clusters_attributes) == set(clusters.keys()))
            self.assertTrue(10 < np.nanmedian(clusters.amps) * 1e6 < 80)  # we expect Volts
            self.assertTrue(0 < np.median(np.abs(clusters.peakToTrough)) < 5)  # we expect ms

            """Check the channels object"""
            channels = alf.io.load_object(probe_folder, 'channels')
            channels_attributes = ['rawInd', 'localCoordinates']
            self.assertTrue(set(channels.keys()) == set(channels_attributes))

            """Check the template object"""
            templates = alf.io.load_object(probe_folder, 'templates')
            templates_attributes = ['amps', 'waveforms', 'waveformsChannels']
            self.assertTrue(set(templates.keys()) == set(templates_attributes))
            self.assertTrue(np.unique([templates[k].shape[0] for k in templates]).size == 1)
            # """Check the probes object"""
            probes_attributes = ['description', 'trajectory']
            probes = alf.io.load_object(session_path.joinpath('alf'), 'probes')
            self.assertTrue(set(probes.keys()) == set(probes_attributes))

            """check sample waveforms and make sure amplitudes check out"""
            swv = alf.io.load_object(probe_folder, '_phy_spikes_subset')
            swv_attributes = ['spikes', 'channels', 'waveforms']
            self.assertTrue(set(swv_attributes) == set(swv.keys()))
            iswv = 20001
            it = spikes.templates[swv.spikes[iswv]]
            _, ics, ict = np.intersect1d(swv.channels[iswv], templates.waveformsChannels[it],
                                         return_indices=True)
            iw = templates.waveforms[it][:, ict] != 0
            self.assertTrue(np.median(np.abs(swv.waveforms[iswv][:, ics][iw])) < 1e-3)
            self.assertTrue(np.median(np.abs(templates.waveforms[it][:, ict][iw])) < 1e-3)

            """(basic) check cross-references"""
            nclusters = clusters.depths.size
            nchannels = channels.rawInd.size
            ntemplates = templates.waveforms.shape[0]
            self.assertTrue(np.all(0 <= spikes.clusters) and
                            np.all(spikes.clusters <= (nclusters - 1)))
            self.assertTrue(np.all(0 <= spikes.templates) and
                            np.all(spikes.templates <= (ntemplates - 1)))
            self.assertTrue(np.all(0 <= clusters.channels) and
                            np.all(clusters.channels <= (nchannels - 1)))
            # check that the site positions channels match the depth with indexing
            self.assertTrue(np.all(clusters.depths == channels.localCoordinates[clusters.channels, 1]))

            """ compare against the cortexlab spikes Matlab code output if fixtures exist """
            for famps in session_path.joinpath(
                    'raw_ephys_data', probe_folder.parts[-1]).rglob('expected_amps_V_matlab.npy'):
                expected_amps = np.load(famps)
                # the difference is within 2 uV
                assert np.max(np.abs((spikes.amps * 1e6 - np.squeeze(expected_amps)))) < 2
                _logger.info('checked ' + '/'.join(famps.parts[-2:]))

            for fdepths in session_path.joinpath(
                    'raw_ephys_data', probe_folder.parts[-1]).rglob('expected_dephts_um_matlab.npy'):
                expected_depths = np.load(fdepths)
                # the difference is within 2 uV
                assert np.nanmax(np.abs((spikes.depths - np.squeeze(expected_depths)))) < .01
                _logger.info('checked ' + '/'.join(fdepths.parts[-2:]))
