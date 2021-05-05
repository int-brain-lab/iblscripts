import logging
import shutil
from pathlib import Path
import numpy as np
import tempfile
from unittest import mock

import alf.io
from ibllib.pipes import local_server
from oneibl.one import ONE

from ci.tests import base

CACHE_DIR = tempfile.TemporaryDirectory()
_logger = logging.getLogger('ibllib')


class TestEphysPipeline(base.IntegrationTest):

    def setUp(self) -> None:
        self.session_path = self.data_path.joinpath("ephys/choice_world/KS022/2019-12-10/001")
        # one = ONE(base_url='http://localhost:8000')
        self.one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
                       username='test_user', password='TapetesBloc18',
                       cache_dir=Path(CACHE_DIR.name))
        self.init_folder = self.data_path.joinpath('ephys', 'choice_world_init')
        if not self.init_folder.exists():
            return
        self.main_folder = self.data_path.joinpath('ephys', 'choice_world')
        if self.main_folder.exists():
            shutil.rmtree(self.main_folder)
        self.main_folder.mkdir(exist_ok=True)
        for ff in self.init_folder.rglob('*.*'):
            link = self.main_folder.joinpath(ff.relative_to(self.init_folder))
            if 'alf' in link.parts:
                continue
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)
        self.session_path.joinpath('raw_session.flag').touch()

    @mock.patch('ibllib.qc.camera.CameraQC')
    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_pipeline_with_alyx(self, mock_vc, _):
        """
        Test the ephys pipeline exactly as it is supposed to run on the local servers
        We stub the QC as it requires a video file and loading frames takes a while.
        We mock the OpenCV video capture class as the camera timestamp extractor inspects the
        video length.
        :param mock_vc: A mock OpenCV VideoCapture class for returning the video length
        :param _: A stub CameraQC object
        :return:
        """
        mock_vc().get.return_value = 40000  # Need a value for number of frames in video
        one = self.one
        # first step is to remove the session and create it anew
        eid = one.eid_from_path(self.session_path, use_cache=False)
        if eid is not None:
            one.alyx.rest('sessions', 'delete', id=eid)

        # create the jobs and run them
        raw_ds = local_server.job_creator(self.session_path,
                                          one=one, max_md5_size=1024 * 1024 * 20)
        eid = one.eid_from_path(self.session_path, use_cache=False)
        self.assertFalse(eid is None)  # the session is created on the database
        # the flag has been erased
        self.assertFalse(self.session_path.joinpath('raw_session.flag').exists())

        eid = one.eid_from_path(self.session_path, use_cache=False)
        subject_path = self.session_path.parents[2]
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
        self.check_spike_sorting_output(self.session_path)

        # quick consistency test on trials length
        trials = alf.io.load_object(self.session_path.joinpath('alf'), 'trials')
        assert alf.io.check_dimensions(trials) == 0

        # check the registration of datasets
        dsets = one.alyx.rest('datasets', 'list', session=eid)
        self.assertEqual(set([ds['url'][-36:] for ds in dsets]),
                         set([ds['id'] for ds in all_datasets + raw_ds]))

        nss = 2
        EXPECTED_DATASETS = [('_iblqc_ephysSpectralDensity.freqs', 4, 4),
                             ('_iblqc_ephysSpectralDensity.power', 4, 4),
                             ('_iblqc_ephysTimeRms.rms', 4, 4),
                             ('_iblqc_ephysTimeRms.timestamps', 4, 4),

                             ('_iblrig_Camera.frame_counter', 3, 3),
                             ('_iblrig_Camera.GPIO', 3, 3),
                             ('_iblrig_Camera.raw', 3, 3),
                             ('_iblrig_Camera.timestamps', 3, 3),
                             ('_iblrig_micData.raw', 1, 1),


                             ('_spikeglx_sync.channels', 2, 3),
                             ('_spikeglx_sync.polarities', 2, 3),
                             ('_spikeglx_sync.times', 2, 3),

                             ('camera.times', 3, 3),

                             ('kilosort.whitening_matrix', nss, nss),
                             ('_kilosort_raw.output', nss, nss),
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
                             # Min is 0 because this session fails extraction proper
                             # extraction test in test_ephys_passive
                             ('_ibl_passivePeriods.intervalsTable', 0, 1),
                             ('_ibl_passiveRFM.times', 0, 1),
                             ('_ibl_passiveGabor.table', 0, 1),
                             ('_ibl_passiveStims.table', 0, 1),
                             ]
        # check that we indeed find expected number of datasets after registration
        # for this we need to get the unique set of datasets
        dids = np.array([d['id'] for d in all_datasets])
        assert set(dids).issubset(set([ds['url'][-36:] for ds in dsets]))
        dtypes = sorted([ds['dataset_type'] for ds in dsets])
        success = True
        for ed in EXPECTED_DATASETS:
            count = sum([1 if ed[0] == dt else 0 for dt in dtypes])
            if not ed[1] <= count <= ed[2]:
                _logger.critical(f'missing dataset types: {ed[0]} found {count}, '
                                 f'expected between [{ed[1]} and {ed[2]}]')
                success = False
            else:
                _logger.info(f'check dataset types registration OK: {ed[0]}')
        self.assertTrue(success)
        # check that the task QC was successfully run
        session_dict = one.alyx.rest('sessions', 'read', id=eid)
        self.assertNotEqual('NOT_SET', session_dict['qc'], 'qc field not updated')
        extended = session_dict['extended_qc']
        self.assertTrue(any(k.startswith('_task_') for k in extended.keys()))
        # also check that the behaviour criterion was set
        assert 'behavior' in extended
        # check that the probes insertions have the json field labeled properly
        pis = one.alyx.rest('insertions', 'list', session=eid)
        for pi in pis:
            assert('n_units' in pi['json'])

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
            clusters_attributes = ['depths', 'channels', 'peakToTrough', 'amps',
                                   'uuids', 'waveforms', 'waveformsChannels', 'metrics']
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
            swv = alf.io.load_object(probe_folder, 'spikes_subset')
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
            self.assertTrue(
                np.all(clusters.depths == channels.localCoordinates[clusters.channels, 1])
            )

            """ compare against the cortexlab spikes Matlab code output if fixtures exist """
            for famps in session_path.joinpath(
                    'raw_ephys_data', probe_folder.parts[-1]).rglob('expected_amps_V_matlab.npy'):
                expected_amps = np.load(famps)
                # the difference is within 2 uV
                assert np.max(np.abs((spikes.amps * 1e6 - np.squeeze(expected_amps)))) < 2
                _logger.info('checked ' + '/'.join(famps.parts[-2:]))

            folder = session_path.joinpath('raw_ephys_data', probe_folder.parts[-1])
            for fdepths in folder.rglob('expected_dephts_um_matlab.npy'):
                expected_depths = np.load(fdepths)
                # the difference is within 2 uV
                assert np.nanmax(np.abs((spikes.depths - np.squeeze(expected_depths)))) < .01
                _logger.info('checked ' + '/'.join(fdepths.parts[-2:]))


if __name__ == "__main__":
    import unittest
    unittest.main(exit=False)
