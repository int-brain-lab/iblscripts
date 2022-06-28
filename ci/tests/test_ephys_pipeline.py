import logging
import shutil
import numpy as np
from unittest import mock
import json

import one.alf.io as alfio
from ibllib.pipes import local_server
import ibllib.pipes.ephys_preprocessing as ephys_tasks
from one.api import ONE

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestEphysSignatures(base.IntegrationTest):
    def setUp(self):
        self.folder_path = self.data_path.joinpath('ephys', 'ephys_signatures')

    def make_new_dataset(self):
        """helper function to use to create a new dataset"""
        folder_path = self.data_path.joinpath('ephys', 'ephys_signatures', 'RawEphysQC')
        for json_file in folder_path.rglob('result.json'):
            with open(json_file) as fid:
                result = json.load(fid)
            if result['outputs'] == True:
                for bin_file in json_file.parent.rglob('*ap.meta'):
                    bin_file.parent.joinpath("_iblqc_ephysChannels.labels.npy").touch()
                    print(bin_file)

    @base.disable_log(level=logging.ERROR, quiet=True)
    def assert_task_inputs_outputs(self, session_paths, EphysTask):
        for session_path in session_paths.iterdir():
            with self.subTest(session=session_path):
                task = EphysTask(session_path)
                if EphysTask.__name__ == 'SpikeSorting':
                    task.signature['input_files'], task.signature['output_files'] = \
                        task.spike_sorting_signature()
                task.get_signatures()
                output_status, _ = task.assert_expected(task.output_files)
                input_status, _ = task.assert_expected_inputs(raise_error=False)
                with open(session_path.joinpath('result.json'), 'r') as f:
                    result = json.load(f)
                self.assertEqual(output_status, result['outputs'],
                                 f"test failed outputs {EphysTask}, {session_path}")
                self.assertEqual(input_status, result['inputs'],
                                 f"test failed inputs {EphysTask}, {session_path}")

    def test_EphysAudio_signatures(self):
        EphysTask = ephys_tasks.EphysAudio
        session_paths = self.folder_path.joinpath('EphysAudio')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysCellQC_signatures(self):
        EphysTask = ephys_tasks.EphysCellsQc
        session_paths = self.folder_path.joinpath('EphysCellsQc')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysMtscomp_signatures(self):
        EphysTask = ephys_tasks.EphysMtscomp
        session_paths = self.folder_path.joinpath('EphysMtscomp', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysMtscomp', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysPassive_signatures(self):
        EphysTask = ephys_tasks.EphysPassive
        session_paths = self.folder_path.joinpath('EphysPassive', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysPassive', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysPulses_signatures(self):
        EphysTask = ephys_tasks.EphysPulses
        session_paths = self.folder_path.joinpath('EphysPulses', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysPulses', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysTrials_signatures(self):
        EphysTask = ephys_tasks.EphysTrials
        session_paths = self.folder_path.joinpath('EphysTrials', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysTrials', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysVideoCompress_signatures(self):
        EphysTask = ephys_tasks.EphysVideoCompress
        session_paths = self.folder_path.joinpath('EphysVideoCompress')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_EphysVideoSyncQC_signatures(self):
        EphysTask = ephys_tasks.EphysVideoSyncQc
        session_paths = self.folder_path.joinpath('EphysVideoSyncQc', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('EphysVideoSyncQc', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_RawEphysQC_signatures(self):
        EphysTask = ephys_tasks.RawEphysQC
        session_paths = self.folder_path.joinpath('RawEphysQC')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

    def test_SpikeSorting_signatures(self):
        EphysTask = ephys_tasks.SpikeSorting
        session_paths = self.folder_path.joinpath('SpikeSorting', '3A')
        self.assert_task_inputs_outputs(session_paths, EphysTask)

        session_paths = self.folder_path.joinpath('SpikeSorting', '3B')
        self.assert_task_inputs_outputs(session_paths, EphysTask)


class TestEphysPipeline(base.IntegrationTest):

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB, cache_dir=self.data_path / 'ephys', cache_rest=None)
        self.init_folder = self.data_path.joinpath('ephys', 'choice_world_init')
        if not self.init_folder.exists():
            raise FileNotFoundError()
        self.main_folder = self.data_path.joinpath('ephys', 'cortexlab', 'Subjects')
        self.session_path = self.main_folder.joinpath('KS022', '2019-12-10', '001')
        if self.main_folder.exists():
            shutil.rmtree(self.main_folder)
        self.main_folder.mkdir(exist_ok=True, parents=True)
        for ff in self.init_folder.rglob('*.*'):
            link = self.main_folder.joinpath(ff.relative_to(self.init_folder))
            if 'alf' in link.parts:
                if 'dlc' in link.name or 'ROIMotionEnergy' in link.name:
                    link.parent.mkdir(exist_ok=True, parents=True)
                    link.symlink_to(ff)
                continue
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)
        self.session_path.joinpath('raw_session.flag').touch()

    @mock.patch('ibllib.qc.camera.CameraQC')
    def test_pipeline_with_alyx(self, _):
        """
        Test the ephys pipeline exactly as it is supposed to run on the local servers
        We stub the QC as it requires a video file and loading frames takes a while.
        :param _: A stub CameraQC object
        :return:
        """

        one = self.one
        # first step is to remove the session and create it anew
        one.alyx.clear_rest_cache()
        eid = one.path2eid(self.session_path, query_type='remote')
        if eid is not None:
            one.alyx.rest('sessions', 'delete', id=eid)

        # create the jobs and run them
        raw_ds = local_server.job_creator(self.session_path,
                                          one=one, max_md5_size=1024 * 1024 * 20)
        one.alyx.clear_rest_cache()
        eid = one.path2eid(self.session_path, query_type='remote')
        self.assertFalse(eid is None)  # the session is created on the database
        # the flag has been erased
        self.assertFalse(self.session_path.joinpath('raw_session.flag').exists())

        subject_path = self.session_path.parents[2]
        tasks_dict = one.alyx.rest('tasks', 'list', session=eid, status='Waiting', no_cache=True)
        # # Hack for PostDLC to ignore DLC status for now, until DLC is actually in the pipeline
        # idx = tasks_dict.index([t for t in tasks_dict if t['name'] == 'EphysPostDLC'][0])
        # id_compress = [t['id'] for t in tasks_dict if t['name'] == 'EphysVideoCompress'][0]
        # tasks_dict[idx]['parents'] = [id_compress]
        # # Hack end, to be removed later
        for td in tasks_dict:
            print(td['name'])
        all_datasets = local_server.tasks_runner(
            subject_path, tasks_dict, one=one, max_md5_size=1024 * 1024 * 20, count=20)

        # check the trajectories and probe info
        self.assertTrue(len(one.alyx.rest('insertions', 'list', session=eid, no_cache=True)) == 2)
        traj = one.alyx.rest('trajectories', 'list',
                             session=eid, provenance='Micro-manipulator', no_cache=True)
        self.assertEqual(len(traj), 2)

        # check the spike sorting output on disk
        self.check_spike_sorting_output(self.session_path)

        # quick consistency test on trials length
        trials = alfio.load_object(self.session_path.joinpath('alf'), 'trials')
        assert alfio.check_dimensions(trials) == 0

        # check the registration of datasets
        dsets = one.alyx.rest('datasets', 'list', session=eid, no_cache=True)
        self.assertEqual(set([ds['url'][-36:] for ds in dsets]),
                         set([ds['id'] for ds in all_datasets + raw_ds]))

        nss = 2
        EXPECTED_DATASETS = [('_iblqc_ephysSpectralDensity.freqs', 4, 4),
                             ('_iblqc_ephysSpectralDensity.power', 4, 4),
                             ('_iblqc_ephysTimeRms.rms', 4, 4),
                             ('_iblqc_ephysTimeRms.timestamps', 4, 4),
                             ('_iblqc_ephysChannels.rawSpikeRates', 2, 2),
                             ('_iblqc_ephysChannels.RMS', 2, 2),
                             ('_iblqc_ephysChannels.labels', 2, 2),

                             ('_iblrig_Camera.frame_counter', 3, 3),
                             ('_iblrig_Camera.GPIO', 3, 3),
                             ('_iblrig_Camera.raw', 3, 3),
                             ('_iblrig_Camera.timestamps', 3, 3),
                             ('_iblrig_micData.raw', 1, 1),

                             ('_spikeglx_sync.channels', 2, 3),
                             ('_spikeglx_sync.polarities', 2, 3),
                             ('_spikeglx_sync.times', 2, 3),

                             ('camera.dlc', 3, 3),
                             ('camera.ROIMotionEnergy', 3, 3),
                             ('ROIMotionEnergy.position', 3, 3),
                             ('camera.times', 3, 3),
                             ('camera.features', 2, 2),
                             ('licks.times', 1, 1),

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
                             ('drift_depths.um', nss, nss),
                             ('drift.times', nss, nss),
                             ('drift.um', nss, nss),
                             ('spikes.amps', nss, nss),
                             ('spikes.clusters', nss, nss),
                             ('spikes.depths', nss, nss),
                             ('spikes.templates', nss, nss),
                             ('spikes.times', nss, nss),
                             ('templates.waveforms', nss, nss),
                             ('templates.waveformsChannels', nss, nss),
                             ('templates.amps', nss, nss),
                             ('_ibl_log.info', nss, nss),

                             ('trials.table', 1, 1),
                             ('trials.goCueTrigger_times', 1, 1),
                             ('trials.intervals', 1, 1),  # intervals_bpod
                             ('trials.stimOff_times', 1, 1),
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
        self.assertTrue(set(dids).issubset(set([ds['url'][-36:] for ds in dsets])))
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
        session_dict = one.alyx.rest('sessions', 'read', id=eid, no_cache=True)
        self.assertNotEqual('NOT_SET', session_dict['qc'], 'qc field not updated')
        extended = session_dict['extended_qc']
        self.assertTrue(any(k.startswith('_task_') for k in extended.keys()))
        # also check that the behaviour criterion was set
        self.assertTrue('behavior' in extended)
        # check that new DLC qc is added properly
        self.assertEqual(extended['_dlcLeft_pupil_diameter_snr'], [True, 12.066])
        self.assertEqual(extended['_dlcRight_pupil_diameter_snr'], [True, 6.53])
        # check that the probes insertions have the json field labeled properly
        pis = one.alyx.rest('insertions', 'list', session=eid, no_cache=True)
        for pi in pis:
            assert 'n_units' in pi['json']
        # check that tasks ran with proper status
        tasks_end = one.alyx.rest('tasks', 'list', session=eid, no_cache=True)
        for t in tasks_end:
            if t['name'] in ['EphysPassive']:
                continue
            assert t['status'] == 'Complete', f"{t['name']} FAILED and shouldn't have for this test"

    def check_spike_sorting_output(self, session_path):
        """ Check the spikes object """
        spikes_attributes = ['depths', 'amps', 'clusters', 'times', 'templates', 'samples']
        probe_folders = list(set([p.parent for p in session_path.joinpath(
            'alf').rglob('spikes.times.npy')]))
        for probe_folder in probe_folders:
            spikes = alfio.load_object(probe_folder, 'spikes')
            self.assertTrue(np.max(spikes.times) > 1000)
            self.assertTrue(alfio.check_dimensions(spikes) == 0)
            # check that it contains the proper keys
            self.assertTrue(set(spikes.keys()).issubset(set(spikes_attributes)))
            self.assertTrue(np.nanmin(spikes.depths) >= 0)
            self.assertTrue(np.nanmax(spikes.depths) <= 3840)
            self.assertTrue(80 < np.median(spikes.amps) * 1e6 < 200)  # we expect Volts

            """Check the clusters object"""
            clusters = alfio.load_object(probe_folder, 'clusters')
            clusters_attributes = ['depths', 'channels', 'peakToTrough', 'amps',
                                   'uuids', 'waveforms', 'waveformsChannels', 'metrics']
            self.assertTrue(np.unique([clusters[k].shape[0] for k in clusters]).size == 1)
            self.assertTrue(set(clusters_attributes) == set(clusters.keys()))
            self.assertTrue(10 < np.nanmedian(clusters.amps) * 1e6 < 80)  # we expect Volts
            self.assertTrue(0 < np.median(np.abs(clusters.peakToTrough)) < 5)  # we expect ms

            """Check the channels object"""
            channels = alfio.load_object(probe_folder, 'channels')
            channels_attributes = ['rawInd', 'localCoordinates']
            self.assertEqual(set(channels.keys()), set(channels_attributes))

            """Check the template object"""
            templates = alfio.load_object(probe_folder, 'templates')
            templates_attributes = ['amps', 'waveforms', 'waveformsChannels']
            self.assertTrue(set(templates.keys()) == set(templates_attributes))
            self.assertTrue(np.unique([templates[k].shape[0] for k in templates]).size == 1)
            # """Check the probes object"""
            probes_attributes = ['description', 'trajectory']
            probes = alfio.load_object(session_path.joinpath('alf'), 'probes')
            self.assertTrue(set(probes.keys()) == set(probes_attributes))

            """check sample waveforms and make sure amplitudes check out"""
            swv = alfio.load_object(probe_folder, 'spikes_subset')
            swv_attributes = ['spikes', 'channels', 'waveforms']
            self.assertTrue(set(swv_attributes) == set(swv.keys()))
            iswv = 10000
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

    def tearDown(self) -> None:
        if self.main_folder.exists():
            shutil.rmtree(self.main_folder.parent)


if __name__ == "__main__":
    import unittest
    unittest.main(exit=False)
