import unittest
from pathlib import Path
import shutil

import numpy as np

import alf.io
from ibllib.pipes import ephys_preprocessing
from oneibl.one import ONE


PATH_TESTS = Path('/mnt/s0/Data/IntegrationTests')
SESSION_PATH = PATH_TESTS.joinpath("ephys/choice_world/KS022/2019-12-10/001")
one = ONE(base_url='http://localhost:8000')


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
        self.assertTrue(np.min(spikes.depths) >= 0)
        self.assertTrue(np.max(spikes.depths) <= 3840)
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
        templates_attributes = ['waveforms', 'waveformsChannels']
        self.assertTrue(set(templates.keys()) == set(templates_attributes))
        self.assertTrue(np.unique([templates[k].shape[0] for k in templates]).size == 1)
        # """Check the probes object"""
        probes_attributes = ['description', 'trajectory']
        probes = alf.io.load_object(session_path.joinpath('alf'), 'probes')
        self.assertTrue(set(probes.keys()) == set(probes_attributes))

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

    def tearDown(self):
        shutil.rmtree(self.main_folder)
        
    def test_pipeline_with_alyx(self):
        eid = one.eid_from_path(SESSION_PATH)

        # prepare by deleting all jobs/tasks related
        jobs = one.alyx.rest('jobs', 'list', session=eid)
        tasks = list(set([j['task'] for j in jobs]))
        [one.alyx.rest('tasks', 'delete', id=task) for task in tasks]

        # create tasks and jobs from scratch
        ephys_pipe = ephys_preprocessing.EphysExtractionPipeline(SESSION_PATH, one=one)
        ephys_pipe.make_graph(show=False)
        alyx_tasks = ephys_pipe.init_alyx_tasks()
        self.assertTrue(len(alyx_tasks) == len(ephys_pipe.jobs))
        alyx_jobs = ephys_pipe.register_alyx_jobs()
        self.assertTrue(len(alyx_jobs) == len(ephys_pipe.jobs))

        # run the pipeline
        job_deck, all_datasets = ephys_pipe.run()
        self.check_spike_sorting_output(self, SESSION_PATH)

