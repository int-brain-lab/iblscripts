import unittest
from pathlib import Path
import logging
import numpy as np
import shutil

import alf.io
from ibllib.io.extractors import ephys_fpga, ephys_trials
_logger = logging.getLogger('ibllib')

BPOD_FILES = [
    '_ibl_trials.choice.npy',
    '_ibl_trials.contrastLeft.npy',
    '_ibl_trials.contrastRight.npy',
    '_ibl_trials.feedbackType.npy',
    '_ibl_trials.goCueTrigger_times_bpod.npy',
    '_ibl_trials.intervals_bpod.npy',
    '_ibl_trials.itiDuration.npy',
    '_ibl_trials.probabilityLeft.npy',
    '_ibl_trials.response_times_bpod.npy',
    '_ibl_trials.rewardVolume.npy'
]

FPGA_FILES = [
    '_ibl_trials.goCue_times.npy',
    '_ibl_trials.stimOn_times.npy',
    '_ibl_trials.intervals.npy',
    '_ibl_trials.feedback_times.npy',
]

ALIGN_BPOD_FPGA_FILES = [
    '_ibl_trials.goCueTrigger_times.npy',
    '_ibl_trials.response_times.npy',
]


class TestEphysTaskExtraction(unittest.TestCase):

    def test_task_extraction_output(self):
        self.root_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/choice_world_init')
        if not self.root_folder.exists():
            return
        self.sessions = [f.parent for f in self.root_folder.rglob('raw_ephys_data')]
        for session_path in self.sessions:
            alf_path = session_path.joinpath('alf')
            shutil.rmtree(alf_path, ignore_errors=True)
            sync, chmap = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)
            bpod_behaviour = ephys_trials.extract_all(session_path, save=True)
            # check that the output is complete
            for f in BPOD_FILES:
                self.assertTrue(alf_path.joinpath(f).exists())
            fpga_behaviour = ephys_fpga.extract_behaviour_sync(sync, chmap=chmap,
                                                               output_path=alf_path, save=True)
            # check that the output is complete
            for f in FPGA_FILES:
                self.assertTrue(alf_path.joinpath(f).exists())
            # checks that all matrices in fpga_behaviour have the same number of trials
            self.assertTrue(np.size(np.unique([fpga_behaviour[k].shape[0]
                                               for k in fpga_behaviour])) == 1)
            # check dimensions after alf load
            self.assertEqual(alf.io.check_dimensions(alf.io.load_object(alf_path, '_ibl_trials'))
                             , 0)
            # all trials have either valve open or error tone in and are mutually exclusive
            self.assertTrue(np.all(np.isnan(fpga_behaviour['valve_open']
                                            * fpga_behaviour['error_tone_in'])))
            self.assertTrue(np.all(np.logical_xor(np.isnan(fpga_behaviour['valve_open']),
                                                  np.isnan(fpga_behaviour['error_tone_in']))))
            # align with the bpod
            ephys_fpga.align_with_bpod(session_path)
            for f in ALIGN_BPOD_FPGA_FILES:
                self.assertTrue(alf_path.joinpath(f).exists())
            shutil.rmtree(alf_path, ignore_errors=True)
