import unittest
from pathlib import Path
import logging
import numpy as np

from ibllib.io.extractors import ephys_fpga
_logger = logging.getLogger('ibllib')


class TestEphysTaskExtraction(unittest.TestCase):

    def test_task_extraction_output(self):
        self.root_folder = Path('/mnt/s0/Data/IntegrationTests/ephys/choice_world_init')
        if not self.root_folder.exists():
            return
        self.sessions = [f.parent for f in self.root_folder.rglob('raw_ephys_data')]
        for session_path in self.sessions:
            sync, chmap = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)
            fpga_behaviour = ephys_fpga.extract_behaviour_sync(sync, chmap=chmap, save=False)

            # checks that all matrices in fpga_behaviour have the same number of trials
            self.assertTrue(np.size(np.unique([fpga_behaviour[k].shape[0]
                                               for k in fpga_behaviour])) == 1)
            # all trials have either valve open or error tone in and are mutually exclusive
            self.assertTrue(np.all(np.isnan(fpga_behaviour['valve_open']
                                            * fpga_behaviour['error_tone_in'])))
            self.assertTrue(np.all(np.logical_xor(np.isnan(fpga_behaviour['valve_open']),
                                                  np.isnan(fpga_behaviour['error_tone_in']))))
