from pathlib import Path
import numpy as np
import unittest

import matplotlib.pyplot as plt
from ibllib.io.extractors import ephys_fpga, training_wheel


class TestWheelExtraction(unittest.TestCase):

    def setUp(self) -> None:
        self.session_path = Path('/mnt/s0/Data/IntegrationTests/wheel/three_clockwise_revolutions')
        if not self.session_path.exists():
            return

    def test_three_clockwise_revolutions_fpga(self):
        sync = ephys_fpga.extract_sync(self.session_path)[0]
        chmap = ephys_fpga.CHMAPS['3B']['nidq']
        fpga_wheel = ephys_fpga.extract_wheel_sync(sync, chmap=chmap, save=False)
        bpod_wheel = training_wheel.get_wheel_data(self.session_path, save=False)
        plt.plot(fpga_wheel['re_ts'], fpga_wheel['re_pos'])

        plt.plot(bpod_wheel['re_ts'], bpod_wheel['re_pos'])
