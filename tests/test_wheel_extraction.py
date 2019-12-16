from pathlib import Path
import numpy as np
import unittest
import shutil
import scipy.interpolate

import alf.io
from ibllib.io.extractors import ephys_fpga, training_wheel, ephys_trials


def compare_wheel_fpga_behaviour(session_path):
    alf_path = session_path.joinpath('alf')
    shutil.rmtree(alf_path, ignore_errors=True)

    sync_path = session_path.joinpath(r'raw_ephys_data')
    sync = alf.io.load_object(sync_path, '_spikeglx_sync', short_keys=True)

    chmap = ephys_fpga.CHMAPS['3B']['nidq']
    fpga_wheel = ephys_fpga.extract_wheel_sync(sync, chmap=chmap, save=False)
    bpod_wheel = training_wheel.get_wheel_data(session_path, save=False)
    ephys_trials.extract_all(session_path, output_path=alf_path, save=True)
    ephys_fpga.extract_behaviour_sync(sync, output_path=alf_path, chmap=chmap, save=True)
    dt = ephys_fpga.align_with_bpod(session_path)
    # resample both traces to the same rate and compute correlation coeff
    tmin = max([np.min(fpga_wheel['re_ts']), np.min(bpod_wheel['re_ts'] + dt)])
    tmax = min([np.max(fpga_wheel['re_ts']), np.max(bpod_wheel['re_ts'] + dt)])
    tscale = np.linspace(tmin, tmax, 500)
    fw = scipy.interpolate.interp1d(fpga_wheel['re_ts'], fpga_wheel['re_pos'])(tscale)
    bw = scipy.interpolate.interp1d(bpod_wheel['re_ts'] + dt, bpod_wheel['re_pos'])(tscale)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpga_wheel['re_ts'], fpga_wheel['re_pos'])
    plt.plot(bpod_wheel['re_ts'] + dt, bpod_wheel['re_pos'])
    plt.plot(tscale, fw)
    plt.plot(tscale, bw)
    # test that the extractions match
    return fpga_wheel, bpod_wheel, fw, bw


# class TestWheelExtractionSimple(unittest.TestCase):
#
#     def setUp(self) -> None:
#         self.session_path = Path('/mnt/s0/Data/IntegrationTests/wheel/three_clockwise_revolutions')
#         if not self.session_path.exists():
#             return
#
#     def test_three_clockwise_revolutions_fpga(self):
#         fpga_wheel, bpod_wheel, fw, bw = compare_wheel_fpga_behaviour(self.session_path)
#         self.assertTrue(np.all(np.abs(fw - bw) < 0.1))
#         # test that the units are in radians: we expect around 9 revolutions clockwise
#         self.assertTrue(0.95 < fpga_wheel['re_pos'][-1] / -(2 * 3.14 * 9) < 1.05)


class TestWheelExtractionSession(unittest.TestCase):

    def setUp(self) -> None:
        self.session_path = Path('/datadisk/Data/IntegrationTests/wheel/KS016_2019_12_05')
        if not self.session_path.exists():
            return

    def test_wheel_extraction_session(self):
        fpga_wheel, bpod_wheel, fw, bw = compare_wheel_fpga_behaviour(self.session_path)
        self.assertTrue(np.all(np.abs(fw - bw) < 0.1))
        # test that the units are in radians: we expect around 9 revolutions clockwise
        self.assertTrue(0.95 < fpga_wheel['re_pos'][-1] / -(2 * 3.14 * 9) < 1.05)
