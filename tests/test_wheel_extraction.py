from pathlib import Path
import numpy as np
import unittest
import shutil
import scipy.interpolate
import matplotlib.pyplot as plt

import alf.io
from ibllib.io.extractors import ephys_fpga, training_wheel, ephys_trials
from ibllib.io.raw_data_loaders import load_settings

DISPLAY = False


def compare_wheel_fpga_behaviour(session_path, display=DISPLAY):
    alf_path = session_path.joinpath('alf')
    shutil.rmtree(alf_path, ignore_errors=True)
    sync_path = session_path.joinpath(r'raw_ephys_data')
    sync = alf.io.load_object(sync_path, '_spikeglx_sync', short_keys=True)
    chmap = ephys_fpga.CHMAPS['3B']['nidq']
    fpga_wheel = ephys_fpga.extract_wheel_sync(sync, chmap=chmap, save=False)
    bpod_wheel = training_wheel.get_wheel_data(session_path, save=False, display=display)
    ephys_trials.extract_all(session_path, output_path=alf_path, save=True)
    ephys_fpga.extract_behaviour_sync(sync, output_path=alf_path, chmap=chmap, save=True)
    bpod2fpga = ephys_fpga.align_with_bpod(session_path)
    # resample both traces to the same rate and compute correlation coeff
    bpod_wheel['re_ts'] = bpod2fpga(bpod_wheel['re_ts'])
    tmin = max([np.min(fpga_wheel['re_ts']), np.min(bpod_wheel['re_ts'])])
    tmax = min([np.max(fpga_wheel['re_ts']), np.max(bpod_wheel['re_ts'])])
    wheel = {'tscale': np.arange(tmin, tmax, 0.01)}
    wheel['fpga'] = scipy.interpolate.interp1d(fpga_wheel['re_ts'], fpga_wheel['re_pos'])(wheel['tscale'])
    wheel['bpod'] = scipy.interpolate.interp1d(bpod_wheel['re_ts'], bpod_wheel['re_pos'])(wheel['tscale'])
    if display:
        plt.figure()
        plt.plot(fpga_wheel['re_ts'] - bpod2fpga(0), fpga_wheel['re_pos'], '*')
        plt.plot(bpod_wheel['re_ts'] - bpod2fpga(0), bpod_wheel['re_pos'], '.')
    return fpga_wheel, bpod_wheel, wheel


class TestWheelExtractionSimpleEphys(unittest.TestCase):

    def setUp(self) -> None:
        self.session_path = Path(
            '/mnt/s0/Data/IntegrationTests/wheel/ephys/three_clockwise_revolutions')
        if not self.session_path.exists():
            return

    def test_three_clockwise_revolutions_fpga(self):
        fpga_wheel, bpod_wheel, wheel = compare_wheel_fpga_behaviour(self.session_path)
        self.assertTrue(np.all(np.abs(wheel['fpga'] - wheel['bpod']) < 0.1))
        # test that the units are in radians: we expect around 9 revolutions clockwise
        self.assertTrue(0.95 < fpga_wheel['re_pos'][-1] / -(2 * 3.14 * 9) < 1.05)


class TestWheelExtractionSessionEphys(unittest.TestCase):

    def setUp(self) -> None:
        self.session_path = Path('/datadisk/Data/IntegrationTests/wheel/ephys/KS016_2019_12_05')
        if not self.session_path.exists():
            return

    def test_wheel_extraction_session(self):
        fpga_wheel, bpod_wheel, wheel = compare_wheel_fpga_behaviour(self.session_path)
        # makes sure that the HF component matches
        from scipy.signal import butter, filtfilt
        b, a = butter(3, 0.0001, btype='high', analog=False)
        fpga = filtfilt(b, a, wheel['fpga'])
        bpod = filtfilt(b, a, wheel['bpod'])
        # plt.figure()
        # plt.plot(wheel['tscale'], fpga)
        # plt.plot(wheel['tscale'], bpod)
        self.assertTrue(np.all(np.abs(fpga - bpod < 0.1)))


class TestWheelExtractionTraining(unittest.TestCase):

    def setUp(self) -> None:
        self.root_path = Path('/datadisk/Data/IntegrationTests/wheel/training')
        if not self.root_path.exists():
            return

    def test_wheel_extraction_training(self):
        for rbf in self.root_path.rglob('raw_behavior_data'):
            session_path = alf.io.get_session_path(rbf)
            bpod_wheel = training_wheel.get_wheel_data(session_path, save=False)
            self.assertTrue(bpod_wheel['re_ts'].size)
