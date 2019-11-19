import unittest
from pathlib import Path

import numpy as np

import alf.io
from ibllib.io import spikeglx
import ibllib.ephys.sync_probes as sync_probes

INTEGRATION_TEST_FOLDER = Path("/mnt/s0/Data/IntegrationTests")


class TestEphysCheckList(unittest.TestCase):
    def setUp(self):
        self.folder3a = INTEGRATION_TEST_FOLDER.joinpath('ephys/sync/sync_3A')
        self.folder3b = INTEGRATION_TEST_FOLDER.joinpath('ephys/sync/sync_3B')

    def test_sync_3A(self):
        return
        if not self.folder3a.exists():
            return
        # the assertion is already in the files
        # test both residual smoothed and linear
        for ses_path in self.folder3a.rglob('raw_ephys_data'):
            self.assertTrue(sync_probes.version3A(ses_path.parent, linear=True, tol=2,
                                                  display=False))
            self.assertTrue(sync_probes.version3A(ses_path.parent, display=False))
            dt = _check_session_sync(ses_path, channel=2)
            self.assertTrue(np.all(np.abs(dt * 30000) < 2))

    def test_sync_3B(self):
        # the assertion is already in the files
        if not self.folder3b.exists():
            return
        for ses_path in self.folder3b.rglob('raw_ephys_data'):
            self.assertTrue(sync_probes.version3B(ses_path.parent, linear=True, tol=10,
                                                  display=False))
            self.assertTrue(sync_probes.version3B(ses_path.parent, display=False))
            dt = _check_session_sync(ses_path, 6)
            # import matplotlib.pyplot as plt
            # plt.plot(dt * 30000)
            self.assertTrue(np.all(np.abs(dt * 30000) < 2))


def _check_session_sync(ses_path, channel):
    """
    Resync the original cam pulses
    :param ses_path:
    :return:
    """
    efiles = spikeglx.glob_ephys_files(ses_path)
    tprobe = []
    tinterp = []
    for ef in efiles:
        if not ef.get('ap'):
            continue
        sync_events = alf.io.load_object(ef.ap.parent, '_spikeglx_sync', short_keys=True)
        sync_file = ef.ap.parent.joinpath(ef.ap.name.replace('.ap.', '.sync.')).with_suffix('.npy')
        t = sync_events.times[sync_events.channels == channel]
        tprobe.append(t)
        tinterp.append(sync_probes.apply_sync(sync_file, t, forward=True))
    return tinterp[0] - tinterp[1]
