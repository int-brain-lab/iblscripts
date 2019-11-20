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
        if not self.folder3a.exists():
            return
        # the assertion is already in the files
        # test both residual smoothed and linear
        for ses_path in self.folder3a.rglob('raw_ephys_data'):
            # we switched to sync using frame2ttl on November 2019
            channel = 12 if '2019-11-05' in str(ses_path) else 2
            self.assertTrue(sync_probes.version3A(ses_path.parent, linear=True, tol=2,
                                                  display=False))
            self.assertTrue(sync_probes.version3A(ses_path.parent, display=True))
            dt = _check_session_sync(ses_path, channel=channel)
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
        # the first step is to construct list arrays with probe sync
        sync_file = ef.ap.parent.joinpath(ef.ap.name.replace('.ap.', '.sync.')).with_suffix('.npy')
        t = sync_events.times[sync_events.channels == channel]
        tsync = sync_probes.apply_sync(sync_file, t, forward=True)
        tprobe.append(t)
        tinterp.append(tsync)
        # the second step is to make sure sample / time_ref files match time / time_ref files
        ts_file = ef.ap.parent.joinpath(ef.ap.name.replace('.ap.', '.timestamps.')
                                        ).with_suffix('.npy')
        fs = spikeglx._get_fs_from_meta(spikeglx.read_meta_data(ef.ap.with_suffix('.meta')))
        tstamp = sync_probes.apply_sync(ts_file, t * fs, forward=True)
        assert(np.all(tstamp - tsync < 1e-12))
    return tinterp[0] - tinterp[1]
