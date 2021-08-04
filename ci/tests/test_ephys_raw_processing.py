import time
import numpy as np
import logging

from ibllib.ephys.neuropixel import trace_header
from ibllib.io import spikeglx
from ibllib.dsp import voltage, rms

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestEphysSpikeSortingPreProc(base.IntegrationTest):

    def test_pre_proc(self):
        # first makes sure the import works properly
        import pyfftw
        # sudo apt-get install -y libfftw3-dev
        # pip install pyFFTW
        _logger.info("pyfftw import ok")
        cbin_file = self.data_path.joinpath('ephys/ephys_spike_sorting/adc_test.ap.cbin')
        sr = spikeglx.Reader(cbin_file, open=True)
        bin_file = cbin_file.with_suffix('.bin')

        ts = time.time()
        voltage.decompress_destripe_cbin(sr)
        te = time.time() - ts
        _logger.info(f'Time elapsed: {te}, length file (secs): {sr.ns / sr.fs}')
        sr_out = spikeglx.Reader(bin_file)

        assert sr.shape == sr_out.shape

        sel_comp = slice(int(65536 * 0.4), int(65536 * 1.6))
        h = trace_header(version=1)
        # create the FFT stencils
        ncv = h['x'].size  # number of channels
        expected = voltage.destripe(sr[sel_comp, :ncv].T, fs=sr.fs).T
        diff = expected - sr_out[sel_comp, :ncv]

        assert np.min(20 * np.log10(rms(diff[10000:-10000, :], axis=0)
                                    / rms(sr_out[sel_comp, :ncv], axis=0))) < 35

        bin_file.unlink()
        # from easyqc.gui import viewseis
        # eqc_o = viewseis(sr[sel_comp, :ncv], si=1 / sr.fs, title='orig', taxis=0)
        # eqc_p = viewseis(sr_out[sel_comp, :ncv], si=1 / sr.fs, title='prod', taxis=0)
        # eqc_e = viewseis(expected, si=1 / sr.fs, title='expected', taxis=0)
        # eqc_d = viewseis(diff, si=1 / sr.fs, title='diff', taxis=0)
