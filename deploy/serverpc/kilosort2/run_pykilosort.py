from pathlib import Path
import argparse
import datetime
import logging
import shutil

import numpy as np
from ibllib.io import spikeglx
from ibllib.ephys import spikes, neuropixel
from pykilosort import add_default_handler, run, Bunch, __version__

_logger = logging.getLogger("pykilosort")


def _sample2v(ap_file):
    md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
    s2v = spikeglx._conversion_sample2v_from_meta(md)
    return s2v["ap"][0]


def run_spike_sorting_ibl(bin_file, delete=True, version=1, ks_output_dir=None, alf_path=None, log_level='INFO'):
    """
    This runs the spike sorting and outputs the raw pykilosort without ALF conversion
    bin_file: binary file full path to run
    delete (true): removes temporary data after run (.pykilosort folder)
    version (1): Neuropixel version
    ks_output_dir (None): location of ks output data, defaults to output folder next to the bin file
    alf_path (None): if specified, performs ks to ALF conversion in the specified folder
    """
    START_TIME = datetime.datetime.now()
    bin_file = Path(bin_file)
    log_file = bin_file.parent.joinpath(f"_{START_TIME.isoformat()}_kilosort.log")
    log_file.parent.mkdir(exist_ok=True, parents=True)

    add_default_handler(level=log_level)
    add_default_handler(level=log_level, filename=log_file)

    h = neuropixel.trace_header(version=version)
    probe = Bunch()
    probe.NchanTOT = 385
    probe.chanMap = np.arange(384)
    probe.xc = h['x']
    probe.yc = h['y']
    probe.kcoords = np.zeros(384)

    try:
        _logger.info(f"Starting Pykilosort version {__version__}, output in {bin_file.parent}")
        run(bin_file, probe=probe, dir_path=bin_file.parent, output_dir=ks_output_dir)
        if delete:
            shutil.rmtree(bin_file.parent.joinpath(".kilosort"))
    except Exception as e:
        _logger.exception("Error in the main loop")
        raise e

    [_logger.removeHandler(h) for h in _logger.handlers]
    ks_output_dir = Path(ks_output_dir) or bin_file.parent.joinpath('output')
    shutil.move(log_file, ks_output_dir.joinpath('spike_sorting_pykilosort.log'))

    # convert the pykilosort output to ALF IBL format
    if alf_path is not None:
        s2v = _sample2v(bin_file)
        alf_path.mkdir(exist_ok=True, parents=True)
        spikes.ks2_to_alf(ks_output_dir, bin_file, alf_path, ampfactor=s2v)


if __name__ == "__main__":
    """
    directory structure example:
        input file: ./CSH_ZAD_029/2020-09-09/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.nidq.cbin
        session_path: ./CSH_ZAD_029/2020-09-09/001
        alf_dir: ./CSH_ZAD_029/2020-09-09/001/alf
        scratch_dir: /mnt/h0/CSH_ZAD_029_2020-09-09_001_probe00
    """

    DELETE = False
    parser = argparse.ArgumentParser(description='Run Kilosort for a bin AP file')
    parser.add_argument('cbin_file', help='compressed binary file with *.cbin extension')
    parser.add_argument('scratch_dir', help='scratch directory')
    args = parser.parse_args()

    # handles input arguments
    cbin_file = Path(args.cbin_file)
    scratch_dir = Path(args.scratch_dir).joinpath()
    assert cbin_file.exists(), f"{cbin_file} not found !"
    _logger.info(f"Spike sorting {cbin_file}")

    scratch_dir.mkdir(exist_ok=True, parents=True)

    # if PRE_PROC:
    #     # run pre-processing
    #     import ibllib.dsp.voltage as voltage
    #     bin_destriped = scratch_dir.joinpath(cbin_file.name).with_suffix('.bin')
    #     if bin_destriped.exists():
    #         print('skip pre-proc')
    #     else:
    #         voltage.decompress_destripe_cbin(sr=cbin_file, output_file=bin_destriped)
    #     if not bin_destriped.with_suffix('.meta').exists():
    #         bin_destriped.with_suffix('.meta').symlink_to(cbin_file.with_suffix('.meta'))
    #
    #     # run pykilosort
    #     run_spike_sorting_ibl(bin_destriped, delete=DELETE, version=1)
    run_spike_sorting_ibl(cbin_file, delete=DELETE, version=1)

    # # mop-up all temporary files
    # shutil.rmtree(bin_destriped.parent)
