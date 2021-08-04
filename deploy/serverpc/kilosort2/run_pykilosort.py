from pathlib import Path
import argparse
import datetime
import logging
import shutil

import numpy as np
from one.alf.files import get_session_path
from ibllib.io import spikeglx
import ibllib.dsp.voltage as voltage
from ibllib.ephys import spikes, neuropixel
from pykilosort import add_default_handler, run, Bunch

_logger = logging.getLogger("pykilosort")


def _sample2v(ap_file):
    md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
    s2v = spikeglx._conversion_sample2v_from_meta(md)
    return s2v["ap"][0]


def run_spike_sorting_ibl(bin_file, delete=True, version=1, alf_path=None):
    """
    This runs the spike sorting and outputs the raw pykilosort without ALF conversion
    """
    START_TIME = datetime.datetime.now()
    bin_file = Path(bin_file)
    log_file = bin_file.parent.joinpath(f"{START_TIME.isoformat()}_kilosort.log")
    log_file.parent.mkdir(exist_ok=True, parents=True)

    add_default_handler(level='DEBUG')
    add_default_handler(level='DEBUG', filename=log_file)

    h = neuropixel.trace_header(version=version)
    probe = Bunch()
    probe.NchanTOT = 385
    probe.chanMap = np.arange(384)
    probe.xc = h['x']
    probe.yc = h['y']
    probe.kcoords = np.zeros(384)

    try:
        _logger.info(f"Starting KS, output in {bin_file.parent}")
        run(bin_file, probe=probe, dir_path=bin_file.parent, n_channels=probe.NchanTOT, dtype=np.int16, sample_rate=3e4)
        if delete:
            shutil.rmtree(bin_file.parent.joinpath(".kilosort"))
    except Exception as e:
        _logger.exception("Error in the main loop")
        raise e

    [_logger.removeHandler(h) for h in _logger.handlers]
    shutil.move(log_file, bin_file.parent.joinpath('output', 'spike_sorting_pykilosort.log'))

    # convert the pykilosort output to ALF IBL format
    if alf_path is not None:
        s2v = _sample2v(bin_file)
        alf_path.mkdir(exist_ok=True, parents=True)
        spikes.ks2_to_alf(bin_file.parent.joinpath('output'), bin_destriped, alf_dir, ampfactor=s2v)


if __name__ == "__main__":
    """
    directory structure example:
        input file: ./CSH_ZAD_029/2020-09-09/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.nidq.cbin
        session_path: ./CSH_ZAD_029/2020-09-09/001
        alf_dir: ./CSH_ZAD_029/2020-09-09/001/alf
        scratch dir: /mnt/h0
        temp_dir: /mnt/h0/CSH_ZAD_029_2020-09-09_001_probe00
    """

    DELETE = True
    parser = argparse.ArgumentParser(description='Run Kilosort for a bin AP file')
    parser.add_argument('cbin_file', help='compressed binary file with *.cbin extension')
    parser.add_argument('scratch_dir', help='scratch directory')
    args = parser.parse_args()

    # handles input arguments
    cbin_file = Path(args.cbin_file)
    scratch_dir = Path(args.scratch_dir).joinpath()
    assert cbin_file.exists(), f"{cbin_file} not found !"
    _logger.info(f"Spike sorting {cbin_file}")

    # create the temporary directory structure
    session_path = get_session_path(cbin_file)
    probe_name = cbin_file.parts[-2]
    alf_dir = session_path.joinpath('alf', probe_name)
    temp_dir = scratch_dir.joinpath('_'.join(list(session_path.parts[-3:]) + [probe_name]))
    temp_dir.mkdir(exist_ok=True, parents=True)

    # run pre-processing
    bin_destriped = temp_dir.joinpath(cbin_file.name).with_suffix('.bin')
    if bin_destriped.exists():
        print('skip pre-proc')
    else:
        voltage.decompress_destripe_cbin(sr=cbin_file, output_file=bin_destriped)
    if not bin_destriped.with_suffix('.meta').exists():
        bin_destriped.with_suffix('.meta').symlink_to(cbin_file.with_suffix('.meta'))

    # run pykilosort
    run_spike_sorting_ibl(bin_destriped, delete=DELETE, version=1)

    # # mop-up all temporary files
    # shutil.rmtree(bin_destriped.parent)
