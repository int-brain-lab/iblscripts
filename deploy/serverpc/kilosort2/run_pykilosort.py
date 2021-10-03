from pathlib import Path
import argparse
import datetime
import logging
import shutil

import numpy as np
from ibllib.io import spikeglx
from ibllib.ephys import spikes, neuropixel
from pykilosort import add_default_handler, run, Bunch, __version__
from pykilosort.params import KilosortParams


_logger = logging.getLogger("pykilosort")


def _sample2v(ap_file):
    md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
    s2v = spikeglx._conversion_sample2v_from_meta(md)
    return s2v["ap"][0]


def run_spike_sorting_ibl(bin_file, scratch_dir=None, delete=True, neuropixel_version=1,
                          ks_output_dir=None, alf_path=None, log_level='INFO'):
    """
    This runs the spike sorting and outputs the raw pykilosort without ALF conversion
    neuroversion (1)
    :param bin_file: binary file full path to
    :param scratch_dir: working directory (home of the .kilosort folder) SSD drive preferred.
    :param delete: bool, optional, defaults to True: whether or not to delete the .kilosort temp folder
    :param neuropixel_version: float, optional, defaults to 1: the Neuropixel probe version
    :param ks_output_dir: string or Path: output directory defaults to None, in which case it will output in the
     scratch directory.
    :param alf_path: strint or Path, optional: if specified, performs ks to ALF conversion in the specified folder
    :param log_level: string, optional, defaults to 'INFO'
    :return:
    """
    START_TIME = datetime.datetime.now()
    # handles all the paths infrastructure
    assert scratch_dir is not None
    bin_file = Path(bin_file)
    scratch_dir.mkdir(exist_ok=True, parents=True)
    ks_output_dir = Path(ks_output_dir) if ks_output_dir is not None else scratch_dir.joinpath('output')
    log_file = scratch_dir.joinpath(f"_{START_TIME.isoformat()}_kilosort.log")
    add_default_handler(level=log_level)
    add_default_handler(level=log_level, filename=log_file)
    # construct the probe geometry information
    ibl_params = ibl_pykilosort_params(neuropixel_version=neuropixel_version)
    try:
        _logger.info(f"Starting Pykilosort version {__version__}, output in {bin_file.parent}")
        run(bin_file, dir_path=scratch_dir, output_dir=ks_output_dir, **ibl_params)
        if delete:
            shutil.rmtree(scratch_dir.joinpath(".kilosort"), ignore_errors=True)
    except Exception as e:
        _logger.exception("Error in the main loop")
        raise e

    [_logger.removeHandler(h) for h in _logger.handlers]
    shutil.move(log_file, ks_output_dir.joinpath('spike_sorting_pykilosort.log'))

    # convert the pykilosort output to ALF IBL format
    if alf_path is not None:
        s2v = _sample2v(bin_file)
        alf_path.mkdir(exist_ok=True, parents=True)
        spikes.ks2_to_alf(ks_output_dir, bin_file, alf_path, ampfactor=s2v)


def ibl_pykilosort_params(neuropixel_version=1):
    h = neuropixel.trace_header(version=neuropixel_version)
    probe = Bunch()
    probe.NchanTOT = 385
    probe.chanMap = np.arange(384)
    probe.xc = h['x']
    probe.yc = h['y']
    probe.kcoords = np.zeros(384)

    params = KilosortParams()
    params.preprocessing_function = 'destriping'
    params.probe = probe
    # params = {k: dict(params)[k] for k in sorted(dict(params))}
    return dict(params)


if __name__ == "__main__":
    """
    directory structure example:
        input file: ./CSH_ZAD_029/2020-09-09/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.nidq.cbin
        session_path: ./CSH_ZAD_029/2020-09-09/001
        alf_dir: ./CSH_ZAD_029/2020-09-09/001/alf/pykilosort
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

    run_spike_sorting_ibl(cbin_file, scratch_dir=scratch_dir, delete=DELETE)

