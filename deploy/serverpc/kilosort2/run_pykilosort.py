from pathlib import Path
import argparse
import datetime
import logging

import numpy as np
import scipy.io
import ibllib  # noqa

from pykilosort import add_default_handler, run, Bunch


def run_spike_sorting_ibl(bin_file):
    _logger = logging.getLogger("pykilosort")
    START_TIME = datetime.datetime.now()

    out_dir = Path("/datadisk/Data/spike_sorting/datasets").joinpath('_'.join(list(bin_file.parts[-6:-3]) + [bin_file.parts[-2]]))
    out_dir.mkdir(exist_ok=True, parents=True)
    add_default_handler(level='DEBUG')
    add_default_handler(level='DEBUG', filename=out_dir.joinpath(f"{START_TIME.isoformat()}_kilosort.log"))
    files_chmap = Path("/home/olivier/Documents/MATLAB/Kilosort2/configFiles/neuropixPhase3A_kilosortChanMap.mat")
    matdata = Bunch(scipy.io.loadmat(files_chmap))

    probe = Bunch()
    probe.NchanTOT = 385
    # WARNING: indexing mismatch with MATLAB hence the -1
    probe.chanMap = (matdata.chanMap - 1).squeeze()
    probe.xc = matdata.xcoords.squeeze()
    probe.yc = matdata.ycoords.squeeze()
    probe.kcoords = probe.yc * 0 + 1

    try:
        _logger.info(f"Starting KS, output in {out_dir}")
        run(bin_file, probe=probe, dir_path=out_dir, n_channels=385, dtype=np.int16, sample_rate=3e4)
    except Exception as e:
        _logger.exception("Error in the main loop")

    [_logger.removeHandler(h) for h in _logger.handlers]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Kilosort for a bin AP file')
    parser.add_argument('bin_file', help='binary file')
    args = parser.parse_args()
    bin_file = Path(args.bin_file)
    assert bin_file.exists(), f"{bin_file} not found !"
    run_spike_sorting_ibl(bin_file)
