from pathlib import Path
import argparse
import logging

from pykilosort.ibl import run_spike_sorting_ibl


_logger = logging.getLogger("pykilosort")


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
    _logger.info(f"Spike sorting {cbin_file}, temp path {scratch_dir}")
    print(f"Spike sorting {cbin_file}, temp path {scratch_dir}")
    run_spike_sorting_ibl(cbin_file, scratch_dir=scratch_dir, delete=DELETE)
