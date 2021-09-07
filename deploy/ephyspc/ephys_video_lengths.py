import argparse
from pathlib import Path

import ibllib.io.extractors.ephys_fpga as ephys_fpga
import alf.folders


def main(session_path):
    session_str = alf.folders.session_path(session_path)
    if session_str is None:
        print("I need a valid session path")
        return
    session_path = Path(session_str)
    if not session_path.exists():
        print("I need a valid session path")
        return
    ephys_fpga.extract_sync(session_path, overwrite=False)
    sync, chmap = ephys_fpga.get_main_probe_sync(session_path)
    fpga_times = ephys_fpga.extract_camera_sync(sync, chmap)
    print([f"{k}: {v.size}" for k, v in fpga_times.items()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count number of sync pulses of raw ephys data")
    parser.add_argument("session_path", help="Session path")
    args = parser.parse_args()
    main(args.session_path)
