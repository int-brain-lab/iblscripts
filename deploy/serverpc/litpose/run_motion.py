import argparse
from pathlib import Path
from iblvideo.motion_energy import motion_energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run motion energy for mp4 file and dlc outputs')
    parser.add_argument('file_mp4', help='mp4 file to compute motion energy from')
    parser.add_argument('dlc_result', help='dlc pqt file to compute motion energy from')
    args = parser.parse_args()

    file_mp4 = Path(args.file_mp4)
    dlc_result = Path(args.dlc_result)

    me_result, me_roi = motion_energy(file_mp4, dlc_result)
