import argparse
from pathlib import Path

from iblvideo.motion_energy import motion_energy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run motion energy for mp4 file and pose outputs')
    parser.add_argument('file_mp4', help='mp4 file to compute motion energy from')
    parser.add_argument('pose_result', help='pose pqt file to compute motion energy from')
    args = parser.parse_args()

    file_mp4 = Path(args.file_mp4)
    pose_result = Path(args.pose_result)

    me_result, me_roi = motion_energy(file_mp4, pose_result)
