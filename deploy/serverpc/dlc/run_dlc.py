import argparse
from pathlib import Path
from iblvideo import download_weights
from iblvideo.choiceworld import dlc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DLC for mp4 file')
    parser.add_argument('file_mp4', help='mp4 file to run DLC on')
    parser.add_argument('overwrite', help='whether to force overwrite existing intermediate outputs')
    args = parser.parse_args()

    file_mp4 = Path(args.file_mp4)
    path_dlc = download_weights()

    dlc_result, _ = dlc(file_mp4, path_dlc=path_dlc, force=args.overwrite)
