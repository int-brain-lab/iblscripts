import argparse
from pathlib import Path

from iblvideo import download_la_models
from iblvideo.segmentation_la import lightning_action


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run lightning action for pose and wheel data')
    parser.add_argument('pose_file', help='pose file to run LA on')
    parser.add_argument('pose_timestamps_file', help='timestamps associated with pose file')
    parser.add_argument('wheel_file', help='wheel file')
    parser.add_argument('wheel_timestamps_file', help='timestamps associated with wheel file')
    parser.add_argument('overwrite', help='whether to force overwrite existing intermediate outputs')
    args = parser.parse_args()

    path_models = download_la_models()

    la_result = lightning_action(
        pose_file=Path(args.pose_file),
        pose_timestamp_file=Path(args.pose_timestamps_file),
        wheel_file=Path(args.wheel_file),
        wheel_timestamp_file=Path(args.wheel_timestamps_file),
        ckpts_path=path_models,
        force=args.overwrite,
    )
