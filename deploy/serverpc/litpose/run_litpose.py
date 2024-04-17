import argparse
from pathlib import Path
# import cProfile
# import pstats

from iblvideo import lightning_pose, download_lit_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run lightning pose for mp4 file')
    parser.add_argument('file_mp4', help='mp4 file to run LP on')
    parser.add_argument('overwrite', help='whether to force overwrite existing intermediate outputs')
    args = parser.parse_args()

    file_mp4 = Path(args.file_mp4)
    path_models = download_lit_model()

    # profile = cProfile.Profile()
    # profile.enable()
    lp_result = lightning_pose(file_mp4, ckpts_path=path_models, force=args.overwrite)
    # profile.disable()
    # profile.create_stats()

    # with open(f'{file_mp4.name}_profile.txt', 'w') as fp:
    #     stats = pstats.Stats(profile, stream=fp)
    #     stats.sort_stats('cumulative')
    #     stats.print_stats()
