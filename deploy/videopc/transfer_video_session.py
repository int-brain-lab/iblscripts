import argparse
import logging
from pathlib import Path

from ibllib.pipes.misc import transfer_video_folders

if __name__ == "__main__":
    # parse user input
    parser = argparse.ArgumentParser(description="Transfer video files to IBL local server")
    parser.add_argument(
        "-l", "--local", default=False, required=False,
        help="Local iblrig_data/Subjects folder")
    parser.add_argument(
        "-r", "--remote", default=False, required=False,
        help="Remote iblrig_data/Subjects folder")
    args = parser.parse_args()

    # logging configuration
    ibllib_log_dir = Path.home() / '.ibl_logs'
    ibllib_log_dir.mkdir() if ibllib_log_dir.exists() is False else None
    log = logging.getLogger("ibllib.pipes.misc")
    log.setLevel(logging.INFO)
    format_str = '%(asctime)s.%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    file_handler = logging.FileHandler(ibllib_log_dir / 'transfer_video_session.log')
    file_format = logging.Formatter(format_str, date_format)
    file_handler.setFormatter(file_format)
    log.addHandler(file_handler)
    log.info("Logging initiated")

    # call rsync_video_folders function in ibllib
    transfer_video_folders(local_folder=args.local, remote_folder=args.remote)