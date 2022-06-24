import argparse
import logging
from pathlib import Path

from ibllib.pipes.misc import check_create_raw_session_flag, create_video_transfer_done_flag, load_videopc_params, \
    subjects_data_folder, transfer_session_folders


def main(local=None, remote=None):
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

    # Determine if user passed in arg for local/remote subject folder locations or pull in from local param file
    local_folder = local if local else load_videopc_params()["DATA_FOLDER_PATH"]
    remote_folder = remote if remote else load_videopc_params()["REMOTE_DATA_FOLDER_PATH"]

    # Check for Subjects folder
    local_subject_folder = subjects_data_folder(local_folder, rglob=True)
    remote_subject_folder = subjects_data_folder(remote_folder, rglob=True)
    log.info(f"Local subjects folder: {local_subject_folder}")
    log.info(f"Remote subjects folder: {remote_subject_folder}")

    # Find all local folders that have 'transfer_me.flag' set and build out list
    local_sessions = sorted(x.parent for x in local_subject_folder.rglob("transfer_me.flag"))
    if local_sessions:
        log.info("The following local session(s) have the 'transfer_me.flag' set:")
        [log.info(i) for i in local_sessions]
    else:
        log.info("No local sessions were found to have the 'transfer_me.flag' set, nothing to transfer.")
        exit(0)

    # call ibllib function to perform generalized user interaction and kick off transfer
    transfer_list, success = transfer_session_folders(
        local_sessions, remote_subject_folder, subfolder_to_transfer="raw_video_data")

    # Create and remove video flag files
    for (entry, ok) in zip(transfer_list, success):
        log.info(f"{entry[0]} -> {entry[1]} - Video file transfer success")

        # Remove local transfer_me flag file
        flag_file = Path(entry[0]) / "transfer_me.flag"
        log.info("Removing local transfer_me flag file - " + str(flag_file))
        try:
            flag_file.unlink()
        except FileNotFoundError as e:
            log.warning("An error occurred when attempting to remove the flag file.\n", e)
        create_video_transfer_done_flag(str(entry[1]))
        check_create_raw_session_flag(str(entry[1]))


if __name__ == "__main__":
    # parse user input
    parser = argparse.ArgumentParser(description="Transfer video files to IBL local server")
    parser.add_argument("-l", "--local", default=False, required=False, help="Local iblrig_data/Subjects folder")
    parser.add_argument("-r", "--remote", default=False, required=False, help="Remote iblrig_data/Subjects folder")
    args = parser.parse_args()
    main(args.local, args.remote)
