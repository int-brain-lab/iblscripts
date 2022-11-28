#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Miles
"""
Simply transfers the specified raw data folders and writes a local 'transferred.flag' file upon
transfer.

Expects data folder to be in the following folder structure:
DATA_FOLDER_PATH/subject/yyyy-mm-dd/nnn/<data_folder>

Example:
    python transfer_data_folder.py raw_sync_data
"""
import argparse
from pathlib import Path
import re

import ibllib.io.flags as flags
from iblutil.util import log_to_file
from ibllib.pipes.misc import (create_basic_transfer_params, subjects_data_folder, transfer_session_folders,
                               create_transfer_done_flag, check_create_raw_session_flag)


def main(data_folder, local=None, remote=None, transfer_done_flag=False):
    # logging configuration
    data_name, = (re.match(r'raw_(\w+)_data', data_folder) or (data_folder,)).groups()
    log = log_to_file(filename=f'transfer_{data_name}_session.log', log='ibllib.pipes.misc')

    # Determine if user passed in arg for local/remote subject folder locations or pull in from
    # local param file or prompt user if missing
    params = create_basic_transfer_params(local_data_path=local, remote_data_path=remote)

    # Check for Subjects folder
    local_subject_folder = subjects_data_folder(params['DATA_FOLDER_PATH'], rglob=True)
    remote_subject_folder = subjects_data_folder(params['REMOTE_DATA_FOLDER_PATH'], rglob=True)
    log.info(f'Local subjects folder: {local_subject_folder}')
    log.info(f'Remote subjects folder: {remote_subject_folder}')

    # Find all local folders that have 'raw_sync_data'
    local_sessions = local_subject_folder.rglob(data_folder)
    # Remove sessions that have a transferred flag file
    local_sessions = filter(lambda x: not any(x.glob('transferred.flag')), local_sessions)
    local_sessions = sorted(x.parent for x in local_sessions)

    if local_sessions:
        log.info('The following local session(s) have yet to be transferred:')
        [log.info(i) for i in local_sessions]
    else:
        log.info('No outstanding local sessions to transfer.')
        return

    # Call ibllib function to perform generalized user interaction and kick off transfer
    transfer_list, success = transfer_session_folders(
        local_sessions, remote_subject_folder, subfolder_to_transfer=data_folder)

    # Create transferred flag files and rename files
    for src, dst in (x for x, ok in zip(transfer_list, success) if ok):
        log.info(f"{src} -> {dst} - {data_name} transfer success")

        # Create flag
        flag_file = src.joinpath(data_folder, 'transferred.flag')
        file_list = map(str, filter(Path.is_file, flag_file.parent.rglob('*')))
        flags.write_flag_file(flag_file, file_list=list(file_list))

        if transfer_done_flag:
            create_transfer_done_flag(str(dst), data_name)
            check_create_raw_session_flag(str(dst))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer raw data folder(s) to IBL local server')
    parser.add_argument('data_folder', help='The raw data folder to transfer, e.g. "raw_sync_data"')
    parser.add_argument('-l', '--local', default=False, required=False, help='Local iblrig_data/Subjects folder')
    parser.add_argument('-r', '--remote', default=False, required=False, help='Remote iblrig_data/Subjects folder')
    parser.add_argument('-f', '--flag', default=False, required=False, help='Create transfer complete flag in remote folder')
    args = parser.parse_args()
    main(args.data_folder, args.local, args.remote, transfer_done_flag=args.flag)
