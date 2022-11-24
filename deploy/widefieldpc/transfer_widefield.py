#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Miles
import argparse
from pathlib import Path
import shutil

import ibllib.io.flags as flags
from iblutil.util import log_to_file
from ibllib.pipes.misc import (create_basic_transfer_params, subjects_data_folder, transfer_session_folders,
                               create_transfer_done_flag, check_create_raw_session_flag)


def main(local=None, remote=None, rename_files=False, data_folder='raw_widefield_data', transfer_done_flag=False):
    DATA_FOLDER = data_folder
    fold = data_folder.split('_')[1]
    # logging configuration
    log = log_to_file(filename=f'transfer_{fold}_sessions.log', log='ibllib.pipes.misc')

    # Determine if user passed in arg for local/remote subject folder locations or pull in from
    # local param file or prompt user if missing
    params = create_basic_transfer_params(local_data_path=local, remote_data_path=remote)

    # Check for Subjects folder
    local_subject_folder = subjects_data_folder(params['DATA_FOLDER_PATH'], rglob=True)
    remote_subject_folder = subjects_data_folder(params['REMOTE_DATA_FOLDER_PATH'], rglob=True)
    log.info(f'Local subjects folder: {local_subject_folder}')
    log.info(f'Remote subjects folder: {remote_subject_folder}')

    # Find all local folders that have 'raw_widefield_data'
    local_sessions = local_subject_folder.rglob(DATA_FOLDER)
    # Remove sessions that have a transferred flag file
    local_sessions = filter(lambda x: not any(x.glob('transferred.flag')), local_sessions)
    local_sessions = sorted(x.parent for x in local_sessions)

    if local_sessions:
        log.info('The following local session(s) have yet to be transferred:')
        [log.info(i) for i in local_sessions]
    else:
        log.info('No outstanding local sessions to transfer.')
        return

    def copy_wiring(wiring_file, filter_pattern):
        default_file = Path(__file__).parent.joinpath('wirings', wiring_file)
        for session_path in filter(lambda x: not any(x.glob(f'{DATA_FOLDER}/{filter_pattern}')), local_sessions):
            destination = session_path.joinpath(DATA_FOLDER, wiring_file)
            log.debug(f'{default_file} -> {destination}')
            shutil.copy(default_file, destination)

    # Ensure each session contains a channels file: copy file over if not present
    log.info('Copying missing wiring files')
    copy_wiring('widefield_wiring.htsv', '*widefield_wiring*')

    # Call ibllib function to perform generalized user interaction and kick off transfer
    transfer_list, success = transfer_session_folders(
        local_sessions, remote_subject_folder, subfolder_to_transfer=DATA_FOLDER)

    # Create transferred flag files and rename files
    for src, dst in (x for x, ok in zip(transfer_list, success) if ok):
        log.info(f"{src} -> {dst} - {fold} transfer success")

        # Create flag
        flag_file = src.joinpath(DATA_FOLDER, 'transferred.flag')
        file_list = map(str, filter(Path.is_file, flag_file.parent.rglob('*')))
        flags.write_flag_file(flag_file, file_list=list(file_list))
        if transfer_done_flag:
            create_transfer_done_flag(str(dst), fold)
            check_create_raw_session_flag(str(dst))

        if rename_files:
            log.info(f'Renaming remote {fold} data files')
            raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer widefield files to IBL local server')
    parser.add_argument('-l', '--local', default=False, required=False, help='Local iblrig_data/Subjects folder')
    parser.add_argument('-r', '--remote', default=False, required=False, help='Remote iblrig_data/Subjects folder')
    parser.add_argument('-f', '--flag', default=True, required=False, help='Create transfer complete flag in remote folder')
    args = parser.parse_args()
    main(args.local, args.remote, data_folder='raw_widefield_data', transfer_done_flag=args.flag)
