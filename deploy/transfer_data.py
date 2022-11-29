#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Miles
"""
Transfers local session data to the server.

This will iterate over sessions in TRANSFERS_PATH or DATA_FOLDER_PATH that contain an
'experiment.description' file and no 'transferred.flag' file.  For each session it will sync the
collections (subfolders) present in the experiment description.

Expects collection to be in the following folder structure:
DATA_FOLDER_PATH/subject/yyyy-mm-dd/nnn/<collection>

Example:
    python transfer_data.py

Parameters:
    in transfer_params there are three parameters:
    - REMOTE_DATA_FOLDER_PATH: The destination location of the session data.
    - DATA_FOLDER_PATH: The location of the local session data.
    - TRANSFERS_PATH: Optional location of the experiment.description files, if not set,
     the DATA_FOLDER_PATH is searched.
    - TRANSFER_LABEL: A unique name for the remote experiment description stub.
"""
import argparse

from one.alf.files import filename_parts
from iblutil.util import log_to_file
import ibllib.io.flags as flags
from ibllib.io import session_params
from ibllib.pipes.misc import \
    create_basic_transfer_params, subjects_data_folder, rsync_paths


def main(local=None, remote=None):
    # logging configuration
    log = log_to_file(filename=f'transfer_session.log', log='ibllib.pipes.misc')

    # Determine if user passed in arg for local/remote subject folder locations or pull in from
    # local param file or prompt user if missing
    params = create_basic_transfer_params(local_data_path=local, remote_data_path=remote)

    # Check for Subjects folder
    local_subject_folder = subjects_data_folder(params['DATA_FOLDER_PATH'], rglob=True)
    remote_subject_folder = subjects_data_folder(params['REMOTE_DATA_FOLDER_PATH'], rglob=True)
    log.info(f'Local subjects folder: {local_subject_folder}')
    log.info(f'Remote subjects folder: {remote_subject_folder}')
    if transfers_path := params.get('TRANSFERS_PATH'):
        log.info(f'Transfers folder: {transfers_path}')
    else:
        transfers_path = local_subject_folder

    # Find all local folders that have an experiment description file
    local_sessions = transfers_path.rglob('_ibl_experiment.description*.yaml')
    # Remove sessions that have a transferred flag file
    local_sessions = sorted(filter(lambda x: not any(x.parent.glob('transferred.flag')), local_sessions))

    if local_sessions:
        log.info('The following local session(s) have yet to be transferred:')
        [log.info(i.parent) for i in local_sessions]
    else:
        log.info('No outstanding local sessions to transfer.')
        return

    for session in local_sessions:
        session_parts = session.parent.as_posix().split('/')[-3:]
        remote_session = remote_subject_folder.joinpath(*session_parts)
        remote_filename = f'{filename_parts(session.name)[3]}.yaml'
        remote_file = remote_session.joinpath('_devices', remote_filename)
        assert remote_file.exists()
        exp_pars = session_params.read_params(session)
        collections = list(session_params.get_collections(exp_pars).values())
        for collection in collections:
            success = rsync_paths(session.with_name(collection), remote_session / collection)
            assert success
            # log.info(f"{src} -> {dst} - {data_name} transfer success")
        session_params.aggregate_device(remote_file, remote_session / '_ibl_experiment.description.yaml', unlink=True)
        if not any(remote_session.joinpath('_devices').glob('*.*')):
            file_list = list(remote_session.rglob('*.*.*'))
            flags.write_flag_file(remote_session.joinpath('raw_session.flag'), file_list=file_list)
        flags.write_flag_file(session.with_name('transferred.flag'), file_list=collections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer raw data folder(s) to IBL local server')
    parser.add_argument('-l', '--local', default=False, required=False, help='Local iblrig_data/Subjects folder')
    parser.add_argument('-r', '--remote', default=False, required=False, help='Remote iblrig_data/Subjects folder')
    args = parser.parse_args()
    main(args.data_folder, args.local)
