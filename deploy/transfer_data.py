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

Workflow:
    1. At the start of acquisition an incomplete experiment description file (a 'stub') is saved on
     the local PC and in the lab server, in a session subfolder called '_devices'.  The filename
     includes the PC's identifier so the copy script knows which stub was saved by which PC.
    2. This copy script is run on each acquisition PC in any order.
    3. The script iterates through the local sessions data (or optionally a separate 'transfers'
     folder) that contain experiment.description stubs.
    4. Session folders containing a 'transferred.flag' file are ignored.
    5. For each session the stub file is read in and rsync is called for each 'collection'
     contained.  If there is a local subfolder that isn't specified in a 'collection' key, it won't
     be copied.
    6. Once rsync succeeds, the remote stub file is merged with the remote experiment.description
     file (or copied over if one doesn't already exist).  The remote stub is deleted.
    7. A 'transferred.flag' file is created in the local session folder.
    8. If no more remote stub files exist for a given session, the empty _devices subfolder is
     deleted and a 'raw_session.flag' file is created in the remote session folder.

Questions:
    - Is this really more robust? Seems similar to our old method but with more points of failure.
    - Dealing with session mismatches - should we include the exp ref in the experiment description
     files?  Do we no longer need the prompts for mismatches?
    - Should all collections defined in the experiment definition exist locally? If so the sync
     should be on the FPGA computer, not the behaviour PC.
    - What should the behaviour be if rsync fails? Error out or move onto the next session?

"""
import argparse
from functools import partial

from one.alf.files import filename_parts
from one.converters import ConversionMixin
from iblutil.util import log_to_file
import ibllib.io.flags as flags
from ibllib.io import session_params
from ibllib.pipes.misc import \
    create_basic_transfer_params, subjects_data_folder, rsync_paths


def main(local=None, remote=None):
    # logging configuration
    log = log_to_file(filename='transfer_session.log', log='ibllib.pipes.misc')

def transfer_session(session, params=None):
    status = True
    if params is None:
        params = create_basic_transfer_params()
    remote_subject_folder = subjects_data_folder(params['REMOTE_DATA_FOLDER_PATH'], rglob=True)
    session_parts = session.parent.as_posix().split('/')[-3:]
    remote_session = remote_subject_folder.joinpath(*session_parts)
    remote_file = session_params.get_remote_stub_name(remote_session, filename_parts(session.name)[3])
    assert remote_file.exists()
    exp_pars = session_params.read_params(session)
    collections = set(session_params.get_collections(exp_pars).values())
    for collection in collections:
        if not session.with_name(collection).exists():
            log.error(f'Collection {session.with_name(collection)} doesn\'t exist')
            status = False
            continue
        log.debug(f'transferring {session_parts} - {collection}')
        status &= rsync_paths(session.with_name(collection), remote_session / collection)
    # if the transfer was successful, merge the stub file with the remote experiment description
    if status:
        main_experiment_file = remote_session / '_ibl_experiment.description.yaml'
        session_params.aggregate_device(remote_file, main_experiment_file, unlink=True)
        # when there is not any remaining stub files, create a flag file in the session folder
        # that indicates the copy is complete
        if not any(remote_session.joinpath('_devices').glob('*.*')):
            file_list = list(map(str, remote_session.rglob('*.*.*')))
            flags.write_flag_file(remote_session.joinpath('raw_session.flag'), file_list=file_list)
        flags.write_flag_file(session.with_name('transferred.flag'), file_list=list(collections))
        log.info(f'{session_parts} transfer success')
    return status


def transfer_sessions(local=None, remote=None):
    """
    Transfer all sessions in the local subject folder that have an experiment.description file
    :param local: the local computer full path to the Subjects folder
    :param remote: the remote server full path to the Subjects folder
    :return:
    """
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
    local_sessions = filter(lambda x: not any(x.parent.glob('transferred.flag')), local_sessions)
    # Sort by date, number and subject name
    local_sessions = sorted(local_sessions, key=partial(ConversionMixin.path2ref, as_dict=False))

    if local_sessions:
        log.info('The following local session(s) have yet to be transferred:')
        [log.info(i.parent) for i in local_sessions]
    else:
        log.info('No outstanding local sessions to transfer.')
        return

    ok = [True] * len(local_sessions)
    for i, session in enumerate(local_sessions):
        session_parts = session.parent.as_posix().split('/')[-3:]
        remote_session = remote_subject_folder.joinpath(*session_parts)
        remote_file = session_params.get_remote_stub_name(remote_session, filename_parts(session.name)[3])
        assert remote_file.exists()
        exp_pars = session_params.read_params(session)
        collections = set(session_params.get_collections(exp_pars).values())
        for collection in collections:
            if not session.with_name(collection).exists():
                log.error(f'Collection {session.with_name(collection)} doesn\'t exist')
                ok[i] = False
                continue
            log.debug(f'transferring {session_parts} - {collection}')
            ok[i] &= rsync_paths(session.with_name(collection), remote_session / collection)
        # if the transfer was successful, merge the stub file with the remote experiment description
        if ok[i]:
            main_experiment_file = remote_session / '_ibl_experiment.description.yaml'
            session_params.aggregate_device(remote_file, main_experiment_file, unlink=True)
            # when there is not any remaining stub files, create a flag file in the session folder
            # that indicates the copy is complete
            if not any(remote_session.joinpath('_devices').glob('*.*')):
                file_list = list(map(str, remote_session.rglob('*.*.*')))
                flags.write_flag_file(remote_session.joinpath('raw_session.flag'), file_list=file_list)
            flags.write_flag_file(session.with_name('transferred.flag'), file_list=list(collections))
            log.info(f'{session_parts} transfer success')
    return local_sessions, ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer raw data folder(s) to IBL local server')
    parser.add_argument('-l', '--local', default=False, required=False, help='Local iblrig_data/Subjects folder')
    parser.add_argument('-r', '--remote', default=False, required=False, help='Remote iblrig_data/Subjects folder')
    args = parser.parse_args()
    main(args.data_folder, args.local)
