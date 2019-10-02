#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, August 13th 2019, 12:10:34 pm
import argparse
import shutil
from pathlib import Path

import alf.folders as folders

from ephyspc_params_file import load_ephyspc_params


# TODO: Move all the code into ibllib and make unittests, import from ibllib and use!
def check_transfer(src_session_path: str or Path, dst_session_path: str or Path):
    src_files = sorted([x for x in Path(src_session_path).rglob('*') if x.is_file()])
    dst_files = sorted([x for x in Path(dst_session_path).rglob('*') if x.is_file()])
    for s, d in zip(src_files, dst_files):
        assert(s.name == d.name)
        assert(s.stat().st_size == d.stat().st_size)
    return


def rename_session(session_path: str) -> Path:
    session_path = Path(folders.session_path(session_path))
    if session_path is None:
        return
    mouse = session_path.parts[-3]
    date = session_path.parts[-2]
    sess = session_path.parts[-1]
    new_mouse = input(
        f"Please insert mouse NAME [current value: {mouse}]> ") or mouse
    new_date = input(
        f"Please insert new session DATE [current value: {date}]> ") or date
    new_sess = input(
        f"Please insert new session NUMBER [current value: {sess}]> ") or sess
    new_session_path = Path(*session_path.parts[:-3]) / new_mouse / new_date / new_sess

    shutil.move(str(session_path), str(new_session_path))
    print(session_path, '--> renamed to:')
    print(new_session_path)

    return new_session_path


def transfer_folder(src: Path, dst: Path, force: bool = False):
    print(f"Attempting to copy:\n{src}\n--> {dst}")
    if force:
        print(f"Removing {dst}")
        shutil.rmtree(dst, ignore_errors=True)
    print(f"Copying all files:\n{src}\n--> {dst}")
    shutil.copytree(src, dst)
    # If folder was created delete the src_flag_file
    if check_transfer(src, dst) is None:
        print("All files copied")


def confirm_remote_folder(local_folder=False, remote_folder=False):
    pars = load_ephyspc_params()

    if not local_folder:
        local_folder = pars['DATA_FOLDER_PATH']
    if not remote_folder:
        remote_folder = pars['REMOTE_DATA_FOLDER_PATH']
    local_folder = Path(local_folder)
    remote_folder = Path(remote_folder)
    # Check for Subjects folder
    local_folder = folders.subjects_data_folder(local_folder, rglob=True)
    remote_folder = folders.subjects_data_folder(remote_folder, rglob=True)

    print('LOCAL:', local_folder)
    print('REMOTE:', remote_folder)
    src_session_paths = [x.parent for x in local_folder.rglob("transfer_me.flag")]

    if not src_session_paths:
        print("Nothing to transfer, exiting...")
        return

    for session_path in src_session_paths:
        print(f"\nFound session: {session_path}")
        flag_file = session_path / 'transfer_me.flag'
        msg = f"Transfer to {remote_folder} with the same name?"
        resp = input(msg + "\n[y]es/[r]ename/[s]kip/[e]xit\n ^\n> ") or 'y'
        resp = resp.lower()
        print(resp)
        if resp not in ['y', 'r', 's', 'e', 'yes', 'rename', 'skip', 'exit']:
            return confirm_remote_folder(local_folder=local_folder, remote_folder=remote_folder)
        elif resp == 'y' or resp == 'yes':
            remote_session_path = remote_folder / Path(*session_path.parts[-3:])
            transfer_folder(
                session_path / 'raw_ephys_data',
                remote_session_path / 'raw_ephys_data',
                force=False)
            flag_file.unlink()
        elif resp == 'r' or resp == 'rename':
            new_session_path = rename_session(session_path)
            remote_session_path = remote_folder / Path(*new_session_path.parts[-3:])
            transfer_folder(
                new_session_path / 'raw_ephys_data',
                remote_session_path / 'raw_ephys_data')
            flag_file.unlink()
        elif resp == 's' or resp == 'skip':
            continue
        elif resp == 'e' or resp == 'exit':
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer files to IBL local server')
    parser.add_argument(
        '-l', '--local', default=False, required=False,
        help='Local iblrig_data/Subjects folder')
    parser.add_argument(
        '-r', '--remote', default=False, required=False,
        help='Remote iblrig_data/Subjects folder')
    args = parser.parse_args()
    confirm_remote_folder(local_folder=args.local, remote_folder=args.remote)
