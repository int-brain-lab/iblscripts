#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, August 13th 2019, 12:10:34 pm
import argparse
import shutil
from pathlib import Path
from shutil import ignore_patterns as ig

import alf.folders as folders
import ibllib.io.flags as flags


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


def transfer_session(src: Path, dst: Path, force: bool = False):
    print(f"Attempting to copy:\n{src}\n--> {dst}")
    src = Path(folders.session_path(src))
    dst_sess = folders.session_path(dst)
    if src is None:
        return
    if dst_sess is None:
        dst = dst / Path(*src.parts[-3:])

    src_flag_file = src / "transfer_me.flag"
    if not src_flag_file.exists():
        return

    flag = flags.read_flag_file(src_flag_file)
    if isinstance(flag, list):
        raise(NotImplementedError)
    else:
        if force:
            shutil.rmtree(dst, ignore_errors=True)
        print(f"Copying all files:\n{src}\n--> {dst}")
        shutil.copytree(src, dst, ignore=ig(str(src_flag_file.name)))
    # If folder was created delete the src_flag_file
    if check_transfer(src / 'raw_video_data', dst / 'raw_video_data') is None:
        print(
            f"{Path(*src.parts[-3:]) / 'raw_video_data'} copied to {dst.parent.parent.parent}")
        src_flag_file.unlink()


def confirm_remote_folder(local_folder, remote_folder):
    local_folder = Path(local_folder)
    remote_folder = Path(remote_folder)
    local_folder = folders.subjects_data_folder(local_folder)
    remote_folder = folders.subjects_data_folder(remote_folder)

    src_session_paths = [x.parent for x in local_folder.rglob(
        "transfer_me.flag")]

    if not src_session_paths:
        print("Nothing to transfer, exiting...")
        return

    for session_path in src_session_paths:
        print(f"\nFound session: {session_path}")
        resp = input(f"Transfer to {remote_folder} with the same name ([y]/n/skip/exit)? ") or 'y'
        print(resp)
        if resp not in ['y', 'n', 'skip', 'exit']:
            return confirm_remote_folder(local_folder, remote_folder)
        elif resp == 'y':
            transfer_session(session_path, remote_folder, force=False)
        elif resp == 'n':
            new_session_path = rename_session(session_path)
            transfer_session(new_session_path, remote_folder)
        elif resp == 'skip':
            continue
        elif resp == 'exit':
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer files to IBL local server')
    parser.add_argument(
        'local_folder', help='Local iblrig_data/Subjects folder')
    parser.add_argument(
        'remote_folder', help='Remote iblrig_data/Subjects folder')
    args = parser.parse_args()
    confirm_remote_folder(args.local_folder, args.remote_folder)
    # local_folder = '/home/nico/Projects/IBL/github/iblrig_data/Subjects'
    # remote_folder = '/home/nico/Projects/IBL/github/iblrig_data_transfer_test/Subjects'
    # confirm_remote_folder(local_folder, remote_folder)
    # src_session_path = '/home/nico/Projects/IBL/github/iblrig_data/Subjects/ZM_335/2018-12-13/001'  # noqa
    # dst_session_path = '/home/nico/Projects/IBL/github/iblrig_data_transfer_test/Subjects/ZM_335/2018-12-13/001'  # noqa
