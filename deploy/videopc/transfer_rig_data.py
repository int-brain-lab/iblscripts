#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, August 13th 2019, 12:10:34 pm
from ibllib.pipes.transfer_rig_data import main
import logging
import argparse
from pathlib import Path

log = logging.getLogger('iblrig')
log.setLevel(logging.INFO)


def confirm_remote_folder(local_folder, remote_folder):  # XXX: TEST THIS!
    local_folder = Path(local_folder)
    remote_folder = Path(remote_folder)

    src_session_paths = [x.parent for x in local_folder.rglob(
        "transfer_me.flag")]

    if not src_session_paths:
        log.info("Nothing to transfer, exiting...")
        return
    for s in src_session_paths:
        local_session_name = s.parts[-3:]
        local_mouse = s.parts[-3]
        local_date = s.parts[-2]
        local_sess = s.parts[-1]
        d = remote_folder / local_mouse / local_date / local_sess

        print(f"Found session: {local_session_name}")
        resp = input(f"Transfer to {remote_folder} with the same name ([y]/n)? ") or 'y'
        if resp not in ['y', 'n']:
            return confirm_remote_folder(local_folder, remote_folder)
        elif resp == 'y':
            main(local_folder, remote_folder)
        elif resp == 'n':
            new_mouse = input(
                f"Please insert mouse NAME [current value: {local_mouse}]> ") or local_mouse
            new_date = input(
                f"Please insert new session DATE [current value: {local_date}]> ") or local_date
            new_sess = input(
                f"Please insert new session NUMBER [current value: {local_sess}]> ") or local_sess
            s.rename(s.parts[:-3] / new_mouse / new_date / new_sess)
            return confirm_remote_folder(local_folder, remote_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer files to IBL local server')
    parser.add_argument(
        'local_folder', help='Local iblrig_data/Subjects folder')
    parser.add_argument(
        'remote_folder', help='Remote iblrig_data/Subjects folder')
    args = parser.parse_args()
    confirm_remote_folder(args.local_folder, args.remote_folder)
