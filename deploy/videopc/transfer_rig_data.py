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


def confirm_remote_folder(local_folder, remote_folder):
    local_folder = Path(local_folder)
    remote_folder = Path(remote_folder)

    src_session_paths = [x.parent for x in local_folder.rglob(
        "transfer_me.flag")]

    if not src_session_paths:
        log.info("Nothing to transfer, exiting...")
        return
    for s in src_session_paths:
        mouse = s.parts[-3]
        date = s.parts[-2]
        sess = s.parts[-1]
        d = remote_folder / mouse / date / sess

        print(f"Found session: {s}")
        resp = (f"Transfer to {d} ([y]/n)? ") or 'y'
        if resp not in ['y', 'n']:
            return confirm_remote_folder(local_folder, remote_folder)
        elif resp == 'y':
            main(s, d)
        elif resp == 'n':
            print("Please insert a different session name e.g. [mouse_name/1990-01-31/001]:")
            new_name = Path(input("> "))
            mouse = new_name.parts[-3]
            date = new_name.parts[-2]
            sess = new_name.parts[-1]
            d = remote_folder / mouse / date / sess
            main(s, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer files to IBL local server')
    parser.add_argument(
        'local_folder', help='Local iblrig_data/Subjects folder')
    parser.add_argument(
        'remote_folder', help='Remote iblrig_data/Subjects folder')
    args = parser.parse_args()
    confirm_remote_folder(args.local_folder, args.remote_folder)
