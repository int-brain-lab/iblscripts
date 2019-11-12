#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, August 13th 2019, 12:10:34 pm
import argparse

from ibllib.pipes.misc import confirm_ephys_remote_folder, get_iblscripts_folder


if __name__ == "__main__":
    iblscripts_folder = get_iblscripts_folder()
    parser = argparse.ArgumentParser(
        description='Transfer files to IBL local server')
    parser.add_argument(
        '-l', '--local', default=False, required=False,
        help='Local iblrig_data/Subjects folder')
    parser.add_argument(
        '-r', '--remote', default=False, required=False,
        help='Remote iblrig_data/Subjects folder')
    parser.add_argument(
        '-f', '--force', default=False, required=False, action='store_true',
        help='Overwrite existing ephys files in dst/remote folder')
    args = parser.parse_args()
    confirm_ephys_remote_folder(
        local_folder=args.local, remote_folder=args.remote, force=args.force,
        iblscripts_folder=iblscripts_folder)
