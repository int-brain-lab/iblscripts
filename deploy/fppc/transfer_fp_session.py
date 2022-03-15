#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mayo Faulkner
# @Date: Tuesday, March 15th 2022, 12:10:34 pm
import argparse

from ibllib.pipes.misc import confirm_fp_remote_folder


if __name__ == "__main__":
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
        help='Overwrite existing video files in dst/remote folder')
    args = parser.parse_args()
    # print(args)
    confirm_fp_remote_folder(local_folder=args.local, remote_folder=args.remote, force=args.force)