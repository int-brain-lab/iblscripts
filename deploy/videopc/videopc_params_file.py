#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, November 8th 2019, 6:16:13 pm
import argparse

from ibllib.pipes.misc import create_videopc_params, create_basic_transfer_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Setup video parmas file')
    parser.add_argument('-f', '--force', default=False, required=False, action='store_true',
                        help='Update parameters')
    args = parser.parse_args()
    params = create_videopc_params(force=args.force)
    create_basic_transfer_params(
        local_data_path=params['DATA_FOLDER_PATH'],
        remote_data_path=params['REMOTE_DATA_FOLDER_PATH'],
        clobber=True
    )
