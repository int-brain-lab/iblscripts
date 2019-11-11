#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, November 8th 2019, 6:16:13 pm
import argparse

from misc import create_ephyspc_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Setup ephys parmas file')
    parser.add_argument('-f', '--force', default=False, required=False, action='store_true',
                        help='Update parameters')
    args = parser.parse_args()
    create_ephyspc_params(force=args.force)
