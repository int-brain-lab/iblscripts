#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, November 8th 2019, 2:02:12 pm
from pathlib import Path

from ibllib.pipes.misc import create_custom_ephys_wirings, get_iblscripts_folder


if __name__ == "__main__":
    iblscripts_folder = get_iblscripts_folder()
    create_custom_ephys_wirings(iblscripts_folder)
