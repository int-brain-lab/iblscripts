# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, February 19th 2019, 11:45:24 am
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 19-02-2019 11:46:07.077
import shutil
import unittest
from pathlib import Path

import ibllib.pipes.experimental_data as iblrig_pipeline
from ibllib.pipes.transfer_rig_data import main as transfer
from oneibl.one import ONE

PATH_TESTS = Path('/mnt/s0/Data/IntegrationTests')


class TestPipeline(unittest.TestCase):

    def test_full_pipeline(self):
        pass
