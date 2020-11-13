#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, November 13th 2020, 5:26:57 pm
import unittest
from pathlib import Path
import logging
import numpy as np
import shutil

import alf.io
from ibllib.io.extractors import ephys_passive
from ibllib.ephys import ephysqc

_logger = logging.getLogger('ibllib')

PATH_TESTS = Path('/mnt/s0/Data/IntegrationTests')
session_path = "/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_054/2020-10-10/001"


class TestEphysPassiveExtraction(unittest.TestCase):

    def setUp(self) -> None:
        self.root_folder = PATH_TESTS.joinpath("ephys", "passive_extraction")
        if not self.root_folder.exists():
            return
        self.session_path = self.root_folder.joinpath("SWC_054", "2020-10-10", "001")

    def test_task_extraction(self):
        ext = ephys_passive.PassiveChoiceWorld(self.session_path)
        data, paths = ext.extract()
        self.assertTrue(paths is None)
        # data tests

    def test_task_extraction_files(self):
        ext = ephys_passive.PassiveChoiceWorld(self.session_path)
        data, paths = ext.extract(save=True)
        self.assertTrue(paths is not None)
        # data tests
        # paths test

    def tearDown(self):
        # remove alf folder
        shutil.rmtree(self.session_path.joinpath("alf"))