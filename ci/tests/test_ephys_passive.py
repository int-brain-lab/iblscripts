#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, November 13th 2020, 5:26:57 pm
import unittest
from pathlib import Path
import logging
import shutil

from ibllib.io.extractors import ephys_passive
from ci.tests import base


log = logging.getLogger("ibllib")


class TestEphysPassiveExtraction(base.IntegrationTest):
    def setUp(self) -> None:
        self.root_folder = self.data_path.joinpath("ephys", "passive_extraction")
        self.session_path = self.root_folder.joinpath("SWC_054", "2020-10-10", "001")
        if not self.root_folder.exists():
            log.error(f"{self.root_folder} does not exist")

    def test_task_extraction(self):
        ext = ephys_passive.PassiveChoiceWorld(self.session_path)
        data, paths = ext.extract()
        self.assertTrue(len(data) == 4)
        self.assertTrue(paths is None)
        # data tests

    def test_task_extraction_files(self):
        ext = ephys_passive.PassiveChoiceWorld(self.session_path)
        data, paths = ext.extract(save=True)
        path_names = [x.name for x in paths]
        expected = [
            "_ibl_passivePeriods.intervalsTable.csv",
            "_ibl_passiveRFM.times.npy",
            "_ibl_passiveGabor.table.csv",
            "_ibl_passiveStims.table.csv",
        ]
        self.assertTrue(all([x in path_names for x in expected]))

        # data tests
        # paths test

    def tearDown(self):
        # remove alf folder
        shutil.rmtree(self.session_path.joinpath("alf"))


if __name__ == "__main__":
    unittest.main(exit=False)
