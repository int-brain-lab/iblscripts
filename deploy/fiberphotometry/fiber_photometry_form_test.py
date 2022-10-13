import os
import shutil
import unittest
from datetime import date
from pathlib import Path

from iblutil.util import get_logger

from fiber_photometry_util import convert_ui_file_to_py, create_data_dirs

log = get_logger(name="fiber_photometry_form", file=True)


class TestFiberPhotometryForm(unittest.TestCase):

    def setUp(self) -> None:
        data_dirs = create_data_dirs(test=True)
        test_data_dir = Path(os.getcwd()) / "test_data" / "2022-09-06"
        test_local_date_dir = data_dirs["fp_local_data_path"] / str(date.today())
        shutil.copytree(test_data_dir, test_local_date_dir)
        log.info(f"\n- TEST DATA LOADED FROM: {test_data_dir}\n- TEST DATA LOADED TO: {test_local_date_dir}")

    def test_convert_ui_file_to_py(self):
        test_file = "fiber_photometry_form.ui"
        log.info(f"Attempting to convert PyQt Designer ui file: {test_file}")
        ui_to_py_file = convert_ui_file_to_py(test_file, "test_output_file.py")
        assert Path(ui_to_py_file).exists()
        log.info(f"PyQt Designer ui file converted to python file named: {ui_to_py_file}")

    def test_create_data_dirs(self):
        data_dirs = create_data_dirs(test=True)
        assert len(data_dirs) == 4
        for x in data_dirs:
            assert data_dirs[x].exists

    def tearDown(self) -> None:
        # self.tempdir.cleanup()
        pass
