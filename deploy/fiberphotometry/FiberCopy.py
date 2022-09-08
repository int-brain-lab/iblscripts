"""
python FiberCopy.py
python FiberCopy.py test
Requirements:
ONE-api
Dataclasses
"""
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from PyQt5 import QtWidgets, QtCore, uic

"""
settings values:
    path_fiber_photometry: str
    path_server_sessions: str
    projects: list[str] = field(default_factory=list)  # settings
    subjects: list[str] = field(default_factory=list)  # settings
"""


class FiberCopy(QtWidgets.QMainWindow):
    """
    Controller
    """
    def __init__(self, *args, **kwargs):
        """
        :param parent:
        :param sr: ibllib.io.spikeglx.Reader instance
        """
        self.model = None
        super(FiberCopy, self).__init__(*args, *kwargs)
        self.settings = QtCore.QSettings('int-brain-lab', 'FiberCopy')
        uic.loadUi(Path(__file__).parent.joinpath('FiberCopy.ui'), self)
        self.actionload_photometry_csv.triggered.connect(self.open_photometry_csv)
        self.show()

    def open_photometry_csv(self, *args, file=None):
        """
        :param args:
        :param file:
        :return:
        """
        if file is None:
            file, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent=self, caption='Select Raw Fiber Photometry recording',
                directory=self.settings.value('path_fiber_photometry'), filter='*.*csv')
        if file == '':
            return
        file = Path(file)
        self.settings.setValue("path_fiber_photometry", str(file.parent))
        self.model = Model(pd.read_csv(file))


@dataclass
class Model:
    """Class to store the necessary data"""
    dataframe: pd.DataFrame

    @property
    def regions(self):
        regions = list(set(self.dataframe.keys()).difference(set(['FrameCounter', 'Timestamp', 'Flags'])))
        regions.sort()
        return regions


def test_model(file_test):
    """
    This test does not require instantiating a GUI, and tests only the model logic
    """
    model = Model(pd.read_csv(file_test))
    expected = ['Region0R', 'Region1G', 'Region2R', 'Region3R', 'Region4R',
                'Region5G', 'Region6G', 'Region7R', 'Region8G']
    assert model.regions == expected


def test_gui(fiber_copy, file_test):
    """
    This requires a GUI instance
    """
    fiber_copy.open_photometry_csv(file=file_test)
    assert (fiber_copy.model is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fiber Photometry Copy GUI')
    parser.add_argument('-t', '--test', default=False, required=False, action='store_true', help='Run tests')
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    fiber_copy = FiberCopy()
    if args.test:
        file_test = Path(__file__).parent.joinpath('fiber_copy_test_fixture.csv')
        test_model(file_test)
        print('model tests pass !!')
        test_gui(fiber_copy, file_test)
        print('gui tests pass !!')
    else:
        sys.exit(app.exec_())
