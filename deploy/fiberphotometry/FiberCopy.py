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
from dataclasses import dataclass, field
import datetime

import pandas as pd

from PyQt5 import QtWidgets, QtCore, uic
"""
settings values:
    path_fiber_photometry: str - last path of fiber photometry csv file
    path_server_sessions: str - destination path for local lab server, i.e. Y:\Subjects\something
    subjects: list[str] = field(default_factory=list) - list of subjects should carry over between sessions
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
        self.dateEdit.setDate(datetime.datetime.now())
        self.pushButton_add_session.clicked.connect(self.add_session)
        self.pushButton_rm_session.clicked.connect(self.remove_session)
        self.pushButton_export.clicked.connect(self.export_file)
        self.comboBox_subjects.addItems(self.settings.value("subjects", []))
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
        # TODO reset table view
        self.listWidget_rois.clear()
        self.listWidget_rois.addItems(self.model.regions)

    def add_session(self):
        str_subject = self.comboBox_subjects.currentText()
        str_date = self.dateEdit.date().toString(QtCore.Qt.ISODate)
        str_number = str(self.spinBox_number.value()).zfill(3)
        list_rois = [i.text() for i in self.listWidget_rois.selectedItems()]
        if str_subject == '' or len(list_rois) == 0:
            return
        else:
            self.settings.setValue("subjects", list(set(self.settings.value("subjects", []) + [str_subject])))
        items = [f"{str_subject}/{str_number}/{str_number}: {roi}" for roi in list_rois]
        items_in_list = [str(self.listWidget_sessions.item(i).text()) for i in range(self.listWidget_sessions.count())]
        self.listWidget_sessions.clear()
        self.listWidget_sessions.addItems(list(set(items + items_in_list)))

    def remove_session(self):
        for i in self.listWidget_sessions.selectedItems():
            self.listWidget_sessions.takeItem(self.listWidget_sessions.row(i))

    def export_file(self):
        pass

@dataclass
class Model:
    """Class to store the necessary data"""
    dataframe: pd.DataFrame

    @property
    def regions(self):
        regions = list(set(self.dataframe.keys()).difference(set(['FrameCounter', 'Timestamp', 'Flags'])))
        regions.sort()
        return regions


def test_model(ft):
    """
    This test does not require instantiating a GUI, and tests only the model logic
    """
    model = Model(pd.read_csv(ft))
    assert(model.regions == ['Region0R', 'Region1G', 'Region2R', 'Region3R', 'Region4R', 'Region5G', 'Region6G', 'Region7R', 'Region8G'])


def test_controller(fiber_copy, file_test):
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
        file_test = Path("fiber_copy_test_fixture.csv")
        # file_test = Path(__file__).parent.joinpath('fiber_copy_test_fixture.csv')
        test_model(file_test)
        print('model tests pass !!')
        test_controller(fiber_copy, file_test)
        print('controller tests pass !!')
    else:
        sys.exit(app.exec_())
