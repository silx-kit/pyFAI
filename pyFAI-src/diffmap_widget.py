# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from __future__ import absolute_import, print_function, division

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/11/2015"
__status__ = "development"
__docformat__ = 'restructuredtext'
__doc__ = """

Module with GUI for diffraction mapping experiments 


"""
from collections import namedtuple
from .gui_utils import QtGui, QtCore, uic
from .utils import float_, int_, str_, get_ui_file
from .integrate_widget import AIWidget
from .io import is_hdf5
import logging
logger = logging.getLogger("diffmap_widget")

DataSetNT = namedtuple("DataSet", ("path", "h5", "nframes"))

class DataSet(object):
    def __init__(self, path, h5=None, nframes=None, shape=None):
        self.path = path
        self.h5 = h5
        self.nframes = nframes
        self.shape = shape

    def as_tuple(self):
        return DataSetNT(self.path, self.h5, self.nframes)

    def is_hdf5(self):
        """Return True if the object is hdf5"""
        if self.h5 is None:
            self.h5 = is_hdf5(self.path)
        return bool(self.h5)
#     def __getitem__(self, item):
#         if item in ("path", "h5", "nframes", "shape"):
#             return self.__getattribute__(item)

class ListDataSet(list):
    pass


class IntegrateWidget(QtGui.QDialog):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self)
        self.widget = AIWidget()
        self.layout = QtGui.QGridLayout(self)
        self.layout.addWidget(self.widget)
        self.widget.okButton.clicked.disconnect()
        self.widget.cancelButton.clicked.disconnect()
        self.widget.okButton.clicked.connect(self.accept)
        self.widget.cancelButton.clicked.connect(self.reject)

    def get_config(self):
        res = self.widget.dump()
        res["method"] = self.widget.get_method()
        return res


class ListModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None, actual_data=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._ref = actual_data
#     def dropMimeData(self, *args, **kwargs):
#         print("ListModel.dropMimeData %s %s" % (args, kwargs))
    def rowCount(self, parent=None):
        return len(self._ref)
    def columnCount(self, parent=None):
        return 3
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if index.row() >= len(self._ref):
                return
            data = self._ref[index.row()]
            if index.column() == 0:
                return data.path
            elif index.column() == 1:
                return data.h5
            if index.column() == 2:
                return data.nframes

    def setData(self, *args, **kwargs):
        return True

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal:
            if role == QtCore.Qt.DisplayRole:
                if section == 0:
                    return "File path"
                elif section == 1:
                    return "h5"
                elif section == 2:
                    return "#"
            elif role == QtCore.Qt.WhatsThisRole:
                if section == 0:
                    return "Path of the file in the computer"
                elif section == 1:
                    return "Internal path in the HDF5 tree"
                elif section == 2:
                    return "Number of frames in the dataset"
            elif role == QtCore.Qt.SizeHintRole:
                if section == 0:
                    return QtCore.QSize(200, 20)
                elif section == 1:
                    return QtCore.QSize(20, 20)
                elif section == 2:
                    return QtCore.QSize(20, 20)



class DiffMapWidget(QtGui.QWidget):

    uif = "diffmap.ui"

    def __init__(self):
        QtGui.QWidget.__init__(self)

        self.integration_config = {}
        self.list_dataset = []  # Contains all datasets to be treated.

        try:
            uic.loadUi(get_ui_file(self.uif), self)
        except AttributeError as error:
            logger.error("I looks like your installation suffers from this bug: http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348")
            raise RuntimeError("Please upgrade your installation of PyQt (or apply the patch)")
        self.aborted = False
        self.listModel = ListModel(self, self.list_dataset)
        self.listFiles.setModel(self.listModel)
        self.listFiles.hideColumn(1)
        self.listFiles.hideColumn(2)
        self.create_connections()
        self.set_validator()
        self.update_number_of_frames()
        self.update_number_of_points()

    def set_validator(self):
        validator = QtGui.QIntValidator(0, 999999, self)
        self.fastMotorPts.setValidator(validator)
        self.slowMotorPts.setValidator(validator)
        self.offset.setValidator(validator)

    def create_connections(self):
        """Signal-slot connection
        """
        self.configureDiffraction.clicked.connect(self.configure_diffraction)
        self.outputFileSelector.clicked.connect(self.configure_output)
        self.runButton.clicked.connect(self.start_processing)
        self.addFiles.clicked.connect(self.input_filer)
        self.sortButton.clicked.connect(self.sort_input)

        self.fastMotorPts.editingFinished.connect(self.update_number_of_points)
        self.slowMotorPts.editingFinished.connect(self.update_number_of_points)
        self.offset.editingFinished.connect(self.update_number_of_points)



    def input_filer(self, *args, **kwargs):
        """
        Called when addFiles clicked: opens a file-brower and populates the 
        listFiles object
        """
        fnames = QtGui.QFileDialog.getOpenFileNames(self,
                         "Select one or more diffraction image files",
                         QtCore.QDir.currentPath(),
                         filter=self.tr("EDF image files (*.edf);;TIFF image files (*.tif);;CBF files (*.cbf);;MarCCD image files (*.mccd);;Any file (*)"))
                         # filter=self.tr("NeXuS files (*.nxs);;HDF5 files (*.h5);;HDF5 files (*.hdf5);;EDF image files (*.edf);;TIFF image files (*.tif);;CBF files (*.cbf);;MarCCD image files (*.mccd);;Any file (*)"))
        for i in fnames:
            self.list_dataset.append(DataSet(i, None, None, None))
        self.listModel.reset()
        self.update_number_of_frames()



    def configure_diffraction(self, *arg, **kwarg):
        """
        """
        logger.info("in configure_diffraction")
        iw = IntegrateWidget(self)
        res = iw.exec_()
        if res == QtGui.QDialog.Accepted:
            self.integration_config = iw.get_config()
        print(self.integration_config)

    def configure_output(self, *args, **kwargs):
        """
        called when clicking on "outputFileSelector"
        """
        fname = QtGui.QFileDialog.getSaveFileName(self, "Output file",
                                                  QtCore.QDir.currentPath(),
                                                  filter=self.tr("NeXuS file (*.nxs);;HDF5 file (*.h5);;HDF5 file (*.hdf5)"))
        self.outputFile.setText(fname)


    def start_processing(self, *arg, **kwarg):
        logger.info("in start_processing")
        if not self.integration_config:
            result = QtGui.QMessageBox.warning(self, "Azimuthal Integration",
                                                   "You need to configure first the Azimuthal integration")
            if result:
                self.configure_diffraction()
            else:
                return
        if not str(self.outputFile.text()):
            result = QtGui.QMessageBox.warning(self, "Destination",
                                                   "You need to configure first the destination file")
            if result:
                self.configure_output()
            else:
                return
        config = self.get_config()
        self.process(config)

    def update_number_of_frames(self):
        cnt = len(self.list_dataset)
        self.numberOfFrames.setText(str(cnt))

    def update_number_of_points(self):
        try:
            slow = int(self.slowMotorPts.text())
        except:
            slow = 1
        try:
            fast = int(self.fastMotorPts.text())
        except:
            fast = 1
        try:
            offset = int(self.offset.text())
        except:
            offset = 0
        self.numberOfPoints.setText(str(slow * fast + offset))

    def sort_input(self):
        self.list_dataset.sort(key=lambda i: i.path)
        self.listModel.reset()

    def get_config(self):
        """
        Return a dict with the plugin configuration which is JSON-serializable 
        """
        res = {
               "ai": self.integration_config,
               "experiment_title": str_(self.experimentTitle.text()).strip(),
               "fast_motor_name": str_(self.fastMotorName.text()).strip(),
               "slow_motor_name": str_(self.slowMotorName.text()).strip(),
               "fast_motor_points": int_(self.fastMotorPts.text()),
               "slow_motor_points": int_(self.slowMotorPts.text()),
               "offset": int_(self.offset.text()),
               "output_file": str_(self.outputFile.text()).strip(),
               "input_data": [i.as_tuple() for i in self.list_dataset]
               }
        return res

    def process(self, config=None):
        """
        Called 
        """
        logger.info("process")
        config_ai = config.get("ai",{})
        diffmap = DiffMap(npt_fast=config.get("fast_motor_points",1), 
                          npt_slow=config.get("slow_motor_points",1), 
                          npt_rad=config_ai.get("nbpt_rad",1000), 
                          npt_azim=config_ai.get("nbpt_azim",1) if config_ai.get("do_2D") else None))
        