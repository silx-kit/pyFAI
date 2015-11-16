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
__date__ = "16/11/2015"
__status__ = "development"
__docformat__ = 'restructuredtext'
__doc__ = """

Module with GUI for diffraction mapping experiments 


"""
# __all__ = ["date", "version_info", "strictversion", "hexversion"]

from .gui_utils import QtGui, QtCore, uic
from .utils import float_, int_, str_, get_ui_file
from .integrate_widget import AIWidget
import logging
logger = logging.getLogger("diffmap_widget")


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
        return self.widget.dump()

# class ListModel(QtCore.QAbstractListModel):
#     def __init__(self, *args, **kwargs):
#         QtCore.QAbstractListModel.__init__(self, *args, **kwargs)
#         self.__list = []
#     def dropMimeData(self, *args, **kwargs):
#         print("ListModel.dropMimeData %s %s" % (args, kwargs))
#     def count(self):
#         return(len(self.__list))
#     def count(self):
#         return(len(self.__list))

class DiffMapWidget(QtGui.QWidget):

    uif = "diffmap.ui"

    def __init__(self):
        QtGui.QWidget.__init__(self)

        self.integration_config = {}

        try:
            uic.loadUi(get_ui_file(self.uif), self)
        except AttributeError as error:
            logger.error("I looks like your installation suffers from this bug: http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348")
            raise RuntimeError("Please upgrade your installation of PyQt (or apply the patch)")
        self.aborted = False
        self.listModel = QtGui.QStringListModel(self)
        self.listFiles.setModel(self.listModel)
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
                         filter=self.tr("NeXuS files (*.nxs);;HDF5 files (*.h5);;HDF5 files (*.hdf5);;EDF image files (*.edf);;TIFF image files (*.tif);;CBF files (*.cbf);;MarCCD image files (*.mccd);;Any file (*)"))
        for i in fnames:
            self.listModel.addItem(i)
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

    def update_number_of_frames(self):
        cnt = len(self.listModel.stringList())
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
        self.numberOfPoints.setText(str(slow * fast - offset))
