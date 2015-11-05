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
__date__ = "05/11/2015"
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
        self.create_connections()

    def create_connections(self):
        """Signal-slot connection
        """
        self.configureDiffraction.clicked.connect(self.configure_diffraction)
        self.runButton.clicked.connect(self.start_processing)

    def configure_diffraction(self, *arg, **kwarg):
        """
        """
        logger.info("in configure_diffraction")
        iw = IntegrateWidget(self)
        res = iw.exec_()
        if res == QtGui.QDialog.Accepted:
            self.integration_config = iw.get_config()
        print(self.integration_config)

    def start_processing(self, *arg, **kwarg):
        logger.info("in start_processing")
        if not self.integration_config:
            dialog = QtGui.QDialog(self)
            lay = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom, dialog)
            lab = QtGui.QLabel("You need to configure first the Azimuthal integration", dialog)
            lay.addWidget(lab)
            buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, dialog)
            lay.addWidget(buttonBox)
            buttonBox.accepted.connect(dialog.accept)
            buttonBox.rejected.connect(dialog.reject)
            result = dialog.exec_()
            if result == QtGui.QDialog.Accepted:
                self.configure_diffraction()
            else:
                return

