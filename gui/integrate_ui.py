#!/usr/bin/python

import sys,logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyFAI")
from PyQt4 import QtCore, QtGui, uic
import pyFAI

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
    logger.info("Socket opened for debugging using rfoo")
except ImportError:
    logger.debug("No socket opened for debugging -> please install rfoo")


window = None
class AIWidget(QtGui.QWidget):
    def __init__(self):
        self.ai=None
        QtGui.QWidget.__init__(self)
        uic.loadUi('integration.ui', self)
        # Connect up the buttons.
#         self.connect(self.ui.okButton, QtCore.SIGNAL("clicked()"),
#                      self, QtCore.SLOT("accept()"))
#         self.connect(self.ui.cancelButton, QtCore.SIGNAL("clicked()"),
#                      self, QtCore.SLOT("reject()"))
        self.connect(self.file_poni, QtCore.SIGNAL("clicked()"), self.select_ponifile)
    def select_ponifile(self):
        ponifile = QtGui.QFileDialog.getOpenFileName()
        self.poni.setText(ponifile)
        try:
            self.ai = pyFAI.load(ponifile)
        except:
            logger.error("file %s does not look like a poni-file" % ponifile)
            return
        self.pixel1.setText(str(self.ai.pixel1))
        self.pixel2.setText(str(self.ai.pixel2))
        self.distance.setText(str(self.ai.dist))
        self.poni1.setText(str(self.ai.poni1))
        self.poni2.setText(str(self.ai.poni2))
        self.rot1.setText(str(self.ai.rot1))
        self.rot2.setText(str(self.ai.rot2))
        self.rot3.setText(str(self.ai.rot3))
        self.splinefile.setText(self.ai.detector.splineFile)
if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    window = AIWidget()
    window.show()
    sys.exit(app.exec_())
