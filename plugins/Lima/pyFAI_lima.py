#!/usr/bin/python

from __future__ import with_statement, print_function
"""
pyFAI_lima

A graphical tool (based on PyQt4) for performing azimuthal integration of images coming from a camera.
No data are saved !

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/10/2013"
__satus__ = "development"

import sys
import time
import signal
import threading
import numpy
import pyFAI.worker
import pyopencl
import os 
op = os.path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyFAI")
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import SIGNAL



UIC = op.join(op.dirname(__file__), "LimaFAI.ui")
window = None



class DoubleView(QtGui.QWidget):
    def __init__(self, ip="192.168.5.19", fps=30, poni=None, json=None):
        QtGui.QWidget.__init__(self)
        try:
            uic.loadUi(UIC, self)
        except AttributeError as error:
            logger.error("I looks like your installation suffers from this bug: http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348")
            raise RuntimeError("Please upgrade your installation of PyQt (or apply the patch)")
        self.ip = str(ip)
        self.fps = float(fps)
        self.label_ip.setText(str(ip))
        self.label_fps.setText(str(fps))
        self.cam = self.iface = self.ctrl = self.acq=None
        self.cam = Basler.Camera(self.ip)
        self.iface = Basler.Interface(self.cam)
        self.ctrl = Core.CtControl(self.iface)
        self.is_playing = False
        self.connect(self.pushButton_play, SIGNAL("clicked()"), self.start_acq)
        self.connect(self.pushButton_stop, SIGNAL("clicked()"), self.stop_acq)
        self.last_frame = None
        self.timer = QtCore.QTimer()
        self.connect(self.timer, SIGNAL("timeout()"), self.update_img)
        if poni:
            worker = pyFAI.worker.Worker(ai=pyFAI.load(poni))
        elif json:
            worker = pyFAI.worker.Worker()
            worker.setJsonConfig(json)
        else:
            worker = None
        self.processLink = FaiLink(worker)
        self.extMgr = self.ctrl.externalOperation()
        self.myOp = extMgr.addOp(Core.USER_LINK_TASK, "pyFAITask", 0)
        self.myOp.setLinkTask(self.processLink)

    def start_acq(self):
        if self.is_playing: return
        self.is_playing = True
        self.acq = self.ctrl.acquisition()
        self.acq.setAcqNbFrames(0)
        self.acq.setAcqExpoTime(1.0/self.fps)
        self.ctrl.prepareAcq()
        self.ctrl.startAcq()
        while self.ctrl.getStatus().ImageCounters.LastImageReady < 1:
            time.sleep(0.1)
        self.last_frame = self.ctrl.getStatus().ImageCounters.LastImageReady
        raw_img = self.ctrl.ReadBaseImage().buffer
        fai_img = self.ctrl.ReadImage().buffer
        self.RawImg.setImage(raw_img)
        self.FaiImg.setImage(fai_img)


    def stop_acq(self):
        if self.is_playing:
            self.is_playing = False
            self.ctrl.stopAcq()
            self.timer.stop()
    
    def update_img(self):
        if self.is_playing:
            raw_img = self.ctrl.ReadBaseImage().buffer
            fai_img = self.ctrl.ReadImage().buffer
            self.RawImg.setImage(raw_img)
            self.FaiImg.setImage(fai_img)

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog [options] "
    version = "%prog " + pyFAI.version
    description = """
    pyFAI-lima is a graphical interface (based on Python/Qt4) to perform azimuthal integration
on a set of files grabbed from a Basler camera using LImA."""
    epilog = """ """
    parser = OptionParser(usage=usage, version=version, description=description, epilog=epilog)
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="switch to verbose/debug mode")
    parser.add_option("-p", "--poni",
                      dest="poni", default=None,
                      help="PONI file containing the setup")
    parser.add_option("-j", "--json",
                      dest="json", default=None,
                      help="json file containing the setup")
    parser.add_option("-f", "--fps",
                      dest="fps", default="30",
                      help="Number of frames per seconds")
    parser.add_option("-i", "--ip",
                      dest="ip", default="192.168.5.19",
                      help="IP address of the Basler camera")
    parser.add_option("-l", "--lima",
                      dest="lima", default=None,
                      help="Base installation of LImA")
    parser.add_option("--no-gui",
                      dest="gui", default=True, action="store_false",
                      help="Process the dataset without showing the user interface.")

    (options, args) = parser.parse_args()
    if options.verbose:
        logger.info("setLevel: debug")
        logger.setLevel(logging.DEBUG)
    if options.lima:
        sys.path.insert(0,options.lima)
    try:
        from Lima import Core, Basler
        from limaFAI import FaiLink
    except ImportError:
        print("Is the PYTHONPATH correctly setup? I did not manage to import Lima")
        sys.exit(1)

    if options.gui:
        app = QtGui.QApplication([])
        window = DoubleView(ip=options.ip, fps=options.fps)
        #window.set_input_data(args)
        window.show()
        sys.exit(app.exec_())
    else:
        raise Exception("No sense!")
        pass
