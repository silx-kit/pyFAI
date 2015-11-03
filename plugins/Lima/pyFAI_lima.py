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
__date__ = "01/11/2015"
__satus__ = "development"

import sys
import time
import signal
import threading
import numpy
import pyFAI.worker
from pyFAI import io
import pyopencl
import os
op = os.path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyFAI")
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import SIGNAL
import pyqtgraph as pg


UIC = op.join(op.dirname(__file__), "pyFAI_lima.ui")
window = None


class DoubleView(QtGui.QWidget):
    def __init__(self, ip="192.168.5.19", fps=30, poni=None, json=None, writer=None, cake=None):
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
        self.cam = self.iface = self.ctrl = self.acq = None
        self.cam = Basler.Camera(self.ip)
        self.iface = Basler.Interface(self.cam)
        self.ctrl = Core.CtControl(self.iface)
        self.is_playing = False
        self.cake = int(cake)
        self.connect(self.pushButton_play, SIGNAL("clicked()"), self.start_acq)
        self.connect(self.pushButton_stop, SIGNAL("clicked()"), self.stop_acq)
        self.last_frame = None
        self.last = time.time()
        if poni:
            worker = pyFAI.worker.Worker(ai=pyFAI.load(poni))
        elif json:
            worker = pyFAI.worker.Worker()
            worker.setJsonConfig(json)
        else:
            worker = None

        self.processLink = LinkPyFAI(worker, writer)
        self.extMgr = self.ctrl.externalOperation()
        self.myOp = self.extMgr.addOp(Core.USER_LINK_TASK, "pyFAILink", 0)
        self.myOp.setLinkTask(self.processLink)

        self.callback = StartAcqCallback(self.ctrl, self.processLink)
        self.myOp.registerCallback(self.callback)
        self.timer = QtCore.QTimer()
        self.connect(self.timer, SIGNAL("timeout()"), self.update_img)
        self.writer = writer
        self.dLayout = QtGui.QHBoxLayout(self.frame)
        if self.cake <= 1:
            self.variablePlot = pg.PlotWidget(parent=self.frame)
        else:
            self.variablePlot = pg.ImageView(parent=self.frame)
        self.dLayout.addWidget(self.variablePlot)

    def start_acq(self):
        if self.is_playing: return
        self.is_playing = True
        self.acq = self.ctrl.acquisition()
        self.acq.setAcqNbFrames(0)
        self.acq.setAcqExpoTime(1.0 / self.fps)
        self.ctrl.prepareAcq()
        if self.cake != self.processLink._worker.nbpt_azim:
            self.processLink._worker.nbpt_azim = int(self.cake)
            self.ctrl.prepareAcq()
        self.ctrl.startAcq()
        while self.ctrl.getStatus().ImageCounters.LastImageReady < 1:
            time.sleep(0.1)
        self.last_frame = self.ctrl.getStatus().ImageCounters.LastImageReady
        raw_img = self.ctrl.ReadBaseImage().buffer
        fai_img = self.ctrl.ReadImage().buffer
        self.RawImg.setImage(raw_img.T)  # , levels=[0, 4096])#, autoLevels=False, autoRange=False)
        if self.cake <= 1:
            for  i in self.variablePlot.plotItem.items[:]:
                self.variablePlot.plotItem.removeItem(i)
            self.variablePlot.plot(fai_img[:, 0], fai_img[:, 1])

        else:
            self.variablePlot.setImage(fai_img.T)  # , levels=[0, 4096])#, autoLevels=False, autoRange=False)
        self.last = time.time()
        self.timer.start(1000.0 / self.fps)

    def stop_acq(self):
        if self.is_playing:
            self.is_playing = False
            self.ctrl.stopAcq()
            self.timer.stop()

    def update_img(self):
        last_frame = self.ctrl.getStatus().ImageCounters.LastImageReady
        if last_frame == self.last_frame:
            return
        if self.is_playing:
            raw_img = self.ctrl.ReadBaseImage().buffer
            fai_img = self.ctrl.ReadImage().buffer
            self.RawImg.setImage(raw_img.T)  # , levels=[0, 4096])#, autoLevels=False, autoRange=False)
            if self.cake <= 1:
                self.variablePlot.plotItem.plot(fai_img[:, 0], fai_img[:, 1])
                self.variablePlot.plotItem.removeItem(self.variablePlot.plotItem.items[0])
            else:
                self.variablePlot.setImage(fai_img.T)  # , levels=[0, 4096])#, autoLevels=False, autoRange=False)
            print("Measured display speed: %5.2f fps" % (1.0 / (time.time() - self.last)))
            self.last = time.time()


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
    parser.add_option("-s", "--scan",
                      dest="scan", default=None,
                      help="Size of scan of the fastest motor")
    parser.add_option("-c", "--cake", action="store", type="int",
                      dest="cake", default=0,
                      help="Perform 2D caking, in so many slices instead of full integration, a reasonable value is 360")

    parser.add_option("--no-gui",
                      dest="gui", default=True, action="store_false",
                      help="Process the dataset without showing the user interface.")

    (options, args) = parser.parse_args()
    if len(args) == 1:
        hurl = args[0]
        if os.path.isdir(hurl):
            # write .dat or .edf files ...
            if options.cake < 2:
                writer = io.AsciiWriter(hurl)
        # Else HDF5
        else:
            if hurl.startswith("hdf5:"):
                hurl = hurl[5:]
            if ":" in hurl:
                hsplit = hurl.split(":")
                hdfpath = hsplit[-1]
                hdffile = ":".join(hsplit[:-1])  # special windows
            else:
                hdfpath = "test_LImA+pyFAI"
                hdffile = hurl
            writer = io.HDF5Writer(hdffile, hdfpath, options.scan)
    elif len(args) > 1:
        logger.error("Specify the HDF5 output file like hdf5:///home/user/filename.h5:/path/to/group")
        sys.exit(1)
    else:
        writer = None

    if options.verbose:
        logger.info("setLevel: debug")
        logger.setLevel(logging.DEBUG)
    if options.lima:
        sys.path.insert(0, options.lima)
    try:
        from Lima import Core, Basler

    except ImportError as error:
        print("Is the PYTHONPATH correctly setup? I did not manage to import Lima and got: %s" % error)
        sys.exit(1)
    from limaFAI import LinkPyFAI, StartAcqCallback
    if options.gui:
        app = QtGui.QApplication([])
        window = DoubleView(ip=options.ip, fps=options.fps, writer=writer, cake=options.cake)
        # window.set_input_data(args)
        window.show()
        sys.exit(app.exec_())
    else:
        raise Exception("No sense!")

