#!/usr/bin/python

import sys, logging, json, os

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
    """
    TODO: 
    - dump & restore method to json
    
    """
    def __init__(self):
        self.ai = None
        QtGui.QWidget.__init__(self)
        uic.loadUi('integration.ui', self)
        self.all_detectors = pyFAI.detectors.ALL_DETECTORS.keys() + ["detector"]
        self.all_detectors.sort()
        self.detector.addItems([i.capitalize() for i in self.all_detectors])
        self.detector.setCurrentIndex(self.all_detectors.index("detector"))
        # Connect up the buttons.
#         self.connect(self.ui.okButton, QtCore.SIGNAL("clicked()"),
#                      self, QtCore.SLOT("accept()"))
#         self.connect(self.ui.cancelButton, QtCore.SIGNAL("clicked()"),
#                      self, QtCore.SLOT("reject()"))
        self.connect(self.file_poni, QtCore.SIGNAL("clicked()"), self.select_ponifile)
        self.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), self.dump)
        saveButton = self.buttonBox.button(QtGui.QDialogButtonBox.Save)
        self.connect(saveButton, QtCore.SIGNAL("clicked()"), self.dump)
        resetButton = self.buttonBox.button(QtGui.QDialogButtonBox.Reset)
        self.connect(resetButton, QtCore.SIGNAL("clicked()"), self.restore)
        self.connect(self.buttonBox, QtCore.SIGNAL("helpRequested()"), self.help)
        self.restore()

    def proceed(self):
        self.dump()
        print("Let's work a bit")

    def die(self):
        print("bye bye")

    def help(self):
        print("Please, help")

    def dump(self, filename=".azimint.json"):
        """
        Dump the status of the current widget to a file in JSON
        
        @param filename: path where to save the config
        @type filename: str
        
        """
        print "Dump!"
        to_save = {"poni": str(self.poni.text()),
                   "detector": str(self.detector.currentText()).lower(),
                   "wavelength":str(self.wavelength.text()),
                   "splinefile":str(self.splinefile.text()),
                   "pixel1": str(self.pixel1.text()),
                   "pixel2":str(self.pixel2.text()),
                   "dist":str(self.dist.text()),
                   "poni1":str(self.poni1.text()),
                   "poni2":str(self.poni2.text()),
                   "rot1":str(self.rot1.text()),
                   "rot2":str(self.rot2.text()),
                   "rot3":str(self.rot3.text()),
                   "do_dummy": bool(self.do_dummy.isChecked()),
                   "do_dark": bool(self.do_dark.isChecked()),
                   "do_flat": bool(self.do_flat.isChecked()),
                   "do_polarization":bool(self.do_polarization.isChecked()),
                   "val_dummy":str(self.val_dummy.text()),
                   "delta_dummy":str(self.delta_dummy.text()),
                   "mask_file":str(self.val_dummy.text()),
                   "dark_current":str(self.val_dummy.text()),
                   "flat_field":str(self.val_dummy.text()),
                   "polarization_factor":str(self.val_dummy.text()),
                   "rad_pt":str(self.rad_pt.text()),
                   "do_2D":bool(self.do_2D.isChecked()),
                   "azim_pt":str(self.rad_pt.text()),
                   }
        if self.q_nm.isChecked():
            to_save["unit"] = "q_nm^-1"
        elif self.tth_deg.isChecked():
            to_save["unit"] = "2th_deg"
        elif self.r_mm.isChecked():
            to_save["unit"] = "r_mm"
        with open(filename, "w") as myFile:
            json.dump(to_save, myFile, indent=4)

    def restore(self, filename=".azimint.json"):
        """
        restore from JSON file the status of the current widget
        
        @param filename: path where the config was saved
        @type filename: str
        
        """
        print("Restore")
        if not os.path.isfile(filename):
            logger.error("No such file: %s" % filename)
        data = json.load(open(filename))
        setup_data = {  "poni": self.poni.setText,
#        "detector": self.all_detectors[self.detector.getCurrentIndex()],
                        "wavelength":self.wavelength.setText,
                        "splinefile":self.splinefile.setText,
                        "pixel1": self.pixel1.setText,
                        "pixel2":self.pixel2.setText,
                        "dist":self.dist.setText,
                        "poni1":self.poni1.setText,
                        "poni2":self.poni2.setText,
                        "rot1":self.rot1.setText,
                        "rot2":self.rot2.setText,
                        "rot3":self.rot3.setText,
                        "do_dummy": self.do_dummy.setChecked,
                        "do_dark": self.do_dark.setChecked,
                        "do_flat": self.do_flat.setChecked,
                        "do_polarization":self.do_polarization.setChecked,
                        "val_dummy":self.val_dummy.setText,
                        "delta_dummy":self.delta_dummy.setText,
                        "mask_file":self.val_dummy.setText,
                        "dark_current":self.val_dummy.setText,
                        "flat_field":self.val_dummy.setText,
                        "polarization_factor":self.val_dummy.setText,
                        "rad_pt":self.rad_pt.setText,
                        "do_2D":self.do_2D.setChecked,
                        "azim_pt":self.rad_pt.setText,
                   }
        for key, value in setup_data.items:
            if key in data:
                value(data[key])
        if "unit" in data:
            unit = data["unit"].lower()
            if unit == "q_nm^-1":
                self.q_nm.setChecked(1)
            elif unit == "2th_deg":
                self.tth_deg.setChecked(1)
            elif unit == "r_mm":
                self.r_mm.setChecked(1)
        if "detector" in data:
            detector = data["detector"].lower()
            if detector in self.all_detectors:
                self.detector.setCurrentIndex(self.all_detectors.index(detector))

    def select_ponifile(self):
        ponifile = QtGui.QFileDialog.getOpenFileName()
        self.poni.setText(ponifile)
        self.set_ponifile(ponifile)

    def set_ponifile(self, ponifile=None):
        if ponifile is None:
            ponifile = self.poni.text()
            print ponifile
        try:
            self.ai = pyFAI.load(ponifile)
        except:
            logger.error("file %s does not look like a poni-file" % ponifile)
            return
        self.pixel1.setText(str(self.ai.pixel1))
        self.pixel2.setText(str(self.ai.pixel2))
        self.dist.setText(str(self.ai.dist))
        self.poni1.setText(str(self.ai.poni1))
        self.poni2.setText(str(self.ai.poni2))
        self.rot1.setText(str(self.ai.rot1))
        self.rot2.setText(str(self.ai.rot2))
        self.rot3.setText(str(self.ai.rot3))
        self.splinefile.setText(self.ai.detector.splineFile or "")
        name = self.ai.detector.name.lower()
        if name in self.all_detectors:
            self.detector.setCurrentIndex(self.all_detectors.index(name))
        else:
            self.detector.setCurrentIndex(self.all_detectors.index("detector"))

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = AIWidget()
    window.show()
    sys.exit(app.exec_())
