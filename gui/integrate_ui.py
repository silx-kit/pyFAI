#!/usr/bin/python

import sys, logging, json, os, time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyFAI")
from PyQt4 import QtCore, QtGui, uic
import pyFAI, fabio

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
    logger.info("Socket opened for debugging using rfoo")
except ImportError:
    logger.debug("No socket opened for debugging -> please install rfoo")


window = None
class AIWidget(QtGui.QWidget):
    """
    TODO: add progress bar at bottom & update when proceeding
    """
    def __init__(self, input_data=None):
        self.ai = pyFAI.AzimuthalIntegrator()
        self.input_data = input_data
        QtGui.QWidget.__init__(self)
        uic.loadUi('integration.ui', self)
        self.all_detectors = pyFAI.detectors.ALL_DETECTORS.keys() + ["detector"]
        self.all_detectors.sort()
        self.detector.addItems([i.capitalize() for i in self.all_detectors])
        self.detector.setCurrentIndex(self.all_detectors.index("detector"))
        self.connect(self.file_poni, QtCore.SIGNAL("clicked()"), self.select_ponifile)
        self.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), self.proceed)
        self.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), self.die)
        saveButton = self.buttonBox.button(QtGui.QDialogButtonBox.Save)
        self.connect(saveButton, QtCore.SIGNAL("clicked()"), self.dump)
        resetButton = self.buttonBox.button(QtGui.QDialogButtonBox.Reset)
        self.connect(resetButton, QtCore.SIGNAL("clicked()"), self.restore)
        self.connect(self.buttonBox, QtCore.SIGNAL("helpRequested()"), self.help)
        self.restore()
        self.progressBar.setValue(0)

    def proceed(self):
        self.dump()
        print("Let's work a bit")
        self.set_ai()



#                   "polarization_factor":str(self.val_dummy.text()).strip(),
#                   "rad_pt":str(self.rad_pt.text()).strip(),
#                   "do_2D":bool(self.do_2D.isChecked()),
#                   "azim_pt":str(self.rad_pt.text()).strip(),
#                   }

        for i in range(100):
            self.progressBar.setValue(i)
            time.sleep(0.1)
        self.die()

    def die(self):
        print("bye bye")
        self.deleteLater()

    def help(self):
        print("Please, help")

    def dump(self, filename=".azimint.json"):
        """
        Dump the status of the current widget to a file in JSON

        @param filename: path where to save the config
        @type filename: str

        """
        print "Dump!"
        to_save = {"poni": str(self.poni.text()).strip(),
                   "detector": str(self.detector.currentText()).lower(),
                   "wavelength":str(self.wavelength.text()).strip(),
                   "splinefile":str(self.splinefile.text()).strip(),
                   "pixel1": str(self.pixel1.text()).strip(),
                   "poni1":str(self.pixel2.text()).strip(),
                   "dist":str(self.dist.text()).strip(),
                   "poni1":str(self.poni1.text()).strip(),
                   "poni2":str(self.poni2.text()).strip(),
                   "rot1":str(self.rot1.text()).strip(),
                   "rot2":str(self.rot2.text()).strip(),
                   "rot3":str(self.rot3.text()).strip(),
                   "do_dummy": bool(self.do_dummy.isChecked()),
                   "do_dark": bool(self.do_dark.isChecked()),
                   "do_flat": bool(self.do_flat.isChecked()),
                   "do_polarization":bool(self.do_polarization.isChecked()),
                   "val_dummy":str(self.val_dummy.text()).strip(),
                   "delta_dummy":str(self.delta_dummy.text()).strip(),
                   "mask_file":str(self.val_dummy.text()).strip(),
                   "dark_current":str(self.val_dummy.text()).strip(),
                   "flat_field":str(self.val_dummy.text()).strip(),
                   "polarization_factor":str(self.val_dummy.text()).strip(),
                   "rad_pt":str(self.rad_pt.text()).strip(),
                   "do_2D":bool(self.do_2D.isChecked()),
                   "azim_pt":str(self.rad_pt.text()).strip().strip(),
                   }
        if self.q_nm.isChecked():
            to_save["unit"] = "q_nm^-1"
        elif self.tth_deg.isChecked():
            to_save["unit"] = "2th_deg"
        elif self.r_mm.isChecked():
            to_save["unit"] = "r_mm"
        with open(filename, "w") as myFile:
            json.dump(to_save, myFile, indent=4)
        print("Saved")

    def restore(self, filename=".azimint.json"):
        """
        restore from JSON file the status of the current widget

        @param filename: path where the config was saved
        @type filename: str

        """
        print("Restore")
        if not os.path.isfile(filename):
            logger.error("No such file: %s" % filename)
            return
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
        for key, value in setup_data.items():
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

    def set_input_data(self, data):
        self.input_data = data

    def _float(self, kw, default=0):
        fval = default
        txtval = str(self.__dict__[kw].text()).strip()
        if txtval:
            try:
                fval = float(txtval)
            except ValueError:
                logger.error("Unable to convert %s to float: %s" % (kw, txtval))
        return fval

    def set_ai(self):
        poni = str(self.poni.text()).strip()
        detector = str(self.detector.currentText()).lower().strip() or "detector"
        self.ai.detector = pyFAI.detectors.detector_factory(detector)

        wavelength = str(self.wavelength.text()).strip()
        if wavelength:
            try:
                fwavelength = float(wavelength)
            except ValueError:
                logger.error("Unable to convert wavelength to float: %s" % wavelength)
            else:
                if fwavelength <= 0 or fwavelength > 1e-6:
                    logger.warning("Wavelength is in meter ... unlikely value %s" % fwavelength)
                self.ai.wavelength = fwavelength

        splinefile = str(self.splinefile.text()).strip()
        if splinefile and os.path.isfile(splinefile):
            self.ai.detector.splineFile = splinefile

        self.ai.pixel1 = self._float("pixel1", 1)
        self.ai.pixel2 = self._float("pixel2", 1)
        self.ai.dist = self._float("dist", 1)
        self.ai.poni1 = self._float("poni1", 0)
        self.ai.poni2 = self._float("poni2", 0)
        self.ai.rot1 = self._float("rot1", 0)
        self.ai.rot2 = self._float("rot2", 0)
        self.ai.rot3 = self._float("rot3", 0)

#                   "do_dummy": bool(self.do_dummy.isChecked()),
#                   "do_polarization":bool(self.do_polarization.isChecked()),
#                   "val_dummy":str(self.val_dummy.text()).strip(),
#                   "delta_dummy":str(self.delta_dummy.text()).strip(),
        mask_file = str(self.val_dummy.text()).strip()
        if mask_file and os.path.exists(mask_file):
            try:
                mask = fabio.open(mask_file).data
            except Exception:
                logger.error("Unable to load mask file %s" % maskfile)
            else:
                self.ai.mask = mask
        dark_files = [i.strip() for i in str(self.val_dummy.text()).split(",")
                      if os.path.isfile(i.strip())]
        if dark_files:
            d0 = fabio.open(dark_files[0]).data
            darks = numpy.zeros(d0.shape[0], d0.shape[1], len(dark_files), dtype=numpy.float32)
            for i, f in enumerate(dark_files):
                darks[:, :, i] = fabio.open(f).data
            self.ai.darkcurrent = darks.mean(axis= -1)

        flat_files = [i.strip() for i in str(self.val_dummy.text()).split(",")
                      if os.path.isfile(i.strip())]
        if flat_files:
            d0 = fabio.open(flat_files[0]).data
            flats = numpy.zeros(d0.shape[0], d0.shape[1], len(flat_files), dtype=numpy.float32)
            for i, f in enumerate(flat_files):
                flats[:, :, i] = fabio.open(f).data
            self.ai.darkcurrent = flats.mean(axis= -1)
#                   "do_dark": bool(self.do_dark.isChecked()),
#                   "do_flat": bool(self.do_flat.isChecked()),

        print self.ai

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = AIWidget()
    window.show()
    sys.exit(app.exec_())
