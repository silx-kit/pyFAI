#!/usr/bin/python

import sys, logging, json, os, time, types, threading
import os.path as op
import numpy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyFAI")
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import SIGNAL
import pyFAI, fabio
from pyFAI.opencl import ocl
from pyFAI.utils import float_, int_

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
    logger.info("Socket opened for debugging using rfoo")
except ImportError:
    logger.debug("No socket opened for debugging -> please install rfoo")


window = None
class AIWidget(QtGui.QWidget):
    """
    """
    def __init__(self, input_data=None):
        self.ai = pyFAI.AzimuthalIntegrator()
        self.input_data = input_data
        self._sem = threading.Semaphore()
        QtGui.QWidget.__init__(self)
        uic.loadUi('integration.ui', self)
        self.all_detectors = pyFAI.detectors.ALL_DETECTORS.keys()
        self.all_detectors.sort()
        self.detector.addItems([i.capitalize() for i in self.all_detectors])
        self.detector.setCurrentIndex(self.all_detectors.index("detector"))
        #connect file selection windows
        self.connect(self.file_poni, SIGNAL("clicked()"), self.select_ponifile)
        self.connect(self.file_splinefile, SIGNAL("clicked()"), self.select_splinefile)
        self.connect(self.file_mask_file, SIGNAL("clicked()"), self.select_maskfile)
        self.connect(self.file_dark_current, SIGNAL("clicked()"), self.select_darkcurrent)
        self.connect(self.file_flat_field, SIGNAL("clicked()"), self.select_flatfield)
        # connect button bar
        self.okButton = self.buttonBox.button(QtGui.QDialogButtonBox.Ok)
        self.saveButton = self.buttonBox.button(QtGui.QDialogButtonBox.Save)
        self.resetButton = self.buttonBox.button(QtGui.QDialogButtonBox.Reset)
        self.connect(self.okButton, SIGNAL("clicked()"), self.proceed)
        self.connect(self.saveButton, SIGNAL("clicked()"), self.dump)
        self.connect(self.buttonBox, SIGNAL("helpRequested()"), self.help)
        self.connect(self.buttonBox, SIGNAL("rejected()"), self.die)
        self.connect(self.resetButton, SIGNAL("clicked()"), self.restore)

        self.connect(self.detector, SIGNAL("currentIndexChanged(int)"), self.detector_changed)
        self.connect(self.do_OpenCL, SIGNAL("clicked()"), self.openCL_changed)
        self.connect(self.platform, SIGNAL("currentIndexChanged(int)"), self.platform_changed)

        self.restore()
        self.progressBar.setValue(0)

    def proceed(self):
        with self._sem:
            out = None
            self.dump()
            logger.debug("Let's work a bit")
            self.set_ai()

    #        Default Keyword arguments
            kwarg = {"unit": "2th_deg",
                     "dummy": None,
                     "delta_dummy": None,
                     "method": "lut",
                     "polarization_factor":0,
                     "filename": None,
                     "safe": False,
                     }

            if self.q_nm.isChecked():
                kwarg["unit"] = "q_nm^-1"
            elif self.tth_deg.isChecked():
                kwarg["unit"] = "2th_deg"
            elif self.r_mm.isChecked():
                kwarg["unit"] = "r_mm"
            else:
                logger.warning("Undefined unit !!! falling back on 2th_deg")


            if bool(self.do_dummy.isChecked()):
                kwarg["dummy"] = float_(self.val_dummy.text())
                delta_dummy = str(self.delta_dummy.text())
                if delta_dummy:
                    kwarg["delta_dummy"] = float(delta_dummy)
                else:
                    kwarg["delta_dummy"] = None
            if bool(self.do_polarization.isChecked()):
                kwarg["polarization_factor"] = float(self.polarization_factor.value())

            kwarg["nbPt_rad"] = int(str(self.rad_pt.text()).strip())
            if self.do_2D.isChecked():
                kwarg["nbPt_azim"] = int(str(self.rad_pt.text()).strip())

            if self.do_OpenCL.isChecked():
                platform = ocl.get_platform(self.platform.currentText())
                pid = platform.id
                did = platform.get_device(self.device.currentText()).id
                if (pid is not None) and (did is not None):
                    kwarg["method"] = "lut_ocl_%i,%i" % (pid, did)
                else:
                    kwarg["method"] = "lut_ocl"

            if self.do_radial_range:
                try:
                    rad_min = float_(self.radial_range_min.text())
                    rad_max = float_(self.radial_range_max.text())
                except ValueError as error:
                    logger.error("error in parsing radial range: %s" % error)
                else:
                    kwarg["radial_range"] = (rad_min, rad_max)

            if self.do_azimuthal_range:
                try:
                    azim_min = float_(self.azimuth_range_min.text())
                    azim_max = float_(self.azimuth_range_max.text())
                except ValueError as error:
                    logger.error("error in parsing azimuthal range: %s" % error)
                else:
                    kwarg["azimuth_range"] = (azim_min, azim_max)

            logger.info("Parameters for integration:%s%s" % (os.linesep,
                            os.linesep.join(["\t%s:\t%s" % (k, v) for k, v in kwarg.items()])))

            logger.debug("processing %s" % self.input_data)
            start_time = time.time()
            if self.input_data in [None, []]:
                logger.warning("No input data to process")
                return

            elif "ndim" in dir(self.input_data) and (self.input_data.ndim == 3):
                # We have a numpy array of dim3
                if "nbPt_azim" in kwarg:
                    out = numpy.zeros((self.input_data.shape[0], kwarg["nbPt_azim"], kwarg["nbPt_rad"]), dtype=numpy.float32)
                    for i in range(self.input_data.shape[0]):
                        self.progressBar.setValue(100.0 * i / self.input_data.shape[0])
                        kwarg["data"] = self.input_data[i]
                        out[i] = self.ai.integrate2d(kwarg)[0]

                else:
                    out = numpy.zeros((self.input_data.shape[0], kwarg["nbPt_rad"]), dtype=numpy.float32)
                    for i in range(self.input_data.shape[0]):
                        self.progressBar.setValue(100.0 * i / self.input_data.shape[0])
                        kwarg["data"] = self.input_data[i]
                        out[i] = self.ai.integrate1d(**kwarg)[1]

            elif "__len__" in dir(self.input_data):
                out = []
                for i, item in enumerate(self.input_data):
                    self.progressBar.setValue(100.0 * i / len(self.input_data))
                    logger.debug("processing %s" % item)
                    if (type(item) in types.StringTypes) and op.exists(item):
                        kwarg["data"] = fabio.open(item).data
                        if "nbPt_azim" in kwarg:
                            kwarg["filename"] = op.splitext(item)[0] + ".azim"
                        else:
                            kwarg["filename"] = op.splitext(item)[0] + ".dat"
                    else:
                        logger.warning("item is not a file ... guessing it is a numpy array")
                        kwarg["data"] = item
                        kwarg["filename"] = None
                    if "nbPt_azim" in kwarg:
                        out.append(self.ai.integrate2d(**kwarg)[0])
                    else:
                        out.append(self.ai.integrate1d(**kwarg)[0])

            logger.info("Processing Done in %.3fs !" % (time.time() - start_time))
            self.progressBar.setValue(100)
        self.die()
        return out

    def die(self):
        logger.debug("bye bye")
        self.deleteLater()

    def help(self):
        logger.debug("Please, help")

    def dump(self, filename=".azimint.json"):
        """
        Dump the status of the current widget to a file in JSON

        @param filename: path where to save the config
        @type filename: str

        """
        print "Dump!"
        to_save = { "poni": str(self.poni.text()).strip(),
                    "detector": str(self.detector.currentText()).lower(),
                    "wavelength":float_(self.wavelength.text()),
                    "splineFile":str(self.splineFile.text()).strip(),
                    "pixel1": float_(self.pixel1.text()),
                    "pixel2":float_(self.pixel2.text()),
                    "dist":float_(self.dist.text()),
                    "poni1":float_(self.poni1.text()).strip(),
                    "poni2":float_(self.poni2.text()).strip(),
                    "rot1":float_(self.rot1.text()).strip(),
                    "rot2":float_(self.rot2.text()).strip(),
                    "rot3":float_(self.rot3.text()).strip(),
                    "do_dummy": bool(self.do_dummy.isChecked()),
                    "do_mask":  bool(self.do_mask.isChecked()),
                    "do_dark": bool(self.do_dark.isChecked()),
                    "do_flat": bool(self.do_flat.isChecked()),
                    "do_polarization":bool(self.do_polarization.isChecked()),
                    "val_dummy":float_(self.val_dummy.text()).strip(),
                    "delta_dummy":float_(self.delta_dummy.text()).strip(),
                    "mask_file":str(self.mask_file.text()).strip(),
                    "dark_current":str(self.dark_current.text()).strip(),
                    "flat_field":str(self.flat_field.text()).strip(),
                    "polarization_factor":float_(self.polarization_factor.value()),
                    "rad_pt":int_(self.rad_pt.text()),
                    "do_2D":bool(self.do_2D.isChecked()),
                    "azim_pt":int_(self.rad_pt.text()),
                    "chi_discontinuity_at_0": bool(self.chi_discontinuity_at_0.isChecked()),
                    "do_radial_range": bool(self.do_radial_range.isChecked()),
                    "do_azimuthal_range": bool(self.do_azimuthal_range.isChecked()),
                    "radial_range_min":float_(self.radial_range_min.text()),
                    "radial_range_max":float_(self.radial_range_max.text()),
                    "azimuth_range_min":float_(self.azimuth_range_min.text()),
                    "azimuth_range_max":float_(self.azimuth_range_max.text()),
                   }
        if self.q_nm.isChecked():
            to_save["unit"] = "q_nm^-1"
        elif self.tth_deg.isChecked():
            to_save["unit"] = "2th_deg"
        elif self.r_mm.isChecked():
            to_save["unit"] = "r_mm"
        with open(filename, "w") as myFile:
            json.dump(to_save, myFile, indent=4)
        logger.debug("Saved")

    def restore(self, filename=".azimint.json"):
        """
        restore from JSON file the status of the current widget

        @param filename: path where the config was saved
        @type filename: str

        """
        logger.debug("Restore")
        if not os.path.isfile(filename):
            logger.error("No such file: %s" % filename)
            return
        data = json.load(open(filename))
        setup_data = {  "poni": self.poni.setText,
#        "detector": self.all_detectors[self.detector.getCurrentIndex()],
                        "wavelength":self.wavelength.setText,
                        "splineFile":self.splineFile.setText,
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
                        "do_polarization": self.do_polarization.setChecked,
                        "val_dummy": self.val_dummy.setText,
                        "delta_dummy": self.delta_dummy.setText,
                        "do_mask":  self.do_mask.setChecked,
                        "mask_file":self.mask_file.setText,
                        "dark_current":self.dark_current.setText,
                        "flat_field":self.flat_field.setText,
                        "polarization_factor":self.polarization_factor.setValue,
                        "rad_pt":self.rad_pt.setText,
                        "do_2D":self.do_2D.setChecked,
                        "azim_pt":self.azim_pt.setText,
                        "chi_discontinuity_at_0": self.chi_discontinuity_at_0.setChecked,
                        "do_radial_range": self.do_radial_range.setChecked,
                        "do_azimuthal_range": self.do_azimuthal_range.setChecked,
                        "radial_range_min":self.radial_range_min.setText,
                        "radial_range_max":self.radial_range_max.setText,
                        "azimuth_range_min":self.azimuth_range_min.setText,
                        "azimuth_range_max":self.azimuth_range_max.setText,
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

    def select_splinefile(self):
        logger.debug("select_splinefile")
        splinefile = str(QtGui.QFileDialog.getOpenFileName())
        if splinefile:
            try:
                self.ai.detector.set_splineFile(splinefile)
                self.pixel1.setText(str(self.ai.pixel1))
                self.pixel2.setText(str(self.ai.pixel2))
                self.splineFile.setText(self.ai.detector.splineFile or "")
            except Exception as error:
                logger.error("failed %s on %s" % (error, splinefile))

    def select_maskfile(self):
        logger.debug("select_maskfile")
        maskfile = str(QtGui.QFileDialog.getOpenFileName())
        if maskfile:
            self.mask_file.setText(maskfile or "")
            self.do_mask.setChecked(True)

    def select_darkcurrent(self):
        logger.debug("select_darkcurrent")
        darkcurrent = str(QtGui.QFileDialog.getOpenFileName())
        if darkcurrent:
            self.dark_current.setText(darkcurrent or "")
            self.do_dark.setChecked(True)

    def select_flatfield(self):
        logger.debug("select_flatfield")
        flatfield = str(QtGui.QFileDialog.getOpenFileName())
        if flatfield:
            self.flat_field.setText(flatfield or "")
            self.do_flat.setChecked(True)

    def set_ponifile(self, ponifile=None):
        if ponifile is None:
            ponifile = self.poni.text()
            print ponifile
        try:
            self.ai = pyFAI.load(ponifile)
        except Exception as error:
            logger.error("file %s does not look like a poni-file, error %s" % (ponifile, error))
            return
        self.pixel1.setText(str(self.ai.pixel1))
        self.pixel2.setText(str(self.ai.pixel2))
        self.dist.setText(str(self.ai.dist))
        self.poni1.setText(str(self.ai.poni1))
        self.poni2.setText(str(self.ai.poni2))
        self.rot1.setText(str(self.ai.rot1))
        self.rot2.setText(str(self.ai.rot2))
        self.rot3.setText(str(self.ai.rot3))
        self.splineFile.setText(self.ai.detector.splineFile or "")
        name = self.ai.detector.name.lower()
        if name in self.all_detectors:
            self.detector.setCurrentIndex(self.all_detectors.index(name))
        else:
            self.detector.setCurrentIndex(self.all_detectors.index("detector"))

    def set_input_data(self, data):
        self.input_data = data

    def _float(self, kw, default=0):
        fval = default
        txtval = str(self.__dict__[kw].text())
        if txtval:
            try:
                fval = float(txtval)
            except ValueError:
                logger.error("Unable to convert %s to float: %s" % (kw, txtval))
        return fval

    def set_ai(self):
        poni = str(self.poni.text()).strip()
        if poni and os.path.isfile(poni):
            self.ai = pyFAI.load(poni)
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

        splineFile = str(self.splineFile.text()).strip()
        if splineFile and os.path.isfile(splineFile):
            self.ai.detector.splineFile = splineFile

        self.ai.pixel1 = self._float("pixel1", 1)
        self.ai.pixel2 = self._float("pixel2", 1)
        self.ai.dist = self._float("dist", 1)
        self.ai.poni1 = self._float("poni1", 0)
        self.ai.poni2 = self._float("poni2", 0)
        self.ai.rot1 = self._float("rot1", 0)
        self.ai.rot2 = self._float("rot2", 0)
        self.ai.rot3 = self._float("rot3", 0)

        if self.chi_discontinuity_at_0.isChecked():
            self.ai.setChiDiscAtZero()

        mask_file = str(self.mask_file.text()).strip()
        if mask_file and os.path.exists(mask_file) and bool(self.do_mask.isChecked()):
            try:
                mask = fabio.open(mask_file).data
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s" % (mask_file, error))
            else:
                self.ai.mask = mask
        dark_files = [i.strip() for i in str(self.dark_current.text()).split(",")
                      if os.path.isfile(i.strip())]
        if dark_files and bool(self.do_dark.isChecked()):
            d0 = fabio.open(dark_files[0]).data
            darks = numpy.zeros(d0.shape[0], d0.shape[1], len(dark_files), dtype=numpy.float32)
            for i, f in enumerate(dark_files):
                darks[:, :, i] = fabio.open(f).data
            self.ai.darkcurrent = darks.mean(axis= -1)

        flat_files = [i.strip() for i in str(self.flat_field.text()).split(",")
                      if os.path.isfile(i.strip())]
        if flat_files and bool(self.do_flat.isChecked()):
            d0 = fabio.open(flat_files[0]).data
            flats = numpy.zeros(d0.shape[0], d0.shape[1], len(flat_files), dtype=numpy.float32)
            for i, f in enumerate(flat_files):
                flats[:, :, i] = fabio.open(f).data
            self.ai.darkcurrent = flats.mean(axis= -1)
        print self.ai

    def detector_changed(self):
        logger.debug("detector_changed")
        detector = str(self.detector.currentText()).lower()
        inst = pyFAI.detectors.detector_factory(detector)
        if inst.force_pixel:
            self.pixel1.setText(str(inst.pixel1))
            self.pixel2.setText(str(inst.pixel2))
            self.splineFile.setText("")
        elif self.splineFile.text():
            splineFile = str(self.splineFile.text()).strip()
            if os.path.isfile(splineFile):
                inst.set_splineFile(splineFile)
                self.pixel1.setText(str(inst.pixel1))
                self.pixel2.setText(str(inst.pixel2))
            else:
                logger.warning("No such spline file %s" % splineFile)
        self.ai.detector = inst

    def openCL_changed(self):
        logger.debug("do_OpenCL")
        do_ocl = bool(self.do_OpenCL.isChecked())
        if do_ocl:
            if ocl is None:
                self.do_OpenCL.setChecked(0)
                return
            if self.platform.count() == 0:
                self.platform.addItems([i.name for i in ocl.platforms])

    def platform_changed(self):
        logger.debug("platform_changed")
        if ocl is None:
            self.do_OpenCL.setChecked(0)
            return
        platform = ocl.get_platform(str(self.platform.currentText()))
        for i in range(self.device.count())[-1::-1]:
            self.device.removeItem(i)
        self.device.addItems([i.name for i in platform.devices])


if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog [options] file1.edf file2.edf ..."
    version = "%prog " + pyFAI.version
    parser = OptionParser(usage=usage, version=version)
    parser.add_option("-v", "--verbose",
                          action="store_true", dest="verbose", default=False,
                          help="switch to verbose/debug mode")
    (options, args) = parser.parse_args()
    # Analyse aruments and options
    args = [i for i in args if os.path.exists(i)]
#        if len(args) != 1:
#            parser.error("incorrect number of arguments")
    if options.verbose:
        logger.info("setLevel: debug")
        logger.setLevel(logging.DEBUG)
    app = QtGui.QApplication([])
    window = AIWidget()
    window.set_input_data(args)
    window.show()
    sys.exit(app.exec_())
