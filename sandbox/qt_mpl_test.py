import logging
import pyFAI
import fabio
from pyFAI.test.utilstest import UtilsTest
from pyFAI.gui.peak_picker import PeakPicker
logging.basicConfig()

from pyFAI.calibrant import get_calibrant
agbh = get_calibrant("AgBh")

img = fabio.open(UtilsTest.getimage("Pilatus1M.edf")).data
pp = PeakPicker(data=img,
                calibrant=agbh,
                wavelength=1e-10,
                detector=pyFAI.detector_factory("Pilatus1M"),
                )
