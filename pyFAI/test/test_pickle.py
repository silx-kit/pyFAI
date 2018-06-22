
import pyFAI
from pyFAI import AzimuthalIntegrator
from pyFAI.detectors import Pilatus1M, Detector
from pickle import dumps, loads
import unittest
from .utilstest import UtilsTest
import fabio
import logging
logger = logging.getLogger(__name__)


class TestPickle(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        img = UtilsTest.getimage("Pilatus1M.edf")
        self.data = fabio.open(img).data
        self.ai = AzimuthalIntegrator(1.58323111834, 0.0334170169115, 0.0412277798782, 0.00648735642526, 0.00755810191106, 0.0, detector=Pilatus1M())
        self.ai.wavelength = 1e-10
        self.npt = 1000

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.data = self.ai = self.npt = None

    def test_Detector_pickle(self):
        det = self.ai.detector  # type: Detector

        print(det, type(det))
        assert dumps(det)
        assert loads(dumps(det))
        self.assertEqual(loads(dumps(det)).shape, Pilatus1M.MAX_SHAPE)


    def test_AzimuthalIntegrator_pickle(self):
        spectra = self.ai.integrate1d(self.data, 1000)  # force lut generation
        dump = dumps(self.ai)
        newai = loads(dump)  # type: AzimuthalIntegrator
        self.assertEqual(newai._cached_array, self.ai._cached_array)
        self.assertEqual(newai.integrate1d(self.data, 1000), spectra)

    def test_Calibrant(self):
        from pyFAI import calibrant
        calibrant = calibrant.CalibrantFactory()('AgBh')
        assert dumps(calibrant)
        assert loads(dumps(calibrant))

def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestPickle))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
