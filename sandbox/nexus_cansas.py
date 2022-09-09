import fabio, pyFAI
import pyFAI.test.utilstest
from pyFAI.io.nexus import save_NXcansas
import os
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_canSAS.nxs")

img = fabio.open(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.edf"))
ai = pyFAI.load(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.poni"))

res = ai.integrate1d(img.data, 1000,
                     unit="q_nm^-1", error_model="poisson",
                     polarization_factor=0.95)
save_NXcansas(dest, res, title="Calibration", sample="AgBh", instrument="Dubble")
print("written to ", dest)
