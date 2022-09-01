import fabio, pyFAI
import pyFAI.test.utilstest
from pyFAI.io.nexus import save_NXmonpd
import os
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_monopd.nxs")

img = fabio.open(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.edf"))
ai = pyFAI.load(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.poni"))

res = ai.integrate1d(img.data, 1000, unit="2th_deg", error_model="poisson", filename="test_monopd.dat")
save_NXmonpd(dest, res, sample="AgBh", instrument="Dubble")
print("written to ",dest)
