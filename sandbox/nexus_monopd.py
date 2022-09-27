import fabio, pyFAI
import pyFAI.test.utilstest
from pyFAI.io.nexus import save_NXmonpd
import os
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_monopd.nxs")

img = fabio.open(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus2M.cbf"))
ai = pyFAI.load(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus2M.poni"))

res = ai.integrate1d(img.data, 1000,
                     unit="2th_deg", error_model="poisson",
                     polarization_factor=0.95)
save_NXmonpd(dest, res, title="Ceria", sample="CeO2", instrument="ID31")
print("written to ", dest)
