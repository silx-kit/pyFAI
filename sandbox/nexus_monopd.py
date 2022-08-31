import fabio, pyFAI
import pyFAI.test.utilstest
from pyFAI.io.nexus import save_NXmonpd

img=fabio.open(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.edf"))
ai=pyFAI.load(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.poni"))

res = ai.integrate1d(img.data, 1000, unit="2th_deg", error_model="poisson")
save_NXmonpd("test_monopd.nxs", res, sample="AgBh", instrument="bm26")