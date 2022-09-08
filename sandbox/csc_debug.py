from concurrent.futures import ThreadPoolExecutor
import time, gc
import numpy
import fabio, pyFAI
import pyFAI.test.utilstest
import os
import scipy.sparse
from pyFAI.ext.CSC_integrator import CscIntegrator

img = fabio.open(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.edf"))
ai = pyFAI.load(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.poni"))

csr = ai.setup_CSR(img.shape, 1000)
scsr = scipy.sparse.csr_matrix(csr.lut)
scsc = scsr.tocsc()
csc = CscIntegrator((scsc.data, scsc.indices, scsc.indptr), img.data.size)

stack = [numpy.random.randint(0, 65000, size=img.shape) for i in range(1000)]
gc.collect()
for t in [1, 2, 4, 8, 16]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=t) as executor:
        results = [i for i in executor.map(csc.integrate_ng, stack)]
    t1 = time.perf_counter()
    print(t, t1 - t0)
    del results
    gc.collect()
