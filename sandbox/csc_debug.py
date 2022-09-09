import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time, gc
import numpy
import fabio, pyFAI
import pyFAI.test.utilstest
import os
import scipy.sparse
from pyFAI.ext.splitBBoxCSR import HistoBBox1d as HistoBBox1dCSR
from pyFAI.ext.splitBBoxCSC import HistoBBox1d as HistoBBox1dCSC

img = fabio.open(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.edf"))
ai = pyFAI.load(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus1M.poni"))

csr = ai.setup_sparse_integrator(img.shape, 1000, algo="CSR")
csc = ai.setup_sparse_integrator(img.shape, 1000, algo="CSC")

ref = csr.integrate_ng(img.data, dummy=-5)
res = csc.integrate_ng(img.data, dummy=-5)
for idx in ("position", 'intensity', 'sigma', 'signal', 'variance', 'normalization', 'count', 'std', 'sem', 'norm_sq'):
    got = numpy.allclose(ref.__getattribute__(idx), res.__getattribute__(idx))
    print(f"Checking {idx}: {'OK' if got else 'FAILED'}")
    if not got:
        print(ref.__getattribute__(idx)[:10])
        print(res.__getattribute__(idx)[:10])
print("#"*50)
print(res.signal[:10])
print(res.normalization[:10])
print(res.signal[:10] / res.normalization[:10])
if 1:
    stack = [numpy.random.randint(0, 65000, size=img.shape) for i in range(1000)]

    gc.collect()
    first = None
    for i in range(int(1 + numpy.ceil(numpy.log(multiprocessing.cpu_count()) / numpy.log(2)))):
        t = 1 << i
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=t) as executor:
            results = [i for i in executor.map(csc.integrate_ng, stack)]
        t1 = time.perf_counter()
        if first is None:
            first = t1 - t0
        print(f"{t} threads: {t1 - t0:.3f} ms/img, speed-up: {first/(t1-t0):.3f}x")
        del results
        gc.collect()
