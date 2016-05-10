import time
import logging
import numpy
import fabio
logging.basicConfig(level=logging.INFO)
import pyFAI, pyFAI.distortion
halfFrelon = "1464/LaB6_0020.edf"
splineFile = "1461/halfccd.spline"
fit2d_cor = "2454/halfccd.fit2d.edf"
from pyFAI.test.utilstest import UtilsTest
fit2dFile = UtilsTest.getimage(fit2d_cor)
halfFrelon = UtilsTest.getimage(halfFrelon)
splineFile = UtilsTest.getimage(splineFile)
det = pyFAI.detectors.FReLoN(splineFile)
img = fabio.open(halfFrelon).data
# det.binning = 5, 8
import numpy
dis = pyFAI.distortion.Distortion(det, det.shape, resize=False,
                                         mask=numpy.zeros(det.shape, "int8"))
pos = dis.calc_pos(False)
from pyFAI.ext import _distortion
print(dis.bin_size)
t0 = time.time()
ref = _distortion.calc_CSR(pos, det.shape, dis.calc_size(), (8, 8))
t1 = time.time()
obt = _distortion.calc_openmp(pos, det.shape, (8,8))
t2 = time.time()
print("ref", t1 - t0, "new", t2 - t1)

def compact_CSR(data, indices, indptr):
    import numpy
    print("Compact CSR...")
    new_data = []
    new_indices = []
    new_indptr = []
    print("    was size %.3fMB" % (sum([i.nbytes for i in (data, indices, indptr)]) / 1e6))
    for i in range(len(indptr)-1):
        start = indptr[i]
        stop = indptr[i+1]
        tmp_data = data[start:stop]
        tmp_indices = indices[start:stop]
        valid = (tmp_data > 0)
        new_data.append(tmp_data[valid])
        new_indices.append(tmp_indices[valid])
    new_indptr = numpy.zeros_like(indptr)
    new_indptr[1:] = numpy.array([i.size for i in new_indices]).cumsum()
    lut = numpy.hstack(new_data), numpy.hstack(new_indices), new_indptr
    print("    new size %.3fMB" % (sum([i.nbytes for i in (lut)]) / 1e6))
    return lut

def compact_LUT(lut):
    import numpy
    print("Compact LUT...")
    print("    was size %.3fMB" % (lut.nbytes / 1e6))
    pos = (lut["coef"] > 0)
    cnt = pos.sum(axis=-1)
    new_lut = numpy.recarray((lut.shape[0], cnt.max()), dtype=lut.dtype)
    for i in range(lut.shape[0]):
        m = numpy.zeros(lut.shape[0], dtype=bool)
        m[i] = True
        m = numpy.atleast_2d(m)
        new_lut[i, :cnt[i]] = lut[numpy.logical_and(pos, (m.T))]
    print("    new size %.3fMB" % (lut.nbytes / 1e6))
    return lut


cmp = compact_CSR(*ref)
print("max error on indexptr %s" % abs(obt[2] - cmp[2]).max())
print("max error on indices %s" % abs(obt[1] - cmp[1]).max())
print("max error on data %s" % abs(obt[1] - cmp[1]).max())

print("*"*80)
t0 = time.time()
ref = _distortion.calc_LUT(pos, det.shape, dis.calc_size(), (8, 8))
t1 = time.time()
obt = _distortion.calc_openmp(pos, det.shape, (8, 8), format="LUT")
t2 = time.time()
print("ref", t1 - t0, "new", t2 - t1)
print(ref.shape, obt.shape)
# cmp = compact_LUT(obt)
# print("max error on index %s" % abs(obt["idx"] - cmp["idx"]).max())
# print("max error on coef %s" % abs(obt["coef"] - cmp["coef"]).max())

if __name__ == "__main__":
    from IPython import embed
    embed()
