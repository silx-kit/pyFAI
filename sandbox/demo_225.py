import time
import logging
import numpy
import fabio
logging.basicConfig(level=logging.INFO)
import pyFAI.distortion


halfFrelon = "LaB6_0020.edf"
splineFile = "halfccd.spline"
fit2d_cor = "halfccd.fit2d.edf"
from pyFAI.test.utilstest import UtilsTest
fit2dFile = UtilsTest.getimage(fit2d_cor)
halfFrelon = UtilsTest.getimage(halfFrelon)
splineFile = UtilsTest.getimage(splineFile)
det = pyFAI.detectors.FReLoN(splineFile)
img = fabio.open(halfFrelon).data
# det.binning = 5, 8

dis = pyFAI.distortion.Distortion(det, det.shape, resize=False,
                                         mask=numpy.zeros(det.shape, "int8"))
pos = dis.calc_pos(False)
import pyFAI.ext._distortion
print(dis.bin_size)
t0 = time.time()
ref = pyFAI.ext._distortion.calc_CSR(pos, det.shape, dis.calc_size(), (8, 8))
t1 = time.time()
obt = pyFAI.ext._distortion.calc_openmp(pos, det.shape, (8, 8))
t2 = time.time()
print("ref", t1 - t0, "new", t2 - t1)

dis = pyFAI.distortion.Distortion(det)
ref = pyFAI.ext._distortion.Distortion(det)
p = ref.calc_LUT()
q = dis.calc_LUT()
delta = (dis.lut["idx"] - ref.LUT["idx"])
bad = 1.0 * dis.lut.size / (delta == 0).sum() - 1
print(bad)

def compact_CSR(data, indices, indptr):
    print("Compact CSR...")
    new_data = []
    new_indices = []
    print("    was size %.3fMB" % (sum([i.nbytes for i in (data, indices, indptr)]) / 1e6))
    for i in range(len(indptr) - 1):
        start = indptr[i]
        stop = indptr[i + 1]
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

from math import floor, ceil, fabs
def calc_area(I1, I2, slope, intercept):
    "Calculate the area between I1 and I2 of a line with a given slope & intercept"
    return 0.5 * (I2 - I1) * (slope * (I2 + I1) + 2 * intercept)


def integrate(box, start, stop, slope, intercept):
    """Integrate in a box a line between start and stop, line defined by its slope & intercept

    @param box: buffer
    """
    if start < stop:  # positive contribution
        P = ceil(start)
        dP = P - start
        if P > stop:  # start and stop are in the same unit
            A = calc_area(start, stop, slope, intercept)
            if A != 0.0:
                AA = fabs(A)
                sign = A / AA
                dA = (stop - start)  # always positive
                h = 0
                while AA > 0:
                    if dA > AA:
                        dA = AA
                        AA = -1
                    box[int(floor(start)), h] += sign * dA
                    AA -= dA
                    h += 1
        else:
            if dP > 0:
                A = calc_area(start, P, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = dP
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[int(floor(P)) - 1, h] += sign * dA
                        AA -= dA
                        h += 1
            # subsection P1->Pn
            for i in range(int(floor(P)), int(floor(stop))):
                A = calc_area(i, i + 1, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA

                    h = 0
                    dA = 1.0
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[i , h] += sign * dA
                        AA -= dA
                        h += 1
            # Section Pn->B
            P = floor(stop)
            dP = stop - P
            if dP > 0:
                A = calc_area(P, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[int(floor(P)), h] += sign * dA
                        AA -= dA
                        h += 1
    elif start > stop:  # negative contribution. Nota if start==stop: no contribution
        P = floor(start)
        if stop > P:  # start and stop are in the same unit
            A = calc_area(start, stop, slope, intercept)
            if A != 0:
                AA = fabs(A)
                sign = A / AA
                dA = (start - stop)  # always positive
                h = 0
                while AA > 0:
                    if dA > AA:
                        dA = AA
                        AA = -1
                    box[int(floor(start)), h] += sign * dA
                    AA -= dA
                    h += 1
        else:
            dP = P - start
            if dP < 0:
                A = calc_area(start, P, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[int(floor(P)), h] += sign * dA
                        AA -= dA
                        h += 1
            # subsection P1->Pn
            for i in range(int(start), int(ceil(stop)), -1):
                A = calc_area(i, i - 1, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = 1
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[i - 1, h] += sign * dA
                        AA -= dA
                        h += 1
            # Section Pn->B
            P = ceil(stop)
            dP = stop - P
            if dP < 0:
                A = calc_area(P, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[int(floor(stop)), h] += sign * dA
                        AA -= dA
                        h += 1

A0, B0, C0, D0 = 0, 0, 0, 0
A1, B1, C1, D1 = 0, 0, 0, 0

offset0 = int(floor(min(A0, B0, C0, D0)))
offset1 = int(floor(min(A1, B1, C1, D1)))
box_size0 = int(ceil(max(A0, B0, C0, D0))) - offset0
box_size1 = int(ceil(max(A1, B1, C1, D1))) - offset1
A0 -= offset0
A1 -= offset1
B0 -= offset0
B1 -= offset1
C0 -= offset0
C1 -= offset1
D0 -= offset0
D1 -= offset1
if B0 != A0:
    pAB = (B1 - A1) / (B0 - A0)
    cAB = A1 - pAB * A0
else:
    pAB = cAB = 0.0
if C0 != B0:
    pBC = (C1 - B1) / (C0 - B0)
    cBC = B1 - pBC * B0
else:
    pBC = cBC = 0.0
if D0 != C0:
    pCD = (D1 - C1) / (D0 - C0)
    cCD = C1 - pCD * C0
else:
    pCD = cCD = 0.0
if A0 != D0:
    pDA = (A1 - D1) / (A0 - D0)
    cDA = D1 - pDA * D0
else:
    pDA = cDA = 0.0
aera = 0.5 * ((C0 - A0) * (D1 - B1) - (C1 - A1) * (D0 - B0))

buffer = numpy.zeros((3, 3))

integrate(buffer, B0, A0, pAB, cAB)
integrate(buffer, C0, B0, pBC, cBC)
integrate(buffer, D0, C0, pCD, cCD)
integrate(buffer, A0, D0, pDA, cDA)
print(buffer)
print(aera, buffer.sum())
"""
cmp = compact_CSR(*ref)
print("max error on indexptr %s" % abs(obt[2] - cmp[2]).max())
print("max error on indices %s" % abs(obt[1] - cmp[1]).max())
print("max error on data %s" % abs(obt[1] - cmp[1]).max())

print("*" * 80)
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
"""
if __name__ == "__main__":
    from IPython import embed
    embed()
