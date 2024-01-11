# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2024-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Module for exporting XRDML powder diffraction files
Inspiration from
"""

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/01/2024"
__docformat__ = 'restructuredtext'

import os
from xml.etree import  ElementTree as et
from .nexus import get_isotime
from .. import version as pyFAI_version

def save_xrdml(filename, result):
    """

    https://www.researchgate.net/profile/Mohamed-Ali-392/post/XRD_Refinement_for_TiO2_Anatase_using_MAUD/attachment/60fa1d85647f3906fc8af2f3/AS%3A1048546157539329%401627004293526/download/sample.xrdml
    """
    now = get_isotime()
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)

    xrdml = et.Element("xrdMeasurements", {"xmlns":"http://www.xrdml.com/XRDMeasurement/1.0",
                                           "xmlns:xsi":"http://www.w3.org/2001/XMLSchema-instance",
                                           "xsi:schemaLocation": "http://www.xrdml.com/XRDMeasurement/1.3 http://www.xrdml.com/XRDMeasurement/1.3/XRDMeasurement.xsd",
                                           "status": "Completed"})
    sample = et.Element("sample", type="To be analyzed")
    xid = et.Element("id")
    xid.text = "000000-0000"
    sample.append(xid)
    name = et.Element("name")
    name.text = "sample"
    sample.append(name)
    xrdml.append(sample)
    measurement = et.Element("xrdMeasurement", {"measurementType":"Scan", "status":"Completed"})
    wavelength = result.poni.wavelength
    if wavelength:
        txtwavelength = str(wavelength*1e10)
        usedWavelength = et.Element("usedWavelength", intended="K-Alpha")
        k_alpha1 = et.Element("kAlpha1", unit="Angstrom")
        k_alpha1.text = txtwavelength
        usedWavelength.append(k_alpha1)
        k_alpha2 = et.Element("kAlpha2", unit="Angstrom")
        k_alpha2.text = txtwavelength
        usedWavelength.append(k_alpha2)
        k_beta = et.Element("kBeta", unit="Angstrom")
        k_beta.text = txtwavelength
        usedWavelength.append(k_beta)
        ratio = et.Element("ratioKAlpha2KAlpha1")
        ratio.text = "0"
        usedWavelength.append(ratio)
        measurement.append(usedWavelength)
    scan = et.Element("scan", {"appendNumber":"0",
                               "mode":"Pre-set time",
                               "measurementType": "Area measurement",
                               "scanAxis":"Gonio",
                               "status":"Completed"})
    header = et.Element("header")
    for stamp in ("startTimeStamp", "endTimeStamp"):
        estamp = et.Element(stamp)
        estamp.text = now
        header.append(estamp)
    author = et.Element("author")
    name = et.Element("name")
    name.text = "pyFAI"
    author.append(name)
    header.append(author)
    source = et.Element("source")
    sw = et.Element("applicationSoftware", version=pyFAI_version)
    sw.text='pyFAI'
    source.append(sw)
    header.append(source)
    scan.append(header)
    datapoints = et.Element("dataPoints")
    positions = et.Element("positions", axis=result.unit.short_name, unit=result.unit.unit_symbol)
    for pos, idx in {"startPosition": 0, "endPosition":-1}.items():
        position = et.Element(pos)
        position.text = str(result.radial[idx])
        positions.append(position)
    datapoints.append(positions)
    ct = et.Element("commonCountingTime", unit="seconds")
    ct.text = "1.00"
    datapoints.append(ct)
    intensities = et.Element("intensities", unit="counts")
    intensities.text = " ".join(str(i) for i in result.intensity)
    datapoints.append(intensities)
    scan.append(datapoints)
    measurement.append(scan)
    xrdml.append(measurement)
    with open(filename, "wb") as w:
        w.write(et.tostring(xrdml))
