#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""
Description of ADSC (Area Detector Systems Corporation) detectors.

The website is no longer available, but can be found throung the
`web archive <https://web.archive.org/web/20150403133907/http://www.adsc-xray.com/>`_.
"""

from __future__ import print_function, division, absolute_import, with_statement

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/08/2018"
__status__ = "production"

from collections import OrderedDict
from ._common import Detector

import logging
logger = logging.getLogger(__name__)


class _ADSC(Detector):
    """Common class for ADSC detector:
    they all share the same constructor signature
    """
    MANUFACTURER = "ADSC"

    def __init__(self, pixel1=51e-6, pixel2=51e-6):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class ADSC_Q315(_ADSC):
    """
    ADSC Quantum 315r detector, 3x3 chips

    Informations from
    http://www.adsc-xray.com/products/ccd-detectors/q315r-ccd-detector/

    Question: how are the gaps handled ?
    """
    force_pixel = True
    MAX_SHAPE = (6144, 6144)
    aliases = ["Quantum 315"]

    def __init__(self, pixel1=51e-6, pixel2=51e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)


class ADSC_Q210(_ADSC):
    """
    ADSC Quantum 210r detector, 2x2 chips

    Informations from
    http://www.adsc-xray.com/products/ccd-detectors/q210r-ccd-detector/

    Question: how are the gaps handled ?
    """
    force_pixel = True
    MAX_SHAPE = (4096, 4096)
    aliases = ["Quantum 210"]

    def __init__(self, pixel1=51e-6, pixel2=51e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)


class ADSC_Q270(_ADSC):
    """
    ADSC Quantum 270r detector, 2x2 chips

    Informations from
    http://www.adsc-xray.com/products/ccd-detectors/q270-ccd-detector/

    Question: how are the gaps handled ?
    """
    force_pixel = True
    MAX_SHAPE = (4168, 4168)
    aliases = ["Quantum 270"]

    def __init__(self, pixel1=64.8e-6, pixel2=64.8e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)


class ADSC_Q4(_ADSC):
    """
    ADSC Quantum 4r detector, 2x2 chips

    Informations from
    http://proteincrystallography.org/detectors/adsc.php

    Question: how are the gaps handled ?
    """
    force_pixel = True
    MAX_SHAPE = (2304, 2304)
    aliases = ["Quantum 4"]

    def __init__(self, pixel1=82e-6, pixel2=82e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)


class HF_130K(_ADSC):
    """
    ADSC HF-130K 1 module

    Informations from
    http://www.adsc-xray.com/products/pixel-array-detectors/hf-130k/

    """
    force_pixel = True
    MAX_SHAPE = (256, 512)
    aliases = ["HF-130k"]

    def __init__(self, pixel1=150e-6, pixel2=150e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)


class HF_262k(_ADSC):
    """
    ADSC HF-262k 2 module

    Informations from
    http://www.adsc-xray.com/products/pixel-array-detectors/hf-262k/

    Nota: gaps between modules is not known/described
    """
    force_pixel = True
    MAX_SHAPE = (512, 512)
    aliases = ["HF-262k"]

    def __init__(self, pixel1=150e-6, pixel2=150e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)


class HF_1M(_ADSC):
    """
    ADSC HF-1M 2x4 modules

    Informations from
    http://www.adsc-xray.com/products/pixel-array-detectors/hf-1m/

    Nota: gaps between modules is not known/described
    """
    force_pixel = True
    MAX_SHAPE = (1024, 1024)
    aliases = ["HF-1M"]

    def __init__(self, pixel1=150e-6, pixel2=150e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)


class HF_2M(_ADSC):
    """
    ADSC HF-1M 3x6 modules

    Informations from
    http://www.adsc-xray.com/products/pixel-array-detectors/hf-2.4m/

    Nota: gaps between modules is not known/described
    """
    force_pixel = True
    MAX_SHAPE = (1536, 1536)
    aliases = ["HF-2.4M"]

    def __init__(self, pixel1=150e-6, pixel2=150e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)


class HF_4M(_ADSC):
    """
    ADSC HF-4M 4x8 modules

    Informations from
    http://www.adsc-xray.com/products/pixel-array-detectors/hf-4m/
    """
    force_pixel = True
    MAX_SHAPE = (2048, 2048)
    aliases = ["HF-4M"]

    def __init__(self, pixel1=150e-6, pixel2=150e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)


class HF_9M(_ADSC):
    """
    ADSC HF-130K 1 module

    Informations from
    http://www.adsc-xray.com/products/pixel-array-detectors/hf-9-4m/

    """
    force_pixel = True
    MAX_SHAPE = (3072, 3072)
    aliases = ["HF-9.4M"]

    def __init__(self, pixel1=150e-6, pixel2=150e-6):
        _ADSC.__init__(self, pixel1=pixel1, pixel2=pixel2)
