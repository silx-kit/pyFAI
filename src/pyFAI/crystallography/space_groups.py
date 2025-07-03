#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Space groups: There are 230 space groups as defined in the internationnal
tables of crystallography (vol.A)

The ReflectionCondition contains selection rules for all of them but not all are correct (yet)
"""

from __future__ import annotations

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "02/07/2025"
__status__ = "production"


class ReflectionCondition:
    """This class contains selection rules for most space-groups

    All methods are static and take a triplet hkl as input representing a familly of Miller plans.
    They return True if the reflection is allowed by symmetry, False otherwise.

    Most of those methods are AI-generated (Co-Pilot) and about 80% of them are still WRONG unless tagged
    "validated" in the docstring.

    Help is welcome to polish this class and fix the non-validated ones.
    """

    @staticmethod
    def group1_p1(h, k, l):
        """Space group 1: P1. No systematic absences. validated"""
        return True

    @staticmethod
    def group2_p_1(h, k, l):
        """Space group 2: P-1. No systematic absences. validated"""
        return True

    @staticmethod
    def group3_p2_b(h, k, l):
        """Space group 3: P2 (unique axis b). No systematic absences. validated"""
        return True

    @staticmethod
    def group4_p21_b(h, k, l):
        """Space group 4: P21 (unique axis b). (0 k 0): k even only. validated"""
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group5_c2_b(h, k, l):
        """Space group 5: C2 (unique axis b). C-centering: (h + k) even. (0 k 0): k even only. validated"""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group6_pm_b(h, k, l):
        """Space group 6: Pm (unique axis b). No systematic absences. validated"""
        return True

    @staticmethod
    def group7_pc_b(h, k, l):
        """Space group 7: Pc (unique axis b). (h 0 l): l even only. validated"""
        if k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group8_cm_b(h, k, l):
        """Space group 8: Cm (unique axis b). C-centering: (h + k) even. validated"""
        return (h + k) % 2 == 0

    @staticmethod
    def group9_cc_b(h, k, l):
        """Space group 9: Cc (unique axis b). C-centering: (h + k) even. (h 0 l): h even only. validated"""
        if k == 0:
            return h % 2 == 0 and l % 2 == 0
        else:
            return (h + k) % 2 == 0
        return True

    @staticmethod
    def group10_p2m_b(h, k, l):
        """Space group 10: P2/m (unique axis b). No systematic absences.validated"""
        return True

    @staticmethod
    def group11_p21m_b(h, k, l):
        """Space group 11: P21/m (unique axis b). (0 k 0): k even only.validated"""
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group12_c2m_b(h, k, l):
        """Space group 12: C2/m (unique axis b). C-centering: (h + k) even. (0 k 0): k even only. validated"""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group13_P2c_b(h, k, l):
        """Space group 13: P 1 2/c 1 (unique axis b). (h 0 l): l even. validated"""
        if k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group14_P21c_b(h, k, l):
        """Space group 14: P 1 21/c 1 (unique axis b). h0l: l even, 0k0: k even. validated"""
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group15_c2c_b(h, k, l):
        """Space group 15: C 1 2/c 1(unique axis b). C-centering: (h + k) even. (0 k 0): k even only. validated"""
        if k == 0:
            return h % 2 == 0 and l % 2 == 0
        return (h + k) % 2 == 0

    @staticmethod
    def group16_P222(h, k, l):
        """Space group 16: P222. No systematic absences.validated"""
        return True

    @staticmethod
    def group17_P2221(h, k, l):
        """Space group 17: P 2 2 21. (0 0 l): l even only. validated"""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group18_P21212(h, k, l):
        """Space group 18: P 21 21 2. (0 0 l): l even only. (0 k 0): k even only. (h 0 0): h even only. validated"""
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group19_p212121(h, k, l):
        """Space group 19: P212121. (0 0 l): l even only. (0 k 0): k even only. (h 0 0): h even only.validated"""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group20_c2221(h, k, l):
        """Space group 20: C2221. C-centering: h + k even, k + l even, h + l even. (0 0 l): l even only. (0 k 0): k even only. (h 0 0): h even only."""
        if (h + k) % 2 != 0 or (k + l) % 2 != 0 or (h + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group21_c222(h, k, l):
        """Space group 21: C222. C-centering: (h + k) even, (k + l) even, (h + l) even."""
        return (h + k) % 2 == 0 and (k + l) % 2 == 0 and (h + l) % 2 == 0

    @staticmethod
    def group22_f222(h, k, l):
        """Space group 22: F222. F-centering: h, k, l all even or all odd. validated"""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group23_i222(h, k, l):
        """Space group 23: I222. I-centering: (h + k + l) even. validated"""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group24_i212121(h, k, l):
        """Space group 24: I212121. I-centering: (h + k + l) even. (h 0 0): h even; (0 k 0): k even; (0 0 l): l even. validated"""
        if (h + k + l) % 2 != 0:
            return False
        if k == 0 and l == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group25_pmm2(h, k, l):
        """Space group 25: Pmm2. No systematic absences.validated"""
        return True

    @staticmethod
    def group26_pmc21(h, k, l):
        """Space group 26: Pmc21. (h 0 l): h even; (0 k 0): k even."""
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group27_pcc2(h, k, l):
        """Space group 27: Pcc2. (h 0 l): h even; (0 k 0): k even."""
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group28_pma2(h, k, l):
        """Space group 28: Pma2. No systematic absences."""
        return True

    @staticmethod
    def group29_pla2(h, k, l):
        """Space group 29: Pla2. (0 k 0): k even; (h 0 l): h even."""
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group30_cmm2(h, k, l):
        """Space group 30: Cmm2. C-centering: (h + k) even. (h 0 l): h even; (0 k 0): k even."""
        if (h + k) % 2 != 0:
            return False
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group31_cmc21(h, k, l):
        """Space group 31: Cmc21. C-centering: (h + k) even. (h 0 l): h even; (0 k 0): k even."""
        if (h + k) % 2 != 0:
            return False
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group32_ccc2(h, k, l):
        """Space group 32: Ccc2. C-centering: (h + k) even, (k + l) even, (h + l) even. (h 0 l): h even; (0 k 0): k even."""
        if (h + k) % 2 != 0 or (k + l) % 2 != 0 or (h + l) % 2 != 0:
            return False
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group33_ama2(h, k, l):
        """Space group 33: Ama2. (0 k 0): k even; (h 0 l): h even."""
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group34_aba2(h, k, l):
        """Space group 34: Aba2. (0 k 0): k even; (h 0 l): h even."""
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group35_fmm2(h, k, l):
        """Space group 35: Fmm2. F-centering: h, k, l all even or all odd. (h 0 l): h even; (0 k 0): k even."""
        if not (h % 2 == k % 2 == l % 2):
            return False
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group36_i_b_m_2(h, k, l):
        """Space group 36: I b m 2. I-centering: (h + k + l) even. (h 0 l): h even; (0 k 0): k even."""
        if (h + k + l) % 2 != 0:
            return False
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group37_i_b_c_2(h, k, l):
        """Space group 37: I b c 2. I-centering: (h + k + l) even. (h 0 l): h even; (0 k 0): k even."""
        if (h + k + l) % 2 != 0:
            return False
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group38_i_b_a_2(h, k, l):
        """Space group 38: I b a 2. I-centering: (h + k + l) even. (h 0 l): h even; (0 k 0): k even."""
        if (h + k + l) % 2 != 0:
            return False
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group39_pmmm(h, k, l):
        """Space group 39: Pmmm. No systematic absences."""
        return True

    @staticmethod
    def group40_pnnm(h, k, l):
        """Space group 40: Pnnm. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group41_pccm(h, k, l):
        """Space group 41: Pccm. (h 0 l): h even; (0 k 0): k even."""
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group42_pban(h, k, l):
        """Space group 42: Pban. (h 0 l): h even; (0 k 0): k even; (h 0 0): h even."""
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group43_pmma(h, k, l):
        """Space group 43: Pmma. (0 0 l): l even; (0 k 0): k even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group44_pmna(h, k, l):
        """Space group 44: Pmna. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group45_pcca(h, k, l):
        """Space group 45: Pcca. (h 0 l): h even; (0 k 0): k even; (h 0 0): h even."""
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group46_pbam(h, k, l):
        """Space group 46: Pbam. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group47_pccn(h, k, l):
        """Space group 47: Pccn. (h 0 l): h even; (0 k 0): k even; (h 0 0): h even."""
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group48_pbcn(h, k, l):
        """Space group 48: Pbcn. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group49_pbca(h, k, l):
        """Space group 49: Pbca. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group50_pnma(h, k, l):
        """Space group 50: Pnma. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group51_pbam(h, k, l):
        """Space group 51: Pbam. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group52_pccn(h, k, l):
        """Space group 52: Pccn. (h 0 l): h even; (0 k 0): k even; (k 0 0): k even."""
        if k == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group53_pbcn(h, k, l):
        """Space group 53: Pbcn. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group54_pbca(h, k, l):
        """Space group 54: Pbca. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group55_pnma(h, k, l):
        """Space group 55: Pnma. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group56_pmmn(h, k, l):
        """Space group 56: Pmmn. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group57_pmmm(h, k, l):
        """Space group 57: Pmmm. No systematic absences."""
        return True

    @staticmethod
    def group58_pnnn(h, k, l):
        """Space group 58: Pnnn. (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group59_cccm(h, k, l):
        """Space group 59: Cccm. C-centering: (h + k) even, (k + l) even, (h + l) even."""
        return (h + k) % 2 == 0 and (k + l) % 2 == 0 and (h + l) % 2 == 0

    @staticmethod
    def group60_ccca(h, k, l):
        """Space group 60: Ccca. C-centering: (h + k) even, (k + l) even, (h + l) even."""
        return (h + k) % 2 == 0 and (k + l) % 2 == 0 and (h + l) % 2 == 0

    @staticmethod
    def group61_fmmm(h, k, l):
        """Space group 61: Fmmm. F-centering: h, k, l all even or all odd."""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group62_fddd(h, k, l):
        """Space group 62: Fddd. F-centering: h, k, l all even or all odd; (h, k, 0): h, k even; (0, k, l): k, l even; (h, 0, l): h, l even."""
        if not (h % 2 == k % 2 == l % 2):
            return False
        if l == 0 and h % 2 != 0:
            return False
        if h == 0 and k % 2 != 0:
            return False
        if k == 0 and l % 2 != 0:
            return False
        return True

    @staticmethod
    def group63_immm(h, k, l):
        """Space group 63: Immm. I-centering: (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group64_ibam(h, k, l):
        """Space group 64: Ibam. I-centering: (h + k + l) even; (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group65_ibca(h, k, l):
        """Space group 65: Ibca. I-centering: (h + k + l) even; (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group66_ibcm(h, k, l):
        """Space group 66: Ibcm. I-centering: (h + k + l) even; (0 0 l): l even; (0 k 0): k even; (h 0 0): h even."""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group67_p4(h, k, l):
        """Space group 67: P4. No systematic absences."""
        return True

    @staticmethod
    def group68_p41(h, k, l):
        """Space group 68: P41. (0, 0, l): l = 4n."""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group69_p42(h, k, l):
        """Space group 69: P42. (0, 0, l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group70_p43(h, k, l):
        """Space group 70: P43. (0, 0, l): l = 4n."""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group71_i4(h, k, l):
        """Space group 71: I4. I-centering: (h + k + l) even.validated"""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group72_i41(h, k, l):
        """Space group 72: I41. I-centering: (h + k + l) even; (0, 0, l): l = 4n."""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group73_p_4(h, k, l):
        """Space group 73: P-4. No systematic absences."""
        return True

    @staticmethod
    def group74_i_4(h, k, l):
        """Space group 74: I-4. I-centering: (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group75_p4_m(h, k, l):
        """Space group 75: P4/m. No systematic absences.validated"""
        return True

    @staticmethod
    def group76_p42_m(h, k, l):
        """Space group 76: P42/m. (0, 0, l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group77_p4_n(h, k, l):
        """Space group 77: P4/n. (h + k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group78_p42_n(h, k, l):
        """Space group 78: P42/n. (h + k) even; (0, 0, l): l even."""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group79_i4_m(h, k, l):
        """Space group 79: I4/m. I-centering: (h + k + l) even.validated"""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group80_i41_a(h, k, l):
        """Space group 80: I41/a. I-centering: (h + k + l) even; (0, 0, l): l = 4n.validated"""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group81_p_42_m(h, k, l):
        """Space group 81: P-42m. No systematic absences.validated"""
        return True

    @staticmethod
    def group82_p_42_c(h, k, l):
        """Space group 82: P-42c. (0, 0, l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group83_p_42_n(h, k, l):
        """Space group 83: P-42n. (h + k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group84_i_42_m(h, k, l):
        """Space group 84: I-42m. I-centering: (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group85_i_42_d(h, k, l):
        """Space group 85: I-42d. I-centering: (h + k + l) even; (0, 0, l): l = 4n."""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group86_p4_2_2(h, k, l):
        """Space group 86: P422. No systematic absences."""
        return True

    @staticmethod
    def group87_p4_21_2(h, k, l):
        """Space group 87: P4_21_2. (0, 0, l): l = 4n."""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group88_p4_32_2(h, k, l):
        """Space group 88: P4_32_2. (0, 0, l): l = 4n."""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group89_p4_3_2(h, k, l):
        """Space group 89: P4_3_2. (0, 0, l): l = 4n."""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group90_p4_1_2(h, k, l):
        """Space group 90: P4_1_2. (0, 0, l): l = 4n."""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group91_p4_12_2(h, k, l):
        """Space group 91: P4_12_2. (0, 0, l): l = 4n.validated"""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group92_p4_32_2(h, k, l):
        """Space group 92: P4_32_2. (0, 0, l): l = 4n."""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group93_p4_3_2(h, k, l):
        """Space group 93: P4_3_2. (0, 0, l): l = 4n."""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group94_p4_2_2(h, k, l):
        """Space group 94: P4_2_2. (0, 0, l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group95_p4_21_2(h, k, l):
        """Space group 95: P4_21_2. (0, 0, l): l = 4n.validated"""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group96_P_43_21_2(h, k, l):
        """Group 96 P 43 21 2, used in lysozyme."""
        if h == 0 and k == 0:
            # 00l: l=4n
            return l % 4 == 0
        elif k == 0 and l == 0:
            # h00: h=2n
            return h % 2 == 0
        # elif h == 0:
        #     # 0kl:
        #     if l % 2 == 1:
        #         # l=2n+1
        #         return True
        #     else:
        #         # 2k+l=4n
        #         return (2 * k + l) % 4 == 0
        return False

    @staticmethod
    def group97_i4_2_2(h, k, l):
        """Space group 97: I422. I-centering: (h + k + l) even.validated"""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group98_i4_12_2(h, k, l):
        """Space group 98: I4_12_2. I-centering: (h + k + l) even; (0, 0, l): l = 4n.validated"""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group99_p4mm(h, k, l):
        """Space group 99: P4mm. No systematic absences.validated"""
        return True

    @staticmethod
    def group100_P4bm(h, k, l):
        """Space group 100: P4bm."""
        # 0kl: k=2n
        if h == 0:
            return k % 2 == 0
        if k == l == 0:
            return h % 2 == 0
        return (h + k) % 2 == 0
        # return True

    @staticmethod
    def group101_p4cc(h, k, l):
        """Space group 101: P4cc. WRONG."""
        return True

    @staticmethod
    def group102_p4nc(h, k, l):
        """Space group 102: P4nc. WRONG."""
        return True

    @staticmethod
    def group103_p42mc(h, k, l):
        """Space group 103: P42mc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group104_p42cm(h, k, l):
        """Space group 104: P42cm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group105_p42cc(h, k, l):
        """Space group 105: P42cc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group106_p42nc(h, k, l):
        """Space group 106: P42nc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group107_p42mmc(h, k, l):
        """Space group 107: P42mmc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group108_p42mcm(h, k, l):
        """Space group 108: P42mcm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group109_p42ccm(h, k, l):
        """Space group 109: P42ccm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group110_p42ncm(h, k, l):
        """Space group 110: P42ncm. (h+k) even; (0,0,l): l even."""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group111_p42mbc(h, k, l):
        """Space group 111: P42mbc. (h+k) even; (0,0,l): l even."""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group112_p42cbm(h, k, l):
        """Space group 112: P42cbm. (h+k) even; (0,0,l): l even."""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group113_p4mmm(h, k, l):
        """Space group 113: P4/mmm. No systematic absences."""
        return True

    @staticmethod
    def group114_p4mcc(h, k, l):
        """Space group 114: P4/mcc. (0,0,l): l even; (h+k) even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if (h + k) % 2 != 0:
            return False
        return True

    @staticmethod
    def group115_p4nbm(h, k, l):
        """Space group 115: P4/nbm. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group116_p4nnc(h, k, l):
        """Space group 116: P4/nnc. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group117_p4mbm(h, k, l):
        """Space group 117: P4/mbm. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group118_p4mnc(h, k, l):
        """Space group 118: P4/mnc. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group119_p4nmm(h, k, l):
        """Space group 119: P4/nmm. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group120_p4ncc(h, k, l):
        """Space group 120: P4/ncc. (h+k) even; (0,0,l): l even."""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group121_p4mm(h, k, l):
        """Space group 121: P4mm. No systematic absences."""
        return True

    @staticmethod
    def group122_p4bm(h, k, l):
        """Space group 122: P4bm. No systematic absences."""
        return True

    @staticmethod
    def group123_p42cm(h, k, l):
        """Space group 123: P42cm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group124_p42nm(h, k, l):
        """Space group 124: P42nm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group125_p42mc(h, k, l):
        """Space group 125: P42mc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group126_p42bc(h, k, l):
        """Space group 126: P42bc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group127_p4cc(h, k, l):
        """Space group 127: P4cc. No systematic absences."""
        return True

    @staticmethod
    def group128_p4nc(h, k, l):
        """Space group 128: P4nc. No systematic absences."""
        return True

    @staticmethod
    def group129_p4mmc(h, k, l):
        """Space group 129: P4mmc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group130_p4mcm(h, k, l):
        """Space group 130: P4mcm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group131_p4ccm(h, k, l):
        """Space group 131: P4ccm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group132_p4ncm(h, k, l):
        """Space group 132: P4ncm. (h+k) even; (0,0,l): l even."""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group133_p4mbc(h, k, l):
        """Space group 133: P4mbc. (h+k) even; (0,0,l): l even."""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group134_p4cbm(h, k, l):
        """Space group 134: P4cbm. (h+k) even; (0,0,l): l even."""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group135_p4mmm(h, k, l):
        """Space group 135: P4/mmm. No systematic absences."""
        return True

    @staticmethod
    def group136_p4mcc(h, k, l):
        """Space group 136: P4/mcc. (0,0,l): l even; (h+k) even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if (h + k) % 2 != 0:
            return False
        return True

    @staticmethod
    def group137_p4nbm(h, k, l):
        """Space group 137: P4/nbm. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group138_p4nnc(h, k, l):
        """Space group 138: P4/nnc. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group139_p4mbm(h, k, l):
        """Space group 139: P4/mbm. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group140_p4mnc(h, k, l):
        """Space group 140: P4/mnc. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group141_p4nmm(h, k, l):
        """Space group 141: P4/nmm. (h+k) even."""
        return (h + k) % 2 == 0

    @staticmethod
    def group142_p4ncc(h, k, l):
        """Space group 142: P4/ncc. (h+k) even; (0,0,l): l even."""
        if (h + k) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group143_p3(h, k, l):
        """Space group 143: P3. No systematic absences. Validated"""
        return True

    @staticmethod
    def group144_p31(h, k, l):
        """Space group 144: P31. No systematic absences."""
        return True

    @staticmethod
    def group145_p32(h, k, l):
        """Space group 145: P32. No systematic absences."""
        return True

    @staticmethod
    def group146_r3(h, k, l):
        """Space group 146: R3 (hexagonal axes). (h - k + l) divisible by 3."""
        return (h - k + l) % 3 == 0

    @staticmethod
    def group147_p3_1_2(h, k, l):
        """Space group 147: P3_1 2. (0,0,l): l = 3n."""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group148_p3_2_1(h, k, l):
        """Space group 148: P3_2 1. (0,0,l): l = 3n."""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group149_r3m(h, k, l):
        """Space group 149: R3m (hexagonal axes). (h - k + l) divisible by 3."""
        return (h - k + l) % 3 == 0

    @staticmethod
    def group150_r3c(h, k, l):
        """Space group 150: R3c (hexagonal axes). (h - k + l) divisible by 3; (0,0,l): l even."""
        if (h - k + l) % 3 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group151_p312(h, k, l):
        """Space group 151: P312. No systematic absences."""
        return True

    @staticmethod
    def group152_p321(h, k, l):
        """Space group 152: P321. No systematic absences."""
        return True

    @staticmethod
    def group153_p3112(h, k, l):
        """Space group 153: P3112. (0,0,l): l = 3n.validated"""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group154_p3121(h, k, l):
        """Space group 154: P3121. (0,0,l): l = 3n.validated"""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group155_p3212(h, k, l):
        """Space group 155: P3212. (0,0,l): l = 3n."""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group156_p3221(h, k, l):
        """Space group 156: P3221. (0,0,l): l = 3n."""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group157_r32(h, k, l):
        """Space group 157: R32 (hexagonal axes). (h - k + l) divisible by 3."""
        return (h - k + l) % 3 == 0

    @staticmethod
    def group158_p3m1(h, k, l):
        """Space group 158: P3m1. No systematic absences."""
        return True

    @staticmethod
    def group159_p31m(h, k, l):
        """Space group 159: P31m. No systematic absences."""
        return True

    @staticmethod
    def group160_p3c1(h, k, l):
        """Space group 160: P3c1. No systematic absences."""
        return True

    @staticmethod
    def group161_p31c(h, k, l):
        """Space group 161: P31c. No systematic absences."""
        return True

    @staticmethod
    def group162_r3m(h, k, l):
        """Space group 162: R3m (hexagonal axes). (h - k + l) divisible by 3."""
        return (h - k + l) % 3 == 0

    @staticmethod
    def group163_r3c(h, k, l):
        """Space group 163: R3c (hexagonal axes). (h - k + l) divisible by 3; (0,0,l): l even."""
        if (h - k + l) % 3 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group164_p_31m(h, k, l):
        """Space group 164: P-31m. No systematic absences. validated"""
        return True

    @staticmethod
    def group165_P_3c1(h, k, l):
        """Space group 165: P-3c1."""
        if h == 0 and k == 0:
            return l % 2 == 0
        if k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group166_R_3m(h, k, l):
        """
        Group 166: R -3 m used in hydrocerusite. Validated
        from http://img.chem.ucl.ac.uk/sgp/large/166bz2.htm"""
        if h == 0 and k == 0:
            # 00l: 3n
            return l % 3 == 0
        elif h == 0 and l == 0:
            # 0k0: k=3n
            return k % 3 == 0
        elif k == 0 and l == 0:
            # h00: h=3n
            return h % 3 == 0
        elif h == k:
            # hhl: l=3n
            return l % 3 == 0
        elif l == 0:
            # hk0: h-k = 3n
            return (h - k) % 3 == 0
        elif k == 0:
            # h0l: h-l = 3n
            return (h - l) % 3 == 0
        elif h == 0:
            # 0kl: h+l = 3n
            return (k + l) % 3 == 0
        else:
            # -h + k + l = 3n
            return (-h + k + l) % 3 == 0

    @staticmethod
    def group167_R_3c(h, k, l):
        """Space group 167: R-3c."""
        if h == k == 0:
            return l % 6 == 0
        elif k == l == 0:
            return h % 3 == 0
        elif k == 0:
            return (h + l) % 3 == 0 and l % 2 == 0
        else:
            return (-h + k + l) % 3 == 0

    @staticmethod
    def group167(h, k, l):
        """Group 167 R -3 c used for Corrundum
        from http://img.chem.ucl.ac.uk/sgp/large/167bz2.htm"""
        if h == 0 and k == 0:
            # 00l: 6n
            return l % 6 == 0
        elif h == 0 and l == 0:
            # 0k0: k=3n
            return k % 3 == 0
        elif k == 0 and l == 0:
            # h00: h=3n
            return h % 3 == 0
        elif h == k:
            # hhl: l=3n
            return l % 3 == 0
        elif l == 0:
            # hk0: h-k = 3n
            return (h - 3) % 3 == 0
        elif k == 0:
            # h0l: l=2n h-l = 3n
            return (l % 2 == 0) and ((h - l) % 3 == 0)
        elif h == 0:
            # 0kl: l=2n h+l = 3n
            return (l % 2 == 0) and ((k + l) % 3 == 0)
        else:
            # -h + k + l = 3n
            return (-h + k + l) % 3 == 0

    @staticmethod
    def group168_P6(h, k, l):
        """Space group 168: P6. No selection. validated"""
        return True

    @staticmethod
    def group168_r_3m(h, k, l):
        """Space group 168: R-3m (hexagonal axes). (h - k + l) divisible by 3."""
        return (h - k + l) % 3 == 0

    @staticmethod
    def group169_P61(h, k, l):
        """Space group 169: P61. Validated"""
        if h == k == 0:
            return l % 6 == 0
        return True

    @staticmethod
    def group169_r_3c(h, k, l):
        """Space group 169: R-3c (hexagonal axes). (h - k + l) divisible by 3; (0,0,l): l even."""
        if (h - k + l) % 3 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group170_p6(h, k, l):
        """Space group 170: P6. No systematic absences."""
        return True

    @staticmethod
    def group171_p61(h, k, l):
        """Space group 171: P61. (0,0,l): l = 6n."""
        if h == 0 and k == 0:
            return l % 6 == 0
        return True

    @staticmethod
    def group172_p65(h, k, l):
        """Space group 172: P65. (0,0,l): l = 6n."""
        if h == 0 and k == 0:
            return l % 6 == 0
        return True

    @staticmethod
    def group173_p62(h, k, l):
        """Space group 173: P62. (0,0,l): l = 3n."""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group174_p64(h, k, l):
        """Space group 174: P64. (0,0,l): l = 3n."""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group175_p63(h, k, l):
        """Space group 175: P63. (0,0,l): l = 2n."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group176_p6_m(h, k, l):
        """Space group 176: P6/m. No systematic absences."""
        return True

    @staticmethod
    def group177_p63_m(h, k, l):
        """Space group 177: P63/m. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group178_p6cc(h, k, l):
        """Space group 178: P6cc. No systematic absences."""
        return True

    @staticmethod
    def group179_p6mc(h, k, l):
        """Space group 179: P6mc. No systematic absences."""
        return True

    @staticmethod
    def group180_p63cm(h, k, l):
        """Space group 180: P63cm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group181_p63_c(h, k, l):
        """Space group 181: P63c. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group182_p6_22(h, k, l):
        """Space group 182: P622. No systematic absences."""
        return True

    @staticmethod
    def group183_p6_21_22(h, k, l):
        """Space group 183: P6_122. (0,0,l): l = 6n."""
        if h == 0 and k == 0:
            return l % 6 == 0
        return True

    @staticmethod
    def group184_p6_522(h, k, l):
        """Space group 184: P6_522. (0,0,l): l = 6n."""
        if h == 0 and k == 0:
            return l % 6 == 0
        return True

    @staticmethod
    def group185_p6_2_22(h, k, l):
        """Space group 185: P6_222. (0,0,l): l = 3n."""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group186_p6_4_22(h, k, l):
        """Space group 186: P6_422. (0,0,l): l = 3n."""
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group187_p6mmm(h, k, l):
        """Space group 187: P6/mmm. No systematic absences.validated"""
        return True

    @staticmethod
    def group188_p6mcc(h, k, l):
        """Space group 188: P6/mcc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group189_p63mc(h, k, l):
        """Space group 189: P63mc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group190_p63cm(h, k, l):
        """Space group 190: P63cm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group191_p63_mcm(h, k, l):
        """Space group 191: P63/mcm. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group192_p63_mmc(h, k, l):
        """Space group 192: P63/mmc. (0,0,l): l even."""
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group193_p23(h, k, l):
        """Space group 193: P23. No systematic absences."""
        return True

    @staticmethod
    def group194_f23(h, k, l):
        """Space group 194: F23. F-centering: h, k, l all even or all odd."""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group195_i23(h, k, l):
        """Space group 195: I23. I-centering: (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group196_p213(h, k, l):
        """Space group 196: P213. No systematic absences."""
        return True

    @staticmethod
    def group197_i213(h, k, l):
        """Space group 197: I213. I-centering: (h + k + l) even.validated"""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group198_pm3(h, k, l):
        """Space group 198: Pm-3. No systematic absences."""
        return True

    @staticmethod
    def group199_pa3(h, k, l):
        """Space group 199: Pa-3. No systematic absences."""
        return True

    @staticmethod
    def group200_fn3(h, k, l):
        """Space group 200: Fm-3. F-centering: h, k, l all even or all odd."""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group201_pn3m(h, k, l):
        """Space group 201: Pn-3m. No systematic absences."""
        return True

    @staticmethod
    def group202_pn3n(h, k, l):
        """Space group 202: Pn-3n. (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group203_pm3m(h, k, l):
        """Space group 203: Pm-3m. No systematic absences."""
        return True

    @staticmethod
    def group204_pm3n(h, k, l):
        """Space group 204: Pm-3n. (h + k + l) even.validated"""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group205_pn3n(h, k, l):
        """Space group 205: Pn-3n. (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group206_fm3m(h, k, l):
        """Space group 206: Fm-3m. F-centering: h, k, l all even or all odd."""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group207_fm3c(h, k, l):
        """Space group 207: Fm-3c. F-centering: h, k, l all even or all odd; (0, 0, l): l even."""
        if not (h % 2 == k % 2 == l % 2):
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group208_fn3m(h, k, l):
        """Space group 208: Fd-3m. F-centering: h, k, l all even or all odd."""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group209_fn3c(h, k, l):
        """Space group 209: Fd-3c. F-centering: h, k, l all even or all odd; (0, 0, l): l even.validated"""
        if not (h % 2 == k % 2 == l % 2):
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group210_im3m(h, k, l):
        """Space group 210: Im-3m. I-centering: (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group211_im3c(h, k, l):
        """Space group 211: Im-3c. I-centering: (h + k + l) even; (0, 0, l): l even.validated"""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group212_pa3(h, k, l):
        """Space group 212: Pa-3. No systematic absences."""
        return True

    @staticmethod
    def group213_ia3(h, k, l):
        """Space group 213: Ia-3. I-centering: (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group214_fm3m(h, k, l):
        """Space group 214: Fm-3m. F-centering: h, k, l all even or all odd."""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group215_fm3c(h, k, l):
        """Space group 215: Fm-3c. F-centering: h, k, l all even or all odd; (0, 0, l): l even."""
        if not (h % 2 == k % 2 == l % 2):
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group216_fd3m(h, k, l):
        """Space group 216: Fd-3m. F-centering: h, k, l all even or all odd.validated"""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group217_fd3c(h, k, l):
        """Space group 217: Fd-3c. F-centering: h, k, l all even or all odd; (0, 0, l): l even."""
        if not (h % 2 == k % 2 == l % 2):
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group218_ia3d(h, k, l):
        """Space group 218: Ia-3d. I-centering: (h + k + l) even; (0, 0, l): l divisible by 4."""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group219_pa3(h, k, l):
        """Space group 219: Pa-3. No systematic absences."""
        return True

    @staticmethod
    def group220_ia3(h, k, l):
        """Space group 220: Ia-3. I-centering: (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group221_pm3m(h, k, l):
        """Space group 221: Pm-3m. No systematic absences.validated"""
        return True

    @staticmethod
    def group222_pm3n(h, k, l):
        """Space group 222: Pm-3n. (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group223_pn3m(h, k, l):
        """Space group 223: Pn-3m. No systematic absences."""
        return True

    @staticmethod
    def group224_pn3n(h, k, l):
        """Space group 224: Pn-3n. (h + k + l) even."""
        return (h + k + l) % 2 == 0

    @staticmethod
    def group225_fm3m(h, k, l):
        """Space group 225: Fm-3m. F-centering: h, k, l all even or all odd.validated"""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group226_fm3c(h, k, l):
        """Space group 226: Fm-3c. F-centering: h, k, l all even or all odd; (0, 0, l): l even."""
        if not (h % 2 == k % 2 == l % 2):
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group227_fd3m(h, k, l):
        """Space group 227: Fd-3m. F-centering: h, k, l all even or all odd."""
        return h % 2 == k % 2 == l % 2

    @staticmethod
    def group228_fd3c(h, k, l):
        """Space group 228: Fd-3c. F-centering: h, k, l all even or all odd; (0, 0, l): l even."""
        if not (h % 2 == k % 2 == l % 2):
            return False
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group229_ia3d(h, k, l):
        """Space group 229: Ia-3d. I-centering: (h + k + l) even; (0, 0, l): l divisible by 4."""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group230_ia3d(h, k, l):
        """Space group 230: Ia-3d. I-centering: (h + k + l) even; (0, 0, l): l divisible by 4."""
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 4 == 0
        return True
