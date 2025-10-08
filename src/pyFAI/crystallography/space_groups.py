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

"""Space groups: There are 230 space groups as defined in the International
Tables of Crystallography (ITC vol.A), some of them have different origins.
For now only the conventional origin is implemented, alternative representation will be
addressed in a second stage.

The ReflectionCondition class contains a function with the selection rules for each
of the 230 space group.
"""

from __future__ import annotations

__authors__ = ["Jérôme Kieffer", "Gudrun Lotze"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/10/2025"
__status__ = "production"


class ReflectionCondition:
    """This class contains selection rules for most space-groups

    All methods are static and take a triplet hkl as input representing a family of Miller plans.
    They return True if the reflection is allowed by symmetry, False otherwise.

    Most of those methods are AI-generated (Co-Pilot) and about 80% of them are still WRONG unless tagged
    "validated" in the docstring.

    Help is welcome to polish this class and fix the non-validated ones.
    """

    @staticmethod
    def group1_P1(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 1: P1. Triclinic.

        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group2_P1bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 2: P1̄. Triclinic.

        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group3_P2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 3: P2. Monoclinic, unique axis b.

        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group4_P21(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 4: P21. Monoclinic, unique axis b.

        Valid reflections must satisfy:
        - 0k0 (h = 0, l = 0): k even

        Source: ITC
        validated
        """
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group5_C2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 5: C2. Monoclinic, unique axis b.

        Valid reflections must satisfy:
        - General hkl:           h + k = 2n
        - h0l (k = 0):           h even
        - 0kl (h = 0):           k even
        - hk0 (l = 0):           h + k even
        - 0k0 (h = 0, l = 0):    k even
        - h00 (k = 0, l = 0):    h even

        Source: ITC
        validated
        """
        # Most specific conditions
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00

        # Semi-specific conditions
        if k == 0:
            return h % 2 == 0  # h0l
        if h == 0:
            return k % 2 == 0  # 0kl
        if l == 0:
            return (h + k) % 2 == 0  # hk0

        # General condition
        return (h + k) % 2 == 0

    @staticmethod
    def group6_Pm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 6: Pm. Monoclinic, unique axis b.

        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group7_Pc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 7: Pc. Monoclinic, unique axis b.

        Valid reflections:
        - h0l (k=0): l even
        - 00l (h=0, k=0): l even

        Source: ITC
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if k == 0:
            return l % 2 == 0  # h0l

        return True

    @staticmethod
    def group8_Cm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 8: Cm. Monoclinic, unique axis b.

        Valid reflections must satisfy:
        - General hkl:           h + k = 2n
        - h0l (k = 0):           h even
        - 0kl (h = 0):           k even
        - hk0 (l = 0):           h + k even
        - 0k0 (h = 0, l = 0):    k even
        - h00 (k = 0, l = 0):    h even

        Source: ITC
        validated
        """
        # Most specific conditions
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0

        # Semi-specific conditions
        if k == 0:
            return h % 2 == 0  # h0l
        if h == 0:
            return k % 2 == 0  # 0kl
        if l == 0:
            return (h + k) % 2 == 0  # hk0

        # General condition
        return (h + k) % 2 == 0  # hkl

    @staticmethod
    def group9_Cc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 9: Cc. Monoclinic, unique axis b.

        Valid reflections must satisfy:
        - General hkl:           h + k = 2n
        - h0l (k = 0):           h, l even
        - 0kl (h = 0):           k even
        - hk0 (l = 0):           h + k even
        - 0k0 (h = 0, l = 0):    k even
        - h00 (k = 0, l = 0):    h even
        - 00l (h = 0, k = 0):    l even

        Source: ITC
        validated
        """
        # Most specific conditions first
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00

        # Semi-specific conditions
        if k == 0:
            return h % 2 == 0 and l % 2 == 0  # h0l
        if h == 0:
            return k % 2 == 0  # 0kl
        if l == 0:
            return (h + k) % 2 == 0  # hk0

        # General condition
        return (h + k) % 2 == 0  # hkl

    @staticmethod
    def group10_P2m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 10: P2/m. Monoclinic, unique axis b.

        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group11_P21m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 11: P2₁/m. Monoclinic, unique axis b.

        Valid reflections must satisfy:
        - 0k0 (h = 0, l = 0):       k even

        Source: ITC
        validated
        """
        # 0k0
        if h == 0 and l == 0:
            return k % 2 == 0
        return True

    @staticmethod
    def group12_C2m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 12: C2/m. Monoclinic, unique axis b.

        Valid reflections must satisfy:
        - General hkl:              h + k even
        - h0l (k = 0):              h even
        - 0kl (h = 0):              k even
        - hk0 (l = 0):              h + k even
        - 0k0 (h = 0, l = 0):       k even
        - h00 (k = 0, l = 0):       h even

        Source: ITC
        validated
        """
        # 0k0
        if h == 0 and l == 0:
            return k % 2 == 0
        # h00
        if k == 0 and l == 0:
            return h % 2 == 0
        # h0l
        if k == 0:
            return h % 2 == 0
        # 0kl
        if h == 0:
            return k % 2 == 0
        # hk0
        if l == 0:
            return (h + k) % 2 == 0
        # General hkl
        if (h + k) % 2 != 0:
            return False
        return True

    @staticmethod
    def group13_P2c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 13: P2/c. Monoclinic, unique axis b.

        Valid reflections must satisfy:
        - h0l (k = 0):              l even
        - 00l (h = 0, k = 0):       l even

        Source: ITC
        validated
        """
        # 00l
        if h == 0 and k == 0:
            return l % 2 == 0
        # h0l
        if k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group14_P21c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 14: P2₁/c. Monoclinic, unique axis b.

        Valid reflections must satisfy:
        - h0l (k = 0):              l even
        - 0k0 (h = 0, l = 0):       k even
        - 00l (h = 0, k = 0):       l even

        Source: ITC
        validated
        """
        # 00l
        if h == 0 and k == 0:
            return l % 2 == 0
        # 0k0
        if h == 0 and l == 0:
            return k % 2 == 0
        # h0l
        if k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group15_C2c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 15: C 2/c. Monoclinic, unique axis b.

        Valid reflections must satisfy:
        - General hkl:           h + k even
        - h0l (k = 0):           h, l even
        - 0kl (h = 0):           k even
        - hk0 (l = 0):           h + k even
        - 0k0 (h = 0, l = 0):    k even
        - h00 (k = 0, l = 0):    h even
        - 00l (h = 0, k = 0):    l even

        Source: https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-hkl?gnum=15
        ITC, p 261
        There are different rules for different cell choices and other unique axis.

        validated
        """
        # 00l
        if h == 0 and k == 0:
            return l % 2 == 0
        # 0k0
        if h == 0 and l == 0:
            return k % 2 == 0
        # h00
        if k == 0 and l == 0:
            return h % 2 == 0
        # h0l
        if k == 0:
            return h % 2 == 0 and l % 2 == 0
        # 0kl
        if h == 0:
            return k % 2 == 0
        # hk0
        if l == 0:
            return (h + k) % 2 == 0
        # General hkl
        if (h + k) % 2 != 0:
            return False
        return True

    @staticmethod
    def group16_P222(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 16: P222. Orthorhombic.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group17_P2221(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 17: P222₁. Orthorhombic.

        Valid reflections must satisfy:
        - 00l (h = 0, k = 0):   l even

        Source: ITC
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        return True

    @staticmethod
    def group18_P21212(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 18: P2₁2₁2. Orthorhombic.

        Valid reflections must satisfy:
        - h00 (k = 0, l = 0) : h even
        - 0k0 (h = 0, l = 0): k even

        Source: ITC
        validated
        """
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        return True

    @staticmethod
    def group19_P212121(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 19: P2₁2₁2₁. Orthorhombic.

        Valid reflections must satisfy:
        - h00 (k = 0, l = 0):    h even
        - 0k0 (h = 0, l = 0):    k even
        - 00l (h = 0, k = 0):    l even

        Source: ITC
        validated
        """

        if h == 0 and k == 0:
            return l % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group20_C2221(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 20: C 2 2 21. Orthorhombic

        Valid reflections must satisfy:
        - General hkl:           h + k even
        - 0kl (h = 0):           k even
        - h0l (k = 0):           h even
        - hk0 (l = 0):           h + k even
        - h00 (k = 0, l = 0):    h even
        - 0k0 (h = 0, l = 0):    k even
        - 00l (h = 0, k = 0):    l even

        Source: https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-hkl?gnum=20
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return k % 2 == 0  # 0kl
        if k == 0:
            return h % 2 == 0  # h0l
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return (h + k) % 2 == 0  # general

    @staticmethod
    def group21_C222(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 21: C 2 2 2. Orthorhombic
        Valid reflections must satisfy:
        - General (hkl):       h + k even
        - 0kl (h=0):           k even
        - h0l (k=0):           h even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even

        Note: Unlike space group 20 (C 2 2 21), there is **no rule for 00l** in this group.
        validated
        """

        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return k % 2 == 0  # 0kl
        if k == 0:
            return h % 2 == 0  # h0l
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return (h + k) % 2 == 0  # general

    @staticmethod
    def group22_F222(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 22: F222. Orthorhombic.

        Valid reflections must satisfy:
        - General hkl:           h + k, h + l, k + l even
        - 0kl (h = 0):           k, l even
        - h0l (k = 0):           h, l even
        - hk0 (l = 0):           h, k even
        - h00 (k = 0, l = 0):    h even
        - 0k0 (h = 0, l = 0):    k even
        - 00l (h = 0, k = 0):    l even

        Source: ITC
        validated
        """
        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return k % 2 == 0 and l % 2 == 0

        # h0l
        if k == 0:
            return h % 2 == 0 and l % 2 == 0

        # hk0
        if l == 0:
            return h % 2 == 0 and k % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        # 0k0
        if h == 0 and l == 0:
            return k % 2 == 0

        # 00l
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group23_I222(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 23: I222. Orthorhombic.

        Valid reflections must satisfy:
        - General hkl:           h + k + l = 2n
        - 0kl (h = 0):           k + l even
        - h0l (k = 0):           h + l even
        - hk0 (l = 0):           h + k even
        - h00 (k = 0, l = 0):    h even
        - 0k0 (h = 0, l = 0):    k even
        - 00l (h = 0, k = 0):    l even

        Source: ITC
        validated
        """
        # General condition
        if (h + k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return (k + l) % 2 == 0

        # h0l
        if k == 0:
            return (h + l) % 2 == 0

        # hk0
        if l == 0:
            return (h + k) % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        # 0k0
        if h == 0 and l == 0:
            return k % 2 == 0

        # 00l
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group24_I212121(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 24: I2₁2₁2₁. Orthorhombic.

        Valid reflections must satisfy:
        - General hkl:           h + k + l = 2n
        - 0kl (h = 0):           k + l even
        - h0l (k = 0):           h + l even
        - hk0 (l = 0):           h + k even
        - h00 (k = 0, l = 0):    h even
        - 0k0 (h = 0, l = 0):    k even
        - 00l (h = 0, k = 0):    l even

        Source: ITC
        validated
        """
        # General condition
        if (h + k + l) % 2 != 0:
            return False
        # 0kl
        if h == 0:
            return (k + l) % 2 == 0
        # h0l
        if k == 0:
            return (h + l) % 2 == 0
        # hk0
        if l == 0:
            return (h + k) % 2 == 0
        # h00
        if k == 0 and l == 0:
            return h % 2 == 0
        # 0k0
        if h == 0 and l == 0:
            return k % 2 == 0
        # 00l
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group25_Pmm2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 25: Pmm2. Primitive lattice.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group26_Pmc21(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 26: Pmc21.
        Valid reflections must satisfy:
        - h0l: l = 2n
        - 00l: l = 2n
        validated
        """
        if k == 0:  # Covers both h0l and 00l cases
            return l % 2 == 0
        return True

    @staticmethod
    def group27_Pcc2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 27: Pcc2.
        Valid reflections must satisfy:
        - General (hkl):       No condition (unrestricted)
        - 0kl (h=0):           l even
        - h0l (k=0):           l even
        - 00l (h=0, k=0):      l even
        No other systematic absences.
        validated
        """
        if h == 0 or k == 0:  # Covers 0kl, h0l, and 00l
            return l % 2 == 0
        return True

    @staticmethod
    def group28_pma2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 28: Pma2
        Valid reflections must satisfy:
        - h0l (k=0):      h even
        - h00 (k=0, l=0): h even
        No other systematic absences.
        validated
        """
        if k == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group29_Pca21(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 29: Pca2₁
        Valid reflections must satisfy:
        - 0kl (h=0):      l even
        - h0l (k=0):      h even
        - h00 (k=0, l=0): h even
        - 00l (h=0, k=0): l even
        No other systematic absences.
        validated
        """
        if h == 0 and k == 0:  # 00l case
            return l % 2 == 0
        if h == 0:  # 0kl case
            return l % 2 == 0
        if k == 0:  # h0l case (includes h00)
            return h % 2 == 0
        return True

    @staticmethod
    def group30_pnc2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 30: Pnc2
        Valid reflections must satisfy:
        - 0kl (h=0):        k + l even
        - h0l (k=0):        l even
        - 0k0 (h=0, l=0):   k even
        - 00l (h=0, k=0):   l even
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            return l % 2 == 0  # h0l
        return True

    @staticmethod
    def group31_pmn21(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 31: Pmn2₁
        Valid reflections must satisfy:
        - h0l (k=0):      h + l even
        - h00 (k=0, l=0): h even
        - 00l (h=0, k=0): l even
        validated
        """
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if k == 0:  # Covers both h0l and h00
            if l == 0:  # h00
                return h % 2 == 0
            return (h + l) % 2 == 0  # h0l
        return True

    @staticmethod
    def group32_pba2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """ "
        Space group 32: Pba2.
        Valid reflections must satisfy:
        - 0kl (h=0):      k even
        - h0l (k=0):      h even
        - h00 (k=0, l=0): h even
        - 0k0 (h=0, l=0): k even
        No other systematic absences.
        validated"""
        if h == 0:
            return k % 2 == 0  # Covers 0kl and 0k0
        if k == 0:
            return h % 2 == 0  # Covers h0l and h00
        return True

    @staticmethod
    def group33_Pna21(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 33: Pna21.
        Valid reflections must satisfy:
        - 0kl (h=0):        k + l even
        - h0l (k=0):        h even
        - h00 (k=0, l=0):   h even
        - 0k0 (h=0, l=0):   k even
        - 00l (h=0, k=0):   l even
        validated"""
        if h == 0:
            return (k + l) % 2 == 0  # Covers 0kl, 0k0, 00l
        if k == 0:
            return h % 2 == 0  # h0l/h00
        return True

    @staticmethod
    def group34_Pnn2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 34: Pnn2. P-centering.
        Valid reflections must satisfy:
        - 0kl (h=0):        k + l even
        - h0l (k=0):        h + l even
        - h00 (k=0, l=0):   h even
        - 0k0 (h=0, l=0):   k even
        - 00l (h=0, k=0):   l even
        validated"""
        if h == 0 or k == 0:
            return (h + k + l) % 2 == 0
        return True

    @staticmethod
    def group35_Cmm2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 35: Cmm2. C-centering.
        Valid reflections must satisfy:
        - General (hkl):       h + k even
        - 0kl (h=0):           k even
        - h0l (k=0):           h even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        validated"""
        if h == 0 or k == 0:
            return (h + k) % 2 == 0
        return (h + k) % 2 == 0

    @staticmethod
    def group36_Cmc21(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 36: Cmc2₁. C-centering.
        Valid reflections must satisfy:
        - General (hkl):       h + k even
        - 0kl (h=0):           k even
        - h0l (k=0):           h and l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if h == 0:
            return (k % 2 == 0) if k != 0 else (l % 2 == 0)  # covers 0kl, 0k0, 00l
        if k == 0:
            return (h % 2 == 0) and (l == 0 or l % 2 == 0)  # covers h0l, h00
        return (h + k) % 2 == 0  # covers hk0 and general case

    @staticmethod
    def group37_Cmm2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """Space group 37: Cmm2. C-centering.
        Valid reflections satisfy:
        - General (hkl):       h + k even
        - 0kl (h=0):           k and l even
        - h0l (k=0):           h and l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated"""
        if h == 0:
            return (k == 0 or k % 2 == 0) and (l == 0 or l % 2 == 0)
        if k == 0:
            return h % 2 == 0 and (l == 0 or l % 2 == 0)
        return (h + k) % 2 == 0

    @staticmethod
    def group38_Amm2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 38: Amm2. A-centering.
        Valid reflections satisfy:
        - General (hkl):       k + l even
        - 0kl (h=0):           k + l even
        - h0l (k=0):           l even
        - hk0 (l=0):           k even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if k == 0:
            return l % 2 == 0  # h0l (includes h00 when l=0)
        if l == 0:
            return k % 2 == 0  # hk0 (includes 0k0 when h=0)
        return (k + l) % 2 == 0  # general and 0kl

    @staticmethod
    def group39_Aem2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 39: Aem2. A-centering.
        Valid reflections must satisfy:
        - General (hkl):       k + l even
        - 0kl (h=0):           k and l even
        - h0l (k=0):           l even
        - hk0 (l=0):           k even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if h == 0:
            return (k == 0 or k % 2 == 0) and (
                l == 0 or l % 2 == 0
            )  # covers 00l, 0k0, 0kl
        if k == 0:
            return l % 2 == 0  # covers h0l (including h00 when l=0)
        if l == 0:
            return k % 2 == 0  # covers hk0 (including 0k0 when h=0)
        return (k + l) % 2 == 0  # general case

    @staticmethod
    def group40_Ama2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 40: Ama2. A-centering.
        Valid reflections must satisfy:
        - General (hkl):       k + l even
        - 0kl (h=0):           k + l even
        - h0l (k=0):           h and l even
        - hk0 (l=0):           k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if h == 0:
            if k == 0:
                return l % 2 == 0
            if l == 0:
                return k % 2 == 0
            return (k + l) % 2 == 0
        if k == 0:
            return h % 2 == 0 and (l == 0 or l % 2 == 0)
        return k % 2 == 0 if l == 0 else (k + l) % 2 == 0

    @staticmethod
    def group41_Aea2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 41: Aea2. A-centering.
        Valid reflections must satisfy:
        - General (hkl):       k + l even
        - 0kl (h=0):           k and l even
        - h0l (k=0):           h and l even
        - hk0 (l=0):           k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated"""
        if h == k == 0:  # 00l
            return l % 2 == 0
        if h == l == 0:  # 0k0
            return k % 2 == 0
        if k == l == 0:  # h00
            return h % 2 == 0
        if h == 0:  # 0kl
            return k % 2 == 0 and l % 2 == 0
        if k == 0:  # h0l
            return h % 2 == 0 and l % 2 == 0
        if l == 0:  # hk0
            return k % 2 == 0
        return (k + l) % 2 == 0  # general hkl

    @staticmethod
    def group42_Fmm2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 42: Fmm2. F-centering.
        Valid reflections must satisfy:
        - General (hkl):       h + k, h + l, and k + l even
        - 0kl (h=0):           k and l even
        - h0l (k=0):           h and l even
        - hk0 (l=0):           h and k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated"""
        if h == k == 0:
            return l % 2 == 0  # 00l
        if h == l == 0:
            return k % 2 == 0  # 0k0
        if k == l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return not (k % 2 or l % 2)  # 0kl
        if k == 0:
            return not (h % 2 or l % 2)  # h0l
        if l == 0:
            return not (h % 2 or k % 2)  # hk0
        return (h + k) % 2 == 0 and (h + l) % 2 == 0 and (k + l) % 2 == 0

    @staticmethod
    def group43_Fdd2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 43: Fdd2. F-centering.
        Valid reflections must satisfy:
        - General (hkl):       h + k, h + l, and k + l even
        - 0kl (h=0):           k and l even, k + l = 4n
        - h0l (k=0):           h and l even, h + l = 4n
        - hk0 (l=0):           h and k even
        - h00 (k=0, l=0):      h % 4 == 0
        - 0k0 (h=0, l=0):      k % 4 == 0
        - 00l (h=0, k=0):      l % 4 == 0
        validated
        """
        if h == k == 0:
            return l % 4 == 0  # 00l
        if h == l == 0:
            return k % 4 == 0  # 0k0
        if k == l == 0:
            return h % 4 == 0  # h00
        if h == 0:
            return k % 2 == 0 and l % 2 == 0 and (k + l) % 4 == 0  # 0kl
        if k == 0:
            return h % 2 == 0 and l % 2 == 0 and (h + l) % 4 == 0  # h0l
        if l == 0:
            return h % 2 == 0 and k % 2 == 0  # hk0
        return (h + k) % 2 == 0 and (h + l) % 2 == 0 and (k + l) % 2 == 0

    @staticmethod
    def group44_Imm2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 44: Imm2. I-centering.
        Valid reflections must satisfy:
        - General (hkl):       h + k + l even
        - 0kl (h=0):           k + l even
        - h0l (k=0):           h + l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if h == k == 0:
            return l % 2 == 0  # 00l
        if h == l == 0:
            return k % 2 == 0  # 0k0
        if k == l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            return (h + l) % 2 == 0  # h0l
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return (h + k + l) % 2 == 0  # general hkl

    @staticmethod
    def group45_Iba2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 45: Iba2. I-centering.
        Valid reflections must satisfy:
        - General (hkl):       h + k + l even
        - 0kl (h=0):           k and l even
        - h0l (k=0):           h and l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if h == k == 0:
            return l % 2 == 0  # 00l
        if h == l == 0:
            return k % 2 == 0  # 0k0
        if k == l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return k % 2 == 0 and l % 2 == 0  # 0kl
        if k == 0:
            return h % 2 == 0 and l % 2 == 0  # h0l
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return (h + k + l) % 2 == 0

    @staticmethod
    def group46_Ima2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 46: Ima2. I-centering.
        Valid reflections must satisfy:
        - General (hkl):       h + k + l even
        - 0kl (h=0):           k + l even
        - h0l (k=0):           h and l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if h == k == 0:
            return l % 2 == 0  # 00l
        if h == l == 0:
            return k % 2 == 0  # 0k0
        if k == l == 0:
            return h % 2 == 0  # h00

        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            return h % 2 == 0 and l % 2 == 0  # h0l
        if l == 0:
            return (h + k) % 2 == 0  # hk0

        return (h + k + l) % 2 == 0  # General

    @staticmethod
    def group47_Pmmm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 47: Pmmm. Primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        validated
        """
        return True

    @staticmethod
    def group48_Pnnn(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 48: Pnnn. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           k + l even
        - h0l (k=0):           h + l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        No general condition on hkl.
        validated
        """
        if h == k == 0:
            return l % 2 == 0  # 00l
        if h == l == 0:
            return k % 2 == 0  # 0k0
        if k == l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            return (h + l) % 2 == 0  # h0l
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return True

    @staticmethod
    def group49_Pccm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 49: Pccm. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           l even
        - h0l (k=0):           l even
        - 00l (h=0, k=0):      l even
        No general condition on hkl.
        validated
        """
        if h == 0 or k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group50_Pban(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 50: Pban. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           k even
        - h0l (k=0):           h even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        No general condition on hkl.
        validated
        """
        if l == 0:
            return (h + k) % 2 == 0  # hk0 (includes h00 & 0k0 when l=0)
        if h == 0:
            return k % 2 == 0  # 0kl
        if k == 0:
            return h % 2 == 0  # h0l
        return True  # general case

    @staticmethod
    def group51_Pmma(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 51: Pmma. Primitive lattice.
        Valid reflections must satisfy:
        - hk0 (l=0):           h even
        - h00 (k=0, l=0):      h even
        No general condition on hkl.
        validated
        """
        if l == 0:
            return h % 2 == 0  # hk0 (includes h00 when k=0)
        return True  # general

    @staticmethod
    def group52_Pnna(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 52: Pnna. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           k + l even
        - h0l (k=0):           h + l even
        - hk0 (l=0):           h even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        No general condition on hkl.
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            return (h + l) % 2 == 0  # h0l
        if l == 0:
            return h % 2 == 0  # hk0
        return True

    @staticmethod
    def group53_Pmna(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 53: Pmna. Primitive lattice.
        Valid reflections must satisfy:
        - h0l (k=0):           h + l even
        - hk0 (l=0):           h even
        - h00 (k=0, l=0):      h even
        - 00l (h=0, k=0):      l even
        No general condition on hkl.
        validated
        """
        if k == 0:
            return (h + l) % 2 == 0 if l != 0 else h % 2 == 0
        if l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group54_Pcca(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 54: Pcca. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           l even
        - h0l (k=0):           l even
        - hk0 (l=0):           h even
        - h00 (k=0, l=0):      h even
        - 00l (h=0, k=0):      l even
        No general condition on hkl.
        validated
        """
        if l == 0:
            return h % 2 == 0
        if h == 0 or k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group55_Pbam(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 55: Pbam. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           k even
        - h0l (k=0):           h even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        No general condition on hkl.
        validated
        """
        if h == 0:
            return k % 2 == 0  # Covers 0k0 and 0kl
        if k == 0:
            return h % 2 == 0  # Covers h00 and h0l
        return True

    @staticmethod
    def group56_Pccn(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 56: Pccn. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           l even
        - h0l (k=0):           l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        No general condition on hkl.
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return l % 2 == 0  # 0kl
        if k == 0:
            return l % 2 == 0  # h0l
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return True

    @staticmethod
    def group57_Pbcm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 57: Pbcm. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           k even
        - h0l (k=0):           l even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        No general condition on hkl.
        validated
        """
        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return k % 2 == 0  # 0kl
        if k == 0:
            return l % 2 == 0  # h0l
        return True

    @staticmethod
    def group58_Pnnm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 58: Pnnm. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           k + l even
        - h0l (k=0):           h + l even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        No general condition on full hkl.
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            return (h + l) % 2 == 0  # h0l
        return True

    @staticmethod
    def group59_Pmmn(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 59: Pmmn. Primitive lattice.
        Valid reflections must satisfy:
        - hk0 (l=0):       h + k even
        - h00 (k=0, l=0):  h even
        - 0k0 (h=0, l=0):  k even
        No general condition on other hkl.
        validated
        """
        if l == 0:
            if h == 0:
                return k % 2 == 0  # 0k0
            if k == 0:
                return h % 2 == 0  # h00
            return (h + k) % 2 == 0  # hk0
        return True

    @staticmethod
    def group60_Pbcn(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 60: Pbcn. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           k even
        - h0l (k=0):           l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        No general condition on full hkl.
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        if h == 0:
            return k % 2 == 0  # 0kl
        if k == 0:
            return l % 2 == 0  # h0l
        return True

    @staticmethod
    def group61_Pbca(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 61: Pbca. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           k even
        - h0l (k=0):           l even
        - hk0 (l=0):           h even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        No general condition on hkl.
        validated
        """
        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            return k % 2 == 0  # 0k0 and 0kl
        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return l % 2 == 0  # h0l
        if l == 0:
            return h % 2 == 0  # hk0
        return True

    @staticmethod
    def group62_Pnma(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 62: Pnma. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):           k + l even
        - hk0 (l=0):           h even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        No general condition on general hkl.
        validated
        """
        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return (k + l) % 2 == 0  # 0kl
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if l == 0:
            return h % 2 == 0  # hk0
        return True

    @staticmethod
    def group63_Cmcm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 63: Cmcm. C-centering.
        Valid reflections must satisfy:
        - general hkl:         h + k even
        - 0kl (h=0):           k even
        - h0l (k=0):           h and l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if (h + k) % 2 != 0:
            return False

        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            return k % 2 == 0  # 0k0 and 0kl

        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return h % 2 == 0 and l % 2 == 0  # h0l
        return True

    @staticmethod
    def group64_Cmce(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 64: Cmce. C-centering.
        Valid reflections must satisfy:
        - general hkl:         h + k even
        - 0kl (h=0):           k even
        - h0l (k=0):           h and l even
        - hk0 (l=0):           h and k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if (h + k) % 2 != 0:
            return False  # General condition for all reflections

        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return k % 2 == 0  # 0kl

        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return h % 2 == 0 and l % 2 == 0  # h0l

        if l == 0:
            return h % 2 == 0 and k % 2 == 0  # hk0

        return True

    @staticmethod
    def group65_Cmmm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 65: Cmmm. C-centering.
        Valid reflections must satisfy:
        - general hkl:         h + k even
        - 0kl (h=0):           k even
        - h0l (k=0):           h even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        validated
        """
        if (h + k) % 2 != 0:
            return False  # general
        if h == 0:
            return k % 2 == 0  #  0kl, 0k0
        if k == 0:
            return h % 2 == 0  #  h0l, h00
        return True

    @staticmethod
    def group66_Cccm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 66: Cccm. C-centering.
        Valid reflections must satisfy:
        - general hkl:         h + k even
        - 0kl (h=0):           k, l even
        - h0l (k=0):           h, l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if h == 0:
            return k % 2 == 0 and l % 2 == 0  # 0kl, 0k0, 00l
        if k == 0:
            return h % 2 == 0 and l % 2 == 0  # h0l, h00
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return (h + k) % 2 == 0  # general hkl

    @staticmethod
    def group67_Cmme(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 67: Cmme. C-centering.
        Valid reflections must satisfy:
        - general hkl:         h + k even
        - 0kl (h=0):           k even
        - h0l (k=0):           h even
        - hk0 (l=0):           h, k even
        validated
        """
        if h == 0:
            return k % 2 == 0  # 0kl, 0k0
        if k == 0:
            return h % 2 == 0  # h0l, h00
        if l == 0:
            return h % 2 == 0 and k % 2 == 0  # hk0
        return (h + k) % 2 == 0  # general hkl

    @staticmethod
    def group68_Ccce(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 68: Ccce. C-centering.
        Valid reflections must satisfy:
        - general hkl:         h + k even
        - 0kl (h=0):           k, l even
        - h0l (k=0):           h, l even
        - hk0 (l=0):           h, k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return k % 2 == 0 and l % 2 == 0  # 0kl
        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return h % 2 == 0 and l % 2 == 0  # h0l
        if l == 0:
            return h % 2 == 0 and k % 2 == 0  # hk0
        return (h + k) % 2 == 0  # general hkl

    @staticmethod
    def group69_Fmmm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 69: Fmmm. F-centering.
        Valid reflections must satisfy:
        - general hkl: h + k, h + l, k + l even
        - 0kl (h=0):   k, l even
        - h0l (k=0):   h, l even
        - hk0 (l=0):   h, k even
        - h00 (k=0,l=0): h even
        - 0k0 (h=0,l=0): k even
        - 00l (h=0,k=0): l even
        validated
        """
        # general
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False
        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return k % 2 == 0 and l % 2 == 0  # 0kl
        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return h % 2 == 0 and l % 2 == 0  # h0l
        if l == 0:
            return h % 2 == 0 and k % 2 == 0  # hk0
        return True

    @staticmethod
    def group70_Fddd(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 70: Fddd. F-centering.
        Valid reflections must satisfy:
        - general hkl:         h + k, h + l, k + l even
        - 0kl (h=0):           k + l = 4n, k, l even
        - h0l (k=0):           h + l = 4n, h, l even
        - hk0 (l=0):           h + k = 4n, h, k even
        - h00 (k=0, l=0):      h = 4n
        - 0k0 (h=0, l=0):      k = 4n
        - 00l (h=0, k=0):      l = 4n
        validated
        """
        # general
        if (h + k) % 2 or (h + l) % 2 or (k + l) % 2:
            return False

        if h == 0:
            if k == 0:
                return l % 4 == 0  # 00l
            if l == 0:
                return k % 4 == 0  # 0k0
            return (k + l) % 4 == 0 and k % 2 == 0 and l % 2 == 0  # 0kl

        if k == 0:
            if l == 0:
                return h % 4 == 0  # h00
            return (h + l) % 4 == 0 and h % 2 == 0 and l % 2 == 0  # h0l

        if l == 0:
            return (h + k) % 4 == 0 and h % 2 == 0 and k % 2 == 0  # hk0

        return True

    @staticmethod
    def group71_Immm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 71: Immm. Body-centered lattice (I-centering).
        Valid reflections must satisfy:
        - general hkl:         h + k, h + l, k + l even
        - 0kl (h=0):           k + l = 4n, k, l even
        - h0l (k=0):           h + l = 4n, h, l even
        - hk0 (l=0):           h + k = 4n, h, k even
        - h00 (k=0, l=0):      h = 4n
        - 0k0 (h=0, l=0):      k = 4n
        - 00l (h=0, k=0):      l = 4n
        validated
        """
        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return (h + l) % 2 == 0  # h0l
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return (h + k + l) % 2 == 0  # general

    @staticmethod
    def group72_Ibam(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 72: Ibam. Body-centered lattice (I-centering).
        Valid reflections must satisfy:
        - general hkl:         h + k + l even
        - 0kl (h=0):           k, l even
        - h0l (k=0):           h, l even
        - hk0 (l=0):           h + k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if (h + k + l) % 2 != 0:
            return False  # general

        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return k % 2 == 0 and l % 2 == 0  # 0kl

        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return h % 2 == 0 and l % 2 == 0  # h0l

        if l == 0:
            return (h + k) % 2 == 0  # hk0

        return True

    @staticmethod
    def group73_Ibca(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 73: Ibca. Body-centered lattice (I-centering).
        Valid reflections must satisfy:
        - general hkl:         h + k + l even
        - 0kl (h=0):           k, l even
        - h0l (k=0):           h, l even
        - hk0 (l=0):           h, k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        if (h + k + l) % 2 != 0:  # general
            return False

        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return k % 2 == 0 and l % 2 == 0  # 0kl

        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return h % 2 == 0 and l % 2 == 0  # h0l

        if l == 0:
            return h % 2 == 0 and k % 2 == 0  # hk0

        return True

    @staticmethod
    def group74_Imma(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 74: Imma. Body-centered lattice (I-centering).
        Valid reflections must satisfy:
        - general hkl:         h + k + l even
        - 0kl (h=0):           k + l even
        - h0l (k=0):           h + l even
        - hk0 (l=0):           h, k even
        - h00 (k=0, l=0):      h even
        - 0k0 (h=0, l=0):      k even
        - 00l (h=0, k=0):      l even
        validated
        """
        # general
        if (h + k + l) % 2 != 0:
            return False

        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return (k + l) % 2 == 0  # 0kl

        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return (h + l) % 2 == 0  # h0l

        if l == 0:
            return h % 2 == 0 and k % 2 == 0  # hk0

        return True

    @staticmethod
    def group75_P4(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 75: P4. Primitive tetragonal.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group76_P41(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 76: P41. Primitive tetragonal.
        Valid reflections must satisfy:
        - 00l (h=0, k=0):       l = 4n
        validated
        """
        if h == k == 0:
            return l % 4 == 0  # 00l
        return True

    @staticmethod
    def group77_P42(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 77: P42. Primitive tetragonal.
        Valid reflections must satisfy:
        - 00l (h=0, k=0):        l = 2n
        validated
        """
        if h == k == 0:
            return l % 2 == 0  # 00l
        return True

    @staticmethod
    def group78_P43(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 78: P43. Primitive tetragonal.
        Valid reflections must satisfy:
        - 00l:         l = 4n
        validated
        """
        if h == k == 0:
            return l % 4 == 0  # 00l
        return True

    @staticmethod
    def group79_I4(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 79: I4. Body-centered lattice (I-centering).
        Valid reflections must satisfy:
        - general hkl:         h + k + l even
        - hk0 (l=0):           h + k even
        - 0kl (h=0):           k + l even
        - hhl (h=k):           l even
        - 00l (h=0, k=0):      l even
        - h00 (k=0, l=0):      h even
        validated
        """
        if (h + k + l) % 2 != 0:
            return False
        if h == 0:
            return (k + l) % 2 == 0 if k else l % 2 == 0
        if k == 0 and l == 0:
            return h % 2 == 0
        if l == 0:
            return (h + k) % 2 == 0
        if h == k:
            return l % 2 == 0
        return True

    @staticmethod
    def group80_I41(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 80: I41. Body-centered tetragonal (I-centering).
        Valid reflections must satisfy:
        - general hkl:         h + k + l even
        - hk0 (l=0):           h + k even
        - 0kl (h=0):           k + l even
        - hhl (h=k):           l even
        - 00l (h=0, k=0):      l = 4n
        - h00 (k=0, l=0):      h even
        validated
        """
        if (h + k + l) % 2:
            return False  # General condition

        if h == 0:
            return l % 4 == 0 if k == 0 else (k + l) % 2 == 0  # 00l or 0kl

        if l == 0:
            return (h + k) % 2 == 0 if k else h % 2 == 0  # hk0 or h00

        return l % 2 == 0 if h == k else True  # hhl or general case

    @staticmethod
    def group81_P4bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 81: P4̅.
        No systematic absences.
        validated
        """
        return True

    @staticmethod
    def group82_I4bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 82: I4̅. Body-centered tetragonal (I-centering).
        Valid reflections must satisfy:
        - hkl:           h + k + l even
        - hk0:           h + k even
        - 0kl:           k + l even
        - hhl:           l even
        - 00l (h=0, k=0): l even
        - h00 (k=0, l=0): h even
        validated
        """
        if (h + k + l) % 2:
            return False

        if h == 0:
            return (k + l) % 2 == 0 if k else l % 2 == 0  # 0kl or 00l

        if l == 0:
            return (h + k) % 2 == 0 if k else h % 2 == 0  # hk0 or h00

        return l % 2 == 0 if h == k else True  # hhl or general

    @staticmethod
    def group83_P4m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 83: P4/m. Tetragonal.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group84_P42m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 84: P42/m. Tetragonal.
        Valid reflections must satisfy:
        - 00l (h=0, k=0):         l even
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group85_P4n(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 85: P4/n. Tetragonal.
        Valid reflections must satisfy:
        - hk0 (l=0):         h + k even
        - h00 (k=0, l=0):    h even
        validated
        """
        if l == 0:
            if k == 0:
                return h % 2 == 0  # h00
            return (h + k) % 2 == 0  # hk0
        return True

    @staticmethod
    def group86_P42n(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 86: P42/n. Tetragonal.
        Valid reflections must satisfy:
        - hk0  (l=0):         h + k even
        - 00l  (h=0, k=0):    l even
        - h00  (k=0, l=0):    h even
        validated
        """
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        return True

    @staticmethod
    def group87_I4m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 87: I4/m. Body-centered tetragonal (I-centering).
        Valid reflections must satisfy:
        - hkl:              h + k + l even
        - hk0  (l=0):       h + k even
        - 0kl  (h=0):       k + l even
        - hhl:              l even
        - 00l  (h=0, k=0):  l even
        - h00  (k=0, l=0):  h even
        validated
        """
        if (h + k + l) % 2:
            return False
        if h == 0:
            return (k + l) % 2 == 0 if k else l % 2 == 0  # 0kl or 00l
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return l % 2 == 0 if h == k else True  # hhl or general

    @staticmethod
    def group88_I41a(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 88: I41/a. Body-centered tetragonal (I-centering).
        Valid reflections must satisfy:
        - hkl:              h + k + l even
        - hk0  (l=0):       h, k even
        - 0kl  (h=0):       k + l even
        - hhl:              l even
        - 00l  (h=0, k=0):  l = 4n
        - h00  (k=0, l=0):  h even
        - hh0  (k=h, l=0):  h even
        validated
        """
        if (h + k + l) % 2 != 0:
            return False

        if h == 0 and k == 0:
            return l % 4 == 0
        if h == 0:
            return (k + l) % 2 == 0
        if l == 0 and k == 0:
            return h % 2 == 0
        if l == 0 and h == k:
            return h % 2 == 0
        if l == 0:
            return h % 2 == 0 and k % 2 == 0
        if h == k:
            return l % 2 == 0
        return True

    @staticmethod
    def group89_P422(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 89: P 4 2 2. Tetragonal.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group90_P4212(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 90: P 4 21 2. Tetragonal.
        Valid reflections must satisfy:
        - h00  (k=0, l=0):  h even
        - 0k0  (h=0, l=0):  k even (a & b are permutable in tetragonal)
        validated
        """
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group91_P4122(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 91: P 41 2 2. Tetragonal
        Valid reflections must satisfy:
        - 00l (h=0, k=0): l = 4n
        validated"""
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group92_P41_21_2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 92: P41 21 2. Tetragonal.
        Valid reflections must satisfy:
        - h00 (k=0, l=0):     h even
        - 0k0 (h=0, l=0):     k even
        - 00l (h=0, k=0):     l = 4n
        validated
        """
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if h == 0 and k == 0:
            return l % 4 == 0  # 00l
        return True

    @staticmethod
    def group93_P42_2_2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 93: P42 2 2. Tetragonal.
        Valid reflections must satisfy:
        - 00l (h=0, k=0):    l even
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        return True

    @staticmethod
    def group94_P42_21_2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 94: P42 21 2. Tetragonal.
        Valid reflections must satisfy:
        - 00l (h=0, k=0):   l even
        - h00 (k=0, l=0):   h even
        - 0k0  (h=0, l=0):  k even (a & b are permutable in tetragonal)
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        return True

    @staticmethod
    def group95_P43_2_2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 95: P43 2 2. Tetragonal.
        Valid reflections must satisfy:
        - 00l (h=0, k=0):    l = 4n
        validated
        """
        if h == 0 and k == 0:
            return l % 4 == 0  # 00l
        return True

    @staticmethod
    def group96_P_43_21_2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 96: P 43 21 2. Tetragonal.
        Valid reflections must satisfy:
        - 00l (h=0, k=0):    l = 4n
        - h00 (k=0, l=0):    h even
        - 0k0 (h=0, l=0):    k even (a & b are permutable in tetragonal)
        Used in lysozyme.
        validated
        """
        if h == 0 and k == 0:
            return l % 4 == 0  # 00l
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        return True

    @staticmethod
    def group97_I422(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 97: I422. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:             h + k + l even
        - hk0 (l=0):       h + k even
        - 0kl (h=0):       k + l even
        - hhl (h=k):       l even
        - 00l (h=0, k=0):  l even
        - h00 (k=0, l=0):  h even
        validated
        """
        if (h + k + l) % 2 != 0:
            return False  # I-centering
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if h == k:
            return l % 2 == 0  # hhl
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        return True

    @staticmethod
    def group98_I4122(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 98: I4122. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:             h + k + l even
        - hk0 (l=0):       h + k even
        - 0kl (h=0):       k + l even
        - hhl (h=k):       l even
        - 00l (h=0, k=0):  l = 4n
        - h00 (k=0, l=0):  h even
        validated
        """
        if (h + k + l) % 2 != 0:
            return False  # I-centering
        if h == 0 and k == 0:
            return l % 4 == 0  # 00l
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if h == k:
            return l % 2 == 0  # hhl
        return True

    @staticmethod
    def group99_P4mm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 99: P4mm. Tetragonal. Primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        validated
        """
        return True

    @staticmethod
    def group100_P4bm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 100: P4bm. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):       k even
        - h0l (k=0):       h even      [implied by symmetry]
        - h00 (k=0, l=0):  h even
        - 0k0 (h=0, l=0):  k even      [implied by symmetry]
        See ITC Vol. A, Section 2.1.3.13 (v) on reflection conditions for full compliance.
        See also http://img.chem.ucl.ac.uk/sgp/large/100az2.htm
        validated
        """
        if h == 0:
            return k % 2 == 0  # 0kl , 0k0
        if k == 0:
            return h % 2 == 0  # h0l, h00
        return True

    @staticmethod
    def group101_P42cm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 101: P42cm. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):       l even
        - h0l (k=0):       l even
        - 00l (h=0, k=0):  l even
        Source for rules: http://img.chem.ucl.ac.uk/sgp/large/101az2.htm
        validated
        """
        if h == 0 or k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group102_P42nm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 102: P42nm. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):        k + l even
        - h0l (k=0):        h + l even
        - h00 (k=0, l=0):   h even
        - 0k0 (h=0, l=0):   k even
        - 00l (h=0, k=0):   l even
        Source for rules: http://img.chem.ucl.ac.uk/sgp/large/102az2.htm
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            return (h + l) % 2 == 0  # h0l
        return True

    @staticmethod
    def group103_P4cc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 103: P4cc. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):        l even
        - h0l (k=0):        l even
        - hhl (h=k):        l even
        - 00l (h=0, k=0):   l even
        Source for rules: http://img.chem.ucl.ac.uk/sgp/large/103az2.htm
        validated
        """
        if k == 0:
            return l % 2 == 0  # h0l, 00l
        if h == 0:
            return l % 2 == 0  # 0kl
        if h == k:
            return l % 2 == 0  # hhl
        return True

    @staticmethod
    def group104_P4nc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 104: P4nc. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):         k + l = 2n
        - h0l (k=0):         h + l = 2n
        - hhl (h=k):         l even
        - h00 (k=0, l=0):    h even
        - 0k0 (h=0, l=0):    k even
        - 00l (h=0, k=0):    l even
        Source for rules: http://img.chem.ucl.ac.uk/sgp/large/104az2.htm
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            return (h + l) % 2 == 0  # h0l
        if h == k:
            return l % 2 == 0  # hhl
        return True

    @staticmethod
    def group105_P42mc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 105: P4₂mc. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - hhl (h = k):       l even
        - 00l (h = 0, k = 0): l even
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == k:
            return l % 2 == 0  # hhl
        return True

    @staticmethod
    def group106_P42bc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 106: P4₂bc. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):       k even
        - h0l (k=0):       h even
        - hhl (h=k):       l even
        - 00l (h=0, k=0):  l even
        - h00 (h≠0, k=0, l=0): h even
        - 0k0 (h=0, k≠0, l=0): k even
        Source for rules: http://img.chem.ucl.ac.uk/sgp/large/106az2.htm
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return k % 2 == 0  # 0kl
        if k == 0:
            return h % 2 == 0  # h0l
        if h == k:
            return l % 2 == 0  # hhl
        return True

    @staticmethod
    def group107_I4mm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 107: I4mm. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - General kl:     h + k + l = 2n
        - hk0 (l=0):      h + k even
        - 0kl (h=0):      k + l even
        - hhl (h=k):      l even
        - 00l (h=0,k=0):  l even
        - h00 (k=0,l=0):  h even
        validated
        """
        if (h + k + l) % 2 != 0:  # I-centering
            return False
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if h == k:
            return l % 2 == 0  # hhl
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        return True

    @staticmethod
    def group108_I4cm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 108: I4cm. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - General hkl:    h + k + l even (I-centering)
        - hk0 (l=0):      h + k even
        - 0kl (h=0):      k, l even
        - hhl (h=k):      l even
        - 00l (h=0,k=0):  l even
        - h00 (k=0,l=0):  h even
        - h0l (k=0):      h, l even
        - 0k0 (h=0,l=0):  k even
        Source for rules: http://img.chem.ucl.ac.uk/sgp/large/108az2.htm
        validated
        """
        if (h + k + l) % 2 != 0:
            return False  # I-centering

        if h == 0:
            if k == 0:
                return l % 2 == 0  # 00l
            if l == 0:
                return k % 2 == 0  # 0k0
            return k % 2 == 0 and l % 2 == 0  # 0kl

        if k == 0:
            if l == 0:
                return h % 2 == 0  # h00
            return h % 2 == 0 and l % 2 == 0  # h0l

        if l == 0:
            return (h + k) % 2 == 0  # hk0

        if h == k:
            return l % 2 == 0  # hhl

        return True

    @staticmethod
    def group109_I41md(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 109: I4₁md. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:            h + k + l even (I-centering)
        - hk0 (l=0):      h + k even
        - 0kl (h=0):      k + l even
        - hhl (h=k):      2h + l= 4n
        - 00l (h=0,k=0):  l= 4n
        - h00 (k=0,l=0):  h even
        - hh0 (h=k,l=0):  h even
        validated
        """
        if (h + k + l) % 2 != 0:  # I-centering (hkl)
            return False
        if h == 0 and k == 0:  # 00l
            return l % 4 == 0
        if l == 0:  # l=0 cases
            if h == k:  # hh0
                return h % 2 == 0
            return (h + k) % 2 == 0  # hk0
        if h == 0:  # 0kl
            return (k + l) % 2 == 0
        if h == k:  # hhl
            return (2 * h + l) % 4 == 0
        return True

    @staticmethod
    def group110_I41cd(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 110: I4₁cd. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:              h + k + l = 2n
        - hk0 (l=0):        h + k even
        - 0kl (h=0):        k, l even
        - hhl (h=k):        2h + l = 4n
        - 00l (h=k=0):      l = 4n
        - h00 (k=l=0):      h even
        - hh̅0 (k=-h, l=0):  h even
        - h0l (k=0):        h, l even
        - 0k0 (h=0, l=0):   k even
        - hh0 (h=k, l=0):   h even
        Source for rules: Combination of ITC and http://img.chem.ucl.ac.uk/sgp/large/110az2.htm
        validated
        """
        if (h + k + l) % 2:  # I-centering
            return False

        if h == 0 and k == 0:
            return l % 4 == 0  # 00l

        if l == 0:
            if h == k:
                return h % 2 == 0  # hh0
            if k == -h:
                return h % 2 == 0  # hh̅0
            if k == 0:
                return h % 2 == 0  # h00
            if h == 0:
                return k % 2 == 0  # 0k0
            return (h + k) % 2 == 0  # hk0

        if h == 0:
            return k % 2 == 0 and l % 2 == 0  # 0kl
        if k == 0:
            return h % 2 == 0 and l % 2 == 0  # h0l
        if h == k:
            return (2 * h + l) % 4 == 0  # hhl

        return True

    @staticmethod
    def group111_P4bar_2m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 111: P4̅2m. Tetragonal. Primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences
        validated
        """
        return True

    @staticmethod
    def group112_P4bar_2c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 112: P4̅2c. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - hhl (h = k):       l even
        - 00l (h = 0, k = 0): l even
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == k:
            return l % 2 == 0  # hhl
        return True

    @staticmethod
    def group113_P4bar_21m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 113: P4̅2₁m. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - h00 (k = 0, l = 0): h even
        - 0k0 (h = 0, l = 0): k even
        Source for rules: ITC and http://img.chem.ucl.ac.uk/sgp/large/113az2.htm
        validated
        """
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        return True

    @staticmethod
    def group114_P4bar_21c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 114: P4̅2₁c. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - hhl (h = k):          l even
        - 00l (h = 0, k = 0):   l even
        - h00 (k = 0, l = 0):   h even
        - 0k0 (h = 0, l = 0):    k even
        Source for rules: ITC and http://img.chem.ucl.ac.uk/sgp/large/114az2.htm
        validated
        """
        if (h, k) == (0, 0) or h == k:  # 00l or hhl
            return l % 2 == 0
        if l == 0:
            if k == 0:  # h00
                return h % 2 == 0
            if h == 0:  # 0k0
                return k % 2 == 0
        return True

    @staticmethod
    def group115_P4bar_m2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 115: P4̅m2. Tetragonal. Primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        validated
        """
        return True

    @staticmethod
    def group116_P4bar_c2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 116: P4̅c2. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h = 0):        l even
        - 00l (h = 0, k = 0): l even
        - h0l (k = 0):        l even
        Source for rules: ITC and http://img.chem.ucl.ac.uk/sgp/large/116az2.htm
        validated
        """
        if h == 0:  # 0kl, 00l
            return l % 2 == 0
        if k == 0:  # h0l
            return l % 2 == 0
        return True

    @staticmethod
    def group117_P4bar_b2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 117: P4̅b2. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):        k even
        - h00 (k=0, l=0):   h even
        - h0l (k=0):        h even
        - 0k0 (h=0, l=0):   k even
        Source for rules: ITC and http://img.chem.ucl.ac.uk/sgp/large/117az2.htm
        validated
        """
        if h == 0:
            return k % 2 == 0
        if k == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group118_P4bar_n2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 118: P4̅n2. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:

        - 0kl (h = 0):        k + l even
        - h0l (k = 0):        h + l even
        - h00 (k = 0, l = 0): h even
        - 0k0 (h = 0, l = 0): k even
        - 00l (h = 0, k = 0): l even

        Source: http://img.chem.ucl.ac.uk/sgp/large/118az2.htm
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if k == 0:
            return (h + l) % 2 == 0  # h0l
        return True

    @staticmethod
    def group119_I4bar_m2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 119: I4̅m2. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:          h + k + l = 2n
        - hk0 (l=0):    h + k even
        - 0kl (h=0):    k + l even
        - hhl (h=k):    l even
        - 00l (h=k=0):  l even
        - h00 (k=l=0):  h even
        Source: ITC
        validated
        """
        if (h + k + l) % 2 != 0:  # I-centering
            return False
        if h == k == 0:  # 00l
            return l % 2 == 0
        if l == 0:
            if k == 0:  # h00
                return h % 2 == 0
            return (h + k) % 2 == 0  # hk0
        if h == 0:  # 0kl
            return (k + l) % 2 == 0
        if h == k:  # hhl
            return l % 2 == 0
        return True

    @staticmethod
    def group120_I4bar_c2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 120: I4̅c2. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:              h + k + l even
        - hk0 (l=0):        h + k even
        - 0kl (h=0):        k even and l even
        - hhl (h=k):        l even
        - 00l (h=k=0):      l even
        - h00 (k=l=0):      h even
        - h0l (k=0):        h + l even
        - 0k0 (h=l=0):      k even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/120az2.htm
        validated
        """
        if (h + k + l) % 2 != 0:  # I-centering
            return False
        if h == k == 0:  # 00l
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == 0:  # 0kl
            return (k % 2 == 0) and (l % 2 == 0)
        if k == 0:  # h0l
            return (h % 2 == 0) and (l % 2 == 0)
        if h == k:  # hhl
            return l % 2 == 0
        return True

    @staticmethod
    def group121_I4bar_2m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 121: I4̅2m. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:              h + k + l even
        - hk0 (l=0):        h + k even
        - 0kl (h=0):        k + l even
        - hhl (h=k):        l even
        - 00l (h=k=0):      l even
        - h00 (k=l=0):      h even
        validated
        """
        if (h + k + l) % 2 != 0:
            return False
        if h == k == 0:
            return l % 2 == 0  # 00l
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        if h == 0:
            return (k + l) % 2 == 0  # 0kl
        if h == k:
            return l % 2 == 0  # hhl
        return True

    @staticmethod
    def group122_I4bar_2d(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 122: I4̅2d. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:              h + k + l even
        - hk0 (l=0):        h + k even
        - 0kl (h=0):        k + l even
        - hhl (h=k):        2h + l = 4n
        - 00l (h=k=0):      l = 4n
        - h00 (k=l=0):      h even
        - hh0 (h=k, l=0):   h even
        - h0l (k=0):        h + l even
        - 0k0 (h=0, l=0):   k even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/122az2.htm
        validated
        """
        if (h + k + l) % 2 != 0:
            return False  # I-centering
        if h == k == 0:  # 00l
            return l % 4 == 0
        if l == 0:
            if h == k:  # hh0
                return h % 2 == 0
            if k == 0:  # h00
                return h % 2 == 0
            return (h + k) % 2 == 0  # hk0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        if k == 0:  # h0l
            return (h + l) % 2 == 0
        if h == 0:  # 0kl
            return (k + l) % 2 == 0
        if h == k:  # hhl
            return (2 * h + l) % 4 == 0
        return True

    @staticmethod
    def group123_P4mmm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 123: P4/mmm. Tetragonal. Primitive lattice.
        Valid reflections must satisfy: — all (h, k, l) allowed
        No systematic absences.
        validated
        """
        return True

    @staticmethod
    def group124_P4mcc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 124: P4/mcc. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h=0):        l = 2n
        - hhl (h=k):        l = 2n
        - 00l (h=k=0):      l = 2n
        - h0l (k=0):        l = 2n
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/124az2.htm
        validated
        """
        if h == 0:  # 0kl
            return l % 2 == 0
        if h == k:  # hhl
            return l % 2 == 0
        if h == k == 0:  # 00l
            return l % 2 == 0
        if k == 0:  # h0l
            return l % 2 == 0
        return True

    @staticmethod
    def group125_P4nbm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 125: P4/nbm. Tetragonal. Primitive lattice..
        Valid reflections must satisfy:
        - hk0 (l=0):        h + k = 2n
        - 0kl (h=0):        k = 2n
        - h00 (k=l=0):      h = 2n
        - h0l (k=0):        h = 2n
        - 0k0 (h=l=0):      k = 2n
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/125az2.htm
        validated
        """
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == 0:  # 0kl
            return k % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if k == 0:  # h0l
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group126_P4nnc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 126: P4/nnc. Tetragonal. Primitive lattice.
        Valid reflections must satisfy:
        - hk0 (l=0):        h + k = 2n
        - 0kl (h=0):        k + l = 2n
        - hhl (h=k):        l = 2n
        - 00l (h=k=0):      l = 2n
        - h00 (k=l=0):      h = 2n
        - h0l (k=0):        h + l = 2n
        - 0k0 (h=l=0):      k = 2n
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/126az2.htm
        validated
        """
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == 0:  # 0kl
            return (k + l) % 2 == 0
        if h == k:  # hhl
            return l % 2 == 0
        if h == k == 0:  # 00l
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if k == 0:  # h0l
            return (h + l) % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group127_P4mbm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 127: P4/mbm. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - 0kl (h=0):        k = 2n
        - h00 (k=l=0):      h = 2n
        - h0l (k=0):        h = 2n
        - 0k0 (h=l=0):      k = 2n
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/127az2.htm
        validated
        """
        if h == 0:  # 0kl
            return k % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if k == 0:  # h0l
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group128_P4mnc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 128: P4/mnc. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - 0kl (h=0):        k + l = 2n
        - hhl (h=k):        l = 2n
        - 00l (h=k=0):      l = 2n
        - h00 (k=l=0):      h = 2n
        - h0l (k=0):        h + l = 2n
        - 0k0 (h=l=0):      k = 2n
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/128az2.htm
        validated
        """
        if h == 0:  # 0kl
            return (k + l) % 2 == 0
        if h == k:  # hhl
            return l % 2 == 0
        if h == k == 0:  # 00l
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if k == 0:  # h0l
            return (h + l) % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group129_P4nmm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 129: P4/nmm. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - hk0 (l=0):        h + k = 2n
        - h00 (k=l=0):      h = 2n
        - 0k0 (h=l=0):      k = 2n
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/129az2.htm
        validated
        """
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if l == 0:
            return (h + k) % 2 == 0  # hk0
        return True

    @staticmethod
    def group130_P4ncc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 130: P4/ncc. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - hk0 (l=0):        h + k = 2n
        - 0kl (h=0):        l = 2n
        - hhl (h=k):        l = 2n
        - 00l (h=k=0):      l = 2n
        - h00 (k=l=0):      h = 2n
        - h0l (k=0):        l = 2n
        - 0k0 (h=l=0):      k = 2n
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/130az2.htm
        validated
        """
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if k == 0:  # h0l
            return l % 2 == 0
        if h == k:  # hhl
            return l % 2 == 0
        if h == 0:  # 0kl
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group131_P42mmc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 131: P42/mmc. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - hhl (h=k):        l even
        - 00l (h=k=0):      l even
        validated
        """
        if h == k:  # hhl
            return l % 2 == 0
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        return True

    @staticmethod
    def group132_P42mcm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 132: P42/mcm. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - 0kl (h=0):         l = 2n
        - 00l (h=k=0):       l = 2n
        - h0l (k=0):         l = 2n
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/132az2.htm
        validated
        """
        if h == 0:  # 0kl
            return l % 2 == 0
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if k == 0:  # h0l
            return l % 2 == 0
        return True

    @staticmethod
    def group133_P42nbc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 133: P42/nbc. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - hk0 (l=0):         h + k even
        - 0kl (h=0):         k even
        - hhl (h=k):         l even
        - 00l (h=k=0):       l even
        - h00 (k=l=0):       h even
        - 0k0 (h=l=0):       k even
        - h0l (k=0):         h even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/133az2.htm
        validated
        """
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == k:  # hhl
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        if k == 0:  # h0l
            return h % 2 == 0
        if h == 0:  # 0kl
            return k % 2 == 0
        return True

    @staticmethod
    def group134_P42nnm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 134: P42/nnm. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - hk0 (l=0):         h + k even
        - 0kl (h=0):         k + l even
        - 00l (h=k=0):       l even
        - h00 (k=l=0):       h even
        - h0l (k=0):         h + l even
        - 0k0 (h=l=0):       k even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/134az2.htm
        validated
        """
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == 0:  # 0kl
            return (k + l) % 2 == 0
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if k == 0:  # h0l
            return (h + l) % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group135_P42mbc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 135: P42/mbc. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - 0kl (h=0):         k even
        - hhl (h=k):         l even
        - 00l (h=k=0):       l even
        - h00 (k=l=0):       h even
        - 0k0 (h=l=0):       k even
        - h0l (k=0):         h even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/135az2.htm
        validated
        """
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if h == k:  # hhl
            return l % 2 == 0
        if h == 0:  # 0kl
            return k % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        if k == 0:  # h0l
            return h % 2 == 0
        return True

    @staticmethod
    def group136_P42mnm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 136: P42/mnm. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - 0kl (h=0):         k + l even
        - 00l (h=k=0):       l even
        - h00 (k=l=0):       h even
        - h0l (k=0):         h + l even
        - 0k0 (h=l=0):       k even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/136az2.htm
        validated
        """
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if h == 0:  # 0kl
            return (k + l) % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if k == 0:  # h0l
            return (h + l) % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group137_P42nmc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 137: P42/nmc. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - hk0 (l=0):         h + k even
        - hhl (h=k):         l even
        - 00l (h=k=0):       l even
        - h00 (k=l=0):       h even
        - 0k0 (h=l=0):       k even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/137az2.htm
        validated
        """
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == k:  # hhl
            return l % 2 == 0
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group138_P42ncm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 138: P42/ncm. Tetragonal. Primitive lattice (P-centering).
        Valid reflections must satisfy:
        - hk0 (l=0):         h + k even
        - 0kl (h=0):         l even
        - 00l (h=k=0):       l even
        - h00 (k=l=0):       h even
        - 0k0 (h=l=0):       k even
        - h0l (k=0):         l even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/138az2.htm
        validated
        """
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == 0 and k != 0:  # 0kl (h=0)
            return l % 2 == 0
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        if k == 0:  # h0l
            return l % 2 == 0
        return True

    @staticmethod
    def group139_I4mmm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 139: I4/mmm. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:            h + k + l even
        - hk0 (l=0):      h + k even
        - 0kl (h=0):      k + l even
        - hhl (h=k):      l even
        - 00l (h=k=0):    l even
        - h00 (k=l=0):    h even
        - h0l (k=0):      h + l even
        - 0k0 (h=l=0):    k even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/139az2.htm
        validated
        """
        # General reflection condition: h + k + l even
        if (h + k + l) % 2 != 0:
            return False
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == 0 and k != 0:  # 0kl (h=0)
            return (k + l) % 2 == 0
        if h == k and h != 0:  # hhl
            return l % 2 == 0
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if k == 0:  # h0l
            return (h + l) % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group140_I4mcm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 140: I4/mcm. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl:            h + k + l even
        - hk0 (l=0):      h + k even
        - 0kl (h=0):      k and l even
        - hhl (h=k):      l even
        - 00l (h=k=0):    l even
        - h00 (k=l=0):    h even
        - h0l (k=0):      h and l even
        - 0k0 (h=l=0):    k even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/140az2.htm
        validated
        """
        # General condition: h + k + l even
        if (h + k + l) % 2 != 0:
            return False
        if l == 0:  # hk0
            return (h + k) % 2 == 0
        if h == 0:  # 0kl
            return k % 2 == 0 and l % 2 == 0
        if h == k and h != 0:  # hhl
            return l % 2 == 0
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if k == 0:  # h0l
            return h % 2 == 0 and l % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        return True

    @staticmethod
    def group141_I41amd(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 141: I41/amd. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl (general):        h + k + l even
        - hk0 (l=0):            h and k even
        - 0kl (h=0):            k + l even
        - hhl (h=k):            2h + l = 4n
        - 00l (h=k=0):          l = 4n
        - h00 (k=l=0):          h even
        - hh0 (h=k, l=0):       h even
        - 0k0 (h=l=0):          k even
        - h0l (k=0):            h + l even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/141az2.htm
        validated
        """
        if (h + k + l) % 2 != 0:  # hkl general condition
            return False
        if l == 0:  # hk0
            if h % 2 != 0 or k % 2 != 0:
                return False
        if h == 0:  # 0kl
            if (k + l) % 2 != 0:
                return False
        if h == k:  # hhl
            if (2 * h + l) % 4 != 0:
                return False
        if h == 0 and k == 0:  # 00l
            if l % 4 != 0:
                return False
        if k == 0 and l == 0:  # h00
            if h % 2 != 0:
                return False
        if h == k and l == 0:  # hh0
            if h % 2 != 0:
                return False
        if h == 0 and l == 0:  # 0k0
            if k % 2 != 0:
                return False
        if k == 0:  # h0l
            if (h + l) % 2 != 0:
                return False
        return True

    @staticmethod
    def group142_I41acd(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 142: I41/acd. Tetragonal. I-centering.
        Valid reflections must satisfy:
        - hkl (general):        h + k + l even
        - hk0 (l=0):            h and k even
        - 0kl (h=0):            k and l even
        - hhl (h=k):            2h + l =4n
        - 00l (h=k=0):          l = 4n
        - h00 (k=l=0):          h even
        - hh0 (h=k, l=0):       h even
        - 0k0 (h=l=0):          k even
        - h0l (k=0):            h and l even
        Source: ITC and http://img.chem.ucl.ac.uk/sgp/large/142az2.htm
        validated
        """
        if (h + k + l) % 2 != 0:
            return False
        if h == 0 and k == 0:
            return l % 4 == 0  # 00l
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == k and l == 0:
            return h % 2 == 0  # hh0
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if h == k:
            return (2 * h + l) % 4 == 0  # hhl
        if l == 0:
            return (h % 2 == 0) and (k % 2 == 0)  # hk0
        if h == 0:
            return (k % 2 == 0) and (l % 2 == 0)  # 0kl
        if k == 0:
            return (h % 2 == 0) and (l % 2 == 0)  # h0l
        return True

    @staticmethod
    def group143_P3(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 143: P3. Trigonal.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        validated
        """
        return True

    @staticmethod
    def group144_P31(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 144: P31. Trigonal.
        Valid reflections must satisfy:
        - 00l (h = k = 0): l = 3n
        Source: http://img.chem.ucl.ac.uk/sgp/large/144az2.htm
        validated
        """
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group145_P32(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 145: P32. Trigonal.
        Valid reflections must satisfy:
        - 00l (h = k = 0): l = 3n
        Source: http://img.chem.ucl.ac.uk/sgp/large/145az2.htm
        validated
        """
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group146_R3(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 146: R3. Trigonal, Rhombohedral (R).
        Valid reflections must satisfy:

        - hkil (general):        -h + k + l = 3n
        - hki0 (l = 0):          -h + k     = 3n
        - hh(-2h)l:              l          = 3n
        - h(-h)0l (i = 0):       k = -h     ⇒ h + l = 3n
        - 000l (h = k = i = 0):  l          = 3n
        - h(-h)00 (i = l = 0):   k = -h,
                                l = 0     ⇒ h = 3n
        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
        using the relation i = -(h + k).
        validated.
        """
        # General condition
        if (-h + k + l) % 3 != 0:
            return False

        # hki0: l = 0 → -h + k = 3n
        if l == 0:
            if (-h + k) % 3 != 0:
                return False

        # 000l: h = k = 0 → l = 3n
        if h == 0 and k == 0:
            if l % 3 != 0:
                return False

        # h-h0l: k = -h → h + l = 3n
        if k == -h:
            if (h + l) % 3 != 0:
                return False

            # h-h00: k = -h, l = 0 → h = 3n
            if l == 0 and h % 3 != 0:
                return False

        return True

    @staticmethod
    def group147_P3bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 147: P-3 (P3̅). Trigonal system.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        Source: ITC
        validated
        """
        return True

    @staticmethod
    def group148_R3bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 148: R-3 (R3̅). Trigonal, Rhombohedral (R).
        Valid reflections must satisfy:
        - hkil (general):         -h + k + l = 3n
        - hki0 (l = 0):           -h + k = 3n
        - hh(-2h)l:               l = 3n
        - h(-h)0l (i = 0):        h + l = 3n
        - 000l (h = k = i = 0):   l = 3n
        - h(-h)00 (i = l = 0):    h = 3n

        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated.
        """
        if (-h + k + l) % 3 != 0:  # hkil general
            return False
        if l == 0 and (-h + k) % 3 != 0:  # hki0
            return False
        if h == k and l % 3 != 0:  # hh(-2h)l
            return False
        if k == -h and l != 0 and (h + l) % 3 != 0:  # h(-h)0l
            return False
        if h == 0 and k == 0 and l % 3 != 0:  # 000l
            return False
        if k == -h and l == 0 and h % 3 != 0:  # h(-h)00
            return False
        return True

    @staticmethod
    def group149_P312(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 149: P3₁2. Trigonal.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        validated.
        """
        return True

    @staticmethod
    def group150_P321(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 150: P3₂1. Trigonal.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        """
        return True

    @staticmethod
    def group151_P3112(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 151: P3₁12. Trigonal.
        Valid reflections must satisfy:
        - 000l (h = k = 0):        l = 3n

        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated.
        """
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group152_P3121(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 152: P3₁21. Trigonal.
        Valid reflections must satisfy:
        - 000l (h = k = 0):        l = 3n

        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated.
        """
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group153_P3212(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 153: P3₂12. Trigonal.
        Valid reflections must satisfy:
        - 000l (h = k = 0):        l = 3n

        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated
        """
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group154_P3221(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 154: P3₂21. Trigonal.
        Valid reflections must satisfy:
        - 000l (h = k = 0):        l = 3n

        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated
        """
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group155_R32(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 155: R32. Trigonal, Rhombohedral (R).
        Valid reflections must satisfy:
        - hkil (general):        -h + k + l = 3n
        - hki0 (l = 0):          -h + k = 3n
        - hh(-2h)l:              l = 3n
        - h(-h)0l (i = 0):       k = -h ⇒ h + l = 3n
        - 000l (h = k = i = 0):  l = 3n
        - h(-h)00 (i = l = 0):   k = -h ⇒ h = 3n

        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated
        """
        if (-h + k + l) % 3 != 0:
            return False
        if l == 0:
            return (-h + k) % 3 == 0
        if h == k:
            return l % 3 == 0  # hh(–2h)l
        if k == -h and l != 0:
            return (h + l) % 3 == 0  # h(–h)0l
        if h == 0 and k == 0:
            return l % 3 == 0  # 000l
        if k == -h and l == 0:
            return h % 3 == 0  # h(–h)00
        return True

    @staticmethod
    def group156_P3m1(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 156: P3m1. Trigonal.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        validated
        """
        return True

    @staticmethod
    def group157_P31m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 157: P31m. Trigonal.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        validated
        """
        return True

    @staticmethod
    def group158_P3c1(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 158: P3c1. Trigonal.
        Valid reflections must satisfy:
        - 0kl (h = 0):        l = 2n
        - h0l (k = 0):        l = 2n
        - h(-h)0l (h = -k):   l = 2n
        - 00l (h = k = 0):    l = 2n

        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
            using the relation i = -(h + k), and
            http://img.chem.ucl.ac.uk/sgp/large/158az2.htm
        validated
        """
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0
        if h == -k:  # h(-h)0l
            return l % 2 == 0
        if h == 0 or k == 0:  # 0kl or h0l
            return l % 2 == 0
        return True

    @staticmethod
    def group159_P31c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 159: P31c. Trigonal.
        Valid reflections must satisfy:
        - hh(-2h)l:               l = 2n
        - 000l (h = k = 0):       l = 2n

        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
                using the relation i = -(h + k)
        validated
        """
        if h == k or (h == 0 and k == 0):  # hh(-2h)l and 000l
            return l % 2 == 0
        return True

    @staticmethod
    def group160_R3m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 160: R3m. Trigonal (Rhombohedral setting, hexagonal axes).
        Valid reflections must satisfy:
        - hkil:                -h + k + l = 3n
        - hki0 (l = 0):                  -h + k = 3n
        - hh(-2h)l:                      l = 3n
        - h(-h)0l (k = -h, l ≠ 0):       h + l = 3n
        - 000l (h = k = 0):              l = 3n
        - h(-h)00 (k = -h, l = 0):       h = 3n
        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
                using the relation i = -(h + k)
                JKC: http://img.chem.ucl.ac.uk/sgp/large/160bz2.htm
        validated
        """
        if (-h + k + l) % 3 != 0:
            return False
        if l == 0:
            return (-h + k) % 3 == 0  # hki0
        if h == k and k == -2 * h:
            return l % 3 == 0  # hh(–2h)l
        if k == -h:
            return (h + l) % 3 == 0  # h(–h)0l and h(–h)00
        if h == 0 and k == 0:
            return l % 3 == 0  # 000l
        return True

    @staticmethod
    def group161_R3c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 161: R3c. Trigonal (Rhombohedral centring, hexagonal axes).
        Valid reflections must satisfy:
        - General hkl:         -h + k + l = 3n
        - 0kl (h = 0):         l = 2n and k + l = 3n
        - h0l (k = 0):         l = 2n and h - l = 3n
        - hk0 (l = 0):         h - k = 3n
        - hhl (h = k):         l = 3n
        - h00 (k = 0, l = 0):  h = 3n
        - 0k0 (h = 0, l = 0):  k = 3n
        - 00l (h = 0, k = 0):  l = 6n

        Source:
            http://img.chem.ucl.ac.uk/sgp/large/161bz2.htm
        validated
        """
        # General condition
        if (-h + k + l) % 3 != 0:
            return False

        # Special cases
        if h == 0 and k == 0:  # 00l
            return l % 6 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 3 == 0
        if k == 0 and l == 0:  # h00
            return h % 3 == 0
        if l == 0:  # hk0
            return (h - k) % 3 == 0
        if h == 0:  # 0kl
            return (l % 2 == 0) and ((k + l) % 3 == 0)
        if k == 0:  # h0l
            return (l % 2 == 0) and ((h - l) % 3 == 0)
        if h == k:  # hhl
            return l % 3 == 0

        return True

    @staticmethod
    def group162_P3bar_m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 162: P3̅1m. Primitive lattice. Trigonal (hexagonal axes).
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        validated
        """
        return True

    @staticmethod
    def group163_P3_1c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 163: P3̅1c. Trigonal (hexagonal axes), primitive lattice.
        Valid reflections must satisfy:
        - hh(-2h)l:                      l = 2n
        - 000l (h = k = 0):              l = 2n

        Source:
            Reflection conditions from ITC (in hkil notation), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 000l
        if h == k:
            return l % 2 == 0  # hh(-2h)l
        return True

    @staticmethod
    def group164_P3bar_m1(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 164: P3̅m1. Primitive lattice. Trigonal (hexagonal axes).
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        validated
        """
        return True

    @staticmethod
    def group165_P3c1(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 165: P3c1. Trigonal (hexagonal axes), primitive lattice.
        Valid reflections must satisfy:
        - h(-h)0l (k = -h):             l = 2n
        - 000l (h = k = 0):             l = 2n
        - 0kl (h = 0):                  l = 2n
        - h0l (k = 0):                  l = 2n

        Source: Reflection conditions from ITC (given in hkil), adapted to (h, k, l)
            using the relation i = -(h + k), and http://img.chem.ucl.ac.uk/sgp/large/165az2.htm
        validated
        """
        if h == 0 and k == 0:
            return l % 2 == 0  # 000l
        if k == -h:
            return l % 2 == 0  # h(-h)0l
        if h == 0:
            return l % 2 == 0  # 0kl
        if k == 0:
            return l % 2 == 0  # h0l
        return True

    @staticmethod
    def group166_R3bar_m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 166: R3̅m. Trigonal (hexagonal axes), rhombohedral lattice.
        Valid reflections must satisfy:
        - hkil:                          -h + k + l = 3n
        - hki0 (l = 0):                  -h + k = 3n
        - hh(-2h)l:                      l = 3n
        - h(-h)0l (i = 0, k = -h):       h + l = 3n
        - 000l (h = k = 0):              l = 3n
        - h(-h)00 (l = 0, k = -h):       h = 3n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/166bz2.htm
        validated
        """
        if (-h + k + l) % 3 != 0:
            return False  # hkil
        if l == 0:
            return (-h + k) % 3 == 0  # hki0
        if h == k:
            return l % 3 == 0  # hh(-2h)l
        if k == -h:
            return (h + l) % 3 == 0  # h(-h)0l (i = 0)
        if h == 0 and k == 0:
            return l % 3 == 0  # 000l
        if k == -h and l == 0:
            return h % 3 == 0  # h(-h)00
        return True

    @staticmethod
    def group167_R3bar_c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 167: R3̅c. Trigonal (hexagonal axes), rhombohedral lattice.
        Used for Corundum.
        Valid reflections must satisfy:
        - hkil:                          -h + k + l = 3n
        - hki0 (l = 0):                  -h + k = 3n
        - hh(-2h)l:                      l = 3n
        - h(-h)0l (i = 0, k = -h):       h + l = 3n and l = 2n
        - 000l (h = k = 0):              l = 6n
        - h(-h)00 (l = 0, k = -h):       h = 3n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated
        """
        # (1) General condition (ITC: -h + k + l = 3n)
        # R-centring condition applies to all reflections.
        if (-h + k + l) % 3 != 0:
            return False

        # (5) Special case: h = k = 0 (ITC: 000l: l = 6n)
        if h == 0 and k == 0:
            return (l % 6) == 0

        # (6) Special case: k = -h, l = 0 (ITC: h(-h)00: h = 3n)
        if k == -h and l == 0:
            return (h % 3) == 0

        # (2) Special case: l = 0 plane (ITC: hki0, l = 0: -h + k = 3n)
        if l == 0:
            return (-h + k) % 3 == 0

        # (3) Special case: h = k (ITC: hh(-2h)l: l = 3n)
        # For l-direction with h = k, l must be multiple of 3
        if h == k:
            return (l % 3) == 0

        # (4) Special case: k = -h (ITC: h(-h)0l: l = 2n and h + l = 3n)
        # i = 0 corresponds to hexagonal h,k,l triple
        if k == -h:
            return l % 2 == 0 and (h + l) % 3 == 0

        # Derived explicit conditions (JKC-style)
        # These are *deductions* from ITC above:
        # Additional explicit forms for 0kl, h0l, and 0k0 follow from the h(-h)0l and h(-h)00
        # conditions by cyclic permutation of indices in the R3̅c hexagonal setting.

        # (7) 0kl plane (h = 0)
        # Derived from the general reflection condition for h(−h)0l (ITC rule (4)).
        # In the hexagonal R-lattice, a 120° rotation about the c-axis cycles
        # the in-plane indices: (h,k,i,l) → (k,i,h,l).
        # Applying this to h(−h)0l (i = 0, k = −h) gives 0kl as a symmetry-equivalent set.
        # The c-glide still requires l to be even (l = 2n),
        # and the centring condition becomes k + l = 3n.
        if h == 0:
            return l % 2 == 0 and (k + l) % 3 == 0

        # (8) h0l plane (k = 0)
        # By 120° rotation of indices (h,k,i,l) → (i,h,k,l), the h(−h)0l condition
        # maps to h0l as a symmetry-equivalent set when k = 0.
        # The c-glide condition still enforces l = 2n (l even),
        # and the R-centring condition becomes h − l = 3n.
        if k == 0:  # h0l plane
            return l % 2 == 0 and (h - l) % 3 == 0

        # (9) 0k0 line (h = 0, l = 0)
        # This follows from h(−h)00 (k = −h, l = 0), which under rotation
        # yields the 0k0 family when h = 0.
        # With l = 0, the only remaining restriction is from R-centring: k = 3n.
        if h == 0 and l == 0:
            return (k % 3) == 0

        return True


    @staticmethod
    def group168_P6(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 168: P6. Hexagonal system. Primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        Source: ITC
        validated
        """
        return True

    @staticmethod
    def group169_P61(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 169: P6₁. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):      l = 6n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 6 == 0
        return True

    @staticmethod
    def group170_P65(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 170: P6₅. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 6n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 6 == 0
        return True

    @staticmethod
    def group171_P62(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 171: P6₂. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 3n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group172_P64(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 172: P6₄. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 3n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group173_P63(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 173: P6₃. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 2n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group174_P6bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 174: P6̅. Hexagonal system, primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        Source: ITC
        validated
        """
        return True

    @staticmethod
    def group175_P6_m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 175: P6/m. Hexagonal system, primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        Source:ITC
        validated
        """
        return True

    @staticmethod
    def group176_P63_m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 176: P6₃/m. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 2n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group177_P622(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 177: P622. Hexagonal system, primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        Source: ITC
        validated
        """
        return True

    @staticmethod
    def group178_P6122(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 178: P6₁22. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 6n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 6 == 0
        return True

    @staticmethod
    def group179_P6522(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 179: P6₅22. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 6n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 6 == 0
        return True

    @staticmethod
    def group180_P6222(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 180: P6₂22. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 3n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group181_P6422(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 181: P6₄22. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 3n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 3 == 0
        return True

    @staticmethod
    def group182_P6322(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 182: P6₃22. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 2n

        Source: ITC
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group183_P6mm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 183: P6mm. Hexagonal system, primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.

        Source: ITC
        validated
        """
        return True

    @staticmethod
    def group184_P6cc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 184: P6cc. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 2n
        - 0kl (h = 0):                      l = 2n
        - h0l (k = 0):                      l = 2n
        - hh(-2h)l (k = h):                 l = 2n
        - h(-h)0l (k = -h):                 l = 2n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/184az2.htm
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 2 == 0

        # 0kl condition
        if h == 0:
            return l % 2 == 0

        # h0l condition
        if k == 0:
            return l % 2 == 0

        # hh(-2h)l condition
        if k == h:
            return l % 2 == 0

        # h(-h)0l condition
        if k == -h:
            return l % 2 == 0

        return True

    @staticmethod
    def group185_P63cm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 185: P6₃cm. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 000l (h = 0, k = 0):              l = 2n
        - h0l (k = 0):                      l = 2n
        - 0kl (h = 0):                      l = 2n
        - h(-h)0l (k = -h):                 l = 2n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/185az2.htm
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 2 == 0

        # h0l condition
        if k == 0:
            return l % 2 == 0

        # 0kl condition
        if h == 0:
            return l % 2 == 0

        # h(-h)0l condition
        if k == -h:
            return l % 2 == 0

        return True

    @staticmethod
    def group186_P63mc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 186: P6₃mc. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - hh(-2h)l (k = h):                 l = 2n
        - 000l (h = 0, k = 0):              l = 2n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 2 == 0

        # hh(-2h)l condition
        if k == h:
            return l % 2 == 0
        return True

    @staticmethod
    def group187_P6bar_m2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 187: P6̅m2. Hexagonal system, primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.

        Source: ITC
        validated
        """
        return True

    @staticmethod
    def group188_P6c2bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 188: P6c2 (P6̅c2). Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - 0kl (h = 0):                    l = 2n
        - h0l (k = 0):                    l = 2n
        - h(-h)0l (k = -h):               l = 2n
        - 000l (h = 0, k = 0):            l = 2n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/188bz2.htm
        validated
        """
        # 0kl condition
        if h == 0:
            return l % 2 == 0

        # h0l condition
        if k == 0:
            return l % 2 == 0

        # h(-h)0l condition
        if k == -h:
            return l % 2 == 0

        # 000l condition
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group189_P6bar_m2(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 189: P6̅2m. Hexagonal system, primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        Source: ITC
        validated
        """
        return True

    @staticmethod
    def group190_P6bar_2c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 190: P6̅2c. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - hh(-2h)l (k = h):                 l = 2n
        - 000l (h = 0, k = 0):              l = 2n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
        validated
        """
        # 000l condition
        if h == 0 and k == 0:
            return l % 2 == 0

        # hh(-2h)l condition
        if k == h:
            return l % 2 == 0

        return True

    @staticmethod
    def group191_P6_mmm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 191: P6/mmm. Hexagonal system, primitive lattice.
        No reflection conditions — all (h, k, l) are allowed.
        No systematic absences.
        Source: ITC
            validated
        """
        return True

    @staticmethod
    def group192_P6_mcc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 192: P6/mcc. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:
        - hh(-2h)l (k = h):     l = 2n
        - h(-h)0l (k = -h):     l = 2n
        - 000l (h = 0, k = 0):  l = 2n
        - 0kl (h = 0):          l = 2n
        - h0l (k = 0):          l = 2n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/192az2.htm
            validated
        """
        # hh(-2h)l condition
        if k == h:
            return l % 2 == 0

        # h(-h)0l condition
        if k == -h:
            return l % 2 == 0

        # 0kl, h0l, and 000l planes
        if h == 0 or k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group193_P63_mcm(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 193: P63/mcm. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:

        - h(-h)0l (k = -h):  l = 2n
        - 000l (h = 0, k = 0): l = 2n
        - 0kl (h = 0):             l = 2n
        - h0l (k = 0):             l = 2n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using i = -(h + k).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/193az2.htm
            validated
        """
        # h(-h)0l condition
        if k == -h:
            return l % 2 == 0

        # 0kl or h0l planes
        if h == 0 or k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group194_P63_mmc(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 194: P63/mmc. Hexagonal system, primitive lattice.
        Valid reflections must satisfy:

        - hh(-2h)l (k = h):     l = 2n
        - 000l (h = 0, k = 0):  l = 2n

        Source:
            Reflection conditions from ITC (in hkil), adapted to (h, k, l)
            using the relation i = -(h + k).
            validated
        """
        # hh(-2h)l condition
        if k == h:
            return l % 2 == 0

        # 000l condition (h = 0, k = 0)
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group195_P23(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 195: P23. Primitive cubic.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group196_F23(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 196: F23. Face-centred cubic.
        Conditions are cyclically permutable.
        Valid reflections must satisfy
        - General hkl:            h + k, h + l, k + l all even
        - 0kl (h=0):              k, l even
        - hhl (h=k):              h + l even
        - h00 (k=0, l=0):         h even

        validated
        """
        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return (k % 2 == 0) and (l % 2 == 0)

        # hhl
        if h == k:
            return (h + l) % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group197_I23(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 197: I23. Body-centred cubic.
        Conditions are cyclically permutable.
        Valid reflections must satisfy
        - General hkl:            h + k + l even
        - 0kl (h=0):              k + l even
        - hhl (h=k):              l even
        - h00 (k=0, l=0):         h even

        validated
        """
        if (h + k + l) % 2:  # general condition
            return False
        if h == 0 and (k + l) % 2:  # 0kl
            return False
        if h == k and l % 2:  # hhl
            return False
        if k == l == 0 and h % 2:  # h00
            return False
        return True

    @staticmethod
    def group198_P213(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 198: P2₁3. Primitive cubic.
        Conditions are cyclically permutable.
        Valid reflections must satisfy
        - h00 (k=0, l=0):  h = 2n
        - 0k0 (h=0, l=0):  k = 2n
        - 00l (h=0, k=0):  l = 2n

        Source: http://img.chem.ucl.ac.uk/sgp/large/198az2.htm
        validated
        """
        if k == l == 0:
            return h % 2 == 0
        if h == l == 0:
            return k % 2 == 0
        if h == k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group199_I213(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 199: I2₁3. Body-centred cubic.
        Conditions are cyclically permutable.

        Valid reflections must satisfy
        - General hkl:    h + k + l = 2n
        - 0kl (h=0):      k + l = 2n
        - hhl (h=k):      l = 2n
        - h00 (k=0,l=0):  h = 2n

        validated
        """
        if (h + k + l) % 2 != 0:  # general condition
            return False
        if h == 0:
            return (k + l) % 2 == 0
        if h == k:
            return l % 2 == 0
        if k == l == 0:
            return h % 2 == 0
        return True

    @staticmethod
    def group200_Pm3bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 200: Pm3̅. Primitive cubic.
        Conditions are cyclically permutable.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group201_Pn3bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 201: Pn3̅. Cubic system, primitive lattice.
        Reflection conditions are cyclically permutable.

        Valid reflections must satisfy:
        - 0kl (h = 0):                    k + l = 2n
        - h00 (k = 0, l = 0):             h = 2n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/201az2.htm
        validated
        """
        # 0kl condition and cyclic permutations
        if h == 0:
            return (k + l) % 2 == 0
        if k == 0:
            return (h + l) % 2 == 0
        if l == 0:
            return (h + k) % 2 == 0

        # h00 condition and cyclic permutations
        if k == 0 and l == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group202_Fm3bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 202: Fm3̅. Cubic system, face-centred lattice.
        Reflection conditions are cyclically permutable.

        Valid reflections must satisfy:
        - General hkl:                      h + k, h + l, k + l = 2n
        - 0kl (h = 0):                      k, l = 2n
        - hhl (h = k):                      h + l = 2n
        - h00 (k = 0, l = 0):               h = 2n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/202az2.htm
            validated
        """
        # General condition
        if not ((h + k) % 2 == 0 and (h + l) % 2 == 0 and (k + l) % 2 == 0):
            return False

        # 0kl
        if h == 0:
            return (k % 2 == 0) and (l % 2 == 0)

        # hhl
        if h == k:
            return (h + l) % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group203_Fd3bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 203: Fd3̅. Cubic system, face-centred lattice.
        Reflection conditions are cyclically permutable.

        Valid reflections must satisfy:
        - General hkl:                   h + k = 2n and h+l, k+l=2n
        - 0kl (h = 0):                   k + l = 4n and k,l=2n
        - hhl:                           h + l = 2n
        - h00 (k = 0, l = 0):            h = 4n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/203az2.htm
        validated
        """
        # General hkl
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl condition, cyclically permutable
        if h == 0:
            return (k + l) % 4 == 0 and k % 2 == 0 and l % 2 == 0
        if k == 0:
            return (h + l) % 4 == 0 and h % 2 == 0 and l % 2 == 0
        if l == 0:
            return (h + k) % 4 == 0 and h % 2 == 0 and k % 2 == 0

        # hhl
        if h == k:
            return (h + l) % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 4 == 0

        return True

    @staticmethod
    def group204_Im3bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 204: Im3̅. Cubic system, body-centred lattice.
        Reflection conditions are cyclically permutable.

        Valid reflections must satisfy:
        - General hkl:            h + k + l even
        - 0kl (h = 0):            k + l even
        - hhl (h = k):            l even
        - h00 (k = 0, l = 0):     h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/204az2.htm
        validated
        """
        # General condition
        if (h + k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return (k + l) % 2 == 0

        # hhl
        if h == k:
            return l % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group205_Pa3bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 205: Pa3̅. Cubic system, primitive lattice.
        Reflection conditions are cyclically permutable.

        Valid reflections must satisfy:
        - 0kl (h = 0):             k even
        - h00 (k = 0, l = 0):      h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/205az2.htm
        validated
        """
        # h00 cyclic permutations
        if k == 0 and l == 0:  # h00
            return h % 2 == 0
        if h == 0 and l == 0:  # 0k0
            return k % 2 == 0
        if h == 0 and k == 0:  # 00l
            return l % 2 == 0

        # 0kl cyclic permutations
        if h == 0:
            return k % 2 == 0  # 0kl
        if k == 0:
            return l % 2 == 0  # h0l
        if l == 0:
            return h % 2 == 0  # hk0

        return True

    @staticmethod
    def group206_Ia3bar(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 206: Ia3̅. Cubic system, body-centred lattice.
        Reflection conditions are cyclically permutable.

        Valid reflections must satisfy:
        - General hkl:            h + k + l = 2n
        - 0kl (h = 0):            k, l = 2n
        - hhl (h = k):            l even
        - h00 (k = 0, l = 0):     h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/206az2.htm
        validated
        """
        # General hkl
        if (h + k + l) % 2 != 0:
            return False

        # 0kl cyclic permutations
        if h == 0:
            return k % 2 == 0 and l % 2 == 0  # 0kl
        if k == 0:
            return h % 2 == 0 and l % 2 == 0  # h0l
        if l == 0:
            return h % 2 == 0 and k % 2 == 0  # hk0

        # hhl
        if h == k:
            return l % 2 == 0

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 2 == 0  # h00
        if h == 0 and l == 0:
            return k % 2 == 0  # 0k0
        if h == 0 and k == 0:
            return l % 2 == 0  # 00l

        return True

    @staticmethod
    def group207_P432(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 207: P432. Primitive cubic.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group208_P4232(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 208: P4₂32. Primitive cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - h00 (k = 0, l = 0): h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/208az2.htm
        validated
        """
        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if h == 0 and k == 0:
            return l % 2 == 0
        return True

    @staticmethod
    def group209_F432(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 209: F432. Face-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k, h + l, k + l all even
        - 0kl (h = 0):           k, l even
        - hhl (h = k):           h + l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/209az2.htm
        validated
        """
        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return k % 2 == 0 and l % 2 == 0

        # hhl
        if h == k:
            return (h + l) % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group210_F4132(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 210: F4₁32. Face-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k = 2n and h + l, k + l = 2n
        - 0kl (h = 0):           k, l even
        - hhl (h = k):           h + l even
        - h00 (k = 0, l = 0):    h = 4n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/210az2.htm
        validated
        """
        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 4 == 0
        if h == 0 and l == 0:
            return k % 4 == 0
        if h == 0 and k == 0:
            return l % 4 == 0

        # 0kl cyclic permutations
        if h == 0:
            return k % 2 == 0 and l % 2 == 0
        if k == 0:
            return h % 2 == 0 and l % 2 == 0
        if l == 0:
            return h % 2 == 0 and k % 2 == 0

        # hhl
        if h == k:
            return (h + l) % 2 == 0

        return True

    @staticmethod
    def group211_I432(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 211: I432. Body-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k + l = 2n
        - 0kl (h = 0):           k + l even
        - hhl (h = k):           l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/211az2.htm
        validated
        """
        # General condition
        if (h + k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return (k + l) % 2 == 0

        # hhl
        if h == k:
            return l % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group212_P4_332(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 212: P4₃32. Primitive cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - h00 (k = 0, l = 0):    h = 4n
        - 0k0 (h = 0, l = 0):    k = 4n
        - 00l (h = 0, k = 0):    l = 4n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # h00
        if k == 0 and l == 0:
            return h % 4 == 0
        # 0k0
        if h == 0 and l == 0:
            return k % 4 == 0
        # 00l
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group213_P4_132(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 213: P4₁32. Primitive cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - h00 (k = 0, l = 0):    h = 4n
        - 0k0 (h = 0, l = 0):    k = 4n
        - 00l (h = 0, k = 0):    l = 4n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # h00
        if k == 0 and l == 0:
            return h % 4 == 0
        # 0k0
        if h == 0 and l == 0:
            return k % 4 == 0
        # 00l
        if h == 0 and k == 0:
            return l % 4 == 0
        return True

    @staticmethod
    def group214_I4_132(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 214: I4₁32. Body-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k + l = 2n
        - 0kl (h = 0):           k + l even
        - hhl (h = k):           l even
        - h00 (k = 0, l = 0):    h = 4n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC: http://img.chem.ucl.ac.uk/sgp/large/214az2.htm
        validated
        """
        # General condition
        if (h + k + l) % 2 != 0:
            return False

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 4 == 0
        if h == 0 and l == 0:
            return k % 4 == 0
        if h == 0 and k == 0:
            return l % 4 == 0

        # hhl
        if h == k:
            return l % 2 == 0

        # 0kl cyclic permutations
        if h == 0:
            return (k + l) % 2 == 0
        if k == 0:
            return (h + l) % 2 == 0
        if l == 0:
            return (h + k) % 2 == 0

        return True

    @staticmethod
    def group215_P4bar_3m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 215: P4̅3m. Primitive cubic.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group216_F4bar_3m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 216: F4̅3m. Face-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k, h + l, k + l even
        - 0kl (h = 0):           k, l even
        - hhl (h = k):           h + l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """

        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return k % 2 == 0 and l % 2 == 0

        # hhl
        if h == k:
            return (h + l) % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group217_I4bar_3m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 217: I4̅3m. Body-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k + l even
        - 0kl (h = 0):           k + l even
        - hhl (h = k):           l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # General condition
        if (h + k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return (k + l) % 2 == 0

        # hhl
        if h == k:
            return l % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group218_P4_3n(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 218: P4̅3n. Primitive cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - hhl (h = k):           l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # hhl cyclic permutations
        if h == k:
            return l % 2 == 0
        if h == l:
            return k % 2 == 0
        if k == l:
            return h % 2 == 0

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group219_F4bar_3c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 219: F4̅3c. Face-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k = 2n and h + l, k + l = 2n
        - 0kl (h = 0):           k, l even
        - hhl (h = k):           h, l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
            JKC:  http://img.chem.ucl.ac.uk/sgp/large/219az2.htm
        validated
        """
        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl cyclic permutations
        if h == 0:
            return k % 2 == 0 and l % 2 == 0
        if k == 0:
            return h % 2 == 0 and l % 2 == 0
        if l == 0:
            return h % 2 == 0 and k % 2 == 0

        # hhl cyclic permutations
        if h == k:
            return h % 2 == 0 and l % 2 == 0
        if h == l:
            return h % 2 == 0 and k % 2 == 0
        if k == l:
            return h % 2 == 0 and k % 2 == 0

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group220_I4bar_3d(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 220: I4̅3d. Body-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k + l = 2n
        - 0kl (h = 0):           k + l even
        - hhl (h = k):           2h + l = 4n
        - h00 (k = 0, l = 0):    h = 4n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # General condition
        if (h + k + l) % 2 != 0:
            return False

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 4 == 0
        if h == 0 and l == 0:
            return k % 4 == 0
        if h == 0 and k == 0:
            return l % 4 == 0

        # hhl cyclic permutations
        if h == k:
            return (2 * h + l) % 4 == 0
        if h == l:
            return (2 * h + k) % 4 == 0
        if k == l:
            return (h + 2 * k) % 4 == 0

        # 0kl cyclic permutationss
        if h == 0:
            return (k + l) % 2 == 0
        if k == 0:
            return (h + l) % 2 == 0
        if l == 0:
            return (h + k) % 2 == 0

        return True

    @staticmethod
    def group221_Pm3bar_m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 221: Pm3̅m. Primitive cubic.
        All reflections are allowed; no systematic absences.
        validated
        """
        return True

    @staticmethod
    def group222_Pn3bar_n(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 222: Pn3̅n. Primitive cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - 0kl (h = 0):           k + l even
        - hhl (h = k):           l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # 0kl cyclic permutations
        if h == 0:
            return (k + l) % 2 == 0
        if k == 0:
            return (h + l) % 2 == 0
        if l == 0:
            return (h + k) % 2 == 0

        # hhl cyclic permutations
        if h == k:
            return l % 2 == 0
        if h == l:
            return k % 2 == 0
        if k == l:
            return h % 2 == 0

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group223_Pm3_n(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 223: Pm3̅n. Primitive cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - hhl (h = k):           l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated (without cyclic permutations)
        """
        # hhl cyclic permutations
        if h == k:
            return l % 2 == 0
        if h == l:
            return k % 2 == 0
        if k == l:
            return h % 2 == 0

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group224_Pn3bar_m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 224: Pn3̅m. Primitive cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - 0kl (h = 0):           k + l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # 0kl cyclic permutations
        if h == 0:
            return (k + l) % 2 == 0
        if k == 0:
            return (h + l) % 2 == 0
        if l == 0:
            return (h + k) % 2 == 0

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 2 == 0
        if h == 0 and l == 0:
            return k % 2 == 0
        if h == 0 and k == 0:
            return l % 2 == 0

        return True

    @staticmethod
    def group225_Fm3bar_m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 225: Fm3̅m. Face-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k, h + l, k + l even
        - 0kl (h = 0):           k, l even
        - hhl (h = k):           h + l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return k % 2 == 0 and l % 2 == 0

        # hhl
        if h == k:
            return (h + l) % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group226_Fm3bar_c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 226: Fm3̅c. Face-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k = 2n and h + l, k + l = 2n
        - 0kl (h = 0):           k, l even
        - hhl (h = k):           h, l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return k % 2 == 0 and l % 2 == 0

        # hhl cyclic permutations
        if h == k:
            return h % 2 == 0 and l % 2 == 0
        if h == l:
            return h % 2 == 0 and k % 2 == 0
        if k == l:
            return k % 2 == 0 and h % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group227_Fd3bar_m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 227: Fd3̅m. Face-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k = 2n and h + l, k + l = 2n
        - 0kl (h = 0):           k + l = 4n and k, l even
        - hhl (h = k):           h + l even
        - h00 (k = 0, l = 0):    h = 4n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl cyclic permutations
        if h == 0:
            return (k + l) % 4 == 0 and k % 2 == 0 and l % 2 == 0
        if k == 0:
            return (h + l) % 4 == 0 and h % 2 == 0 and l % 2 == 0
        if l == 0:
            return (h + k) % 4 == 0 and h % 2 == 0 and k % 2 == 0

        # hhl cyclic permutations
        if h == k:
            return (h + l) % 2 == 0
        if h == l:
            return (h + k) % 2 == 0
        if k == l:
            return (k + h) % 2 == 0

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 4 == 0
        if h == 0 and l == 0:
            return k % 4 == 0
        if h == 0 and k == 0:
            return l % 4 == 0

        return True

    @staticmethod
    def group228_Fd3bar_c(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 228: Fd3̅c. Face-centred cubic.
        Reflection conditions are permutable.

        Valid reflections must satisfy:
        - General hkl:           h + k = 2n and h + l, k + l = 2n
        - 0kl (h = 0):           k + l = 4n and k, l even
        - hhl (h = k):           h, l even
        - h00 (k = 0, l = 0):    h = 4n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # General condition
        if (h + k) % 2 != 0 or (h + l) % 2 != 0 or (k + l) % 2 != 0:
            return False

        # 0kl cyclic permutations
        if h == 0:
            return (k + l) % 4 == 0 and k % 2 == 0 and l % 2 == 0
        if k == 0:
            return (h + l) % 4 == 0 and h % 2 == 0 and l % 2 == 0
        if l == 0:
            return (h + k) % 4 == 0 and h % 2 == 0 and k % 2 == 0

        # hhl cyclic permutations
        if h == k:
            return h % 2 == 0 and l % 2 == 0
        if h == l:
            return h % 2 == 0 and k % 2 == 0
        if k == l:
            return k % 2 == 0 and h % 2 == 0

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 4 == 0
        if h == 0 and l == 0:
            return k % 4 == 0
        if h == 0 and k == 0:
            return l % 4 == 0

        return True

    @staticmethod
    def group229_Im3bar_m(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 229: Im3̅m. Body-centred cubic.
        Reflection conditions, without permutations.

        Valid reflections must satisfy:
        - General hkl:           h + k + l = 2n
        - 0kl (h = 0):           k + l even
        - hhl (h = k):           l even
        - h00 (k = 0, l = 0):    h even

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # General condition
        if (h + k + l) % 2 != 0:
            return False

        # 0kl
        if h == 0:
            return (k + l) % 2 == 0

        # hhl
        if h == k:
            return l % 2 == 0

        # h00
        if k == 0 and l == 0:
            return h % 2 == 0

        return True

    @staticmethod
    def group230_Ia3bar_d(h: int, k: int, l: int) -> bool:  # noqa: E741
        """
        Space group 230: Ia3̅d. Body-centred cubic.
        Reflection conditions, without permutations.

        Valid reflections must satisfy:
        - General hkl:           h + k + l = 2n
        - 0kl (h = 0):           k, l even
        - hhl (h = k):           2h + l = 4n
        - h00 (k = 0, l = 0):    h = 4n

        Source:
            Reflection conditions from ITC, adapted to (h, k, l).
        validated
        """
        # General condition
        if (h + k + l) % 2 != 0:
            return False

        # h00 cyclic permutations
        if k == 0 and l == 0:
            return h % 4 == 0
        if h == 0 and l == 0:
            return k % 4 == 0
        if h == 0 and k == 0:
            return l % 4 == 0

        # hhl cyclic permutations
        if h == k:
            return (2 * h + l) % 4 == 0
        if h == l:
            return (2 * h + k) % 4 == 0
        if k == l:
            return (2 * k + h) % 4 == 0

        # 0kl cyclic permutations
        if h == 0:
            return k % 2 == 0 and l % 2 == 0
        if k == 0:
            return h % 2 == 0 and l % 2 == 0
        if l == 0:
            return h % 2 == 0 and k % 2 == 0

        return True
