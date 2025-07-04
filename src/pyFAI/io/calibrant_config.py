# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module to read and sometimes write calibration files"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/07/2025"
__status__ = "development"
__docformat__ = "restructuredtext"


import os
from dataclasses import field
from ..containers import Reflection, Miller, dataclass
from typing import Optional

@dataclass
class CalibrantConfig:
    name: str = ""
    filename: str = ""
    cell: str = ""
    space_group: str = ""
    reference: str = ""
    reflections: list = field(default_factory=list)

    def __repr__(self):
        out = [f"# Calibrant: {self.name}",
               f"# Cell: {self.cell} ({self.space_group})",
               f"# Ref: {self.reference}",
               "",
               "# d_spacing  # (h k l) mult intensity"]
        for ref in self.reflections:
            out.append(f"{ref.d_spacing:12.8f} # {ref.hkl} {ref.multiplicity:2d} {ref.intensity}")
        return os.linesep.join(out)

    @classmethod
    def from_DIF(cls, filename: str):
        """Alternative constructor from dif-file, as provided by the American Mineralogist database

            https://rruff.geo.arizona.edu/AMS/amcsd.php
            https://www.rruff.net/amcsd/

        :param filename: name of the diff-file as string
        :return: CalibrantConfig instance
        """
        raw = []
        with open(filename) as fd:
            for line in fd:
                raw.append(line.strip())
        reflections = []
        started = False
        for line in raw:
            if line.startswith("2-THETA") and not started:
                started = True
                continue
            if started:
                if line.startswith("=" * 10):
                    break
                words = line.split()
                if len(words) >= 7:
                    reflections.append(
                        Reflection(
                            float(words[2]),
                            float(words[1]),
                            Miller(int(words[3]), int(words[4]), int(words[5])),
                            int(words[6]),
                        )
                    )
        if reflections:
            reflections.sort(key=lambda r: r.d_spacing, reverse=True)
            #read the other metadata ...
            name = raw[0]
            reference = raw[2]
            for line in raw:
                if line.startswith("CELL PARAMETERS:"):
                    cell = line.split(":")[1].strip()
                if line.startswith("SPACE GROUP:"):
                    space_group = line.split(":")[1].strip()
            return  cls(name,
                        filename,
                        cell,
                        space_group,
                        reference,
                        reflections)
        raise ValueError(f"Unable to parse `{filename}` as DIF-file.")

    @classmethod
    def from_dspacing(self, filename: str):
        """Alternative constructor from d-spacing file, pyFAI historical calibrant files

        :param filename: name of the D-file
        :return CalibrationConfig instance
        """
        intensities = []  # list of peak intensities, same length as dSpacing
        multiplicities = []  # list of multiplicities, same length as dSpacing
        hkls = []  # list of miller indices for each reflection, same length as dSpacing
        metadata = []  # headers
        self._dSpacing = []
        header = True
        generic = False
        with open(filename) as f:
            for line in f:
                stripped = line.strip()
                if header and stripped.startswith("#"):
                    metadata.append(stripped.strip("# \t"))
                    continue
                header = False
                words = stripped.split()
                if generic:
                    self._dSpacing += [float(i) for i in words]
                    continue
                try:
                    hash_pos = words.index("#")
                except ValueError:
                    self._dSpacing += [float(i) for i in words]
                    generic = True
                    continue

                if hash_pos == 1 and generic is False:
                    if words[0].startswith("#"):
                        continue
                    ds = float(words[0])
                    self._dSpacing.append(ds)
                    start_miller = end_miller = None
                    for i, j in enumerate(words[2:], start=2):
                        if j.startswith("("):
                            start_miller = i
                            continue
                        if j.endswith(")"):
                            end_miller = i
                            break
                    if start_miller and end_miller:
                        hkls.append(" ".join(words[start_miller:end_miller+1]))
                        if len(words)>end_miller:
                            multiplicities.append(int(words[end_miller+1]))
        print(self.metadata)
