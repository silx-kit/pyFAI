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
__date__ = "09/07/2025"
__status__ = "development"
__docformat__ = "restructuredtext"


import os
from dataclasses import field
from ..containers import Reflection, Miller, dataclass


@dataclass
class CalibrantConfig:
    name: str = ""
    description: str = ""
    filename: str = ""
    cell: str = ""
    space_group: str = ""
    reference: str = ""
    reflections: list = field(default_factory=list)

    def __str__(self):
        out = [
            f"# Calibrant: {self.description or self.name}" + (f" ({self.name})" if self.description else ""),
            f"# Cell: {self.cell}" + (f" ({self.space_group})" if self.space_group else ""),
            f"# Ref: {self.reference}",
            "",
            "# d_spacing  # (h k l)  mult intensity"]
        for ref in self.reflections:
            if ref.intensity is not None:
                out.append(f"{ref.dspacing:12.8f} # {str(ref.hkl):10s} {ref.multiplicity:2d} {ref.intensity}")
            elif ref.multiplicity:
                out.append(
                    f"{ref.dspacing:12.8f} # {str(ref.hkl):10s} {ref.multiplicity:2d}"
                )
            elif ref.hkl:
                out.append(f"{ref.dspacing:12.8f} # {str(ref.hkl):10s}")
            else:
                out.append(f"{ref.dspacing:12.8f}")
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
            reflections.sort(key=lambda r: r.dspacing, reverse=True)
            # read the other metadata ...
            name = raw[0]
            reference = raw[2]
            for line in raw:
                if line.startswith("CELL PARAMETERS:"):
                    cell = line.split(":")[1].strip()
                if line.startswith("SPACE GROUP:"):
                    space_group = line.split(":")[1].strip()

            return cls(
                name=name,
                filename=filename,
                cell=cell,
                space_group=space_group,
                reference=reference,
                reflections=reflections,
            )
        raise ValueError(f"Unable to parse `{filename}` as DIF-file.")

    @classmethod
    def from_dspacing(cls, filename: str):
        """Alternative constructor from d-spacing file, pyFAI historical calibrant files

        :param filename: name of the D-file
        :return CalibrationConfig instance
        """
        generic = False
        begining = True
        self = cls(filename=filename)
        raw = []
        with open(filename) as f:
            for line in f:
                raw.append(line.strip())

        has_weak_reflection = "weak" in " ".join(raw).lower()

        for line in raw:
            if begining and line.startswith("#"):
                line = line.strip("# \t")
                if "Calibrant:" in line:
                    name = line.split(":", 1)[1].strip()
                    if "(" in name:
                        idx = name.index("(")
                        self.description = name[:idx].strip()
                        # There could be several (): `Vanadinite (Pb5(BO4)3Cl)`
                        cnt = 0
                        lname = []
                        for c in name[idx:]:
                            lname.append(c)
                            if c == "(":
                                cnt += 1
                            elif c == ")":
                                cnt -= 1
                            if cnt == 0:
                                break
                        self.name = "".join(lname[1:-1]).strip()
                    else:
                        self.name = name.strip()
                    continue
                elif "Ref:" in line:
                    self.reference = line.split(":", 1)[1].strip()
                    continue
                elif "Cell:" in line:
                    cell = line.split(":", 1)[1].strip()
                    if ("(" in cell) and (")" in cell):
                        idx = cell.index("(")
                        self.space_group = cell[idx + 1 : cell.index(")")].strip()
                        self.cell = cell[:idx].strip()
                    else:
                        self.cell = cell
                    continue
                else:
                    if not self.cell:
                        self.cell = line
                continue
            begining = False
            words = line.split()
            if not words:
                continue
            if generic:
                for word in words:
                    if word.startswith("#"):
                        break
                    try:
                        value = float(word)
                    except ValueError:
                        break
                    else:
                        self.reflections.append(Reflection(dspacing=value))
                continue
            try:
                hash_pos = words.index("#")
            except ValueError:
                self.reflections += [Reflection(dspacing=float(i)) for i in words]
                generic = True
                continue
            if hash_pos == 1 and generic is False:
                if words[0].startswith("#"):
                    continue
                reflection = Reflection(dspacing=float(words[0]))
                if has_weak_reflection:
                    reflection.intensity = 1.0
                self.reflections.append(reflection)
                start_miller = end_miller = None
                for i, j in enumerate(words[2:], start=2):
                    if j.startswith("("):
                        start_miller = i
                        if j.endswith(")"):
                            end_miller = i
                            break
                        continue
                    if j.endswith(")"):
                        end_miller = i
                        break
                if start_miller and end_miller:
                    reflection.hkl = Miller.parse(" ".join(words[start_miller : end_miller + 1]))
                    if len(words) > end_miller + 1:
                        mult = words[end_miller + 1]
                        if mult.startswith("#"):
                            continue
                        elif mult.isdecimal():
                            reflection.multiplicity = int(mult)
                    if len(words) > end_miller + 2:
                        intensity = words[end_miller + 2]
                        if intensity.startswith("#"):
                            continue
                        try:
                            value = float(intensity)
                        except ValueError:
                            if "weak" in intensity.lower():
                                reflection.intensity = 0.0
                        else:
                            reflection.intensity = value
        return self

    def save(self, filename: str = None):
        """Save the calibrant structure into a D-spaacing file

        :param filename: name of the output file. If not provided, can re-use the previous one.
        """
        if filename is None:
            filename = self.filename

        self.filename = filename

        if not filename.lower().endswith(".d"):
            filename += ".D"
        with open(filename, "w", encoding="utf-8") as fd:
            fd.write(str(self))
