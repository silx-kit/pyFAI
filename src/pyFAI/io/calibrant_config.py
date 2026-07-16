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
__date__ = "24/06/2026"
__status__ = "development"
__docformat__ = "restructuredtext"


import os
import re
import logging
from dataclasses import field
from ..containers import Reflection, Miller, dataclass

logger = logging.getLogger(__name__)


def _parse_cell(text: str) -> dict:
    """Extract the lattice name and the cell parameters from the textual
    representation of a cell, e.g. as produced by ``str(Cell)``:
    `Face centered cubic cell a=5.41165Å b=5.41165Å c=5.41165Å α=90° β=90° γ=90°`

    :param text: cell description
    :return: dict with keys among lattice, a, b, c, alpha, beta, gamma
    """
    result = {}
    lower = text.lower()
    for lattice in ("cubic", "tetragonal", "hexagonal", "rhombohedral",
                    "orthorhombic", "monoclinic", "triclinic"):
        if lattice in lower:
            result["lattice"] = lattice
            break
    for name, symbol in (("a", "a"), ("b", "b"), ("c", "c"),
                         ("alpha", "(?:\N{GREEK SMALL LETTER ALPHA}|alpha)"),
                         ("beta", "(?:\N{GREEK SMALL LETTER BETA}|beta)"),
                         ("gamma", "(?:\N{GREEK SMALL LETTER GAMMA}|gamma)")):
        match = re.search(rf"\b{symbol}=([-+0-9.eE]+)", text)
        if match:
            result[name] = float(match.group(1))
    return result


@dataclass
class CalibrantConfig:
    name: str = ""
    description: str = ""
    filename: str = ""
    cell: str = ""
    space_group: str = ""
    reference: str = ""
    reflections: list = field(default_factory=list)
    eos: object = None
    "Optional EquationOfState describing the variation of the cell with pressure and temperature"

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
        if not self.reflections:
            raise ValueError(f"No valid reflections found in calibrant file '{filename}'")
        return self

    @classmethod
    def from_JCPDS(cls, filename: str):
        """Alternative constructor from a JCPDS file (version 4), the format
        used by the high-pressure community (Dioptas, GSECARS, ...).

        The compression parameters (K0, K0P, DK0DT, DK0PDT) and the thermal
        expansion (ALPHAT, DALPHAT) are turned into an EquationOfState
        instance stored in the ``eos`` attribute: a PVT composite when both
        are present, the bare model otherwise.

        :param filename: name of the jcpds-file
        :return: CalibrantConfig instance
        """
        from ..crystallography.cell import Cell  # lazy loading to prevent cyclic imports
        from ..crystallography.eos import BirchMurnaghan, PVT, VolumeExpansion

        version = None
        symmetry = None
        comments = []
        values = {}
        reflections = []
        with open(filename) as fd:
            for line in fd:
                if ":" not in line:
                    continue
                tag, value = line.split(":", 1)
                tag = tag.strip().upper()
                value = value.strip()
                if tag == "VERSION":
                    version = value
                elif tag == "COMMENT":
                    comments.append(value)
                elif tag == "SYMMETRY":
                    symmetry = value.upper()
                elif tag == "DIHKL":
                    words = value.replace(",", " ").split()
                    if len(words) >= 5:
                        reflections.append(Reflection(dspacing=float(words[0]),
                                                      intensity=float(words[1]),
                                                      hkl=Miller(int(words[2]), int(words[3]), int(words[4]))))
                else:
                    try:
                        values[tag] = float(value)
                    except ValueError:
                        logger.warning("Unable to parse JCPDS line: %s", line.strip())
        if version is None or int(float(version)) != 4:
            raise ValueError(f"Only version-4 JCPDS files are supported, `{filename}` is version {version}")
        if symmetry is None or "A" not in values:
            raise ValueError(f"No symmetry or cell parameter found in JCPDS file `{filename}`")

        a = values["A"]
        if symmetry == "CUBIC":
            cell = Cell.cubic(a)
        elif symmetry == "TETRAGONAL":
            cell = Cell.tetragonal(a, values["C"])
        elif symmetry == "HEXAGONAL":
            cell = Cell.hexagonal(a, values["C"])
        elif symmetry in ("RHOMBOHEDRAL", "TRIGONAL"):
            cell = Cell.rhombohedral(a, values["ALPHA"])
        elif symmetry == "ORTHORHOMBIC":
            cell = Cell.orthorhombic(a, values["B"], values["C"])
        elif symmetry == "MONOCLINIC":
            cell = Cell.monoclinic(a, values["B"], values["C"], values["BETA"])
        else:
            cell = Cell(a, values.get("B", a), values.get("C", a),
                        values.get("ALPHA", 90.0), values.get("BETA", 90.0), values.get("GAMMA", 90.0))

        isothermal = thermal = eos = None
        if "K0" in values:
            isothermal = BirchMurnaghan(k0=values["K0"], k0p=values.get("K0P", 4.0))
        if values.get("ALPHAT"):
            coefficients = [values["ALPHAT"]]
            if values.get("DALPHAT"):
                coefficients.append(values["DALPHAT"])
            thermal = VolumeExpansion(coefficients)
        if isothermal and thermal:
            eos = PVT(isothermal, thermal,
                      dk0dt=values.get("DK0DT", 0.0),
                      dk0pdt=values.get("DK0PDT", 0.0),
                      v0=cell.volume)
        elif isothermal or thermal:
            eos = isothermal or thermal
            eos.v0 = cell.volume

        return cls(name=os.path.splitext(os.path.basename(filename))[0],
                   description=" ".join(comments),
                   filename=filename,
                   cell=str(cell),
                   reflections=reflections,
                   eos=eos)

    def to_JCPDS(self) -> str:
        """Serialize as a JCPDS file (version 4), the format used by the
        high-pressure community (Dioptas, GSECARS, ...).

        Requires a parseable ``cell`` description. The ``eos`` attribute, when
        present, must map onto the JCPDS parametrization: a Birch-Murnaghan
        compression and/or a polynomial volume expansion (possibly combined in
        a PVT composite); other models raise a ValueError.

        :return: content of the JCPDS file as a string
        """
        from ..crystallography.eos import (BirchMurnaghan, LatticeExpansion, PVT,
                                           ThermalExpansion, VolumeExpansion)

        lines = ["VERSION: 4"]
        if self.description or self.name:
            lines.append(f"COMMENT: {self.description or self.name}")

        isothermal = thermal = None
        dk0dt = dk0pdt = 0.0
        if isinstance(self.eos, PVT):
            isothermal, thermal = self.eos.isothermal, self.eos.thermal
            dk0dt, dk0pdt = self.eos.dk0dt, self.eos.dk0pdt
        elif isinstance(self.eos, (ThermalExpansion, LatticeExpansion, VolumeExpansion)):
            thermal = self.eos
        elif self.eos is not None:
            isothermal = self.eos
        if isothermal is not None:
            if not isinstance(isothermal, BirchMurnaghan):
                raise ValueError(f"JCPDS assumes a Birch-Murnaghan compression model, unable to export {isothermal!r}")
            lines.append(f"K0: {isothermal.k0:.10g}")
            lines.append(f"K0P: {isothermal.k0p:.10g}")
            if dk0dt:
                lines.append(f"DK0DT: {dk0dt:.10g}")
            if dk0pdt:
                lines.append(f"DK0PDT: {dk0pdt:.10g}")
        if thermal is not None:
            dalphat = 0.0
            if isinstance(thermal, VolumeExpansion):
                if len(thermal.coefficients) > 2:
                    raise ValueError("JCPDS supports at most 2 volume-expansion coefficients")
                alphat = thermal.coefficients[0]
                if len(thermal.coefficients) > 1:
                    dalphat = thermal.coefficients[1]
            elif isinstance(thermal, ThermalExpansion) and not thermal.alpha1 and not thermal.alpha2:
                logger.warning("Exponential thermal expansion approximated as linear in the JCPDS file")
                alphat = thermal.alpha0
            elif isinstance(thermal, LatticeExpansion) and len(thermal.coefficients) <= 2:
                logger.warning("Lattice expansion converted to a volume expansion, truncated at 2nd order, in the JCPDS file")
                coefficients = thermal.coefficients + [0.0]
                alphat = 3.0 * coefficients[0]
                dalphat = 3.0 * coefficients[1] + 3.0 * coefficients[0] ** 2
            else:
                raise ValueError(f"Unable to export thermal model {thermal!r} to JCPDS")
            lines.append(f"ALPHAT: {alphat:.10g}")
            if dalphat:
                lines.append(f"DALPHAT: {dalphat:.10g}")

        cell = _parse_cell(self.cell)
        if "lattice" not in cell or "a" not in cell:
            raise ValueError(f"Unable to parse the cell description `{self.cell}` for JCPDS export")
        lattice = cell["lattice"]
        lines.append(f"SYMMETRY: {lattice.upper()}")
        lines.append(f"A: {cell['a']:.10g}")
        if lattice in ("tetragonal", "hexagonal"):
            keys = ("c",)
        elif lattice == "rhombohedral":
            keys = ("alpha",)
        elif lattice == "orthorhombic":
            keys = ("b", "c")
        elif lattice == "monoclinic":
            keys = ("b", "c", "beta")
        elif lattice == "triclinic":
            keys = ("b", "c", "alpha", "beta", "gamma")
        else:  # cubic
            keys = ()
        for key in keys:
            lines.append(f"{key.upper()}: {cell[key]:.10g}")

        for reflection in self.reflections:
            hkl = reflection.hkl if reflection.hkl else (0, 0, 0)
            intensity = 100.0 if reflection.intensity is None else reflection.intensity
            lines.append(f"DIHKL: {reflection.dspacing:.8f} {intensity:g} {hkl[0]} {hkl[1]} {hkl[2]}")
        return os.linesep.join(lines)

    def save_JCPDS(self, filename: str):
        """Save the calibrant structure into a JCPDS (version 4) file.

        :param filename: name of the output file
        """
        with open(filename, "w", encoding="utf-8") as fd:
            fd.write(self.to_JCPDS())

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
