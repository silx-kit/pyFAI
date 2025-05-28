# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module for writing HDF5 in the Nexus style"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/05/2025"
__status__ = "production"
__docformat__ = 'restructuredtext'

import os
import sys
import time
import logging
import json
import posixpath
from ..utils.decorators import deprecated
from ..containers import Integrate1dResult, ErrorModel
from .. import version
from .ponifile import PoniFile
from ..method_registry import IntegrationMethod
logger = logging.getLogger(__name__)
try:
    import h5py
except ImportError as error:
    h5py = None
    logger.error("h5py module missing")
else:
    try:
        h5py._errors.silence_errors()
    except AttributeError:  # old h5py
        pass


def get_isotime(forceTime=None):
    """
    :param forceTime: enforce a given time (current by default)
    :type forceTime: float
    :return: the current time as an ISO8601 string
    :rtype: string
    """
    if forceTime is None:
        forceTime = time.time()
    localtime = time.localtime(forceTime)
    gmtime = time.gmtime(forceTime)
    tz_h = localtime.tm_hour - gmtime.tm_hour
    tz_m = localtime.tm_min - gmtime.tm_min
    return "%s%+03i:%02i" % (time.strftime("%Y-%m-%dT%H:%M:%S", localtime), tz_h, tz_m)


def from_isotime(text, use_tz=False):
    """
    :param text: string representing the time is iso format
    """
    if len(text) == 1:
        # just in case someone sets as a list
        text = text[0]
    try:
        text = text.decode("ascii")
    except (UnicodeError, AttributeError):
        text = str(text)
    if len(text) < 19:
        logger.warning("Not a iso-time string: %s", text)
        return
    base = text[:19]
    if use_tz and len(text) == 25:
        sgn = 1 if text[:19] == "+" else -1
        tz = 60 * (60 * int(text[20:22]) + int(text[23:25])) * sgn
    else:
        tz = 0
    return time.mktime(time.strptime(base, "%Y-%m-%dT%H:%M:%S")) + tz


def is_hdf5(filename):
    """
    Check if a file is actually a HDF5 file

    :param filename: this file has better to exist
    """
    signature = [137, 72, 68, 70, 13, 10, 26, 10]
    if not os.path.exists(filename):
        raise IOError("No such file %s" % (filename))
    with open(filename, "rb") as f:
        raw = f.read(8)
    sig = [ord(i) for i in raw] if sys.version_info[0] < 3 else [int(i) for i in raw]
    return sig == signature


class Nexus(object):
    """
    Writer class to handle Nexus/HDF5 data

    Manages:

    - entry

        - pyFAI-subentry

            - detector

    TODO: make it thread-safe !!!
    """

    def __init__(self, filename, mode=None, creator=None, start_time=None):
        """
        Constructor

        :param filename: name of the hdf5 file containing the nexus
        :param mode: can be 'r', 'a', 'w', '+' ....
        :param creator: set as attr of the NXroot
        :param start_time: set as attr of the NXroot
        """
        self.filename = os.path.abspath(filename)
        self.mode = mode
        if not h5py:
            logger.error("h5py module missing: NeXus not supported")
            raise RuntimeError("H5py module is missing")

        pre_existing = os.path.exists(self.filename)
        if self.mode is None:
            if pre_existing:
                self.mode = "r"
            else:
                self.mode = "a"

        if self.mode == "r" and h5py.version.version_tuple >= (2, 9):
            self.file_handle = open(self.filename, mode=self.mode + "b")
            self.h5 = h5py.File(self.file_handle, mode=self.mode)
        else:
            self.file_handle = None
            self.h5 = h5py.File(self.filename, mode=self.mode)
        self.to_close = []

        if not pre_existing or "w" in mode:
            self.h5.attrs["NX_class"] = "NXroot"
            self.h5.attrs["file_time"] = get_isotime(start_time)
            self.h5.attrs["file_name"] = self.filename
            self.h5.attrs["HDF5_Version"] = h5py.version.hdf5_version
            self.h5.attrs["creator"] = creator or self.__class__.__name__

    def __del__(self):
        self.close()

    def close(self, end_time=None):
        """
        Close the file and update all entries.
        """
        try:
            if self.mode != "r":
                if self.h5:
                    end_time = get_isotime(end_time)
                    while self.to_close:
                        entry = self.to_close.pop()
                        entry["end_time"] = end_time
                    self.h5.attrs["file_update_time"] = get_isotime()
        except Exception as error:
            sys.stderr.write(f"{type(error)}: {error},\nwhile finalizing Nexus file\n")

        try:
            if self.h5:
                self.h5.close()
            if self.file_handle:
                self.file_handle.close()
        except Exception as error:
            sys.stderr.write(f"Error closing file: {error}\n")

    # Context manager for "with" statement compatibility
    def __enter__(self, *arg, **kwarg):
        return self

    def __exit__(self, *arg, **kwarg):
        self.close()

    def flush(self):
        if self.h5:
            self.h5.flush()

    def get_entry(self, name):
        """
        Retrieves an entry from its name

        :param name: name of the entry to retrieve
        :return: HDF5 group of NXclass == NXentry
        """
        for grp_name in self.h5:
            if grp_name == name:
                grp = self.h5[grp_name]
                if isinstance(grp, h5py.Group) and \
                   ("start_time" in grp) and  \
                   self.get_attr(grp, "NX_class") == "NXentry":
                        return grp

    def get_entries(self):
        """
        retrieves all entry sorted the latest first.

        :return: list of HDF5 groups
        """
        entries = [(grp, from_isotime(self.h5[grp + "/start_time"][()]))
                   for grp in self.h5
                   if isinstance(self.h5[grp], h5py.Group) and
                   ("start_time" in self.h5[grp]) and
                   self.get_attr(self.h5[grp], "NX_class") == "NXentry"]
        entries.sort(key=lambda a: a[1], reverse=True)  # sort entries in decreasing time
        return [self.h5[i[0]] for i in entries]

    def find_detector(self, all=False):
        """
        Tries to find a detector within a NeXus file, takes the first compatible detector

        :param all: return all detectors found as a list
        """
        result = []
        for entry in self.get_entries():
            for instrument in self.get_class(entry, "NXsubentry") + self.get_class(entry, "NXinstrument"):
                for detector in self.get_class(instrument, "NXdetector"):
                    if all:
                        result.append(detector)
                    else:
                        return detector
        return result

    def new_entry(self, entry="entry", program_name="pyFAI",
                  title=None, force_time=None, force_name=False):
        """
        Create a new entry

        :param entry: name of the entry
        :param program_name: value of the field as string
        :param title: description of experiment as str
        :param force_time: enforce the start_time (as string!)
        :param force_name: force the entry name as such, without numerical suffix.
        :return: the corresponding HDF5 group
        """
        if not force_name:
            nb_entries = len(self.get_entries())
            entry = "%s_%04i" % (entry, nb_entries)
        entry_grp = self.h5
        for i in entry.split("/"):
            if i:
                entry_grp = entry_grp.require_group(i)
        self.h5.attrs["default"] = entry_grp.name.strip("/")
        entry_grp.attrs["NX_class"] = "NXentry"
        if title is not None:
            entry_grp["title"] = str(title)
        entry_grp["program_name"] = str(program_name)
        if force_time:
            entry_grp["start_time"] = str(force_time)
        else:
            entry_grp["start_time"] = get_isotime()
        self.to_close.append(entry_grp)
        return entry_grp

    def new_instrument(self, entry="entry", instrument_name="id00",):
        """
        Create an instrument in an entry or create both the entry and the instrument if
        """
        if not isinstance(entry, h5py.Group):
            entry = self.new_entry(entry)
        return self.new_class(entry, instrument_name, "NXinstrument")
#        howto external link
        # myfile['ext link'] = h5py.ExternalLink("otherfile.hdf5", "/path/to/resource")

    def new_class(self, grp, name, class_type="NXcollection"):
        """
        create a new sub-group with  type class_type
        :param grp: parent group
        :param name: name of the sub-group
        :param class_type: NeXus class name
        :return: subgroup created
        """
        sub = grp.require_group(name)
        sub.attrs["NX_class"] = str(class_type)
        return sub

    def new_detector(self, name="detector", entry="entry", subentry="pyFAI"):
        """
        Create a new entry/pyFAI/Detector

        :param detector: name of the detector
        :param entry: name of the entry
        :param subentry: all pyFAI description of detectors should be in a pyFAI sub-entry
        """
        entry_grp = self.new_entry(entry)
        pyFAI_grp = self.new_class(entry_grp, subentry, "NXsubentry")
        pyFAI_grp["definition_local"] = str("pyFAI")
        pyFAI_grp["definition_local"].attrs["version"] = str(version)
        det_grp = self.new_class(pyFAI_grp, name, "NXdetector")
        return det_grp

    def get_class(self, grp, class_type="NXcollection"):
        """
        return all sub-groups of the given type within a group

        :param grp: HDF5 group
        :param class_type: name of the NeXus class
        """
        coll = [grp[name] for name in grp
                if isinstance(grp[name], h5py.Group) and
                self.get_attr(grp[name], "NX_class") == class_type]
        return coll

    @deprecated(reason="WRONG", since_version="0.20")
    def get_data(self, grp, class_type="NXdata"):
        """
        return all dataset of the the NeXus class NXdata
        WRONG, do not use...

        :param grp: HDF5 group
        :param class_type: name of the NeXus class
        """
        coll = [grp[name] for name in grp
                if isinstance(grp[name], h5py.Dataset) and
                self.get_attr(grp[name], "NX_class") == class_type]
        return coll

    def get_dataset(self, grp, attr=None, value=None):
        """return list of dataset of the group matching
        the given attribute having the given value

        :param grp: HDF5 group
        :param attr: name of an attribute
        :param value: requested value for the attribute
        :return: list of dataset
        """
        coll = [grp[name] for name in grp
                if isinstance(grp[name], h5py.Dataset) and
                self.get_attr(grp[name], attr) == value]
        return coll

    def get_default_NXdata(self):
        """Return the default plot configured in the nexus structure.

        :return: the group with the default plot or None if not found
        """
        entry_name = self.h5.attrs.get("default")
        if entry_name:
            entry_grp = self.h5.get(entry_name)
            nxdata_name = entry_grp.attrs.get("default")
            if nxdata_name:
                if nxdata_name.startswith("/"):
                    return self.h5.get(nxdata_name)
                else:
                    return entry_grp.get(nxdata_name)

    def deep_copy(self, name, obj, where="/", toplevel=None, excluded=None, overwrite=False):
        """
        perform a deep copy:
        create a "name" entry in self containing a copy of the object

        :param where: path to the toplevel object (i.e. root)
        :param  toplevel: firectly the top level Group
        :param excluded: list of keys to be excluded
        :param overwrite: replace content if already existing
        """
        if (excluded is not None) and (name in excluded):
            return
        if not toplevel:
            toplevel = self.h5[where]
        if isinstance(obj, h5py.Group):
            if name not in toplevel:
                grp = toplevel.require_group(name)
                for k, v in obj.attrs.items():
                        grp.attrs[k] = v
        elif isinstance(obj, h5py.Dataset):
            if name in toplevel:
                if overwrite:
                    del toplevel[name]
                    logger.warning("Overwriting %s in %s", toplevel[name].name, self.filename)
                else:
                    logger.warning("Not overwriting %s in %s", toplevel[name].name, self.filename)
                    return
            toplevel[name] = obj[()]
            for k, v in obj.attrs.items():
                toplevel[name].attrs[k] = v

    @classmethod
    def get_attr(cls, dset, name, default=None):
        """Return the attribute of the dataset

        Handles the ascii -> unicode issue in python3 #275

        :param dset: a HDF5 dataset (or a group)
        :param name: name of the attribute
        :param default: default value to be returned
        :return: attribute value decoded in python3 or default
        """
        dec = default
        if name in dset.attrs:
            raw = dset.attrs[name]
            if (sys.version_info[0] > 2) and ("decode" in dir(raw)):
                dec = raw.decode()
            else:
                dec = raw
        return dec


def load_nexus(filename):
    """Tried to read-back a file from a Nexus file written by pyFAI

    :param filename: the name of the nexus file
    :return: parsed result
    """
    autodecode = lambda ma: ma.decode() if isinstance(ma, bytes) else ma
    with Nexus(filename, mode="r") as nxs:
        entry = nxs.get_entries()[0]
        ad = autodecode(entry["definition"][()])
        process_grp = entry["pyFAI"]
        cfg_grp = process_grp["poni"]
        poni = PoniFile(json.loads(cfg_grp["data"][()]))
        cfg_grp = process_grp["integrate"]

        if ad == "NXmonopd":
            result = Integrate1dResult(entry["results/polar_angle"][()],
                                       entry["results/data"][()],
                                       entry["results/errors"][()] if "results/errors" in entry else None)
            result._set_unit(entry["results/polar_angle"].attrs["units"])
            detector_grp = nxs.h5[posixpath.split(entry["results/polar_angle"].attrs["target"])[0]]
            result._set_sum_signal(detector_grp["raw"][()])
            result._set_sum_variance(detector_grp["sum_variance"][()])
            norm_grp = entry["normalization"]
            result._set_sum_normalization(norm_grp["integral"][()])
            result._set_sum_normalization2(norm_grp["integral_sq"][()])
            result._set_count(norm_grp["pixels"][()])
        elif ad == "NXcanSAS":
            data_grp = entry[entry.attrs["default"]]
            result = Integrate1dResult(data_grp["Q"][()],
                                       data_grp["I"][()],
                                       data_grp["Idev"][()] if "Idev" in data_grp else None)
            unit = data_grp["Q"].attrs["units"]
            unit = unit if "_" in unit else f"q_{unit[2:]}^-1"
            result._set_unit(unit)
            result._set_sum_signal(data_grp["sum_signal"][()])
            if "sum_variance" in data_grp:
                result._set_sum_variance(data_grp["sum_variance"][()])
            if "sum_normalization" in data_grp:
                result._set_sum_normalization(data_grp["sum_normalization"][()])
            if "sum_normalization2" in data_grp:
                result._set_sum_normalization2(data_grp["sum_normalization2"][()])
            if "count" in data_grp:
                result._set_count(data_grp["count"][()])
        else:
            raise RuntimeError(f"Unsupported application definition: {ad}, please fill in a bug")

        result._set_compute_engine(autodecode(cfg_grp["compute_engine"][()]))
        result._set_has_solidangle_correction(cfg_grp["has_solidangle_correction"][()])
        if "integration_method" in cfg_grp:
            result._set_method(IntegrationMethod.select_method(**json.loads(cfg_grp["integration_method"][()]))[0])
        result._set_has_dark_correction(cfg_grp["has_dark_correction"][()])
        result._set_has_flat_correction(cfg_grp["has_flat_correction"][()])
        result._set_has_mask_applied(autodecode(cfg_grp["has_mask_applied"][()]))
        pf = autodecode(cfg_grp["polarization_factor"][()])
        result._set_polarization_factor(None if pf == "None" else pf)
        result._set_normalization_factor(cfg_grp["normalization_factor"][()])
        result._set_method_called(autodecode(cfg_grp["method_called"][()]))
        result._set_metadata(json.loads(cfg_grp["metadata"][()]))
        result._set_error_model(ErrorModel.parse(autodecode(cfg_grp["error_model"][()])))
        result._set_poni(poni)
        result._set_weighted_average(cfg_grp["weighted_average"][()])

        return result


def _save_pyFAI(nexus, entry_grp, result):
    """Write the NXprocess into the entry

    :param nexus: Opened Nexus file
    :param entry: HDF5 group containing the NXentry
    :param result: instance of Integrate1dResult
    :return: the NXprocess group
    """
    process = nexus.new_class(entry_grp, "pyFAI", "NXprocess")
    process["sequence_index"] = 1
    process["program"] = "pyFAI"
    process["version"] = str(version)
    process["date"] = get_isotime()
    cfg_grp = nexus.new_class(process, "poni", "NXnote")
    cfg_grp.create_dataset("data", data=json.dumps(result.poni.as_dict(), indent=2, separators=(",\r\n", ": ")))
    cfg_grp.create_dataset("format", data="text/json")
    col_grp = nexus.new_class(process, "integrate", "NXcollection")
    pf = float(result.polarization_factor) if result.polarization_factor is not None else "None"
    pol_ds = col_grp.create_dataset("polarization_factor", data=pf)
    pol_ds.attrs["doc"] = "Between -1 and +1, 0 for circular, None for no-correction"
    nf = float(result.normalization_factor) if result.normalization_factor is not None else "None"
    nf_ds = col_grp.create_dataset("normalization_factor", data=nf)
    nf_ds.attrs["doc"] = "User-provided normalization factor, usually to account for incident flux"
    col_grp["has_mask_applied"] = result.has_mask_applied
    col_grp["has_dark_correction"] = result.has_dark_correction
    col_grp["has_flat_correction"] = result.has_flat_correction
    col_grp["has_solidangle_correction"] = result.has_solidangle_correction
    col_grp["metadata"] = json.dumps(result.metadata)
    if result.percentile is not None:
        col_grp.create_dataset("percentile", data=result.percentile).attrs["doc"] = "used with median-filter like reduction"
    col_grp.create_dataset("method_called", data=result.method_called).attrs["doc"] = "name of the function called of AzimuthalIntegrator"
    col_grp.create_dataset("compute_engine", data=result.compute_engine).attrs["doc"] = "name of the compute engine selected by pyFAI"
    col_grp.create_dataset("error_model", data=result.error_model.as_str()).attrs["doc"] = "how is variance assessed from signal ?"
    col_grp.create_dataset("weighted_average", data=result.weighted_average).attrs["doc"] = "use the weighted average (as in -ng engines) or the unweighted one (as in -legacy) ?"

    if result.method:
        col_grp.create_dataset("integration_method", data=json.dumps(result.method.method._asdict() or {})).\
            attrs["doc"] = "dict with the type of splitting, the kind of algorithm and its implementation"
    return process


def save_NXmonpd(filename, result,
                 title="monopd",
                 entry="entry",
                 instrument="beamline",
                 source_name="ESRF",
                 source_type="synchotron",
                 source_probe="x-ray",
                 sample="sample",
                 extra=None):
    """Save integrated data into a HDF5-file following
    the Nexus powder diffraction application definition:
    https://manual.nexusformat.org/classes/applications/NXmonopd.html

    :param filename: name of the file to be written
    :param result: instance of Integrate1dResult
    :param title: title of the experiment
    :param entry: name of the entry
    :param instrument: name/brand of the instrument
    :param source_name: name/brand of the particule source
    :param source_type: kind of source as a string
    :param source_probe: Any of these values: 'neutron' | 'x-ray' | 'electron'
    :param sample: sample name
    :param extra: extra metadata as a dict
    """
    with Nexus(filename, mode="w") as nxs:
        entry_grp = nxs.new_entry(entry=entry, program_name="pyFAI",
                  title=title, force_time=None, force_name=True)
        entry_grp["definition"] = "NXmonopd"
        entry_grp["definition"].attrs["version"] = "3.1"

        process = _save_pyFAI(nxs, entry_grp, result)

        # sample
        sample_grp = nxs.new_class(entry_grp, sample, "NXsample")
        sample_grp["name"] = sample

        # instrument
        instrument_grp = nxs.new_instrument(entry_grp, instrument)
        # source
        source_grp = nxs.new_class(instrument_grp, source_name, "NXsource")
        source_grp["type"] = str(source_type)
        source_grp["name"] = str(source_name)
        source_grp["probe"] = str(source_probe)
        if result.poni and  result.poni.wavelength:
            crystal_grp = nxs.new_class(instrument_grp, "monochromator", "NXcrystal")
            crystal_grp["wavelength"] = float(result.poni.wavelength * 1e10)
            crystal_grp["wavelength"].attrs["units"] = "angstrom"
        # detector
        detector = result.poni.detector.__class__.__name__ if result.poni else "Detector"
        detector_grp = nxs.new_class(instrument_grp, detector, "NXdetector")
        detector_grp["name"] = detector
        if result.poni:
            detector_grp["config"] = json.dumps(result.poni.detector.get_config())
        polar_angle_ds = detector_grp.create_dataset("polar_angle", data=result.radial)
        polar_angle_ds.attrs["axis"] = "1"
        polar_angle_ds.attrs["units"] = str(result.unit)
        polar_angle_ds.attrs["long_name"] = result.unit.label
        polar_angle_ds.attrs["target"] = polar_angle_ds.name
        polar_angle_ds.attrs["interpretation"] = "spectrum"
        intensities_ds = detector_grp.create_dataset("data", data=result.intensity)
        intensities_ds.attrs["doc"] = "weighted average of all pixels in a bin"
        intensities_ds.attrs["signal"] = "1"
        intensities_ds.attrs["interpretation"] = "spectrum"
        intensities_ds.attrs["target"] = intensities_ds.name
        raw_ds = detector_grp.create_dataset("raw", data=result.sum_signal)
        raw_ds.attrs["doc"] = "Sum of signal of all pixels in a bin"
        raw_ds.attrs["interpretation"] = "spectrum"

        # normalization
        nrm_grp = nxs.new_class(entry_grp, "normalization", "NXmonitor")
        nrm_grp["mode"] = "monitor"
        # nrm_grp["preset"] = ?
        nrm_ds = nrm_grp.create_dataset("integral", data=result.sum_normalization)
        nrm_ds.attrs["doc"] = "sum of normalization of all pixels in a bin"
        nrm_ds.attrs["units"] = "relative to the PONI position"
        nrm_ds.attrs["interpretation"] = "spectrum"
        nrm2_ds = nrm_grp.create_dataset("integral_sq", data=result.sum_normalization2)
        nrm2_ds.attrs["doc"] = "sum of normalization squarred of all pixels in a bin"
        nrm2_ds.attrs["interpretation"] = "spectrum"
        cnt_ds = nrm_grp.create_dataset("pixels", data=result.count)
        cnt_ds.attrs["doc"] = "Number of pixels contributing to each bin"
        cnt_ds.attrs["units"] = "pixels"
        cnt_ds.attrs["interpretation"] = "spectrum"

        # Results available as links
        integration_data = nxs.new_class(entry_grp, "results", "NXdata")
        integration_data["polar_angle"] = polar_angle_ds
        integration_data["data"] = intensities_ds
        if result.sum_variance is not None:
            errors_ds = detector_grp.create_dataset("errors", data=result.sem)
            errors_ds.attrs["target"] = errors_ds.name
            errors_ds.attrs["doc"] = "standard error of the mean"
            errors_ds.attrs["interpretation"] = "spectrum"
            integration_data["errors"] = errors_ds
            vari_ds = detector_grp.create_dataset("sum_variance", data=result.sum_variance)
            vari_ds.attrs["doc"] = "Propagated variance, prior to normalization"
            vari_ds.attrs["interpretation"] = "spectrum"
        integration_data.attrs["signal"] = "data"
        integration_data.attrs["axes"] = ["polar_angle"]
        integration_data.attrs["polar_angle_indices"] = 0
        integration_data["title"] = f"Powder diffraction pattern of {sample}"
        entry_grp.attrs["default"] = posixpath.relpath(integration_data.name, entry_grp.name)

        if extra:
            extra_grp = nxs.new_class(entry_grp, "extra", "NXnote")
            extra_grp.create_dataset("data", data=json.dumps(extra, indent=2, separators=(",\r\n", ": ")))
            extra_grp.create_dataset("format", data="text/json")


def save_NXcansas(filename, result,
                  title="something descriptive yet short",
                  run="run-number",
                  entry="entry",
                  instrument="beamline",
                  source_name="ESRF",
                  source_type="synchotron",
                  source_probe="x-ray",
                  sample="sample",
                  extra=None):
    """Save integrated data into a HDF5-file following
    the Nexus canSAS application definition:
    https://manual.nexusformat.org/classes/applications/NXcanSAS.html

    :param filename: name of the file to be written
    :param result: instance of Integrate1dResult
    :param title: title of the experiment
    :param entry: name of the entry
    :param instrument: name/brand of the instrument
    :param source_name: name/brand of the particule source
    :param source_type: kind of source as a string
    :param source_probe: Any of these values: 'neutron' | 'x-ray' | 'electron'
    :param sample: sample name
    :param extra: extra metadata as a dict
    """
    with Nexus(filename, mode="w") as nxs:
        entry_grp = nxs.new_entry(entry=entry, program_name="pyFAI",
                  title=title, force_time=None, force_name=True)

        entry_grp.attrs["canSAS_class"] = "SASentry"
        entry_grp.attrs["version"] = "1.1"

        entry_grp["definition"] = "NXcanSAS"
        entry_grp["definition"].attrs["version"] = "1.1"
        entry_grp["run"] = run

        process = _save_pyFAI(nxs, entry_grp, result)
        process.attrs["canSAS_class"] = "SASprocess"
        process["name"] = "Azimuthal integration"

        # sample
        sample_grp = nxs.new_class(entry_grp, sample, "NXsample")
        sample_grp["name"] = sample

        # Instrument:
        instrument_grp = nxs.new_instrument(entry_grp, instrument)
        instrument_grp.attrs["canSAS_class"] = "SASinstrument"
        instrument_grp["name"] = str(instrument)
        # NXaperture ?
        # NXcollimator?
        # Detector
        detector = result.poni.detector.__class__.__name__ if result.poni else "Detector"
        detector_grp = nxs.new_class(instrument_grp, detector, "NXdetector")
        detector_grp.attrs["canSAS_class"] = "SASdetector"
        detector_grp["name"] = detector
        if result.poni:
            detector_grp["config"] = json.dumps(result.poni.detector.get_config())
            detector_grp.create_dataset("SDD", data=result.poni.dist).attrs["units"] = "m"
            detector_grp.create_dataset("x_position", data=-result.poni.poni2).attrs["units"] = "m"
            detector_grp.create_dataset("y_position", data=-result.poni.poni1).attrs["units"] = "m"
            detector_grp.create_dataset("roll", data=-result.poni.rot3).attrs["units"] = "rad"
            detector_grp.create_dataset("pitch", data=-result.poni.rot2).attrs["units"] = "rad"
            detector_grp.create_dataset("yaw", data=-result.poni.rot1).attrs["units"] = "rad"
            detector_grp.create_dataset("x_pixel_size", data=result.poni.detector.pixel2).attrs["units"] = "m"
            detector_grp.create_dataset("y_pixel_size", data=result.poni.detector.pixel1).attrs["units"] = "m"
            # this is approximate but should be enough for SAXS where tilts are small:
            detector_grp.create_dataset("beam_center_x", data=result.poni.poni2 / result.poni.detector.pixel2).attrs["units"] = "m"
            detector_grp.create_dataset("beam_center_y", data=result.poni.poni1 / result.poni.detector.pixel1).attrs["units"] = "m"

        # source
        source_grp = nxs.new_class(instrument_grp, source_name, "NXsource")
        source_grp.attrs["canSAS_class"] = "SASsource"
        source_grp["type"] = str(source_type)
        source_grp["name"] = str(source_name)
        source_grp["probe"] = str(source_probe)
        if result.poni and  result.poni.wavelength:
            source_grp.create_dataset("incident_wavelength", data=float(result.poni.wavelength * 1e10)).attrs["units"] = "angstrom"

        # data
        integration_data = nxs.new_class(entry_grp, "sasdata", "NXdata")
        integration_data.attrs["canSAS_class"] = "SASdata"
        integration_data.attrs["signal"] = "I"
        integration_data.attrs["axes"] = ["Q"]
        integration_data.attrs["I_axes"] = "Q"
        integration_data.attrs["timestamp"] = get_isotime()
        entry_grp.attrs["default"] = posixpath.relpath(integration_data.name, entry_grp.name)

        q_ds = integration_data.create_dataset("Q", data=result.radial)
        q_ds.attrs["axis"] = "1"
        q_ds.attrs["units"] = ("1/" + str(result.unit).split("_")[-1].split("^")[0]) if str(result.unit).startswith("q") else str(result.unit)
        q_ds.attrs["long_name"] = result.unit.label
        # q_ds.attrs["target"] = q_ds.name
        q_ds.attrs["interpretation"] = "spectrum"
        intensities_ds = integration_data.create_dataset("I", data=result.intensity)
        intensities_ds.attrs["doc"] = "weighted average of all pixels in a bin"
        intensities_ds.attrs["signal"] = "1"
        intensities_ds.attrs["interpretation"] = "spectrum"
        intensities_ds.attrs["units"] = "arbitrary"

        # intensities_ds.attrs["target"] = intensities_ds.name
        raw_ds = integration_data.create_dataset("sum_signal", data=result.sum_signal)
        raw_ds.attrs["doc"] = "Sum of signal of all pixels in a bin"
        raw_ds.attrs["interpretation"] = "spectrum"
        nrm_ds = integration_data.create_dataset("sum_normalization", data=result.sum_normalization)
        nrm_ds.attrs["doc"] = "sum of normalization of all pixels in a bin"
        nrm_ds.attrs["units"] = "relative to the PONI position"
        nrm_ds.attrs["interpretation"] = "spectrum"
        nrm2_ds = integration_data.create_dataset("sum_normalization2", data=result.sum_normalization2)
        nrm2_ds.attrs["doc"] = "sum of normalization squarred of all pixels in a bin"
        nrm2_ds.attrs["interpretation"] = "spectrum"
        cnt_ds = integration_data.create_dataset("count", data=result.count)
        cnt_ds.attrs["doc"] = "Number of pixels contributing to each bin"
        cnt_ds.attrs["units"] = "pixels"
        cnt_ds.attrs["interpretation"] = "spectrum"
        if result.sum_variance is not None:
            errors_ds = integration_data.create_dataset("Idev", data=result.sem)
            intensities_ds.attrs["uncertainties"] = "Idev"
            errors_ds.attrs["target"] = errors_ds.name
            errors_ds.attrs["doc"] = "standard error of the mean"
            errors_ds.attrs["interpretation"] = "spectrum"
            errors_ds.attrs["units"] = "arbitrary"
            integration_data["errors"] = errors_ds
            vari_ds = integration_data.create_dataset("sum_variance", data=result.sum_variance)
            vari_ds.attrs["doc"] = "Propagated variance, prior to normalization"
            vari_ds.attrs["interpretation"] = "spectrum"

        if extra:
            extra_grp = nxs.new_class(entry_grp, "extra", "NXnote")
            extra_grp.create_dataset("data", data=json.dumps(extra, indent=2, separators=(",\r\n", ": ")))
            extra_grp.create_dataset("format", data="text/json")
