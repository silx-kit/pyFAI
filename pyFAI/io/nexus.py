# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2020 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "19/11/2020"
__status__ = "production"
__docformat__ = 'restructuredtext'

import os
import sys
import time
import logging
import numpy
from ..utils.decorators import deprecated
from .. import version
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

    def __init__(self, filename, mode=None, creator=None):
        """
        Constructor

        :param filename: name of the hdf5 file containing the nexus
        :param mode: can be 'r', 'a', 'w', '+' ....
        :param creator: set as attr of the NXroot
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
            self.h5.attrs["file_time"] = get_isotime()
            self.h5.attrs["file_name"] = self.filename
            self.h5.attrs["HDF5_Version"] = h5py.version.hdf5_version
            self.h5.attrs["creator"] = creator or self.__class__.__name__

    def close(self, end_time=None):
        """
        close the filename and update all entries
        """
        if self.mode != "r":
            end_time = get_isotime(end_time)
            for entry in self.to_close:
                entry["end_time"] = end_time
            self.h5.attrs["file_update_time"] = get_isotime()
        self.h5.close()
        if self.file_handle:
            self.file_handle.close()

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
        self.h5.attrs["default"] = entry
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
