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
"""case_insensitive_dataclass decorators"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/11/2025"
__status__ = "development"
__docformat__ = 'restructuredtext'


import sys
import logging
import typing
from dataclasses import dataclass as _dataclass


logger = logging.getLogger(__name__)

# User defined dataclasses
if sys.version_info >= (3, 10):
    dataclass = _dataclass(slots=True)
else:
    dataclass = _dataclass


class CaseInsensitiveMeta(type):
    """ Avoid using metaclasses !"""
    def __new__(mcls, name, bases, namespace, **kw):
        cls = super().__new__(mcls, name, bases, namespace, **kw)

        # Only process classes that are dataclasses (they have __dataclass_fields__)
        if hasattr(cls, '__dataclass_fields__'):
            ci_map = {}
            for f, v in cls.__dataclass_fields__.items():
                if v.type == typing.ClassVar:
                    continue
                lowered = f.lower()
                if lowered in ci_map:
                    raise ValueError(
                        f"Case-insensitive field name clash in {cls.__name__}: "
                        f"'{f}' and '{ci_map[lowered]}' differ only by case."
                    )
                ci_map[lowered] = f
            cls._ci_map = ci_map
        return cls


class CaseInsensitiveMixin(metaclass=CaseInsensitiveMeta):
    """
    All dataclasses that inherit from this mixin become case-insensitive.
    The mixin does **not** interfere with normal positional arguments.
    """

    @classmethod
    def _ci_resolve(cls, key: str) -> str:
        """Return the canonical field name for a case-insensitive key."""
        try:
            return cls._ci_map[key.lower()]
        except (AttributeError, KeyError):
            raise AttributeError(f"{cls.__name__!r} has no field named {key!r}")

    def __init__(self, *args, **kwargs) -> None:
        """
        Positional arguments are handed to the normal dataclass __init__.
        Keyword arguments are first normalized to lower case, then matched
        against the real field names.
        """
        cls = self.__class__
        # First deal with positional args
        new_kw = {}
        for k, v in zip(cls._ci_map.values(), args):
            new_kw[k] = v
        # then with keyword args
        for k, v in kwargs.items():
            try:
                new_k = cls._ci_map[k.lower()]
            except (AttributeError, KeyError):
                raise AttributeError(f"{cls.__name__!r} has no field named {k!r}")
            new_kw[new_k] = v
        #finally call the constructor of the dataclass
        super().__init__(**new_kw)

    def __getattr__(self, name: str):
        # Called only when normal attribute lookup fails.
        # Translate the name and retry.
        field_name = self._ci_resolve(name)
        return object.__getattribute__(self, field_name)

    def __setattr__(self, name: str, value) -> None:
        # If the attribute already exists (including private ones like _ci_map)
        # we let the normal path handle it.
        if name.startswith('_') or name in self.__dict__:
            object.__setattr__(self, name, value)
            return

        # Otherwise treat it as a field name – case-insensitively.
        field_name = self._ci_resolve(name)
        object.__setattr__(self, field_name, value)

    def __delattr__(self, name: str) -> None:
        field_name = self._ci_resolve(name)
        object.__delattr__(self, field_name)


def case_insensitive_dataclass(_cls=None, *,
                                init: bool = True,
                                repr: bool = True,
                                eq: bool = True,
                                order: bool = False,
                                unsafe_hash: bool = False,
                                frozen: bool = False,
                                match_args: bool =True,
                                kw_only: bool =False,
                                slots: bool =False):
    """
    Use instead of the builtin ``@dataclass``:

        @case_insensitive_dataclass
        class Person:
            Name: str
            Age: int = 0

    The generated class automatically inherits from ``CaseInsensitiveMixin``.
    All arguments are forwarded to the standard ``dataclasses.dataclass``.
    """
    def wrap(cls):
        # Run the regular dataclass decorator on the new class.
        dc = _dataclass(cls, init=init, repr=repr, eq=eq,
                        order=order, unsafe_hash=unsafe_hash,
                        frozen=frozen, match_args=match_args,
                        kw_only=kw_only, slots=slots)
        cidc = type(dc.__name__, (CaseInsensitiveMixin, dc), {"__init__": CaseInsensitiveMixin.__init__})
        return cidc

    # The decorator can be used with or without parentheses.
    if _cls is None:
        return wrap
    else:
        return wrap(_cls)

# TODO CaseInsensitiveNamedTuple ?