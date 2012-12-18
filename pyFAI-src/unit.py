#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author: Picca Frédéric-Emmanuel <picca@synchrotron-soleil.fr>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
__author__ = "Picca Frédéric-Emmanuel"
__contact__ = "picca@synchrotron-soleil.fr"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/12/2012"
__status__ = "beta"
__docformat__ = 'restructuredtext'


class UnitEnum:
    class UNDEFINED:
        REPR = '?'

    class TTH:
        REPR = "2th_deg"

    class TTH_RAD:
        REPR = "2th_rad"

    class Q:
        REPR = "q_nm^-1"

    class Q_A:
        REPR = "q_A^-1"

    class R:
        REPR = "r_mm"

