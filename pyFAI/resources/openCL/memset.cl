/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Preprocessing program
 *
 *
 *   Copyright (C) 2012-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 19/01/2017
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * \file
 *
 * \brief OpenCL kernels for image array casting, array mem-setting and normalizing
 *
 * Constant to be provided at build time:
 *   NBINS:  number of output bins for histograms
 *
 */

#include "for_eclipse.h"

/**
 * \brief Sets the values of 3 float output arrays to zero.
 *
 * Gridsize = size of arrays + padding.
 *
 * - array0: float Pointer to global memory with the outMerge array
 * - array1: float Pointer to global memory with the outCount array
 * - array2: float Pointer to global memory with the outData array
 */
__kernel void
memset_out(__global float *array0,
           __global float *array1,
           __global float *array2
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NBINS)
  {
    array0[i] = 0.0f;
    array1[i] = 0.0f;
    array2[i] = 0.0f;
  }
}
