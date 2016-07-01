/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Kernel with full pixel-split using a LUT
 *
 *
 *   Copyright (C) 2012-2014 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 11/10/2014
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
 * \brief OpenCL kernels for 1D azimuthal integration
 *
 * Needed constant:
 *   NLUT: maximum size of the LUT
 *   NBINS: number of output bins for histograms
 *   ON_CPU: 0 for GPU, 1 for CPU and probably Xeon Phi 
 */


#include "for_eclipse.h"


struct lut_point_t
{
    int idx;
    float coef;
};



/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT
 *
 * An image instensity value is spread across the bins according to the positions stored in the LUT.
 * The lut is an 2D-array of index (contains the positions of the pixel in the input array)
 * and coeficients (fraction of pixel going to the bin)
 * Values of 0 in the mask are processed and values of 1 ignored as per PyFAI
 *
 * This implementation is especially efficient on CPU where each core reads adjacents memory.
 * the use of local pointer can help on the CPU.
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param lut         Pointer to an 2D-array of (unsigned integers,float) containing the index of input pixels and the fraction of pixel going to the bin
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param delta_dummy Float: precision for bad pixel value
 * @param do_dark     Bool/int: shall dark-current correction be applied ?
 * @param dark        Float pointer to global memory storing the dark image.
 * @param do_flat     Bool/int: shall flat-field correction be applied ? (could contain polarization corrections)
 * @param flat        Float pointer to global memory storing the flat image.
 * @param outData     Float pointer to the output 1D array with the weighted histogram
 * @param outCount    Float pointer to the output 1D array with the unweighted histogram
 * @param outMerged   Float pointer to the output 1D array with the diffractogram
 *
 */
__kernel void
lut_integrate(  const     __global    float              *weights,
                const     __global    struct lut_point_t *lut,
                const                 int                 do_dummy,
                const                 float               dummy,
                          __global    float              *outData,
                          __global    float              *outCount,
                          __global    float              *outMerge
                )
{
    int idx, k, j, i= get_global_id(0);
    float sum_data = 0.0f;
    float sum_count = 0.0f;
    float cd = 0.0f;
    float cc = 0.0f;
    float t, y;
    const float epsilon = 1e-10f;
    float coef, data;
    if(i < NBINS)
    {
        for (j=0;j<NLUT;j++)
        {
            if (ON_CPU){
                //On CPU best performances are obtained  when each single thread reads adjacent memory
                k = i*NLUT+j;

            }
            else{
                //On GPU best performances are obtained  when threads are reading adjacent memory
                k = j*NBINS+i;
            }

            idx = lut[k].idx;
            coef = lut[k].coef;
            if((idx <= 0) && (coef <= 0.0f))
              break;
            data = weights[idx];
            if( (!do_dummy) || (data!=dummy) )
            {
                //sum_data +=  coef * data;
                //sum_count += coef;
                //Kahan summation allows single precision arithmetics with error compensation
                //http://en.wikipedia.org/wiki/Kahan_summation_algorithm
                y = coef*data - cd;
                t = sum_data + y;
                cd = (t - sum_data) - y;
                sum_data = t;
                y = coef - cc;
                t = sum_count + y;
                cc = (t - sum_count) - y;
                sum_count = t;
            };//end if dummy
        };//for j
        outData[i] = sum_data;
        outCount[i] = sum_count;
        if (sum_count > epsilon)
            outMerge[i] =  sum_data / sum_count;
        else
            outMerge[i] = dummy;
  };//if NBINS
};//end kernel
