/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Kernel with full pixel-split using a LUT
 *
 *
 *   Copyright (C) 2012-2018 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 20/01/2017
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
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
 * @param coef_power  Set to 2 for variance propagation, leave to 1 for mean calculation
 * @param sum_data     Float pointer to the output 1D array with the weighted histogram
 * @param sum_count    Float pointer to the output 1D array with the unweighted histogram
 * @param mergedd   Float pointer to the output 1D array with the diffractogram
 *
 */
__kernel void
lut_integrate(  const     global    float              *weights,
                const     global    struct lut_point_t *lut,
                const               char                do_dummy,
                const               float               dummy,
                const               int                 coef_power,
                          global    float              *sum_data,
                          global    float              *sum_count,
                          global    float              *merged
                )
{
    int idx, k, j, bin_num= get_global_id(0);

    // we use _K suffix to highlight it is float2 used for Kahan summation
    float2 sum_data_K = (float2)(0.0f, 0.0f);
    float2 sum_count_K = (float2)(0.0f, 0.0f);
    const float epsilon = 1e-10f;
    float coef, data;
    if(bin_num < NBINS)
    {
        for (j=0;j<NLUT;j++)
        {
            if (ON_CPU){
                //On CPU best performances are obtained  when each single thread reads adjacent memory
                k = bin_num*NLUT+j;

            }
            else{
                //On GPU best performances are obtained  when threads are reading adjacent memory
                k = j*NBINS+bin_num;
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
                sum_data_K = kahan_sum(sum_data_K, ((coef_power == 2) ? coef*coef: coef) * data);
                sum_count_K = kahan_sum(sum_count_K, coef);
            };//end if dummy
        };//for j
        sum_data[bin_num] = sum_data_K.s0;
        sum_count[bin_num] = sum_count_K.s0;
        if (sum_count_K.s0 > epsilon)
            merged[bin_num] =  sum_data_K.s0 / sum_count_K.s0;
        else
            merged[bin_num] = dummy;
  };//if NBINS
};//end kernel
