/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Preprocessing program
 *
 *
 *   Copyright (C) 2024-2024 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 20/11/2024
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
 *
 * \brief OpenCL kernels performing median filtering | quantile averages
 *
 * Constant to be provided at build time:
 *
 * Files to be co-built:
 *   collective/reduction.cl
 *   collective/scan.cl
 *   collective/comb_sort.cl
 */

#include "for_eclipse.h"

/**
 * \brief Performs sigma clipping in azimuthal rings based on a LUT in CSR form for background extraction
 *
 * Grid: 1D grid with one workgroup processes one bin in a collaborative manner,
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param indices     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param indptr      Integer pointer to global memory holding the pointers to the coefs and indices for the CSR matrix
 * @param cutoff      Discard any value with |value - mean| > cutoff*sigma
 * @param cycle       number of cycle
 * @param error_model 0:disable, 1:variance, 2:poisson, 3:azimuthal, 4:hybrid
 * @param summed      contains all the data
 * @param averint     Average signal
 * @param stdevpix    Float pointer to the output 1D array with the propagated error (std)
 * @param stdevpix    Float pointer to the output 1D array with the propagated error (sem)
 * @param shared      Buffer of shared memory of size WORKGROUP_SIZE * 8 * sizeof(float)
 */

kernel void
csr_medfilt    (  const   global  float4  *data4,
                          global  float4  *work4,
                  const   global  float   *coefs,
                  const   global  int     *indices,
                  const   global  int     *indptr,
                  const           float    quant_min,
                  const           float    quant_max,
                  const           char     error_model,
                  const           float    empty,
                          global  float8  *summed,
                          global  float   *averint,
                          global  float   *stdevpix,
                          global  float   *stderrmean,
                 volatile local   float8  *shared8
                          )
{
    int bin_num = get_group_id(0);
    int wg = get_local_size(0);
    int tid = get_local_id(0);
    int start = indptr[bin_num];
    int stop = indptr[bin_num+1];
    int cnt;
    char curr_error_model=error_model;
    volatile local int counter[1];
    float8 result;

    // first populate the work4 array from data4
    for (int i=start+tid; i<stop; i+=wg)
    {
        float4 r4, w4;
        int idx = indices[i];
        float coef = coefs[i];
        r4 = data4[idx];

//        work0[:, 0] = pixels[:, 0]/ pixels[:, 2]
//        work0[:, 1] = pixels[:, 0] * csr_data
//        work0[:, 2] = pixels[:, 1] * csr_data2
//        work0[:, 3] = pixels[:, 2] * csr_data

        w4.s0 = r4.s0 / r4.s2;
        w4.s1 = r4.s0 * coef;
        w4.s2 = r4.s1 * coef * coef;
        w4.s3 = r4.s2 * coef;

        work4[i] = w4;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // then perform the sort in the work space along the s0 component

    //TODO


    barrier(CLK_GLOBAL_MEM_FENCE);

    if (get_local_id(0) == 0) {
        summed[bin_num] = result;
        if (result.s6 > 0.0f) {
            averint[bin_num] =  aver;
            stdevpix[bin_num] = std;
            stderrmean[bin_num] = sqrt(result.s2) / result.s4;
        }
        else {
            averint[bin_num] = empty;
            stderrmean[bin_num] = empty;
            stdevpix[bin_num] = empty;
        }
    }
} //end csr_sigma_clip4 kernel
