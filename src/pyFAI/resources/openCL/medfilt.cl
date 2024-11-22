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
 * Grid: 2D grid with one workgroup processes one bin in a collaborative manner,
 *       dim 0: index of bin, size=1
 *       dim 1: collaboarative working group size, probably optimal in the range 32-128
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
                          local   int*    shared_int // size of the workgroup size
                          local   int*    shared_float // size of the workgroup size
                          )
{
    int bin_num = get_group_id(0);
    int wg = get_local_size(1);
    int tid = get_local_id(1);
    int start = indptr[bin_num];
    int stop = indptr[bin_num+1];
    int size = stop-start;
    int cnt, step=11;
    int niter, idx;
    char curr_error_model=error_model;
    float8 result;
    float sum=0.0f, ratio=1.3f;
    float2 acc_sig, acc_nrm, acc_var;

    // first populate the work4 array from data4
    for (int i=start+tid; i<stop; i+=wg)
    {
        float4 r4, w4;
        int idx = indices[i];
        float coef = coefs[i];
        r4 = data4[idx];

        w4.s0 = r4.s0 / r4.s2;
        w4.s1 = r4.s0 * coef;
        w4.s2 = r4.s1 * coef * coef;
        w4.s3 = r4.s2 * coef;

        work4[i] = w4;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // then perform the sort in the work space along the s0 component
    step = first_step(step, size, ratio);

    for (step=step; step>0; step=previous_step(step, ratio))
        cnt = passe_float4(&work4[start], size, step, shared);

    while (cnt)
        cnt = passe_float4(&work4[start], size, 1, shared);

    // Then perform the cumsort of the weights to s0
    // In blelloch scan, one workgroup can process 2wg in size.
    niter = (size + 2*wg-1)/(2*wg);
    sum = 0.0f;
    for (int i=0; i<niter; i+=1)
    {
        idx += start + tid + 2*wg*i;

        shared_float[tid] = (idx<stop)?work4[idx].s3:0.0f;
        shared_float[tid+wg] = ((idx+wg)<stop)?work4[idx+wg].s3:0.0f;

        blelloch_scan_float(shared_float);

        if (idx<size)
            work4[idx].s0 = sum + shared[lid];
        if (i+ws<size)
            work4[idx+wg].s0 = sum + shared[lid+g];
        sum += shared[2*ws-1];

    }
    // Perform the sum for accumulator of signal, variance, normalization and count

    cnt = 0;
    acc_sig = (float2)(0.0f, 0.0f);
    acc_nrm = (float2)(0.0f, 0.0f);
    acc_var = (float2)(0.0f, 0.0f);

    float qmin =
    float qmax =

    for (int i=start+tid; i<stop; i+=wg)
    {
        if valid ...
    }


    // Finally store the accumulated value

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
