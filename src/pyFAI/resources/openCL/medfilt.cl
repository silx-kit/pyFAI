/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Preprocessing program
 *
 *
 *   Copyright (C) 2024-2024 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 06/12/2024
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

float2 inline sum_float2_reduction(local float* shared)
{
    int wg = get_local_size(0) * get_local_size(1);
    int tid = get_local_id(0) + get_local_size(0)*get_local_id(1);

    // local reduction based implementation
    for (int stride=wg>>1; stride>0; stride>>=1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid<stride) && ((tid+stride)<wg))
        {
            int pos_here, pos_there;
            float2 here, there;
            pos_here = 2*tid;
            pos_there = pos_here + 2*stride;
            here.s0 = shared[pos_here];
            here.s1 = shared[pos_here+1];
            there.s0 = shared[pos_there];
            there.s1 = shared[pos_there+1];
            here = dw_plus_dw(here, there);
            shared[2*tid] = here.s0;
            shared[2*tid+1] = here.s1;
        }

    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 res = (float2)(shared[0], shared[1]);
    barrier(CLK_LOCAL_MEM_FENCE);
    return res;
}


/**
 * \brief Performs sigma clipping in azimuthal rings based on a LUT in CSR form for background extraction
 *
 * Grid: 2D grid with one workgroup processes one bin in a collaborative manner,
 *       dim 0: collaboarative working group size, probably optimal in the range 32-128
 *       dim 1: index of bin, size=1
 *
 * @param weights      Float pointer to global memory storing the input image.
 * @param coefs        Float pointer to global memory holding the coeficient part of the LUT
 * @param indices      Integer pointer to global memory holding the corresponding index of the coeficient
 * @param indptr       Integer pointer to global memory holding the pointers to the coefs and indices for the CSR matrix
 * @param quant_min    start percentile/100 to use. Use 0.5 for the median
 * @param quant_max    stop percentile/100 to use. Use 0.5 for the median
 * @param error_model  0:disable, 1:variance, 2:poisson, 3:azimuthal, 4:hybrid
 * @param empty        Value for empty bins, i.e. those without pixels (can be NaN)
 * @param summed       contains all the data
 * @param averint      Average signal
 * @param stdevpix     Float pointer to the output 1D array with the propagated error (std)
 * @param stdevpix     Float pointer to the output 1D array with the propagated error (sem)
 * @param shared_int   Buffer of shared memory of size WORKGROUP_SIZE * sizeof(int)
 * @param shared_float Buffer of shared memory of size WORKGROUP_SIZE * 2 * sizeof(float)
 * */



kernel void
csr_medfilt    (  const   global  float4  *data4,
                          global  float4  *work4,
                  const   global  float   *coefs,
                  const   global  int     *indices,
                  const   global  int     *indptr,
                  const           float    quant_min,
                                  float    quant_max,
                  const           char     error_model,
                  const           float    empty,
                          global  float8  *summed,
                          global  float   *averint,
                          global  float   *stdevpix,
                          global  float   *stderrmean,
                          local   int*    shared_int,  // size of the workgroup size
                          local   float*  shared_float // size of 2x the workgroup size
                          )
{
    int bin_num = get_group_id(1);
    int wg = get_local_size(0);
    int tid = get_local_id(0);
    int start = indptr[bin_num];
    int stop = indptr[bin_num+1];
    int size = stop-start;
    int sum_cnt, cnt, step=11;
    int idx;
    char curr_error_model=error_model;
    float8 result;
    float sum=0.0f, ratio=1.3f;
    float2 acc_sig, acc_nrm, acc_var, acc_nrm2;

    // ensure the last element is always taken
    if (quant_max == 1.0f)
        quant_max = 1.000001f;

    if (size==0)
    { // Nothing to do since no pixel contribute to bin.
        if (tid == 0)
        {
            averint[bin_num] = empty;
            stderrmean[bin_num] = empty;
            stdevpix[bin_num] = empty;
        }
        return;
    } // Early exit

    // first populate the work4 array from data4
    for (int i=start+tid; i<stop; i+=wg)
    {
        float4 r4, w4;
        int idx = indices[i];
        float coef = (coefs == ZERO)?1.0f:coefs[i];
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
        cnt = passe_float4(&work4[start], size, step, shared_int);

    while (cnt)
        cnt = passe_float4(&work4[start], size, 1, shared_int);
    // Then perform the cumsort of the weights to s0
    // In blelloch scan, one workgroup can process 2wg in size.

    barrier(CLK_GLOBAL_MEM_FENCE);

    sum = 0.0f;
    for (int i=0; i<(size + 2*wg-1)/(2*wg); i++)
    {
        idx = start + tid + 2*wg*i;

        shared_float[tid] = (idx<stop)?work4[idx].s3:0.0f;
        shared_float[tid+wg] = ((idx+wg)<stop)?work4[idx+wg].s3:0.0f;

        blelloch_scan_float(shared_float);

        if (idx<stop)
            work4[idx].s0 = sum + shared_float[tid];
        if ((idx+wg)<stop)
            work4[idx+wg].s0 = sum + shared_float[tid+wg];
        sum += shared_float[2*wg-1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Perform the sum for accumulator of signal, variance, normalization and count

    cnt = 0;
    acc_sig = (float2)(0.0f, 0.0f);
    acc_var = (float2)(0.0f, 0.0f);
    acc_nrm = (float2)(0.0f, 0.0f);
    acc_nrm2 = (float2)(0.0f, 0.0f);

    float qmin = quant_min * sum;
    float qmax = quant_max * sum;

    for (int i=start+tid; i<stop; i+=wg)
    {
        float q_last = (i>start)?work4[i-1].s0:0.0f;
        float4 w = work4[i];
        if((((q_last>=qmin) && (w.s0<=qmax))  // case several contribution
        || ((q_last<=qmin) && (w.s0>=qmax))) // case unique contribution qmin==qmax
        && (w.s3)) {                         // non empty
            cnt ++;
            acc_sig = dw_plus_fp(acc_sig, w.s1);
            acc_var = dw_plus_fp(acc_var, w.s2);
            acc_nrm = dw_plus_fp(acc_nrm, w.s3);
            acc_nrm2 = dw_plus_dw(acc_nrm2, fp_times_fp(w.s3, w.s3));

        }
    }

    //  Now parallel reductions, one after the other :-/

    shared_int[tid] = cnt;
    cnt = sum_int_reduction(shared_int);

    shared_float[2*tid] = acc_sig.s0;
    shared_float[2*tid+1] = acc_sig.s1;
    acc_sig = sum_float2_reduction(shared_float);

    shared_float[2*tid] = acc_var.s0;
    shared_float[2*tid+1] = acc_var.s1;
    acc_var = sum_float2_reduction(shared_float);

    shared_float[2*tid] = acc_nrm.s0;
    shared_float[2*tid+1] = acc_nrm.s1;
    acc_nrm = sum_float2_reduction(shared_float);

    shared_float[2*tid] = acc_nrm2.s0;
    shared_float[2*tid+1] = acc_nrm2.s1;
    acc_nrm2 = sum_float2_reduction(shared_float);

    // Finally store the accumulated value

    if (tid == 0)
    {
        summed[bin_num] = (float8)(acc_sig.s0, acc_sig.s1,
                                acc_var.s0, acc_var.s1,
                                acc_nrm.s0, acc_nrm.s1,
                                (float)cnt, acc_nrm2.s0);
        if (acc_nrm2.s0 > 0.0f)
        {
            averint[bin_num] = acc_sig.s0/acc_nrm.s0 ;
            stdevpix[bin_num] = sqrt(acc_var.s0/acc_nrm2.s0) ;
            stderrmean[bin_num] = sqrt(acc_var.s0) / acc_nrm.s0;
        }
        else {
            averint[bin_num] = empty;
            stderrmean[bin_num] = empty;
            stdevpix[bin_num] = empty;
        }
    }
} //end csr_medfilt kernel
