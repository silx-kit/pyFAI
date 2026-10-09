/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Preprocessing program
 *
 *
 *   Copyright (C) 2024-2025 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 05/10/2026
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
            here = (float2)(shared[pos_here], shared[pos_here+1]);
            there = (float2)(shared[pos_there], shared[pos_there+1]);
            here = dw_plus_dw(here, there);
            shared[pos_here] = here.s0;
            shared[pos_here+1] = here.s1;
        }

    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 res = (float2)(shared[0], shared[1]);
    barrier(CLK_LOCAL_MEM_FENCE);
    return res;
}


float2 inline sum_float2_sum(local float* shared)
{
    int wg = get_local_size(0) * get_local_size(1);
    int tid = get_local_id(0) + get_local_size(0)*get_local_id(1);

    float2 here, there;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid==0)
    {
        here = (float2)(shared[0], shared[1]);
        for (int pos_there=2; pos_there<wg; pos_there+=2)
        {
            there = (float2)(shared[pos_there],shared[pos_there+1]);
            here = dw_plus_dw(here, there);
        }
        shared[0] = here.s0;
        shared[1] = here.s1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    here = (float2)(shared[0], shared[1]);
    return here;
}


/**
 * \brief Quantile-average (median filtering) in azimuthal rings, from a LUT in CSR form
 *
 * The kernel never sorts the bin. It only needs the two keys which bound the
 * weighted-quantile window, and those come from a weighted radix-select: eight
 * read-only passes of four bits, no scratch array, no limit on the size of a bin.
 * A single Blelloch scan then resolves the pixels whose key sits exactly on one of
 * the two bounds, which a threshold alone cannot split.
 *
 * Grid: 2D grid, one workgroup per bin, processed collaboratively
 *       dim 0: collaborative workgroup size, must be a power of two
 *       dim 1: index of bin, size=1
 *
 * @param data4        Float4 pointer to the preprocessed image (signal, variance, norm, cnt)
 * @param pairs        Float2 pointer to global memory, one entry per non-zero element of
 *                     the CSR matrix, holding (key, weight) for the bin being processed
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
 * @param stderrmean   Float pointer to the output 1D array with the propagated error (sem)
 * @param shared_int   Buffer of shared memory of size WORKGROUP_SIZE * sizeof(int)
 * @param shared_float Buffer of shared memory of size WORKGROUP_SIZE * 4 * sizeof(float)
 * @param shared_hist  Buffer of shared memory of size WORKGROUP_SIZE * 16 * sizeof(float)
 * */

/* Order preserving map float -> uint, so that the bits of a key can be walked from
 * the most significant down, the way a radix-select needs.
 */
uint inline sortable_key(float f)
{
    uint u = as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* Weighted radix-select over pairs[0..size), s0 being the key and s1 the weight.
 *
 * Returns the key K at which the cumulative weight crosses `target`, and leaves in
 * *below the weight of every key strictly under K:
 *      below <= target < below + weight(K)
 * Four bits are resolved per pass, so eight passes, each one read-only.
 * `hist` is a buffer of 16 * workgroup-size floats.
 */
uint inline weighted_select(const global float2 *pairs,
                            int size,
                            float target,
                            local float *hist,
                            float *below)
{
    int wg = get_local_size(0);
    int tid = get_local_id(0);
    uint prefix = 0u;
    float base = 0.0f;

    for (int shift=28; shift>=0; shift-=4)
    {
        float local_hist[16];
        for (int d=0; d<16; d++)
            local_hist[d] = 0.0f;
        // on the first pass every key is a candidate, afterwards only those whose
        // higher bits match the prefix resolved so far
        uint mask = (shift >= 28) ? 0u : (0xFFFFFFFFu << (shift + 4));

        for (int i=tid; i<size; i+=wg)
        {
            float2 pair = pairs[i];
            uint key = sortable_key(pair.s0);
            if ((key & mask) == (prefix & mask))
                local_hist[(key >> shift) & 15u] += pair.s1;
        }
        for (int d=0; d<16; d++)
            hist[d*wg + tid] = local_hist[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride=wg>>1; stride>0; stride>>=1)
        {
            if (tid < stride)
                for (int d=0; d<16; d++)
                    hist[d*wg + tid] += hist[d*wg + tid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // every thread walks the same 16 totals, which avoids a broadcast
        float acc = base;
        int digit = 15;
        for (int d=0; d<16; d++)
        {
            float next = acc + hist[d*wg];
            if (next > target)
            {
                digit = d;
                break;
            }
            acc = next;
        }
        prefix |= ((uint)digit) << shift;
        base = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    *below = base;
    return prefix;
}


kernel void
csr_medfilt    (  const   global  float4  *data4,
                          global  float2  *pairs,
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
                          local   int*    shared_int,  // workgroup size
                          local   float*  shared_float,// 4x the workgroup size
                          local   float*  shared_hist  // 16x the workgroup size
                          )
{
    int bin_num = get_group_id(1);
    int wg = get_local_size(0);
    int tid = get_local_id(0);
    int start = indptr[bin_num];
    int stop = indptr[bin_num+1];
    int size = stop-start;
    int cnt;
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

    // Lay out (key, weight) once, and total the weights on the way.
    float partial = 0.0f;
    for (int i=tid; i<size; i+=wg)
    {
        int idx = indices[start+i];
        float coef = (coefs == ZERO)?1.0f:coefs[start+i];
        float4 r4 = data4[idx];
        float weight = r4.s2 * coef;
        pairs[start+i] = (float2)(r4.s0 / r4.s2, weight);
        partial += weight;
    }
    shared_float[tid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride=wg>>1; stride>0; stride>>=1)
    {
        if (tid < stride)
            shared_float[tid] += shared_float[tid+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sum = shared_float[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float qmin = quant_min * sum;
    float qmax = quant_max * sum;

    float below_lo, below_hi;
    uint key_lo = weighted_select(&pairs[start], size, qmin, shared_hist, &below_lo);
    uint key_hi;
    if (qmin == qmax)
    {   // median: both bounds are the same order statistic, one select is enough
        key_hi = key_lo;
        below_hi = below_lo;
    }
    else
        key_hi = weighted_select(&pairs[start], size, qmax, shared_hist, &below_hi);

    /* A pixel is kept when its slice of cumulated weight lies inside [qmin, qmax],
     * or, in the degenerate case quant_min == quant_max, when it spans the whole
     * window. Written against the keys:
     *   key_lo < key < key_hi            -> always kept, no cumulated weight needed
     *   key < key_lo  or  key > key_hi   -> never kept
     *   key == key_lo or key == key_hi   -> depends on where the pixel falls inside
     *                                       its group of equal keys, hence the scan
     * The scan runs over the bin in its natural order, so the split of a group of
     * equal keys is reproducible, which the sort it replaces was not.
     */
    cnt = 0;
    acc_sig = (float2)(0.0f, 0.0f);
    acc_var = (float2)(0.0f, 0.0f);
    acc_nrm = (float2)(0.0f, 0.0f);
    acc_nrm2 = (float2)(0.0f, 0.0f);

    local float *scan_lo = shared_float;
    local float *scan_hi = shared_float + 2*wg;
    float run_lo = 0.0f, run_hi = 0.0f;
    bool split_hi = (key_hi != key_lo);

    for (int block=0; block<(size + 2*wg-1)/(2*wg); block++)
    {
        int i0 = tid + 2*wg*block;
        int i1 = i0 + wg;
        float2 p0 = (i0<size)?pairs[start+i0]:(float2)(0.0f, 0.0f);
        float2 p1 = (i1<size)?pairs[start+i1]:(float2)(0.0f, 0.0f);
        uint k0 = sortable_key(p0.s0), k1 = sortable_key(p1.s0);
        float w0 = (i0<size)?p0.s1:0.0f, w1 = (i1<size)?p1.s1:0.0f;

        scan_lo[tid]      = (k0 == key_lo)?w0:0.0f;
        scan_lo[tid+wg]   = (k1 == key_lo)?w1:0.0f;
        scan_hi[tid]      = (split_hi && (k0 == key_hi))?w0:0.0f;
        scan_hi[tid+wg]   = (split_hi && (k1 == key_hi))?w1:0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        blelloch_scan_float(scan_lo);
        blelloch_scan_float(scan_hi);

        for (int upper=0; upper<2; upper++)
        {   // each thread owns one element in each half of the block
            int i = upper?i1:i0;
            if (i>=size) continue;
            uint key = upper?k1:k0;
            float weight = upper?w1:w0;
            int slot = upper?(tid+wg):tid;
            bool keep;

            if (key == key_lo || (split_hi && key == key_hi))
            {   // inside a group of equal keys: the inclusive scan gives what comes before
                float below = (key == key_lo)?below_lo:below_hi;
                local float *scan = (key == key_lo)?scan_lo:scan_hi;
                float run = (key == key_lo)?run_lo:run_hi;
                float q_last = below + run + scan[slot] - weight;
                float q_here = q_last + weight;
                keep = ((q_last>=qmin) && (q_here<=qmax))
                    || ((q_last<=qmin) && (q_here>=qmax));
            }
            else
                keep = (key>key_lo) && (key<key_hi);

            if (keep && weight)
            {
                int idx = indices[start+i];
                float coef = (coefs == ZERO)?1.0f:coefs[start+i];
                float4 r4 = data4[idx];
                cnt ++;
                acc_sig = dw_plus_fp(acc_sig, r4.s0 * coef);
                acc_var = dw_plus_fp(acc_var, r4.s1 * coef * coef);
                acc_nrm = dw_plus_fp(acc_nrm, weight);
                acc_nrm2 = dw_plus_dw(acc_nrm2, fp_times_fp(weight, weight));
            }
        }
        run_lo += scan_lo[2*wg-1];
        run_hi += scan_hi[2*wg-1];
        barrier(CLK_LOCAL_MEM_FENCE);
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
