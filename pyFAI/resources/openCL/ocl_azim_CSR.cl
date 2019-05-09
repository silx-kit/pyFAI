/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Kernel with full pixel-split using a CSR sparse matrix
 *
 *
 *   Copyright (C) 2012-2018 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 02/10/2018
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
 * \brief OpenCL kernels for 1D azimuthal integration using CSR sparse matrix representation
 * 
 * Constant to be provided at build time:
 *   WORKGROUP_SIZE
 */

#include "for_eclipse.h"
/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT in CSR form
 *
 * An image instensity value is spread across the bins according to the positions stored in the LUT.
 * The lut is represented by a set of 3 arrays (coefs, row_ind, col_ptr)
 * Values of 0 in the mask are processed and values of 1 ignored as per PyFAI
 *
 * This implementation is especially efficient on CPU where each core reads adjacent memory.
 * the use of local pointer can help on the CPU.
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param row_ind     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param col_ptr     Integer pointer to global memory holding the pointers to the coefs and row_ind for the CSR matrix
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param coef_power  Set to 2 for variance propagation, leave to 1 for mean calculation
 * @param sum_data    Float pointer to the output 1D array with the weighted histogram
 * @param sum_count   Float pointer to the output 1D array with the unweighted histogram
 * @param merged      Float pointer to the output 1D array with the diffractogram
 *
 */
kernel void
csr_integrate(  const   global  float   *weights,
                const   global  float   *coefs,
                const   global  int     *row_ind,
                const   global  int     *col_ptr,
                const           char     do_dummy,
                const           float    dummy,
                const           int      coef_power,
                        global  float   *sum_data,
                        global  float   *sum_count,
                        global  float   *merged
             )
{
    // each workgroup (ideal size: warp) is assigned to 1 bin
    int bin_num = get_group_id(0);
    int thread_id_loc = get_local_id(0);
    int active_threads = get_local_size(0);
    int2 bin_bounds = (int2) (col_ptr[bin_num], col_ptr[bin_num + 1]);
    int bin_size = bin_bounds.y - bin_bounds.x;
    // we use _K suffix to highlight it is float2 used for Kahan summation
    float2 sum_data_K = (float2)(0.0f, 0.0f);
    float2 sum_count_K = (float2)(0.0f, 0.0f);
    const float epsilon = 1e-10f;
    float coef, coefp, data;
    int idx, k, j;

    for (j=bin_bounds.x;j<bin_bounds.y;j+=WORKGROUP_SIZE)
    {
        k = j+thread_id_loc;
        if (k < bin_bounds.y)
        {
               coef = coefs[k];
               idx = row_ind[k];
               data = weights[idx];
               if  (! isfinite(data))
                   continue;

               if( (!do_dummy) || (data!=dummy) )
               {
                   //sum_data +=  coef * data;
                   //sum_count += coef;
                   //Kahan summation allows single precision arithmetics with error compensation
                   //http://en.wikipedia.org/wiki/Kahan_summation_algorithm
                   // defined in kahan.cl
                   sum_data_K = kahan_sum(sum_data_K, ((coef_power == 2) ? coef*coef: coef) * data);
                   sum_count_K = kahan_sum(sum_count_K, coef);
               };//end if dummy
       } //end if k < bin_bounds.y
       };//for j
/*
 * parallel reduction
 */

// REMEMBER TO PASS WORKGROUP_SIZE AS A CPP DEF
    local float2 super_sum_data[WORKGROUP_SIZE];
    local float2 super_sum_count[WORKGROUP_SIZE];
    
    int index;
    
    if (bin_size < WORKGROUP_SIZE)
    {
        if (thread_id_loc < bin_size)
        {
            super_sum_data[thread_id_loc] = sum_data_K;
            super_sum_count[thread_id_loc] = sum_count_K;
        }
        else
        {
            super_sum_data[thread_id_loc] = (float2)(0.0f, 0.0f);
            super_sum_count[thread_id_loc] = (float2)(0.0f, 0.0f);
        }
    }
    else
    {
        super_sum_data[thread_id_loc] = sum_data_K;
        super_sum_count[thread_id_loc] = sum_count_K;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (active_threads != 1)
    {
        active_threads /= 2;
        if (thread_id_loc < active_threads)
        {
            index = thread_id_loc + active_threads;
            super_sum_data[thread_id_loc] = compensated_sum(super_sum_data[thread_id_loc], super_sum_data[index]);
            super_sum_count[thread_id_loc] = compensated_sum(super_sum_count[thread_id_loc], super_sum_count[index]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_id_loc == 0)
    {
        sum_data[bin_num] = super_sum_data[0].s0;
        sum_count[bin_num] = super_sum_count[0].s0;
        if (sum_count[bin_num] > epsilon)
            merged[bin_num] =  sum_data[bin_num] / sum_count[bin_num];
        else
            merged[bin_num] = dummy;
    }
};//end kernel


/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT in CSR form
 *  Unlike the former kernel, it works with a workgroup size of ONE
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param row_ind     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param col_ptr     Integer pointer to global memory holding the pointers to the coefs and row_ind for the CSR matrix
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param coef_power  Set to 2 for variance propagation, leave to 1 for mean calculation
 * @param sum_data    Float pointer to the output 1D array with the weighted histogram
 * @param sum_count   Float pointer to the output 1D array with the unweighted histogram
 * @param merged      Float pointer to the output 1D array with the diffractogram
 *
 */
kernel void
csr_integrate_single(  const   global  float   *weights,
                       const   global  float   *coefs,
                       const   global  int     *row_ind,
                       const   global  int     *col_ptr,
                       const           char     do_dummy,
                       const           float    dummy,
                       const           int      coef_power,
                               global  float   *sum_data,
                               global  float   *sum_count,
                               global  float   *merged)
{
    // each workgroup of size=warp is assinged to 1 bin
    int bin_num = get_group_id(0);
    // we use _K suffix to highlight it is float2 used for Kahan summation
    float2 sum_data_K = (float2)(0.0f, 0.0f);
    float2 sum_count_K = (float2)(0.0f, 0.0f);
    const float epsilon = 1e-10f;
    float coef, data;
    int idx, j;

    for (j=col_ptr[bin_num];j<col_ptr[bin_num+1];j++)
    {
        coef = coefs[j];
        idx = row_ind[j];
        data = weights[idx];
        if  (! isfinite(data))
            continue;
        if( (!do_dummy) || (data!=dummy) )
        {
            //sum_data +=  coef * data;
            //sum_count += coef;
            //Kahan summation allows single precision arithmetics with error compensation
            //http://en.wikipedia.org/wiki/Kahan_summation_algorithm
            // defined in kahan.cl
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
};//end kernel

