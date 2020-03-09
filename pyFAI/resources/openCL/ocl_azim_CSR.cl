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
 * \brief OpenCL workgroup function for sparse matrix-dense vector multiplication
 *
 * The CSR matrix is represented by a set of 3 arrays (coefs, indices, indptr)
 *
 * The returned value is a float2 with the main result in s0 and the remainder in s1
 *
 * @param vector      float2 array in global memory storing the data as signal/normalization.
 * @param data        float  array in global memory holding the coeficient part of the LUT
 * @param indices     integer array in global memory holding the corresponding column index of the coeficient
 * @param indptr      Integer array in global memory holding the index of the start of the nth line
 * @param super_sum   Local array of float2 of size WORKGROUP_SIZE: mandatory as a static function !
 * @return (sum_main, sum_neg)
 *
 */

static inline float2 CSRxVec(const   global  float   *vector,
                             const   global  float   *data,
                             const   global  int     *indices,
                             const   global  int     *indptr,
                                     local   float2  *super_sum)
{
    // each workgroup (ideal size: 1 warp or slightly larger) is assigned to 1 bin
    int bin_num = get_group_id(0);
    int thread_id_loc = get_local_id(0);
    int active_threads = get_local_size(0);
    int2 bin_bounds = (int2) (indptr[bin_num], indptr[bin_num + 1]);
    int bin_size = bin_bounds.y - bin_bounds.x;
    // we use _K suffix to highlight it is float2 used for Kahan summation
    float2 sum_K = (float2)(0.0f, 0.0f);
    float coef, signal;
    int idx, k, j;

    for (j=bin_bounds.x; j<bin_bounds.y; j+=WORKGROUP_SIZE) {
        k = j+thread_id_loc;
        if (k < bin_bounds.y) {
               coef = data[k];
               idx = indices[k];
               signal = vector[idx];
               if (isfinite(signal)) {
                   // defined in kahan.cl
                   sum_K = kahan_sum(sum_K, coef * signal);
               };//end if finite
       } //end if k < bin_bounds.y
     }//for j
  
    // parallel reduction

    int index;
    if (bin_size < WORKGROUP_SIZE) {
        if (thread_id_loc < bin_size) 
            super_sum[thread_id_loc] = sum_K;
        else 
            super_sum[thread_id_loc] = (float2)(0.0f, 0.0f);
    }
    else 
        super_sum[thread_id_loc] = sum_K;

    barrier(CLK_LOCAL_MEM_FENCE);

    while (active_threads > 1) {
        active_threads /= 2; 
        if (thread_id_loc < active_threads) {
            index = thread_id_loc + active_threads;
            super_sum[thread_id_loc] = compensated_sum(super_sum[thread_id_loc], super_sum[index]);
        } //end active thread
        barrier(CLK_LOCAL_MEM_FENCE);
    } // end reduction
    return super_sum[0];
}


/**
 * \brief OpenCL function for 1d azimuthal integration based on CSR matrix multiplication
 *
 * The CSR matrix is represented by a set of 3 arrays (coefs, indices, indptr)
 *
 * @param data        float2 array in global memory storing the data as signal/normalization.
 * @param coefs       float  array in global memory holding the coeficient part of the LUT
 * @param indices     integer array in global memory holding the corresponding column index of the coeficient
 * @param indptr      Integer array in global memory holding the index of the start of the nth line
 * @param super_sum   Local array of float4 of size WORKGROUP_SIZE: mandatory as a static function !
 * @return (sum_signal_main, sum_signal_neg, sum_norm_main, sum_norm_neg)
 *
 */

static inline float4 CSRxVec2(const   global  float2   *data,
                              const   global  float    *coefs,
                              const   global  int      *indices,
                              const   global  int      *indptr,
                                      local   float4   *super_sum)
{
    // each workgroup (ideal size: 1 warp or slightly larger) is assigned to 1 bin
    int bin_num = get_group_id(0);
    int thread_id_loc = get_local_id(0);
    int active_threads = get_local_size(0);
    int2 bin_bounds = (int2) (indptr[bin_num], indptr[bin_num + 1]);
    int bin_size = bin_bounds.y - bin_bounds.x;
    // we use _K suffix to highlight it is float2 used for Kahan summation
    float2 sum_signal_K = (float2)(0.0f, 0.0f);
    float2 sum_norm_K = (float2)(0.0f, 0.0f);
    int idx, k, j;

    for (j=bin_bounds.x; j<bin_bounds.y; j+=WORKGROUP_SIZE) {
        k = j+thread_id_loc;
        if (k < bin_bounds.y) {
               float coef = coefs[k];
               idx = indices[k];
               float signal = data[idx].s0;
               float norm = data[idx].s1;
               if (isfinite(signal) && isfinite(norm)) {
                   // defined in kahan.cl
                   sum_signal_K = kahan_sum(sum_signal_K, coef * signal);
                   sum_norm_K = kahan_sum(sum_norm_K, coef * norm);
               };//end if finite
       } //end if k < bin_bounds.y
    } //for j
    
    // parallel reduction
    if (bin_size < WORKGROUP_SIZE) {
        if (thread_id_loc < bin_size) {
            super_sum[thread_id_loc] = (float4)(sum_signal_K, sum_norm_K);
        }
        else {
            super_sum[thread_id_loc] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    else {
        super_sum[thread_id_loc] = (float4)(sum_signal_K, sum_norm_K);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (active_threads > 1) {
        active_threads /= 2; 
        if (thread_id_loc < active_threads) {
            float4 here = super_sum[thread_id_loc];
            float4 there = super_sum[thread_id_loc + active_threads];
            sum_signal_K = compensated_sum((float2)(here.s0, here.s1), (float2)(there.s0, there.s1));
            sum_norm_K = compensated_sum((float2)(here.s2, here.s3), (float2)(there.s2, there.s3));
            super_sum[thread_id_loc] = (float4) (sum_signal_K, sum_norm_K);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return super_sum[0];
}

/**
 * \brief OpenCL function for 1d azimuthal integration based on CSR matrix multiplication after normalization !
 *
 * The CSR matrix is represented by a set of 3 arrays (coefs, indices, indptr)
 *
 * @param data        float4 array in global memory storing the data as signal/variance/normalization/count.
 * @param coefs       float  array in global memory holding the coeficient part of the LUT
 * @param indices     integer array in global memory holding the corresponding column index of the coeficient
 * @param indptr      Integer array in global memory holding the index of the start of the nth line
 * @param super_sum   Local array of float8 of size WORKGROUP_SIZE: mandatory as a static function !
 * @return (sum_signal_main, sum_signal_neg, sum_variance_main,sum_variance_neg,
 *          sum_norm_main, sum_norm_neg, sum_count_main, sum_count_neg)
 *
 */


static inline float8 CSRxVec4(const   global  float4   *data,
                              const   global  float    *coefs,
                              const   global  int      *indices,
                              const   global  int      *indptr,
                                      local   float8   *super_sum)
{
    // each workgroup (ideal size: 1 warp or slightly larger) is assigned to 1 bin
    int bin_num = get_group_id(0);
    int thread_id_loc = get_local_id(0);
    int active_threads = get_local_size(0);
    int2 bin_bounds = (int2) (indptr[bin_num], indptr[bin_num + 1]);
    int bin_size = bin_bounds.y - bin_bounds.x;
    // we use _K suffix to highlight it is float2 used for Kahan summation
    float2 sum_signal_K = (float2)(0.0f, 0.0f);
    float2 sum_variance_K = (float2)(0.0f, 0.0f);
    float2 sum_norm_K = (float2)(0.0f, 0.0f); 
    float2 sum_count_K = (float2)(0.0f, 0.0f);
    int idx, k, j;

    for (j=bin_bounds.x; j<bin_bounds.y; j+=WORKGROUP_SIZE) {
        k = j+thread_id_loc;
        if (k < bin_bounds.y) {
               float coef = coefs[k];
               idx = indices[k];
               float4 quatret = data[idx];
               float signal = quatret.s0;
               float variance = quatret.s1;
               float norm = quatret.s2;
               float count = quatret.s3;
               if (isfinite(signal) && isfinite(variance) && isfinite(norm) && (count > 0))
               {
                   // defined in kahan.cl
                   sum_signal_K = kahan_sum(sum_signal_K, coef * signal);
                   sum_variance_K = kahan_sum(sum_variance_K, coef * coef * variance);
                   sum_norm_K = kahan_sum(sum_norm_K, coef * norm);
                   sum_count_K = kahan_sum(sum_count_K, coef * count);
               };//end if finite
       } //end if k < bin_bounds.y
    };//for j
/*
 * parallel reduction
 */
    if (bin_size < WORKGROUP_SIZE) {
        if (thread_id_loc < bin_size) {
            super_sum[thread_id_loc] = (float8)(sum_signal_K, sum_variance_K, sum_norm_K, sum_count_K);
        }
        else {
            super_sum[thread_id_loc] = (float8)(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    else {
        super_sum[thread_id_loc] = (float8)(sum_signal_K, sum_variance_K, sum_norm_K, sum_count_K);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (active_threads > 1) {
        active_threads /= 2;
        if (thread_id_loc < active_threads) {
            float8 here =  super_sum[thread_id_loc];
            float8 there = super_sum[thread_id_loc + active_threads];
            sum_signal_K = compensated_sum((float2)(here.s0, here.s1), (float2)(there.s0, there.s1));
            sum_variance_K = compensated_sum((float2)(here.s2, here.s3), (float2)(there.s2, there.s3));
            sum_norm_K = compensated_sum((float2)(here.s4, here.s5), (float2)(there.s4, there.s5));
            sum_count_K = compensated_sum((float2)(here.s6, here.s7), (float2)(there.s6, there.s7));
            super_sum[thread_id_loc] = (float8)(sum_signal_K, sum_variance_K, sum_norm_K, sum_count_K);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return super_sum[0];
}

/**
 * \brief OpenCL function for sigma clipping CSR look up table. Sets count to NAN
 *
 * The CSR matrix is represented by a set of 3 arrays (coefs, indices, indptr)
 *
 * @param data        float4 array in global memory storing the data as signal/variance/normalization/count.
 * @param coefs       float  array in global memory holding the coeficient part of the LUT
 * @param indices     integer array in global memory holding the corresponding column index of the coeficient
 * @param indptr      Integer array in global memory holding the index of the start of the nth line
 * @param aver        average over the region
 * @param std         standard deviation of the average
 * @param cutoff      cut values above so many sigma, set count to NAN 
 * @return (sum_signal_main, sum_signal_neg, sum_variance_main,sum_variance_neg,
 *          sum_norm_main, sum_norm_neg, sum_count_main, sum_count_neg)
 *
 */


static inline int _sigma_clip4(         global  float4   *data,
                                const   global  float    *coefs,
                                const   global  int      *indices,
                                const   global  int      *indptr,
                                                float    aver,
                                                float    std,
                                                float    cutoff,
                                        local   int      *counter){
    // each workgroup (ideal size: 1 warp or slightly larger) is assigned to 1 bin
    int cnt, j, k, idx;
    counter[0] = 0;
    int bin_num = get_group_id(0);
    int thread_id_loc = get_local_id(0);
    int active_threads = get_local_size(0);
    int2 bin_bounds = (int2) (indptr[bin_num], indptr[bin_num + 1]);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for (j=bin_bounds.s0; j<bin_bounds.s1; j+=WORKGROUP_SIZE){
        k = j + thread_id_loc;
        if (k < bin_bounds.s1){
            idx = indices[k];
            float4 quatret = data[idx];
            // Check validity (on cnt, i.e. s3) and normalisation (in s2) value to avoid zero division 
            if (isfinite(quatret.s3) && (quatret.s2 > 0.0f)){
                float signal = quatret.s0 / quatret.s2;
                if (fabs(signal-aver) > cutoff*std){
                    data[idx].s3 = NAN;
                    atomic_inc(counter);
                }
                     
            } // if finite
        }// in bounds
    }// loop
    barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);
    return counter[0];
}// functions

/* Calculate the standard deviation of the signal within a bin assuming an azimuthal symmetry
 * 
 * 
 * @return: 2tuple containing std=sqrt(sum_error_squared/count) & sem=sqrt(sum_error_squared)/count
 */
static inline float2 _azimuthal_deviation(        global  float4   *data, 
                                          const   global  float    *coefs,
                                          const   global  int      *indices,
                                          const   global  int      *indptr,
                                          const           float    aver,
                                                 local   float4  *shared4)
{
    // each workgroup (ideal size: 1 warp or slightly larger) is assigned to 1 bin
    int bin_num = get_group_id(0);
    int thread_id_loc = get_local_id(0);
    int active_threads = get_local_size(0);
    int2 bin_bounds = (int2) (indptr[bin_num], indptr[bin_num + 1]);
    int bin_size = bin_bounds.y - bin_bounds.x;
    // we use _K suffix to highlight it is float2 used for Kahan summation
    float4 sum_K = (float4)(0.0f, 0.0f,0.0f, 0.0f);
    float coef, signal;
    int idx, k, j;
    float2 sum_variance_K = (float2)(0.0f, 0.0f);
    float2 sum_count_K = (float2)(0.0f, 0.0f);

    // each thread processes a few points according to the LUT. 
    for (j=bin_bounds.x; j<bin_bounds.y; j+=WORKGROUP_SIZE){
        k = j+thread_id_loc;
        if (k < bin_bounds.y){
               coef = coefs[k];
               idx = indices[k];
               float4 quatret = data[idx];
               if (isfinite(quatret.s0) && isfinite(quatret.s1) && (quatret.s2>0) && (quatret.s3>0)){
                   // defined in kahan.cl
                   float delta = coef * (aver - (quatret.s0/quatret.s2));
                   sum_variance_K = kahan_sum(sum_variance_K, delta*delta);
                   sum_count_K = kahan_sum(sum_count_K, coef * quatret.s3);
               }//end if finite
       } //end if k < bin_bounds.y
    }//for j

    // parallel reduction between threads in a workgoup
    int index;
    if (bin_size < WORKGROUP_SIZE) {
        if (thread_id_loc < bin_size)
            shared4[thread_id_loc] = (float4)(sum_variance_K, sum_count_K);
        else
            shared4[thread_id_loc] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
    else
        shared4[thread_id_loc] = (float4)(sum_variance_K, sum_count_K);

    barrier(CLK_LOCAL_MEM_FENCE);

    while (active_threads > 1) {
        active_threads /= 2;
        if (thread_id_loc < active_threads) {            
            float4 here =  shared4[thread_id_loc];
            float4 there = shared4[thread_id_loc + active_threads];
            sum_variance_K = compensated_sum((float2)(here.s0, here.s1), (float2)(there.s0, there.s1));
            sum_count_K = compensated_sum((float2)(here.s2, here.s3), (float2)(there.s2, there.s3));
            shared4[thread_id_loc] = (float4)(sum_variance_K, sum_count_K);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return (float2)(sqrt(shared4[0].s0/shared4[0].s2),
                    sqrt(shared4[0].s0)/shared4[0].s2);
}

/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT in CSR form
 *
 * An image instensity value is spread across the bins according to the positions stored in the LUT.
 * The lut is represented by a set of 3 arrays (coefs, indices, indptr)
 * Values of 0 in the mask are processed and values of 1 ignored as per PyFAI
 *
 * This implementation is especially efficient on CPU where each core reads adjacent memory.
 * the use of local pointer can help on the CPU.
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param indices     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param indptr     Integer pointer to global memory holding the pointers to the coefs and indices for the CSR matrix
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
                const   global  int     *indices,
                const   global  int     *indptr,
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
    int2 bin_bounds = (int2) (indptr[bin_num], indptr[bin_num + 1]);
    int bin_size = bin_bounds.y - bin_bounds.x;
    // we use _K suffix to highlight it is float2 used for Kahan summation
    float2 sum_data_K = (float2)(0.0f, 0.0f);
    float2 sum_count_K = (float2)(0.0f, 0.0f);
    const float epsilon = 1e-10f;
    float coef, coefp, data;
    int idx, k, j;

    for (j=bin_bounds.x;j<bin_bounds.y;j+=WORKGROUP_SIZE) {
        k = j+thread_id_loc;
        if (k < bin_bounds.y) {
               coef = coefs[k];
               idx = indices[k];
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
    }//for j

    // parallel reduction

    // REMEMBER TO PASS WORKGROUP_SIZE AS A CPP DEF
    local float2 super_sum_data[WORKGROUP_SIZE];
    local float2 super_sum_count[WORKGROUP_SIZE];

    int index;

    if (bin_size < WORKGROUP_SIZE) {
        if (thread_id_loc < bin_size) {
            super_sum_data[thread_id_loc] = sum_data_K;
            super_sum_count[thread_id_loc] = sum_count_K;
        }
        else {
            super_sum_data[thread_id_loc] = (float2)(0.0f, 0.0f);
            super_sum_count[thread_id_loc] = (float2)(0.0f, 0.0f);
        }
    }
    else {
        super_sum_data[thread_id_loc] = sum_data_K;
        super_sum_count[thread_id_loc] = sum_count_K;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (active_threads > 1) {
        active_threads /= 2 ;
        if (thread_id_loc < active_threads) {
            index = thread_id_loc + active_threads;
            super_sum_data[thread_id_loc] = compensated_sum(super_sum_data[thread_id_loc], super_sum_data[index]);
            super_sum_count[thread_id_loc] = compensated_sum(super_sum_count[thread_id_loc], super_sum_count[index]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_id_loc == 0) { 
        //Only thread 0 works 
        sum_data[bin_num] = super_sum_data[0].s0;
        sum_count[bin_num] = super_sum_count[0].s0;
        if (sum_count[bin_num] > epsilon)
            merged[bin_num] =  sum_data[bin_num] / sum_count[bin_num];
        else
            merged[bin_num] = dummy;
    } //end thread 0
}//end kernel


/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT in CSR form
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param indices     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param indptr     Integer pointer to global memory holding the pointers to the coefs and indices for the CSR matrix
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
                       const   global  int     *indices,
                       const   global  int     *indptr,
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

    for (j=indptr[bin_num];j<indptr[bin_num+1];j++) {
        coef = coefs[j];
        idx = indices[j];
        data = weights[idx];

        if( isfinite(data) && ((!do_dummy) || (data!=dummy))) {
            //sum_data +=  coef * data;
            //sum_count += coef;
            //Kahan summation allows single precision arithmetics with error compensation
            //http://en.wikipedia.org/wiki/Kahan_summation_algorithm
            // defined in kahan.cl
            sum_data_K = kahan_sum(sum_data_K, ((coef_power == 2) ? coef*coef: coef) * data);
            sum_count_K = kahan_sum(sum_count_K, coef);
        }//end if dummy
    }//for j
    sum_data[bin_num] = sum_data_K.s0;
    sum_count[bin_num] = sum_count_K.s0;
    if (sum_count_K.s0 > epsilon)
        merged[bin_num] =  sum_data_K.s0 / sum_count_K.s0;
    else
        merged[bin_num] = dummy;
}//end kernel

/**
 * \brief Performs 1d azimuthal integration based on CSR sparse matrix multiplication on preprocessed data
 *  Unlike the former kernel, it works with a workgroup size of ONE (tailor made form MacOS bug)
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param indices     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param indptr     Integer pointer to global memory holding the pointers to the coefs and indices for the CSR matrix
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param coef_power  Set to 2 for variance propagation, leave to 1 for mean calculation
 * @param sum_data    Float pointer to the output 1D array with the weighted histogram
 * @param sum_count   Float pointer to the output 1D array with the unweighted histogram
 * @param merged      Float pointer to the output 1D array with the diffractogram
 *
 */
kernel void
csr_integrate4(  const   global  float4  *weights,
                 const   global  float   *coefs,
                 const   global  int     *indices,
                 const   global  int     *indptr,
                         global  float8  *summed,
                         global  float   *averint,
                         global  float   *stderr)
{
    int bin_num = get_group_id(0);
    local float8 shared[WORKGROUP_SIZE];
    float8 result = CSRxVec4(weights, coefs, indices, indptr, shared);
    if (get_local_id(0)==0) {
        summed[bin_num] = result;
        if (result.s4 > 0.0f) {
            averint[bin_num] =  result.s0 / result.s4;
            stderr[bin_num] = sqrt(result.s2) / result.s4;
        }
        else {
            averint[bin_num] = NAN;
            stderr[bin_num] = NAN;
        } //end else
    } // end if thread0 
};//end kernel


/**
 * \brief Performs 1d azimuthal integration based on CSR sparse matrix multiplication on preprocessed data
 *  Unlike the former kernel, it works with a workgroup size of ONE (tailor made form MacOS bug)
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param indices     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param indptr     Integer pointer to global memory holding the pointers to the coefs and indices for the CSR matrix
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param coef_power  Set to 2 for variance propagation, leave to 1 for mean calculation
 * @param sum_data    Float pointer to the output 1D array with the weighted histogram
 * @param sum_count   Float pointer to the output 1D array with the unweighted histogram
 * @param merged      Float pointer to the output 1D array with the diffractogram
 *
 */
kernel void
csr_integrate4_single(  const   global  float4  *weights,
                        const   global  float   *coefs,
                        const   global  int     *indices,
                        const   global  int     *indptr,
                                global  float8  *summed,
                                global  float   *averint,
                                global  float   *stderr)
{
    // each workgroup of size=warp is assinged to 1 bin
    int bin_num = get_group_id(0);
    // we use _K suffix to highlight it is float2 used for Kahan summation
    float2 sum_signal_K = (float2)(0.0f, 0.0f);
    float2 sum_variance_K = (float2)(0.0f, 0.0f);
    float2 sum_norm_K = (float2)(0.0f, 0.0f);
    float2 sum_count_K = (float2)(0.0f, 0.0f);
    const float epsilon = 1e-10f;

    for (int j=indptr[bin_num];j<indptr[bin_num+1];j++) {
        float coef = coefs[j];
        int idx = indices[j];
        float4 tmp = weights[idx];
        float signal = tmp.s0;
        float variance = tmp.s1;
        float norm = tmp.s2;
        float count = tmp.s3;

        if( isfinite(signal) && isfinite(variance) && isfinite(norm) && (count > 0)) {
            //Kahan summation allows single precision arithmetics with error compensation
            //http://en.wikipedia.org/wiki/Kahan_summation_algorithm
            // defined in kahan.cl
            sum_signal_K = kahan_sum(sum_signal_K ,coef * signal);
            sum_variance_K = kahan_sum(sum_variance_K, coef * coef * variance);
            sum_norm_K = kahan_sum(sum_norm_K, coef * norm);
            sum_count_K = kahan_sum(sum_count_K, coef * count);
        }//end if finite
    }//for j

    summed[bin_num] = (float8)(sum_signal_K, sum_variance_K, sum_norm_K, sum_count_K);
    if (sum_norm_K.s0 > 0.0f) {
        averint[bin_num] = sum_signal_K.s0 / sum_norm_K.s0;
        stderr[bin_num] = sqrt(sum_variance_K.s0) / sum_norm_K.s0;
    }
    else {
        averint[bin_num] = NAN;
        stderr[bin_num] = NAN;
    }
}//end kernel

/**
 * \brief Performs sigma clipping in azimuthal rings based on a LUT in CSR form for background extraction
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param indices     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param indptr      Integer pointer to global memory holding the pointers to the coefs and indices for the CSR matrix
 * @param cutoff      Discard any value with |value - mean| > cutoff*sigma
 * @param cycle       number of cycle 
 * @param azimuthal   set to 1 to calculate the variance from the difference to the average in the bin, left to 0 to propagate as usual 
 * @param summed      contains all the data
 * @param averint     Average signal
 * @param stdevpix    Standard deviation of the pixel
 * @param stderrmean  Standard error of the mean
 *
 */

kernel void
csr_sigma_clip4(          global  float4  *data4,
                  const   global  float   *coefs,
                  const   global  int     *indices,
                  const   global  int     *indptr,
                  const           float    cutoff,
                  const           int      cycle,
                  const           int      azimuthal,
                          global  float8  *summed,
                          global  float   *averint,
                          global  float   *stdevpix,
                          global  float   *stderrmean) {
    int bin_num = get_group_id(0);
    float aver, std, sem;
    int cnt;
    local float8 shared8[WORKGROUP_SIZE];
    
    // first calculation of azimuthal integration to initialize aver & std
    
    float8 result = CSRxVec4(data4, coefs, indices, indptr, shared8);
    if (result.s4 > 0.0f){
        aver = result.s0 / result.s4;
        if (azimuthal){
            float2 res2 = _azimuthal_deviation(data4, coefs, indices, indptr, aver, (__local float4*) shared8);
            std = res2.s0;
            sem = res2.s1;
        }             
        else {
            std = sqrt(result.s2 / result.s4);
            sem = sqrt(result.s2) / result.s4;
        }
            
    }
    else {
        aver = NAN;
        std = NAN;
        sem = NAN;
    }

    for (int i=0; i<cycle; i++) {
        if ( ! (isfinite(aver) && isfinite(std)))
            break;

        cnt = _sigma_clip4(data4, coefs, indices, indptr, aver, std, cutoff, (__local int*) shared8);

        if (! cnt) 
            break;
        
        result = CSRxVec4(data4, coefs, indices, indptr, shared8);

        if (result.s4 > 0.0f) {
            aver = result.s0 / result.s4;
            if (azimuthal){
            	float2 res2 = _azimuthal_deviation(data4, coefs, indices, indptr, aver, (__local float4*) shared8);
            	std = res2.s0;
            	sem = res2.s1;
            }                
            else {
            	std = sqrt(result.s2 / result.s4);
            	sem = sqrt(result.s2) / result.s4;
            }
                
        }
        else {
            aver = NAN;
            std = NAN;
            sem = NAN;
            break;
        }
    }
        
    if (get_local_id(0) == 0) {
        summed[bin_num] = result;
        averint[bin_num] = aver;
        //Note the standard error of the mean, SEM,  differs from std by sqrt of the normalization factor
        stdevpix[bin_num] = std;
        stderrmean[bin_num] = sem;
    }
} //end kernel
