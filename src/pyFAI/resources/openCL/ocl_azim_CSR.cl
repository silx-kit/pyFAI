/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Kernel with full pixel-split using a CSR sparse matrix
 *
 *
 *   Copyright (C) 2012-2023 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 08/07/2022
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
 * Constant to be provided at build time: None
 */

#include "for_eclipse.h"
#define ZERO 0
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
    int idx;

    for (int k=bin_bounds.x+thread_id_loc; k<bin_bounds.y; k+=active_threads) {
        coef = (data == ZERO)?1.0f:data[k];
        idx = indices[k];
        signal = vector[idx];
        if (isfinite(signal)) {
           // defined in silx:doubleword.cl
           sum_K = dw_plus_fp(sum_K, coef * signal);
        };//end if finite
     }//for k

    // parallel reduction

    int index;
    if (bin_size < active_threads) {
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
            super_sum[thread_id_loc] = dw_plus_dw(super_sum[thread_id_loc], super_sum[index]);
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

    for (j=bin_bounds.x; j<bin_bounds.y; j+=active_threads) {
        k = j+thread_id_loc;
        if (k < bin_bounds.y) {
               float coef, signal, norm;
               coef = (coefs == ZERO)?1.0f:coefs[k];
               idx = indices[k];
               signal = data[idx].s0;
               norm = data[idx].s1;
               if (isfinite(signal) && isfinite(norm)) {
                   // defined in silx:doubleword.cl
                   sum_signal_K = dw_plus_fp(sum_signal_K, coef * signal);
                   sum_norm_K = dw_plus_fp(sum_norm_K, coef * norm);
               };//end if finite
       } //end if k < bin_bounds.y
    } //for j

    // parallel reduction
    if (bin_size < active_threads) {
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
            sum_signal_K = dw_plus_dw((float2)(here.s0, here.s1), (float2)(there.s0, there.s1));
            sum_norm_K = dw_plus_dw((float2)(here.s2, here.s3), (float2)(there.s2, there.s3));
            super_sum[thread_id_loc] = (float4) (sum_signal_K, sum_norm_K);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return super_sum[0];
}


/* _accumulate_poisson: accumulate one value assuming a known error model like the poissonian model
 *
 * All processings are compensated for error accumulation
 *
 * :param accumulator8: float8 containing (sumX_hi, sumX_lo, sumV_hi, sumV_lo, sumW_hi, sumW_lo, cnt_hi, cnt_lo)
 * :param value4: one quadret read from the preproc4 array containing Signal, Variance, normalization and 1 if the pixel is valid.
 * :param coef: how much of that pixel contributes due to pixel splitting [0-1]
 * :retuen: the updated accumulator8
 * */

static inline float8 _accumulate_poisson(float8 accum8,
                                         float4 value4,
                                         float coef){

    float signal, variance, norm, count;
    signal = value4.s0;
    variance = value4.s1;
    norm = value4.s2;
    count = value4.s3;

    if (isfinite(signal) && isfinite(variance) && isfinite(norm) && (count > 0.0f)) {
        float sum_count, sum_norm_2;
        float2 sum_signal_K, sum_variance_K, sum_norm_1, w, coef2;

        sum_signal_K = (float2)(accum8.s0, accum8.s1);
        sum_variance_K = (float2)(accum8.s2, accum8.s3);
        sum_norm_1 = (float2)(accum8.s4, accum8.s5);
        sum_count = accum8.s6;
        sum_norm_2 = accum8.s7;
        // defined in silx:doubleword.cl
        sum_signal_K = dw_plus_dw(sum_signal_K, fp_times_fp(coef, signal));
        coef2 = fp_times_fp(coef,  coef);
        sum_variance_K = dw_plus_dw(sum_variance_K, dw_times_fp(coef2, variance));
        w = fp_times_fp(coef, norm);
        sum_norm_1 = dw_plus_dw(w, sum_norm_1);
        sum_norm_2 = dw_plus_fp(dw_times_dw(w, w), sum_norm_2).s0;

//        sum_norm_1 += coef * norm;
//        sum_norm_2 += coef * coef * norm * norm;
        sum_count = fma(coef, count, sum_count);
        accum8 = (float8)(sum_signal_K, sum_variance_K, sum_norm_1, sum_count, sum_norm_2);
    }
    return accum8;
}

/* _accumulate_azimuthal: accumulate one value considering the azimuthal regions
 *
 * All processings are compensated for error accumulation
 * See Schubert & Gretz 2018, especially Eq22.
 *
 * :param accumulator8: float8 containing (sumX_hi, sumX_lo, sumV_hi, sumV_lo, sumW_hi, sumW_lo, cnt_hi, cnt_lo)
 * :param value4: one quadret read from the preproc4 array containing Signal, (unused variance), normalization and 1 if the pixel is valid.
 * :param coef: how much of that pixel contributes due to pixel splitting [0-1]
 * :retuen: the updated accumulator8
 *
 * Nota: here the variance of the pixel is not used, the variance is calculated from the distance from the pixel to the mean.
 * This algorithm is not numerically as stable as the `poisson` variant
 *
 * */

static inline float8 _accumulate_azimuthal(float8 accum8,
                                           float4 value4,
                                           float coef){

    float signal, norm, count;
    signal = value4.s0;
//    variance = quatret.s1;
    norm = value4.s2;
    count = value4.s3;

    if (isfinite(signal) && isfinite(norm) && (count > 0) && (coef > 0))
    {
        if (accum8.s7 <= 0.0f){
            // Initialize the accumulator with data from the pixel
            float2 w = fp_times_fp(coef, norm);
            accum8 = (float8)(fp_times_fp(coef, signal),
                              0.0f, 0.0f,
                              w,
                              coef * count,
                              dw_times_dw(w, w).s0);
        }
        else{
            //The accumulator is already initialized
            float2 sum_signal_K, sum_variance_K, sum_norm_1, sum_norm_2;
            float sum_count, omega2_A;
            float2 x, delta, omega1_A, delta2, omega1_B, omega2_B, omega3, OmegaAOmegaB;

            sum_signal_K = (float2)(accum8.s0, accum8.s1);
            sum_variance_K = (float2)(accum8.s2, accum8.s3);
            omega1_A = (float2)(accum8.s4, accum8.s5);
            sum_count = accum8.s6;
            omega2_A = accum8.s7;
            // defined in silx:doubleword.cl
            omega1_B = fp_times_fp(coef, norm);
            omega2_B = dw_times_dw(omega1_B, omega1_B);
            sum_norm_1 = dw_plus_dw(omega1_B, omega1_A);
            sum_norm_2 = dw_plus_fp(omega2_B, omega2_A);

            x = dw_div_fp((float2)(signal, 0.0f), norm);


            // VV(AUb) = VV(A) + w_b^2*(b-<A>)*(b-<AUb>)
            delta = dw_plus_dw(dw_div_dw(sum_signal_K, omega1_A), -x);
            sum_signal_K = dw_plus_dw(sum_signal_K, fp_times_fp(coef, signal));
            delta2 = dw_plus_dw(dw_div_dw(sum_signal_K, sum_norm_1), -x);
            sum_variance_K = dw_plus_dw(sum_variance_K, dw_times_dw(omega2_B, dw_times_dw(delta2, delta)));

            // at the end as X_A is used in the variance XX_A

            sum_count = fma(coef, count, sum_count);
            accum8 = (float8)(sum_signal_K, sum_variance_K, sum_norm_1, sum_count, sum_norm_2.s0);
        }
    }
    return accum8;
}

/* _merge_poisson: Merge two partial dataset assuming a known error model like the poissonian model
 *
 * All processings are compensated for error accumulation
 *
 * :param here, there: two float8-accumulators containing (sumX_hi, sumX_lo, sumV_hi, sumV_lo, sumW_hi, sumW_lo, cnt_hi, cnt_lo)
 * :return: the updated accumulator8
 * */

static inline float8 _merge_poisson(float8 here,
                                    float8 there){
    float2 sum_signal_K, sum_variance_K, sum_omega;
    sum_signal_K = dw_plus_dw((float2)(here.s0, here.s1),
                                   (float2)(there.s0, there.s1));
    sum_variance_K = dw_plus_dw((float2)(here.s2, here.s3),
                                     (float2)(there.s2, there.s3));
    sum_omega = dw_plus_dw((float2)(here.s4, here.s5),
                           (float2)(there.s4, there.s5));
    return (float8)(sum_signal_K,
                    sum_variance_K,
                    sum_omega,
                    here.s6 + there.s6,
                    here.s7 + there.s7);
}

/* _merge_azimuthal: Merge two partial dataset considering the azimuthal regions
 *
 * All processings are compensated for error accumulation
 * See Schubert & Gretz 2018, especially Eq22.
 *
 * :param here, there: two float8-accumulators containing (sumX_hi, sumX_lo, sumV_hi, sumV_lo, sumW_hi, sumW_lo, cnt_hi, cnt_lo)
 * :return: the updated accumulator8
 *
 * Nota: here the variance of the pixel is not used, the variance is calculated from the distance from the pixel to the mean.
 * This algorithm is not numerically as stable as the `poisson` variant
 *
 * */

static inline float8 _merge_azimuthal(float8 here,
                                      float8 there){
    if (here.s7 <= 0.0f){ // Check the counter is not ZERO
        return there;
    }
    else if (there.s7 <= 0.0f){
        return here;
    }
    float2 sum_signal_K, sum_variance_K, V_A, V_B, delta, delta2, omega3, sum_norm_1, sum_norm_2, omega1_A, omega1_B;
    float sum_count, omega2_A, omega2_B;


    sum_variance_K = dw_plus_dw((float2)(here.s2, here.s3),
                                (float2)(there.s2, there.s3));

    // Omaga_A > Omaga_B (better numerical stability since algo is non symetrical)
    if (here.s4<there.s4){
        V_A = (float2)(there.s0, there.s1);
        V_B = (float2)(here.s0, here.s1);

        omega1_A = (float2)(there.s4, there.s5);
        omega1_B = (float2)(here.s4, here.s5);

        omega2_A = there.s7;
        omega2_B = here.s7;
    }
    else{
        V_A = (float2)(here.s0, here.s1);
        V_B = (float2)(there.s0, there.s1);

        omega1_A = (float2)(here.s4, here.s5);
        omega1_B = (float2)(there.s4, there.s5);

        omega2_A = here.s7;
        omega2_B = there.s7;
    }
    sum_signal_K = dw_plus_dw(V_A, V_B);

    sum_norm_1 = dw_plus_dw(omega1_A, omega1_B);
    sum_norm_2 = fp_plus_fp(omega2_A, omega2_B);
    sum_count = here.s6 + there.s6;

    // Add the cross-ensemble part
    // If one of the sub-ensemble is empty, the cross-region term is empty as well
    delta = dw_plus_dw(dw_times_dw(V_A, omega1_B), - dw_times_dw(V_B, omega1_A));
    //delta2 = dw_times_dw(delta, delta);
    delta2 = dw_times_fp(dw_times_dw(delta, delta), omega2_B);
    //omega3 = dw_times_dw(sum_norm_2, dw_times_dw(dw_times_dw(omega1_A,omega1_A), dw_times_dw(omega1_B,omega1_B)));
    omega3 = dw_times_dw(sum_norm_1, dw_times_dw(omega1_A,dw_times_dw(omega1_B,omega1_B)));
    sum_variance_K = dw_plus_dw(sum_variance_K, dw_div_dw(delta2, omega3));

    return (float8)(sum_signal_K, sum_variance_K, sum_norm_1, sum_count, sum_norm_2.s0);
}

/**
 * \brief CSRxVec4 OpenCL function for 1d azimuthal integration based on CSR matrix multiplication after normalization !
 *
 * The CSR matrix is represented by a set of 3 arrays (coefs, indices, indptr)
 *
 * @param data        float4 array in global memory storing the data as signal/variance/normalization/count.
 * @param coefs       float  array in global memory holding the coeficient part of the LUT
 * @param indices     integer array in global memory holding the corresponding column index of the coeficient
 * @param indptr      Integer array in global memory holding the index of the start of the nth line
 * @param error_model 0:diable, 1:variance, 2:poisson, 3:azimuthal 4:hybrid
 * @param super_sum   Local array of float8 of size WORKGROUP_SIZE: mandatory as a static function !
 * @return (sum_signal_main, sum_signal_neg, sum_variance_main,sum_variance_neg,
 *          sum_norm_main, sum_norm_neg, sum_count_main, sum_count_neg)
 *
 */
static inline float8 CSRxVec4(const   global  float4   *data,
                              const   global  float    *coefs,
                              const   global  int      *indices,
                              const   global  int      *indptr,
                              const           char     error_model,
                              volatile local  float8   *super_sum)
{
    // each workgroup (ideal size: 1 warp or slightly larger) is assigned to 1 bin
    int bin_num = get_group_id(0);
    int thread_id_loc = get_local_id(0);
    int active_threads = get_local_size(0);
    int2 bin_bounds = (int2) (indptr[bin_num], indptr[bin_num + 1]);
    int bin_size = bin_bounds.y - bin_bounds.x;
    // we use _K suffix to highlight it is float2 used for Kahan summation
    volatile float8 accum8 = (float8) (0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    int idx, k, j;

    for (int k=bin_bounds.x+thread_id_loc; k<bin_bounds.y; k+=active_threads) {
        float coef, signal, variance, norm, count;
        coef = (coefs == ZERO)?1.0f: coefs[k];
        idx = indices[k];
        float4 quatret = data[idx];
        if (error_model==AZIMUTHAL){
           accum8 = _accumulate_azimuthal(accum8, quatret, coef);
        }
        else{
           accum8 = _accumulate_poisson(accum8, quatret, coef);
        }
    };//for k
/*
 * parallel reduction
 */
    super_sum[thread_id_loc] = accum8;
    barrier(CLK_LOCAL_MEM_FENCE);
    while (active_threads > 1) {
        active_threads /= 2;
        if (thread_id_loc < active_threads) {
            if (error_model==AZIMUTHAL){
                accum8 = _merge_azimuthal(super_sum[thread_id_loc],
                                          super_sum[thread_id_loc + active_threads]);

            }//if azimuthal
            else{
                accum8 = _merge_poisson(super_sum[thread_id_loc],
                                        super_sum[thread_id_loc + active_threads]);

            }//if poisson
            super_sum[thread_id_loc] = accum8;
        } //therad active
        barrier(CLK_LOCAL_MEM_FENCE);
    } // loop reduction
    accum8 = super_sum[0];
    return accum8;
}


/**
 * \brief CSRxVec4_single OpenCL function for 1d azimuthal integration based on CSR matrix multiplication after normalization !
 * Single threaded version for Apple-CPU driver
 *
 * The CSR matrix is represented by a set of 3 arrays (coefs, indices, indptr)
 *
 * @param data        float4 array in global memory storing the data as signal/variance/normalization/count.
 * @param coefs       float  array in global memory holding the coeficient part of the LUT
 * @param indices     integer array in global memory holding the corresponding column index of the coeficient
 * @param indptr      Integer array in global memory holding the index of the start of the nth line
 * @param error_model 0:diable, 1:variance, 2:poisson, 3:azimuthal 4:hybrid
 * @return (sum_signal_main, sum_signal_neg, sum_variance_main,sum_variance_neg,
 *          sum_norm_main, sum_norm_neg, sum_count_main, sum_count_neg)
 *
 */
static inline float8 CSRxVec4_single(const   global  float4   *data,
                                     const   global  float    *coefs,
                                     const   global  int      *indices,
                                     const   global  int      *indptr,
                                     const           char     error_model)
{
    // each workgroup (size== 1) is assigned to 1 bin
    int bin_num = get_global_id(0);
    float8 accum8 = (float8)(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    for (int j=indptr[bin_num];j<indptr[bin_num+1];j++) {
        float coef = (coefs == ZERO)?1.0f:coefs[j];
        int idx = indices[j];
        float4 quatret = data[idx];
        if (error_model==AZIMUTHAL){
            accum8 = _accumulate_azimuthal(accum8, quatret, coef);
        }
        else{
            accum8 = _accumulate_poisson(accum8, quatret, coef);
        }
    }//for j
    return accum8;
} //end CSRxVec4_single function


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
 * @return number of pixel discarded in workgroup
 *
 */


static inline int _sigma_clip4(         global  float4   *data,
                                const   global  float    *coefs,
                                const   global  int      *indices,
                                const   global  int      *indptr,
                                                float    aver,
                                                float    cutoff,
                               volatile local   int      *counter){
    // each workgroup  is assigned to 1 bin
    int cnt, j, k, idx;
    counter[0] = 0;
    int bin_num = get_group_id(0);
    int thread_id_loc = get_local_id(0);
    int active_threads = get_local_size(0);
    int2 bin_bounds = (int2) (indptr[bin_num], indptr[bin_num + 1]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (j=bin_bounds.s0; j<bin_bounds.s1; j+=active_threads){
        k = j + thread_id_loc;
        if (k < bin_bounds.s1){
            idx = indices[k];
            float4 quatret = data[idx];
            // Check validity (on cnt, i.e. s3) and normalisation (in s2) value to avoid zero division
            if (isfinite(quatret.s3) && (quatret.s2 != 0.0f)){
                float signal = quatret.s0 / quatret.s2;
                if (fabs(signal-aver) > cutoff){
                    data[idx].s3 = NAN;
                    atomic_inc(counter);
                }
            } // if finite
        }// in bounds
    }// loop
    barrier(CLK_LOCAL_MEM_FENCE);
    return counter[0];
}// functions

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
 * @param shared      Buffer of shared memory of size WORKGROUP_SIZE * 4 * sizeof(float)
 */
kernel void
csr_integrate(  const   global  float   *weights,
                const   global  float   *coefs,
                const   global  int     *indices,
                const   global  int     *indptr,
                const           int      nbins,
                const           char     do_dummy,
                const           float    dummy,
                const           int      coef_power,
                        global  float   *sum_data,
                        global  float   *sum_count,
                        global  float   *merged,
                        local   float4  *shared)
{
    // each workgroup (ideal size: warp) is assigned to 1 bin
    int bin_num = get_group_id(0);
    int thread_id_loc = get_local_id(0);
    int active_threads = get_local_size(0);

    if (bin_num<nbins){
        int2 bin_bounds = (int2) (indptr[bin_num], indptr[bin_num + 1]);
        int bin_size = bin_bounds.y - bin_bounds.x;
        // we use _K suffix to highlight it is float2 used for Kahan summation
        float2 sum_data_K = (float2)(0.0f, 0.0f);
        float2 sum_count_K = (float2)(0.0f, 0.0f);
        float coef, coefp, data;
        int idx;

        for (int k=bin_bounds.x+thread_id_loc; k<bin_bounds.y; k+=active_threads) {
           coef = (coefs == ZERO)?1.0f:coefs[k];;
           idx = indices[k];
           data = weights[idx];
           if  (! isfinite(data))
               continue;

           if( (!do_dummy) || (data!=dummy) )
           {
               // defined in silx:doubleword.cl
               sum_data_K = dw_plus_fp(sum_data_K, ((coef_power == 2) ? coef*coef: coef) * data);
               sum_count_K = dw_plus_fp(sum_count_K, coef);
           };//end if dummy
        }//for k

        // parallel reduction

        int index;

        if (bin_size < active_threads) {
            if (thread_id_loc < bin_size) {
                shared[thread_id_loc] = (float4)(sum_data_K,sum_count_K);
            }
            else {
                shared[thread_id_loc] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        else {
            shared[thread_id_loc] = (float4)(sum_data_K, sum_count_K);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        while (active_threads > 1) {
            active_threads /= 2 ;
            if (thread_id_loc < active_threads) {
                index = thread_id_loc + active_threads;
                float4 value4here =  shared[thread_id_loc];
                float4 value4there =  shared[index];
                shared[thread_id_loc] = (float4)(dw_plus_dw(value4here.lo, value4there.lo),
                                                 dw_plus_dw(value4here.hi, value4there.hi));
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (thread_id_loc == 0) {
            //Only thread 0 works
            sum_data[bin_num] = shared[0].s0;
            sum_count[bin_num] = shared[0].s2;
            if (sum_count[bin_num] > 0.0f)
                merged[bin_num] =  sum_data[bin_num] / sum_count[bin_num];
            else
                merged[bin_num] = dummy;
        } //end thread 0

    } // valid bin
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
                       const           int      nbins,
                       const           char     do_dummy,
                       const           float    dummy,
                       const           int      coef_power,
                               global  float   *sum_data,
                               global  float   *sum_count,
                               global  float   *merged)
{
    // each workgroup of size does not matter, the smallest reasonably possible: highly divergent
    int bin_num = get_global_id(0);
    if (bin_num<nbins){
        // we use _K suffix to highlight it is float2 used for Kahan summation
        float2 sum_data_K = (float2)(0.0f, 0.0f);
        float2 sum_count_K = (float2)(0.0f, 0.0f);
        const float epsilon = 1e-10f;
        float coef, data;
        int idx, j;

        for (j=indptr[bin_num];j<indptr[bin_num+1];j++) {
            coef = (coefs == ZERO)?1.0f:coefs[j];
            idx = indices[j];
            data = weights[idx];

            if( isfinite(data) && ((!do_dummy) || (data!=dummy))) {
                // defined in silx:doubleword.cl
                sum_data_K = dw_plus_fp(sum_data_K, ((coef_power == 2) ? coef*coef: coef) * data);
                sum_count_K = dw_plus_fp(sum_count_K, coef);
            }//end if dummy
        }//for j
        sum_data[bin_num] = sum_data_K.s0;
        sum_count[bin_num] = sum_count_K.s0;
        if (sum_count_K.s0 > epsilon)
            merged[bin_num] =  sum_data_K.s0 / sum_count_K.s0;
        else
            merged[bin_num] = dummy;
    } // if valid bin
}//end kernel


/**
 *  \brief csr_integrate4a: Performs 1d azimuthal integration based on CSR sparse matrix multiplication on preprocessed data
 *
 *
 * @param weights     Float pointer to global memory storing the input image after preprocessing. Contains (signal, variance, normalisation, count) as float4.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param indices     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param indptr      Integer pointer to global memory holding the pointers to the coefs and indices for the CSR matrix
 * @param empty       Float: value for bad pixels, NaN is a good guess
 * @param error_model 0:diable, 1:variance, 2:poisson, 3:azimuthal 4:hybrid
 * @param summed      Float pointer to the output with all 4 histograms in Kahan representation
 * @param averint     Float pointer to the output 1D array with the averaged signal
 * @param stdevpix    Float pointer to the output 1D array with the propagated error (std)
 * @param stdevpix    Float pointer to the output 1D array with the propagated error (sem)
 * @param shared      Buffer of shared memory of size WORKGROUP_SIZE * 8 * sizeof(float)
 *
 */
kernel void
csr_integrate4(  const   global  float4  *weights,
                 const   global  float   *coefs,
                 const   global  int     *indices,
                 const   global  int     *indptr,
                 const           int      nbins,
                 const           float    empty,
                 const           char     error_model,
                         global  float8  *summed,
                         global  float   *averint,
                         global  float   *stdevpix,
                         global  float   *stderrmean,
                         local   float8  *shared)
{
    int bin_num = get_group_id(0);
    if (bin_num<nbins){
        float8 result = CSRxVec4(weights, coefs, indices, indptr, error_model, shared);
        if (get_local_id(0)==0) {
            summed[bin_num] = result;
            if (result.s6 > 0.0f) {
                averint[bin_num] =  result.s0 / result.s4;
                stdevpix[bin_num] = sqrt(result.s2 / result.s7);
                stderrmean[bin_num] = sqrt(result.s2) / result.s4;
            }
            else {
                averint[bin_num] = empty;
                stderrmean[bin_num] = empty;
                stdevpix[bin_num] = empty;
            } //end else
        } // end if thread0
    } // if valid bin
};//end kernel


/**
 * \brief Performs 1d azimuthal integration based on CSR sparse matrix multiplication on preprocessed data
 *  Unlike the former kernel, it works with a any workgroup size, especialy  ONE (tailor made form MacOS bug)
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param indices     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param indptr      Integer pointer to global memory holding the pointers to the coefs and indices for the CSR matrix
 * @param empty       Float: value for bad pixels, NaN is a good guess
 * @param error_model 0:diable, 1:variance, 2:poisson, 3:azimuthal 4:hybrid
 * @param summed      Float pointer to the output with all 4 histograms in Kahan representation
 * @param averint     Float pointer to the output 1D array with the averaged signal
 * @param stdevpix    Float pointer to the output 1D array with the propagated error (std)
 * @param stdevpix    Float pointer to the output 1D array with the propagated error (sem)
 * @param shared      Buffer of shared memory, unused
 *
 */
kernel void
csr_integrate4_single(  const   global  float4  *weights,
                        const   global  float   *coefs,
                        const   global  int     *indices,
                        const   global  int     *indptr,
                        const           int      nbins,
                        const           float    empty,
                        const           char     error_model,
                                global  float8  *summed,
                                global  float   *averint,
                                global  float   *stdevpix,
                                global  float   *stderrmean,
                                local   float8  *shared)
{
    // each workgroup of size==1 is assinged to 1 bin
    int bin_num = get_global_id(0);
    if (bin_num<nbins){
        float8 accum8 = CSRxVec4_single(weights, coefs, indices, indptr, error_model);
        summed[bin_num] = accum8;
        if (accum8.s6 > 0.0f) {
            averint[bin_num] = accum8.s0 / accum8.s4;
            stdevpix[bin_num] = sqrt(accum8.s2 / accum8.s7);
            stderrmean[bin_num] = sqrt(accum8.s2) / accum8.s4;
        }
        else {
            averint[bin_num] = empty;
            stderrmean[bin_num] = empty;
            stdevpix[bin_num] = empty;
        }
    } // if validd bin
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
 * @param error_model 0:disable, 1:variance, 2:poisson, 3:azimuthal, 4:hybrid
 * @param summed      contains all the data
 * @param averint     Average signal
 * @param stdevpix    Float pointer to the output 1D array with the propagated error (std)
 * @param stdevpix    Float pointer to the output 1D array with the propagated error (sem)
 * @param shared      Buffer of shared memory of size WORKGROUP_SIZE * 8 * sizeof(float)
 */

kernel void
csr_sigma_clip4(          global  float4  *data4,
                  const   global  float   *coefs,
                  const   global  int     *indices,
                  const   global  int     *indptr,
                  const           float    cutoff,
                  const           int      cycle,
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
    float aver, std;
    int cnt, nbpix;
    char curr_error_model=error_model;
    volatile local int counter[1];
    float8 result;

    // Number of pixel in this bin.
    // Used to calulate the minimum reasonnable cut-off according to Chauvenet criterion.
    // 3 is the smallest integer above sqrt(2pi) -> math domain error
    nbpix = max(3, indptr[bin_num + 1] - indptr[bin_num]);

    // special case: no cycle and hybrid mode, should be best handled at host side
    if (error_model==HYBRID){
        if (cycle==0)
            curr_error_model = POISSON;
        else
            curr_error_model = AZIMUTHAL;
    }
//    if ((get_local_id(0) == 0) && (bin_num==400))
//        printf("%d em %d cem %d\n", bin_num, error_model, curr_error_model);

    // first calculation of azimuthal integration to initialize aver & std
    result = (wg==1? CSRxVec4_single(data4, coefs, indices, indptr, curr_error_model):
                            CSRxVec4(data4, coefs, indices, indptr, curr_error_model, shared8));

    if (result.s6 > 0.0f){
        aver = result.s0 / result.s4;
        std = sqrt(result.s2 / result.s7);
    }
    else {
        aver = NAN;
        std = NAN;
    }

    for (int i=0; i<cycle; i++) {
        if ( ! (isfinite(aver) && isfinite(std))){
            break;
        }

        float chauvenet_cutoff = max(cutoff, sqrt(2.0f*log((float)nbpix/sqrt(2.0f*M_PI_F))));
        cnt = _sigma_clip4(data4, coefs, indices, indptr, aver, std *chauvenet_cutoff, counter);

//        if ((get_local_id(0) == 0) && (bin_num==400))
//            printf("bin %d i %d cycle %d cnt %d em %d cem %d\n", bin_num, i, cycle, cnt, error_model, curr_error_model);

        if ((cnt == 0) && (error_model!=HYBRID)){
            break;
        }

        nbpix = max(3, nbpix - cnt);
        if ((error_model==HYBRID) && ((cycle==i+1) || (cnt==0))) {
                curr_error_model = POISSON;
//                if ((get_local_id(0) == 0) && (bin_num==400))
//                printf("Change model em %d cem %d\n",error_model, curr_error_model);
        }
        result = (wg==1? CSRxVec4_single(data4, coefs, indices, indptr, curr_error_model):
                                CSRxVec4(data4, coefs, indices, indptr, curr_error_model, shared8));

        if (result.s6 > 0.0f) {
            aver = result.s0 / result.s4;
            std = sqrt(result.s2 / result.s7);
        }
        else {
            aver = NAN;
            std = NAN;
            break;
        }
        if ((cnt == 0) && (error_model==HYBRID)) {
            break;
        }
    } // clipping loop
//    if ((get_local_id(0) == 0) && (bin_num==400))
//        printf("end %d em %d cem %d\n\n", bin_num, error_model, curr_error_model);

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
