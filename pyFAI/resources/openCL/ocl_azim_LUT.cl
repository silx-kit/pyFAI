/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Kernel with full pixel-split using a LUT
 *
 *
 *   Copyright (C) 2012-2023 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 20/01/2023
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
                sum_data_K = dw_plus_fp(sum_data_K, ((coef_power == 2) ? coef*coef: coef) * data);
                sum_count_K = dw_plus_fp(sum_count_K, coef);
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


/**
 * \brief OpenCL function for 1d azimuthal integration based on LUT matrix multiplication after normalization !
 *
 * @param data        float4 array in global memory storing the data as signal/variance/normalization/count.
 * @param lut         lut_point 2d array in global memory holding the coeficient part of the LUT
 * @return (sum_signal_main, sum_signal_neg, sum_variance_main, sum_variance_neg,
 *          sum_norm_main, sum_norm_neg, sum_count_main, sum_count_neg)
 *
 */
float8 static inline LUTxVec4(const   global  float4  *data,
                              const   global    struct lut_point_t *lut)
{

	int bin_num, k, j;
	bin_num= get_global_id(0);
    float2 sum_signal_K = (float2)(0.0f, 0.0f);
    float2 sum_variance_K = (float2)(0.0f, 0.0f);
    float2 sum_norm_1 = (float2)(0.0f, 0.0f);
    float sum_count = 0.0f;
    float sum_norm_2 = 0.0f;
    if(bin_num < NBINS){
        for (j=0;j<NLUT;j++){
            if (ON_CPU){
                //On CPU best performances are obtained  when each single thread reads adjacent memory
                k = bin_num*NLUT+j;
            }
            else{
                //On GPU best performances are obtained  when threads are reading adjacent memory
                k = j*NBINS+bin_num;
            }

            int idx = lut[k].idx;
            float coef = lut[k].coef;
            if((coef != 0.0f) && (idx >= 0)){//
                   float4 quatret = data[idx];
                   float signal = quatret.s0;
                   float variance = quatret.s1;
                   float norm = quatret.s2;
                   float count = quatret.s3;
                   if (isfinite(signal) && isfinite(variance) && isfinite(norm) && (count > 0))
                   {
                       // defined in kahan.cl
                       sum_signal_K = dw_plus_fp(sum_signal_K, coef * signal);
                       sum_variance_K = dw_plus_fp(sum_variance_K, coef * coef * variance);
                       float w = coef * norm;
                       sum_norm_1 = dw_plus_fp(sum_norm_1, w);
                       sum_count = fma(coef, count, sum_count);
                       sum_norm_2 = fma(w, w, sum_norm_2);
                   };//end if finite
            } //end if valid point
        }//end for j
    }// if bin_num
    return (float8)(sum_signal_K, sum_variance_K, sum_norm_1, sum_count, sum_norm_2);
}//end function

/**
 * \brief Performs 1d azimuthal integration based on LUT sparse matrix multiplication on preprocessed data
 *
* @param weights      Float pointer to global memory storing the input image after preprocessing. Contains (signal, variance, normalisation, count) as float4.
 * @param lut         Pointer to an 2D-array of (unsigned integers,float) containing the index of input pixels and the fraction of pixel going to the bin
 * @param empty       value given for empty bins, NaN is a good guess
 * @param summed      Pointer to the output 1D array with all the histograms: (sum_signal_Kahan, sum_variance_Kahan, sum_norm_Kahan, sum_count_Kahan)
 * @param averint     Float pointer to the output 1D array with the averaged signal
 * @param stderr      Float pointer to the output 1D array with the propagated error
 *
 */
kernel void
lut_integrate4( const   global  float4  *weights,
                const   global  struct  lut_point_t *lut,
                const           float   empty,
                        global  float8  *summed,
                        global  float   *averint,
                        global  float   *stdevpix,
                        global  float   *stderr)
{
    int bin_num = get_global_id(0);
    if(bin_num < NBINS){
    	float8 result = LUTxVec4(weights, lut);
		summed[bin_num] = result;
		if (result.s4 > 0.0f) {
				averint[bin_num] =  result.s0 / result.s4;
				stdevpix[bin_num] = sqrt(result.s2 / result.s7);
				stderr[bin_num] = sqrt(result.s2) / result.s4;
		}
		else {
				averint[bin_num] = empty;
				stdevpix[bin_num] = empty;
				stderr[bin_num] = empty;
		} //end else
    }
}//end kernel
