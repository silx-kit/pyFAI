/*
 *   Project: Azimuthal regrouping OpenCL kernel for PyFAI.
 *            1D & 2D histogram based on atomic adds
 *
 *
 *   Copyright (C) 2014-2018 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 2019-03-15
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
.
 */

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
   union {
       unsigned int u32;
       float        f32;
   } next, expected, current;
   current.f32    = *addr;
   do {
       expected.f32 = current.f32;
       next.f32     = expected.f32 + val;
       current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr,
                                      expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
}


inline void atomicAdd_g_kahan(volatile global float2 *addr, float val)
{
   union {
       unsigned long u64;
       float2        f64;
   } next, expected, current;
   current.f64    = *addr;
   do {
       expected.f64 = current.f64;
       next.f64     = kahan_sum(expected.f64, val);
       current.u64  = atomic_cmpxchg( (volatile __global unsigned long *)addr,
                                      expected.u64, next.u64);
   } while( current.u64 != expected.u64 );
}

/**
 * \brief Calculate the 1D weighted histogram of positions
 *
 * - position: array of position
 * - weight: array of weights
 * - histo: contains the resulting histogram
 * - size: of the image/weights/position
 * - nbins: size of histo
 * - mini: lower bound of the histogram
 * - maxi: upper boubd of the histogram (excluded)
 *
 * This is a 1D histogram
 */


kernel void histogram_1d(global float* position,
                         global float* weight,
                         global float* histo,
                         int size,
                         int nbins,
                         float mini,
                         float maxi)
{
    int id = get_global_id(0);
    if (id<size)
    {
        int target = (int) (nbins * (position[id] - mini) / (maxi-mini));
        atomicAdd_g_f(&histo[target], weight[id]);
    }
    return;
}

/**
 * \brief Calculate the 1D weighted histogram of positions for preprocessed data
 *
 * - position: array of position
 * - weight: array of weights containing: signal, variance, normalization, count
 * - histo_sig: contains the resulting histogram for the signal
 * - histo_var: contains the resulting histogram for the variance
 * - histo_nrm: contains the resulting histogram for the normalization
 * - histo_cnt: contains the resulting histogram for the pixel count
 * - size: of the image/weights/position
 * - nbins: size of histograms
 * - mini: lower bound of the histogram
 * - maxi: upper boubd of the histogram (excluded)
 *
 * This is a 1D histogram
 */

kernel void histogram_1d_preproc(global float* position,
                                 global float4* weight,
                                 global float2* histo_sig,
                                 global float2* histo_var,
                                 global float2* histo_nrm,
                                 global float2* histo_cnt,
                                 int size,
                                 int nbins,
                                 float mini,
                                 float maxi)
{
    size_t id = get_global_id(0);
    if (id<size)
    {// pixel in the image
        int target = (int) (nbins * (position[id] - mini) / (maxi-mini));
        if (target >= 0) && (target < nbins)
        {
            float4 value = weight[id];
            atomicAdd_g_kahan(&histo_sig[target], value.s0);
            atomicAdd_g_kahan(&histo_var[target], value.s1);
            atomicAdd_g_kahan(&histo_nrm[target], value.s2);
            atomicAdd_g_kahan(&histo_cnt[target], value.s3);
        } // else discard value
    }
    return;
}

/**
 * \brief Calculate the 2D weighted histogram of positions for preprocessed data
 *
 * - radial: array of position in the radial direction
 * - azimuthal: array of position in the azimuthal direction
 * - weight: array of weights containing: signal, variance, normalization, count
 * - histo_sig: contains the resulting histogram for the signal
 * - histo_var: contains the resulting histogram for the variance
 * - histo_nrm: contains the resulting histogram for the normalization
 * - histo_cnt: contains the resulting histogram for the pixel count
 * - size: of the image/weights/positions
 * - nbins_radial: size of histograms
 * - mini: lower bound of the histogram
 * - maxi: upper boubd of the histogram (excluded)
 *
 * This is a 1D histogram
 */

kernel void histogram_2d_preproc(global float* radial,
                                 global float* azimuthal,
                                 global float4* weight,
                                 global float2* histo_sig,
                                 global float2* histo_var,
                                 global float2* histo_nrm,
                                 global float2* histo_cnt,
                                 int size,
                                 int nbins_rad,
                                 int nbins_azim,
                                 float mini_rad,
                                 float maxi_rad,
                                 float mini_azim,
                                 float maxi_azim,)
{
    int id = get_global_id(0);
    if (id<size)
    {// we are in the image
        int target_rad = (int) (nbins_rad * (radial[id] - mini_rad) / (maxi_rad-mini_rad));
        if ((target_rad >= 0) && (target_rad < nbins_rad))
        {
            int target_azim = (int) (nbins_azim * (azimuthal[id] - mini_azim) / (maxi_azim-mini_azim));
            if ((target_azim >= 0) && (target_azim < nbins_azim))
            {
                int target = ...
            }

            atomicAdd_g_kahan(&histo_sig[target], weight[id].s0);
            atomicAdd_g_kahan(&histo_var[target], weight[id].s1);
            atomicAdd_g_kahan(&histo_nrm[target], weight[id].s2);
            atomicAdd_g_kahan(&histo_cnt[target], weight[id].s3);

        }
    }
    return;
}


