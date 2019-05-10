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


inline void atomic_add_global_float(volatile global float *addr, float val)
{
   union {
       uint  u32;
       float f32;
   } next, expected, current;
   current.f32    = *addr;
   do {
       expected.f32 = current.f32;
       next.f32     = expected.f32 + val;
       current.u32  = atomic_cmpxchg( (volatile global uint *)addr,
                                      expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
}

#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

inline void atomic_add_global_kahan(volatile global float2 *addr, float val)
{
   union {
       unsigned long u64;
       float2        f64;
   } next, expected, current;
   current.f64    = *addr;
   do {
       expected.f64 = current.f64;
       next.f64     = kahan_sum(expected.f64, val);
       current.u64  = atomic_cmpxchg( (volatile global unsigned long *)addr,
                                      expected.u64, next.u64);
   } while( current.u64 != expected.u64 );
}
#else
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable

inline void atomic_add_global_kahan(volatile global float2 *addr, float val)
{
   union {
       uint2   u64;
       float2 f64;
   } next, expected, current;
   current.f64    = *addr;
   do {
       expected.f64 = current.f64;
       next.f64     = kahan_sum(expected.f64, val);
       current.u64.s0  = atomic_cmpxchg( (volatile global uint *)addr,
                                        expected.u64.s0, next.u64.s0);
   } while( current.u64.s0 != expected.u64.s0 );
}
#endif

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
                         unsigned int size,
                         unsigned int nbins,
                         float mini,
                         float maxi)
{
    unsigned int idx = get_global_id(0);
    if (idx<size)
    {// pixel in the image
        float pvalue = position[idx];
        if ((pvalue>=mini) && (pvalue<maxi))
        {// position in range
            unsigned int target = (unsigned int) (nbins * (pvalue - mini) / (maxi-mini));
            atomic_add_global_float(&histo[target], weight[idx]);
        }// else discard value
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
 * - histo_cnt: contains the resulting histogram for the pixel count -> as integrers
 * - size: of the image/weights/position
 * - nbins: size of histograms
 * - mini: lower bound of the histogram
 * - maxi: upper boubd of the histogram (excluded)
 *
 * This is a 1D histogram
 */

kernel void histogram_1d_preproc(global float* position,
                                 global float4* weights,
                                 global float2* histo_sig,
                                 global float2* histo_var,
                                 global float2* histo_nrm,
                                 global unsigned int* histo_cnt,
                                 int size,
                                 int nbins,
                                 float mini,
                                 float maxi)
{
    unsigned int idx = get_global_id(0);
    if (idx<size)
    {// pixel in the image
        float pvalue = position[idx];
        if ((pvalue>=mini)&&(pvalue<maxi))
        { // position in range
            unsigned int target = (unsigned int) (nbins * (position[idx] - mini) / (maxi-mini));
            float4 value = weights[idx];
            atomic_add_global_kahan(&histo_sig[target], value.s0);
            atomic_add_global_kahan(&histo_var[target], value.s1);
            atomic_add_global_kahan(&histo_nrm[target], value.s2);
            atomic_add(&histo_cnt[target], (unsigned int)(value.s3 + 0.5f));
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

kernel void histogram_2d_preproc(global float * radial,
                                 global float * azimuthal,
                                 global float4 * weights,
                                 global float2 * histo_sig,
                                 global float2 * histo_var,
                                 global float2 * histo_nrm,
                                 global unsigned int * histo_cnt,
                                 unsigned int size,
                                 unsigned int nbins_rad,
                                 unsigned int nbins_azim,
                                 float mini_rad,
                                 float maxi_rad,
                                 float mini_azim,
                                 float maxi_azim)
{
    unsigned int idx = get_global_id(0);
    if (idx<size)
    {// we are in the image
        float rvalue = radial[idx];
        float avalue = azimuthal[idx];
        if ((rvalue>=mini_rad)&&(rvalue<maxi_azim)&&
            (avalue>mini_azim)&&(avalue<maxi_azim))
        { //pixel position is the range
            unsigned int target_rad = (unsigned int) (nbins_rad * (radial[idx] - mini_rad) / (maxi_rad-mini_rad));
            unsigned int target_azim = (unsigned int) (nbins_azim * (azimuthal[idx] - mini_azim) / (maxi_azim-mini_azim));
            unsigned int target = target_rad * nbins_azim + target_azim;
            float4 value = weights[idx];
            atomic_add_global_kahan(&histo_sig[target], value.s0);
            atomic_add_global_kahan(&histo_var[target], value.s1);
            atomic_add_global_kahan(&histo_nrm[target], value.s2);
            atomic_add(&histo_cnt[target], (unsigned int) (value.s3 + 0.5f));
        }
    }
}


/**
 * \brief Sets the values of 4 float output arrays to zero.
 *
 * Gridsize = size of arrays + padding.
 *
 * - array0: float2 array
 * - array1: float2 array
 * - array2: float2 array
 * - array3: integer array ... global memory with the outData array
 * - nbins :
 */
kernel void
memset_histograms(global float2 *histo_sig,
                  global float2 *histo_var,
                  global float2 *histo_nrm,
                  global int *histo_cnt,
                  unsigned int nbins)
{
  unsigned int idx = get_global_id(0);
  //Global memory guard for padding
  if (idx < nbins)
  {
     histo_sig[idx] = (float2)(0.0f, 0.0f);
     histo_var[idx] = (float2)(0.0f, 0.0f);
     histo_nrm[idx] = (float2)(0.0f, 0.0f);
     histo_cnt[idx] = 0;
  }
}

/*
 * \brief Post-process the various histogram arrays to export intensity and errors
 *
 * param:
 *  - histo_sig: sum of signal among bins
 *  - histo_var: sum of variance among bins
 *  - histo_nrm: sum of normalization amoung bins
 *  - histo_cnt: number of pixels per bins
 *  - nbin: number of bins
 *  - empty: value for empty bins (i.e. nrm=0 or cnt=0)
 *  - intensities: result array with sig/norm
 *  - errors: result array with standard deviation
 */

kernel void
histogram_postproc(global float2 * histo_sig,
                   global float2 * histo_var,
                   global float2 * histo_nrm,
                   global unsigned int * histo_cnt,
                   unsigned int nbins,
                   float empty,
                   global float *intensities,
                   global float *errors)
{
    unsigned int idx = get_global_id(0);
    //Global memory guard for padding
    if (idx < nbins)
    {
        float nrm = histo_nrm[idx].s0 + histo_nrm[idx].s1;
        if ((histo_cnt[idx]==0) || (nrm == 0.0f))
        {
            intensities[idx] = empty;
            errors[idx] = empty;
        }
        else
        {
            intensities[idx] = (histo_sig[idx].s0 + histo_sig[idx].s1) / nrm;
            errors[idx] = sqrt(histo_var[idx].s0 + histo_var[idx].s1)/ nrm;
        }
    }
}
