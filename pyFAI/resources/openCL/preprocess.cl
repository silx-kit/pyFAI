/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Preprocessing program
 *
 *
 *   Copyright (C) 2012 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 04/09/2014
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * \file
 *
 * \brief OpenCL kernels for image array casting, array mem-setting and normalizing
 *
 * Constant to be provided at build time:
 *   NIMAGE: size of the image
 *   NBINS:  number of output bins for histograms
 *
 */

#include "for_eclipse.h"

/**
 * \brief cast values of an array of int8 into a float output array.
 *
 * - array_s8: Pointer to global memory with the input data as signed8 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
s8_to_float(__global char  *array_s8,
            __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_s8[i];
}


/**
 * \brief cast values of an array of uint8 into a float output array.
 *
 * - array_u8: Pointer to global memory with the input data as unsigned8 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u8_to_float(__global unsigned char  *array_u8,
            __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_u8[i];
}


/**
 * \brief cast values of an array of int16 into a float output array.
 *
 * - array_s16: Pointer to global memory with the input data as signed16 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
s16_to_float(__global short *array_s16,
             __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_s16[i];
}


/**
 * \brief cast values of an array of uint16 into a float output array.
 *
 * - array_u16: Pointer to global memory with the input data as unsigned16 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u16_to_float(__global unsigned short  *array_u16,
             __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_u16[i];
}

/**
 * \brief cast values of an array of uint32 into a float output array.
 *
 * - array_u32: Pointer to global memory with the input data as unsigned32 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u32_to_float(__global unsigned int  *array_u32,
             __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_u32[i];
}

/**
 * \brief convert values of an array of int32 into a float output array.
 *
 * - array_int:  Pointer to global memory with the data as unsigned32 array
 * - array_float:  Pointer to global memory with the data float array
 */
__kernel void
s32_to_float(__global int  *array_int,
             __global float  *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)(array_int[i]);
}


/**
 * \brief Sets the values of 3 float output arrays to zero.
 *
 * Gridsize = size of arrays + padding.
 *
 * - array0: float Pointer to global memory with the outMerge array
 * - array1: float Pointer to global memory with the outCount array
 * - array2: float Pointer to global memory with the outData array
 */
__kernel void
memset_out(__global float *array0,
           __global float *array1,
           __global float *array2
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NBINS)
  {
    array0[i] = 0.0f;
    array1[i] = 0.0f;
    array2[i] = 0.0f;
  }
}

// Functions to be called from an actual kernel.

static float3 _preproc3(const __global float  *image,
                        const __global float  *variance,
                        const          char   do_mask,
                        const __global char   *mask,
                        const          char   do_dark,
                        const __global float  *dark,
                        const          char   do_dark_variance,
                        const __global float  *dark_variance,
                        const          char   do_flat,
                        const __global float  *flat,
                        const          char   do_solidangle,
                        const __global float  *solidangle,
                        const          char   do_polarization,
                        const __global float  *polarization,
                        const          char   do_absorption,
                        const __global float  *absorption,
                        const          char   do_dummy,
                        const          float  dummy,
                        const          float  delta_dummy,
                        const          float  normalization_factor)
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        if (do_mask && (!mask[i]))
        {
            result.s0 = image[i];
            if (variance != 0)
                result.s1 = variance[i];
            result.s2 = normalization_factor;
            if ( (!do_dummy)
                    ||((delta_dummy!=0.0f) && (fabs(result.s0-dummy) > delta_dummy))
                    ||((delta_dummy==0.0f) && (result.s0!=dummy)))
            {
                if (do_dark)
                    result.s0 -= dark[i];
                if (do_dark_variance)
                    result.s1 += dark_variance[i];
                if (do_flat)
                    result.s2 *= flat[i];
                if (do_solidangle)
                    result.s2 *= solidangle[i];
                if (do_polarization)
                    result.s2 *= polarization[i];
                if (do_absorption)
                    result.s2 *= absorption[i];
                if (isnan(result.s0) || isnan(result.s1) || isnan(result.s2) || (result.s2 <= 0))
                    result = (float3)(0.0, 0.0, 0.0);
            }
            else
            {
                result = (float3)(0.0, 0.0, 0.0);
            }//end if do_dummy
        } // end if mask
    };//end if NIMAGE
    return result;
};//end function


/**
 * \brief Performs the normalization of input image by dark subtraction,
 *        flatfield, solid angle, polarization and absorption division.
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction
 *  - Solid angle correction (division)
 *  - polarization correction (division)
 *  - flat fiels correction (division)
 * Corrections are made in place unless the pixel is dummy.
 * Dummy pixels are left untouched so that they remain dummy
 *
 * - image           Float pointer to global memory storing the input image.
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall flat-field correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_dummy        Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy           Float: value for bad pixels
 * - delta_dummy     Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
**/

__kernel void
corrections(      __global float  *image,
            const          char   do_mask,
            const __global char   *mask,
            const          char   do_dark,
            const __global float  *dark,
            const          char   do_flat,
            const __global float  *flat,
            const          char   do_solidangle,
            const __global float  *solidangle,
            const          char   do_polarization,
            const __global float  *polarization,
			const          char   do_absorption,
			const __global float  *absorption,
            const          char   do_dummy,
            const          float  dummy,
            const          float  delta_dummy,
            const          float  normalization_factor
            )
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        result = _preproc3(image,
                            0,
                            0,
                            0,
                            do_dark,
                            dark,
                            0,
                            0,
                            do_flat,
                            flat,
                            do_solidangle,
                            solidangle,
                            do_polarization,
                            polarization,
                            do_absorption,
                            absorption,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor);
        if ((result.s0 != 0.0) && (result.s2 > 0.0))
            image[i] = result.s0/result.s2;
        else
            image[i] = dummy;
    };//end if NIMAGE

};//end kernel


/**
 * \brief Performs Normalization of input image with float2 output (num,denom)
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction for the data
 *  - Solid angle correction (denominator)
 *  - polarization correction (denominator)
 *  - flat fiels correction (denominator)
 *
 * Corrections are made out of place.
 * Dummy pixels set both the numerator and denominator to 0
 *
 * - image           Float pointer to global memory storing the input image.
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall flat-field correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_dummy          Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy             Float: value for bad pixels
 * - delta_dummy       Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
 *
**/
__kernel void
corrections2(const __global float  *image,
             const          char   do_dark,
             const __global float  *dark,
             const          char   do_flat,
             const __global float  *flat,
             const          char   do_solidangle,
             const __global float  *solidangle,
             const          char   do_polarization,
             const __global float  *polarization,
             const          char   do_absorption,
             const __global float  *absorption,
             const          char   do_dummy,
             const          float  dummy,
             const          float  delta_dummy,
             const          float  normalization_factor,
                   __global float2  *output
            )
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        result = _preproc3(image,
                            0,
                            0,
                            0,
                            do_dark,
                            dark,
                            0,
                            0,
                            do_flat,
                            flat,
                            do_solidangle,
                            solidangle,
                            do_polarization,
                            polarization,
                            do_absorption,
                            absorption,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor);
        output[i] = (float2)(result.s0, result.s2);
    };//end if NIMAGE
};//end kernel

/**
 * \brief Performs Normalization of input image with float3 output (signal, variance, normalization) assuming poissonian signal
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction for the data
 *  - Solid angle correction (denominator)
 *  - polarization correction (denominator)
 *  - flat fiels correction (denominator)
 *
 * Corrections are made out of place.
 * Dummy pixels set both the numerator and denominator to 0
 *
 * - image           Float pointer to global memory storing the input image.
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall flat-field correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_dummy          Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy             Float: value for bad pixels
 * - delta_dummy       Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
 *
**/
__kernel void
corrections3Poisson( const __global float  *image,
                     const          char   do_dark,
                     const __global float  *dark,
                     const          char   do_flat,
                     const __global float  *flat,
                     const          char   do_solidangle,
                     const __global float  *solidangle,
                     const          char   do_polarization,
                     const __global float  *polarization,
                     const          char   do_absorption,
                     const __global float  *absorption,
                     const          char   do_dummy,
                     const          float  dummy,
                     const          float  delta_dummy,
                     const          float  normalization_factor,
                           __global float3  *output
            )
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        result = _preproc3(image,
                           image,
                            0,
                            0,
                            do_dark,
                            dark,
                            do_dark,
                            dark,
                            do_flat,
                            flat,
                            do_solidangle,
                            solidangle,
                            do_polarization,
                            polarization,
                            do_absorption,
                            absorption,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor);
        output[i] = result;
    };//end if NIMAGE
};//end kernel


/**
 * \brief Performs Normalization of input image with float3 output (signal, variance, normalization)
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction for the data
 *  - Solid angle correction (division)
 *  - polarization correction (division)
 *  - flat fiels correction (division)
 * Corrections are made in place unless the pixel is dummy.
 * Dummy pixels are left untouched so that they remain dummy
 *
 * - image              Float pointer to global memory storing the input image.
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall flat-field correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_dummy          Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy             Float: value for bad pixels
 * - delta_dummy       Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
 *
**/

__kernel void
corrections3(const __global float  *image,
             const __global float  *variance,
             const          char   do_dark,
             const __global float  *dark,
             const          char   do_dark_variance,
             const __global float  *dark_variance,
             const          char   do_flat,
             const __global float  *flat,
             const          char   do_solidangle,
             const __global float  *solidangle,
             const          char   do_polarization,
             const __global float  *polarization,
             const          char   do_absorption,
             const __global float  *absorption,
             const          char   do_dummy,
             const          float  dummy,
             const          float  delta_dummy,
             const          float  normalization_factor,
                   __global float3  *output
            )
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        result = _preproc3(image,
                           variance,
                            0,
                            0,
                            do_dark,
                            dark,
                            do_dark_variance,
                            dark_variance,
                            do_flat,
                            flat,
                            do_solidangle,
                            solidangle,
                            do_polarization,
                            polarization,
                            do_absorption,
                            absorption,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor);
        output[i] = result;
    };//end if NIMAGE
};//end kernel


