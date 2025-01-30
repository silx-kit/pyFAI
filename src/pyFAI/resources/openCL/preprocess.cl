/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Preprocessing program
 *
 *
 *   Copyright (C) 2012-2024 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 19/11/2024
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
 * \brief OpenCL kernels for image array casting, array mem-setting and normalizing
 *
 * Constant to be provided at build time:
 *   NIMAGE: size of the image
 *
 */

#include "for_eclipse.h"
enum ErrorModel { NO_VAR=0, VARIANCE=1, POISSON=2, AZIMUTHAL=3, HYBRID=4 };

/**
 * \brief cast values of an array of int8 into a float output array.
 *
 * - array_s8: Pointer to global memory with the input data as signed8 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
kernel void
s8_to_float(global char  *array_s8,
            global float *array_float
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
kernel void
u8_to_float(global unsigned char  *array_u8,
            global float *array_float
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
kernel void
s16_to_float(global short *array_s16,
             global float *array_float
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
kernel void
u16_to_float(global unsigned short  *array_u16,
             global float *array_float
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
kernel void
u32_to_float(global unsigned int  *array_u32,
             global float *array_float
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
kernel void
s32_to_float(global int  *array_int,
             global float  *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)(array_int[i]);
}

/* Function reading at the given position.
 * Dtype is 1/-1 for char/uchar .... 8/-8 for int64/uint64 and 32/64 for float/double.
 */
static float _any2float(const global uchar* input,
                       size_t position,
                       char dtype){
    float value=0.0f;
    if (dtype == 1){
        uchar ival =  input[position];
        value = convert_float(ival);
    }
    else if (dtype == -1){
        char ival =  as_char(input[position]);
        value = convert_float(ival);
    }
    else if (dtype == 2){
        uchar2 rval =  (uchar2) (input[2*position],input[2*position+1]);
        ushort ival = as_ushort(rval);
        value = convert_float(ival);
    }
    else if (dtype == -2){
        uchar2 rval =  (uchar2) (input[2*position],input[2*position+1]);
        short ival = as_short(rval);
        value = convert_float(ival);
    }
    else if (dtype == 4){
        uchar4 rval =  (uchar4) (input[4*position],input[4*position+1], input[4*position+2],input[4*position+3]);
        uint ival = as_uint(rval);
        value = convert_float(ival);
    }
    else if (dtype == -4){
        uchar4 rval =  (uchar4) (input[4*position],input[4*position+1], input[4*position+2],input[4*position+3]);
        int ival = as_int(rval);
        value = convert_float(ival);
    }
    else if (dtype == 8){
        uchar8 rval =  (uchar8) (input[8*position],input[8*position+1], input[8*position+2],input[8*position+3],
                                 input[8*position+4],input[8*position+5], input[8*position+6],input[8*position+7]);
        ulong ival = as_ulong(rval);
        value = convert_float(ival);
    }
    else if (dtype == -8){
        uchar8 rval =  (uchar8) (input[8*position],input[8*position+1], input[8*position+2],input[8*position+3],
                              input[8*position+4],input[8*position+5], input[8*position+6],input[8*position+7]);
        long ival = as_long(rval);
        value = convert_float(ival);
    }
    else if (dtype == 32){
        uchar4 rval =  (uchar4) (input[4*position], input[4*position+1], input[4*position+2], input[4*position+3]);
        value = as_float(rval);
    }
    else if (dtype == 64){
#ifdef cl_khr_fp64
    #if cl_khr_fp64
        uchar8 rval =  (uchar8) (input[8*position],input[8*position+1], input[8*position+2],input[8*position+3],
                              input[8*position+4],input[8*position+5], input[8*position+6],input[8*position+7]);
        value = convert_float(as_double(rval));
    #else
        if (get_global_id(0)==0)printf("Double precision arithmetics is not supported on this device !\n");
    #endif
#else
        if (get_global_id(0)==0)printf("Double precision arithmetics is not supported on this device !\n");
#endif
    }

    return value;
}

/**
 * Internal functions pixel wise function.
 *
 * Performs the normalization of input image by dark subtraction,
 *        variance is propagated to second member of the float4
 *        flatfield, solid angle, polarization and absorption are stored in
 *        third member. the last member contains 1 for valid pixels.
 *
 * Invalid/Dummy pixels will all have the 3rd/4th-member set to 0, i.e. no weight
 *
 * - image           Float pointer to global memory storing the input image.
 * - error_model     If POISSON or HYBRID, initialize variance with raw signal
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall polarization correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_absorption   Bool/int: shall absorption correction be applied ?
 * - absorption      Float pointer to global memory storing the effective absoption of each pixel.
 * - do_mask         perform mask correction ?
 * - mask            Bool/char pointer to mask array
 * - do_dummy        Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy           Float: value for bad pixels
 * - delta_dummy     Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
**/

static float4 _preproc4(const float  value,
                        const          char   error_model,
                        const global float  *variance,
                        const          char   do_dark,
                        const global float  *dark,
                        const          char   do_dark_variance,
                        const global float  *dark_variance,
                        const          char   do_flat,
                        const global float  *flat,
                        const          char   do_solidangle,
                        const global float  *solidangle,
                        const          char   do_polarization,
                        const global float  *polarization,
                        const          char   do_absorption,
                        const global float  *absorption,
                        const          char   do_mask,
                        const global char   *mask,
                        const          char   do_dummy,
                        const          float  dummy,
                        const          float  delta_dummy,
                        const          float  normalization_factor,
                        const          char   apply_normalization)
{
    size_t i = get_global_id(0);
    float4 result = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < NIMAGE){
        if ((!do_mask) || (!mask[i])){
            float v = 0.0f;
            switch(error_model)
            {
            case VARIANCE:
                v = variance[i];
                break;
            case NO_VAR:
                v = 0.0f;
                break;
            default:
                v = max(value, 1.0f);
            }
            result = (float4)(value, v, normalization_factor, 1.0f);

            if ( (!do_dummy)
                  ||((delta_dummy != 0.0f) && (fabs(result.s0-dummy) > delta_dummy))
                  ||((delta_dummy == 0.0f) && (result.s0 != dummy))){
                if (do_dark)
                    result.s0 -= dark[i];
                if (do_dark_variance)
                    result.s1 += dark_variance[i];
                if (do_flat){
                    float one_flat = flat[i];
                    if ( (!do_dummy)
                         ||((delta_dummy != 0.0f) && (fabs(one_flat-dummy) > delta_dummy))
                         ||((delta_dummy == 0.0f) && (one_flat != dummy)))
                        result.s2 *= one_flat;
                    else
                        result = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                }
                if (do_solidangle)
                    result.s2 *= solidangle[i];
                if (do_polarization)
                    result.s2 *= polarization[i];
                if (do_absorption)
                    result.s2 *= absorption[i];
                if (isnan(result.s0) || isnan(result.s1) || isnan(result.s2) || (result.s2 == 0.0f))
                    result = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                else if (apply_normalization){
                    result.s0 /= result.s2;
                    result.s1 /= result.s2*result.s2;
                    result.s2 = 1.0f;
                }
            }
            else{
                result = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            }//end if do_dummy

        } // end if mask
    }//end if NIMAGE
    return result;
}//end function


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
 * - do_polarization Bool/int: shall polarization correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_absorption   Bool/int: shall absorption correction be applied ?
 * - absorption      Float pointer to global memory storing the effective absoption of each pixel.
 * - do_mask         perform mask correction ?
 * - mask            Bool/char pointer to mask array
 * - do_dummy        Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy           Float: value for bad pixels
 * - delta_dummy     Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 * - apply_normalization: divide signal by norm at preprocessing stage (for an unweighted mean, default being weighted mean)
 * - output:         Destination array
**/

kernel void
corrections(const global float  *image,
            const          char   do_dark,
            const global float  *dark,
            const          char   do_flat,
            const global float  *flat,
            const          char   do_solidangle,
            const global float  *solidangle,
            const          char   do_polarization,
            const global float  *polarization,
			const          char   do_absorption,
			const global float  *absorption,
            const          char   do_mask,
            const global char   *mask,
            const          char   do_dummy,
            const          float  dummy,
            const          float  delta_dummy,
            const          float  normalization_factor,
            const          char   apply_normalization,
                  global float  *output){
    size_t i= get_global_id(0);
    if (i < NIMAGE) {
        float4 result;
        result = _preproc4(_any2float( (const global uchar*) image, i, 32),
                           0,
                           image,
                           do_dark,
                           dark,
                           0,
                           dark,
                           do_flat,
                           flat,
                           do_solidangle,
                           solidangle,
                           do_polarization,
                           polarization,
                           do_absorption,
                           absorption,
                           do_mask,
                           mask,
                           do_dummy,
                           dummy,
                           delta_dummy,
                           normalization_factor,
                           apply_normalization);
        if (result.s2 != 0.0f)
            output[i] = result.s0 / result.s2;
        else
            output[i] = dummy;
    }//end if NIMAGE
}//end kernel


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
kernel void
corrections2(const global float  *image,
             const          char   do_dark,
             const global float  *dark,
             const          char   do_flat,
             const global float  *flat,
             const          char   do_solidangle,
             const global float  *solidangle,
             const          char   do_polarization,
             const global float  *polarization,
             const          char   do_absorption,
             const global float  *absorption,
             const          char   do_mask,
             const global char   *mask,
             const          char   do_dummy,
             const          float  dummy,
             const          float  delta_dummy,
             const          float  normalization_factor,
             const          char   apply_normalization,
                   global float2  *output
            )
{
    size_t i = get_global_id(0);

    if (i < NIMAGE)
    {
        float4 result;
        result = _preproc4(_any2float((const global uchar*)image, i, 32),
                           0,
                           image,
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
                           do_mask,
                           mask,
                           do_dummy,
                           dummy,
                           delta_dummy,
                           normalization_factor,
                           apply_normalization);
        output[i] = (float2)(result.s0, result.s2);
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

kernel void
corrections3(const global float  *image,
             const          char   error_model,
             const global float  *variance,
             const          char   do_dark,
             const global float  *dark,
             const          char   do_dark_variance,
             const global float  *dark_variance,
             const          char   do_flat,
             const global float  *flat,
             const          char   do_solidangle,
             const global float  *solidangle,
             const          char   do_polarization,
             const global float  *polarization,
             const          char   do_absorption,
             const global float  *absorption,
             const          char   do_mask,
             const global char   *mask,
             const          char   do_dummy,
             const          float  dummy,
             const          float  delta_dummy,
             const          float  normalization_factor,
             const          char   apply_normalization,
                   global float3  *output
            )
{
    size_t i = get_global_id(0);

    if (i < NIMAGE){
        float4 result;
        result = _preproc4( _any2float((const global uchar*)image, i, 32),
                            error_model,
                            variance,
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
                            do_mask,
                            mask,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor,
                            apply_normalization);
        output[i] = (float3)(result.s0, result.s1, result.s2);
    };//end if NIMAGE
};//end kernel


/**
 * \brief Performs Normalization of input image with float4 output (signal, variance, normalization, count)
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction for the data
 *  - Solid angle correction (division)
 *  - polarization correction (division)
 *  - flat fiels correction (division)
 * Corrections are made in place unless the pixel is dummy.
 * Dummy pixels are left untouched so that they remain dummy
 *
 * - image           Float pointer to global memory storing the input image.
 * - error_model     error_model, int defined in the ErrorModel enum
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
 *
**/

kernel void
corrections4(const global float  *image,
             const          char   error_model,
             const global float  *variance,
             const          char   do_dark,
             const global float  *dark,
             const          char   do_dark_variance,
             const global float  *dark_variance,
             const          char   do_flat,
             const global float  *flat,
             const          char   do_solidangle,
             const global float  *solidangle,
             const          char   do_polarization,
             const global float  *polarization,
             const          char   do_absorption,
             const global float  *absorption,
             const          char   do_mask,
             const global char   *mask,
             const          char   do_dummy,
             const          float  dummy,
             const          float  delta_dummy,
             const          float  normalization_factor,
             const          char   apply_normalization,
                   global float4  *output
            )
{
    size_t i = get_global_id(0);
    float4 result = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < NIMAGE)
    {
        result = _preproc4( _any2float((const global uchar*)image, i, 32),
                            error_model,
                            variance,
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
                            do_mask,
                            mask,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor,
                            apply_normalization);
        output[i] = result;
    };//end if NIMAGE
};//end kernel

/**
 * \brief Performs Normalization of input image with float4 output (signal, variance, normalization, count)
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction for the data
 *  - Solid angle correction (division)
 *  - polarization correction (division)
 *  - flat fiels correction (division)
 * Corrections are made in place unless the pixel is dummy.
 * Dummy pixels are left untouched so that they remain dummy
 *
 * - image           Float pointer to global memory storing the input image.
 * - image_dtype     integer containing the coding to the datatype ((+/)-1 for (u)char, ... (+/)-4 for (u)int, 32 and 64 for float and double)
 * - error_model     error_model, int defined in the ErrorModel enum
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
 *
**/

kernel void
corrections4a(const global uchar  *image,
             const          char  dtype,
             const          char   error_model,
             const global float  *variance,
             const          char   do_dark,
             const global float  *dark,
             const          char   do_dark_variance,
             const global float  *dark_variance,
             const          char   do_flat,
             const global float  *flat,
             const          char   do_solidangle,
             const global float  *solidangle,
             const          char   do_polarization,
             const global float  *polarization,
             const          char   do_absorption,
             const global float  *absorption,
             const          char   do_mask,
             const global char   *mask,
             const          char   do_dummy,
             const          float  dummy,
             const          float  delta_dummy,
             const          float  normalization_factor,
             const          char   apply_normalization,
                   global float4  *output
            )
{
    size_t i = get_global_id(0);
    float4 result = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < NIMAGE)
    {
        float value = _any2float(image, i, dtype);
        result = _preproc4( value,
                            error_model,
                            variance,
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
                            do_mask,
                            mask,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor,
                            apply_normalization);
        output[i] = result;
    };//end if NIMAGE
};//end kernel
