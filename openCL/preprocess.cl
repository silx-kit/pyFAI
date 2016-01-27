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
 * @param array_s8: Pointer to global memory with the input data as signed8 array
 * @param array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
s8_to_float(__global char  *array_s8,
		     __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NIMAGE)
	array_float[i]=(float)array_s8[i];
}


/**
 * \brief cast values of an array of uint8 into a float output array.
 *
 * @param array_u8: Pointer to global memory with the input data as unsigned8 array
 * @param array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u8_to_float(__global unsigned char  *array_u8,
		     __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NIMAGE)
	array_float[i]=(float)array_u8[i];
}


/**
 * \brief cast values of an array of int16 into a float output array.
 *
 * @param array_s16: Pointer to global memory with the input data as signed16 array
 * @param array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
s16_to_float(__global short  *array_s16,
		     __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NIMAGE)
	array_float[i]=(float)array_s16[i];
}


/**
 * \brief cast values of an array of uint16 into a float output array.
 *
 * @param array_u16: Pointer to global memory with the input data as unsigned16 array
 * @param array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u16_to_float(__global unsigned short  *array_u16,
		     __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NIMAGE)
	array_float[i]=(float)array_u16[i];
}

/**
 * \brief cast values of an array of uint32 into a float output array.
 *
 * @param array_u32: Pointer to global memory with the input data as unsigned32 array
 * @param array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u32_to_float(__global unsigned int  *array_u32,
		     __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NIMAGE)
	array_float[i]=(float)array_u32[i];
}

/**
 * \brief convert values of an array of int32 into a float output array.
 *
 * @param array_int:  Pointer to global memory with the data as unsigned32 array
 * @param array_float:  Pointer to global memory with the data float array
 */
__kernel void
s32_to_float(	__global int  *array_int,
				__global float  *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NIMAGE)
	array_float[i] = (float)(array_int[i]);
}



/**
 * \brief Sets the values of 3 float output arrays to zero.
 *
 * Gridsize = size of arrays + padding.
 *
 * @param array0: float Pointer to global memory with the outMerge array
 * @param array1: float Pointer to global memory with the outCount array
 * @param array2: float Pointer to global memory with the outData array
 */
__kernel void
memset_out(__global float *array0,
		   __global float *array1,
		   __global float *array2
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NBINS)
  {
	array0[i]=0.0f;
	array1[i]=0.0f;
	array2[i]=0.0f;
  }
}


/**
 * \brief Performs Normalization of input image
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction
 *  - Solid angle correction (division)
 *  - polarization correction (division)
 *  - flat fiels correction (division)
 * Corrections are made in place unless the pixel is dummy.
 * Dummy pixels are left untouched so that they remain dummy
 *
 * @param image	          Float pointer to global memory storing the input image.
 * @param do_dark         Bool/int: shall dark-current correction be applied ?
 * @param dark            Float pointer to global memory storing the dark image.
 * @param do_flat         Bool/int: shall flat-field correction be applied ?
 * @param flat            Float pointer to global memory storing the flat image.
 * @param do_solidangle   Bool/int: shall flat-field correction be applied ?
 * @param solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * @param do_polarization Bool/int: shall flat-field correction be applied ?
 * @param polarization    Float pointer to global memory storing the polarization of each pixel.
 * @param do_dummy    	  Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       	  Float: value for bad pixels
 * @param delta_dummy 	  Float: precision for bad pixel value
 *
**/
__kernel void
corrections( 		__global float 	*image,
			const			 int 	do_dark,
			const 	__global float 	*dark,
			const			 int	do_flat,
			const 	__global float 	*flat,
			const			 int	do_solidangle,
			const 	__global float 	*solidangle,
			const			 int	do_polarization,
			const 	__global float 	*polarization,
			const		 	 int   	do_dummy,
			const			 float 	dummy,
			const		 	 float 	delta_dummy,
			const		 	 float 	normalization_factor
			)
{
	float data;
	int i= get_global_id(0);
	if(i < NIMAGE)
	{
		data = image[i];
		if( (!do_dummy) || ((delta_dummy!=0.0f) && (fabs(data-dummy) > delta_dummy))|| ((delta_dummy==0.0f) && (data!=dummy)))
		{
			if(do_dark)
				data-=dark[i];
			if(do_flat)
				data/=flat[i];
			if(do_solidangle)
				data/=solidangle[i];
			if(do_polarization)
				data/=polarization[i];
			image[i] = data/normalization_factor;
		}else{
			image[i] = dummy;
		}//end if do_dummy
	};//end if NIMAGE
};//end kernel


