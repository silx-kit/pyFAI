/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Kernel with full pixel-split using a LUT
 *
 *
 *   Copyright (C) 2012 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 26/10/2012
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published
 *   by the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   and the GNU Lesser General Public License  along with this program.
 *   If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * \file
 * \brief OpenCL kernels for 1D azimuthal integration
 */

//OpenCL extensions are silently defined by opencl compiler at compile-time:
#ifdef cl_amd_printf
  #pragma OPENCL EXTENSION cl_amd_printf : enable
  //#define printf(...)
#elif defined(cl_intel_printf)
  #pragma OPENCL EXTENSION cl_intel_printf : enable
#else
  #define printf(...)
#endif

#ifdef ENABLE_FP64
//	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	typedef double bigfloat_t;
#else
//	#pragma OPENCL EXTENSION cl_khr_fp64 : disable
	typedef float bigfloat_t;
#endif

#define GROUP_SIZE BLOCK_SIZE

struct lut_point_t
{
		uint idx;
	    float coef;
};

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
			const			 uint 	size,
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
			const		 	 float 	delta_dummy
			)
{
	float data;
	uint i= get_global_id(0);
	if(i < size)
	{
		data = image[i];
		if( (!do_dummy) || (delta_dummy && (fabs(data-dummy) > delta_dummy))|| (!delta_dummy && (data!=dummy)))
		{
			if(do_dark)
				data-=dark[i];
			if(do_flat)
				data/=flat[i];
			if(do_solidangle)
				data/=solidangle[i];
			if(do_polarization)
				data/=polarization[i];
			image[i] = data;
		}//end if do_dummy
	};//end if bins
};//end kernel


/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT
 *
 * An image instensity value is spread across the bins according to the positions stored in the LUT.
 * The lut_index contains the positions of the pixel in the input array
 * Values of 0 in the mask are processed and values of 1 ignored as per PyFAI
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param bins        Unsigned int: number of output bins wanted (and pre-calculated)
 * @param lut_size    Unsigned int: dimension of the look-up table
 * @param lut_idx     Unsigned integers pointer to an array of with the index of input pixels
 * @param lut_coef    Float pointer to an array of coefficients for each input pixel
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param delta_dummy Float: precision for bad pixel value
 * @param do_dark     Bool/int: shall dark-current correction be applied ?
 * @param dark        Float pointer to global memory storing the dark image.
 * @param do_flat     Bool/int: shall flat-field correction be applied ? (could contain polarization corrections)
 * @param flat        Float pointer to global memory storing the flat image.
 * @param outData     Float pointer to the output 1D array with the weighted histogram
 * @param outCount    Float pointer to the output 1D array with the unweighted histogram
 * @param outMerged   Float pointer to the output 1D array with the diffractogram
 * 
 */
__kernel void
lut_integrate_orig(	const 	__global float 	*weights,
				const			 	uint 	bins,
				const			 	uint 	lut_size,
				const 	__global 	uint 	*lut_idx,
				const 	__global 	float 	*lut_coef,
				const			 	int   	do_dummy,
				const			 	float 	dummy,
				const			 	float 	delta_dummy,
						__global 	float	*outData,
						__global 	float	*outCount,
						__global 	float	*outMerge
		        )
{
	int idx, k, j, i= get_global_id(0);
	float sum_data = 0.0f;
	float sum_count = 0.0f;
	float cd = 0.0f;
	float cc = 0.0f;
	float t, y;
	const float epsilon = 1e-10f;
	float coef, data;
	if(i < bins)
	{
		for (j=0;j<lut_size;j++)
		{
			k = i*lut_size+j;
			idx = lut_idx[k];
			coef = lut_coef[k];
			if((idx <= 0) && (coef <= 0.0f))
			  break;
			data = weights[idx];
			if( (!do_dummy) || (delta_dummy && (fabs(data-dummy) > delta_dummy))|| (!delta_dummy && (data!=dummy)))
			{
				//sum_data +=  coef * data;
				//sum_count += coef;
				//	Kahan summation allows single precision arithmetics with error compensation
				//	http://en.wikipedia.org/wiki/Kahan_summation_algorithm
				y = coef*data - cd;
				t = sum_data + y;
				cd = (t - sum_data) - y;
				sum_data = t;
				y = coef - cc;
				t = sum_count + y;
				cc = (t - sum_count) - y;
				sum_count = t;

			};//test if dummy
		};//for j
		outData[i] = (float) sum_data;
		outCount[i] = (float) sum_count;
		if (sum_count > epsilon)
			outMerge[i] = (float) sum_data / sum_count;
		else
			outMerge[i] = 0.0f;
  };//if bins
};//end kernel

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
 * @param bins        Unsigned int: number of output bins wanted (and pre-calculated)
 * @param lut_size    Unsigned int: dimension of the look-up table
 * @param lut         Pointer to an 2D-array of (unsigned integers,float) containing the index of input pixels and the fraction of pixel going to the bin 
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param delta_dummy Float: precision for bad pixel value
 * @param do_dark     Bool/int: shall dark-current correction be applied ?
 * @param dark        Float pointer to global memory storing the dark image.
 * @param do_flat     Bool/int: shall flat-field correction be applied ? (could contain polarization corrections)
 * @param flat        Float pointer to global memory storing the flat image.
 * @param outData     Float pointer to the output 1D array with the weighted histogram
 * @param outCount    Float pointer to the output 1D array with the unweighted histogram
 * @param outMerged   Float pointer to the output 1D array with the diffractogram
 * 
 */
__kernel void
lut_integrate_single(	const 	__global 	float 		*weights,
						const			 	uint 		bins,
						const			 	uint 		lut_size,
						const 	__global struct lut_point_t *lut,
						const			 	int   		do_dummy,
						const			 	float 		dummy,
						const			 	float 		delta_dummy,
						__global 	float		*outData,
						__global 	float		*outCount,
						__global 	float		*outMerge
		        )
{
	int idx, k, j, i= get_global_id(0);
	float sum_data = 0.0f;
	float sum_count = 0.0f;
	float cd = 0.0f;
	float cc = 0.0f;
	float t, y;
	const float epsilon = 1e-10f;
	float coef, data;
	if(i < bins)
	{
		for (j=0;j<lut_size;j++)
		{
			k = i*lut_size+j;
			idx = lut[k].idx;
			coef = lut[k].coef;
			if((idx <= 0) && (coef <= 0.0f))
			  break;
			data = weights[idx];
			if( (!do_dummy) || (delta_dummy && (fabs(data-dummy) > delta_dummy))|| (!delta_dummy && (data!=dummy)))
			{
				//sum_data +=  coef * data;
				//sum_count += coef;
				//Kahan summation allows single precision arithmetics with error compensation
				//http://en.wikipedia.org/wiki/Kahan_summation_algorithm
				y = coef*data - cd;
				t = sum_data + y;
				cd = (t - sum_data) - y;
				sum_data = t;
				y = coef - cc;
				t = sum_count + y;
				cc = (t - sum_count) - y;
				sum_count = t;
			};//end if dummy
		};//for j
		outData[i] = sum_data;
		outCount[i] = sum_count;
		if (sum_count > epsilon)
			outMerge[i] =  sum_data / sum_count;
		else
			outMerge[i] = 0.0f;
  };//if bins
};//end kernel

/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT
 *
 * An image instensity value is spread across the bins according to the positions stored in the LUT.
 * The lut is an 2D-array of index (contains the positions of the pixel in the input array) 
 * and coeficients (fraction of pixel going to the bin)
 * Values of 0 in the mask are processed and values of 1 ignored as per PyFAI
 *
 * This implementation is especially efficient on GPU where adjacent cores reads adjacents memory at the same time
 * On GPU, textures can be usefull but this prevents the code from compiling under certain OpenCL implementations.  
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param bins        Unsigned int: number of output bins wanted (and pre-calculated)
 * @param lut_size    Unsigned int: dimension of the look-up table
 * @param lut         Pointer to an 2D-array of (unsigned integers,float) containing the index of input pixels and the fraction of pixel going to the bin 
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param delta_dummy Float: precision for bad pixel value
 * @param do_dark     Bool/int: shall dark-current correction be applied ?
 * @param dark        Float pointer to global memory storing the dark image.
 * @param do_flat     Bool/int: shall flat-field correction be applied ? (could contain polarization corrections)
 * @param flat        Float pointer to global memory storing the flat image.
 * @param outData     Float pointer to the output 1D array with the weighted histogram
 * @param outCount    Float pointer to the output 1D array with the unweighted histogram
 * @param outMerged   Float pointer to the output 1D array with the diffractogram
 * 
 */
__kernel void
lut_integrate_lutT(	const 	__global 	float 		*weights,
						const			 	uint 		bins,
						const			 	uint 		lut_size,
						const 	__global struct lut_point_t *lut,
						const			 	int   		do_dummy,
						const			 	float 		dummy,
						const			 	float 		delta_dummy,
						__global 			float		*outData,
						__global 			float		*outCount,
						__global 			float		*outMerge
		        )
{
	uint idx, k, j, i= get_global_id(0);
	float sum_data = 0.0f;
	float sum_count = 0.0f;
	float cd = 0.0f;
	float cc = 0.0f;
	float t, y;
	const float epsilon = 1e-10f;
	float coef, data;
//	const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	if(i < bins)
	{
		for (j=0;j<lut_size;j++)
		{
			k = j*bins+i;
			idx = lut[k].idx;
			coef = lut[k].coef;
			if((idx == 0) && (coef <= 0.0f))
			  break;
//			data = read_imagef(weights, sampler, (int2)(idx%dimY , idx/dimY)).s0;
			data = weights[idx];
			if( (!do_dummy) || (delta_dummy && (fabs(data-dummy) > delta_dummy))|| (data!=dummy) )
			{
				//sum_data +=  coef * data;
				//sum_count += coef;
				//Kahan summation allows single precision arithmetics with error compensation
				//http://en.wikipedia.org/wiki/Kahan_summation_algorithm
				y = coef*data - cd;
				t = sum_data + y;
				cd = (t - sum_data) - y;
				sum_data = t;
				y = coef - cc;
				t = sum_count + y;
				cc = (t - sum_count) - y;
				sum_count = t;
			};//test dummy
		};//for j
		outData[i] =  sum_data;
		outCount[i] = sum_count;
		if (sum_count > epsilon)
			outMerge[i] = (sum_data / sum_count);
		else
			outMerge[i] = 0.0f;
  };//if bins
};//end kernel
