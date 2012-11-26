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
  uint i = get_global_id(0);
  //Global memory guard for padding
  if(i < NBINS)
  {
	array0[i]=0.0f;
	array1[i]=0.0f;
	array2[i]=0.0f;
  }
}

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
			image[i] = data;
		}else{
			image[i] = dummy;
		}//end if do_dummy
	};//end if NIMAGE
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
lut_integrate(	const 	__global	float	*weights,
				const 	__global	struct lut_point_t *lut,
				const				int   		do_dummy,
				const			 	float 		dummy,
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
	if(i < NBINS)
	{
		for (j=0;j<NLUT;j++)
		{
			if (ON_CPU){
				//On CPU best performances are obtained  when each single thread reads adjacent memory
				k = i*NLUT+j;

			}
			else{
				//On GPU best performances are obtained  when threads are reading adjacent memory
				k = j*NBINS+i;
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
			outMerge[i] = dummy;
  };//if NBINS
};//end kernel
