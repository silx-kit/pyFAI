/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Kernel with full pixel-split using a LUT
 *
 *
 *   Copyright (C) 2012 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 11/10/2012
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published
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
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
  #pragma OPENCL EXTENSION cl_khr_fp64 : disable
#endif

#define GROUP_SIZE BLOCK_SIZE


/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT
 *
 * An image instensity value is spread across the bins according to the positions stored in the LUT.
 * The lut_index contains the positions of the pixel in the input array
 * Values of 0 in the mask are processed and values of 1 ignored as per PyFAI
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param do_dummy    bint: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param binarray    UINTType Pointer to global memory with the uweights array.
 * @param tth_min_max Float pointer to global memory of size 2 (vector) storing the min and max values
 *                     for 2th +- d2th.
 * @param intensity   Float pointer to global memory where the input image resides.
 * @param histogram   UINTType Pointer to global memory with the uhistogram array.
 * @param span_range  Float pointer to global memory with the max values of spans per group.
 * @param mask        Int pointer to global memory with the mask to be used.
 * @param tth_range   Float pointer to global memory of size 2 (vector) storing the min and max for integration.
 *                     If tth range is not specified the this array points to tth_min_max.
 */
__kernel void
lut_integrate(	const 	__global 	float 	*weights,
				const 	__global 	uint 	bins,
				const 	__global 	uint 	lut_size,
				const 	__global 	uint 	*lut_idx,
				const 	__global 	float 	*lut_coef,
				const 				int   	do_dummy,
				const 				float 	dummy,
				const 				float 	delta_dummy,
				const 				int 	do_dark, 
				const 	__global 	float 	*dark,
				const 		 		int		do_flat,
				const 	__global 	float 	*flat,
						__global 	double	*outData,
						__global 	double	*outCount,
						__global 	double	*outMerge
		        )
{  
	
	uint k, j, i= get_global_id(0);
	int idx
	double sum_data = 0.0;
	double sum_count = 0.0;
	const double epsilon = 1e-10
	float coef, data
	if(gid < bins)
	{
		for (j=0;j<lut_size;j++)
		{
			k = i*lut_size+j;
			idx = lut_idx[k];
			coef = lut_coef[k];
			if((idx <= 0) && (coef <= 0.0))
			  break;
			data = weight[idx];
			if(do_dummy)
			{
				if(delta_dummy)
				{
					if (fabs(data-dummy)<=delta_dummy)
						continue;            	
				}
				else
				{
					if(data==dummy)
						continue;
				}
			}
		  if(do_dark)
			  data -= dark[idx];
		  if do_flat:
			  data /= flat[idx];
			  
		  sum_data +=  coef * data;
		  sum_count += coef;
		}
	  outData[i] = sum_data
	  outCount[i] = sum_count
	  if (sum_count > epsilon)
		  outMerge[i] = sum_data / sum_count

  }//if bins
}