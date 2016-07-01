/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Kernel with full pixel-split using a CSR sparse matrix
 *
 *
 *   Copyright (C) 2012-2014 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 10/10/2014
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
 * \brief OpenCL kernels for 1D azimuthal integration using CSR sparse matrix representation
 * 
 * Constant to be provided at build time:
 *   WORKGROUP_SIZE
 */

#include "for_eclipse.h"
/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT in CSR form
 *
 * An image instensity value is spread across the bins according to the positions stored in the LUT.
 * The lut is represented by a set of 3 arrays (coefs, row_ind, col_ptr)
 * Values of 0 in the mask are processed and values of 1 ignored as per PyFAI
 *
 * This implementation is especially efficient on CPU where each core reads adjacents memory.
 * the use of local pointer can help on the CPU.
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param row_ind     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param col_ptr     Integer pointer to global memory holding the pointers to the coefs and row_ind for the CSR matrix
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param outData     Float pointer to the output 1D array with the weighted histogram
 * @param outCount    Float pointer to the output 1D array with the unweighted histogram
 * @param outMerged   Float pointer to the output 1D array with the diffractogram
 *
 */
__kernel void
csr_integrate(	const 	__global	float	*weights,
                const   __global    float   *coefs,
                const   __global    int     *row_ind,
                const   __global    int     *col_ptr,
				const				int   	do_dummy,
				const			 	float 	dummy,
						__global 	float	*outData,
						__global 	float	*outCount,
						__global 	float	*outMerge
		     )
{
    int thread_id_loc = get_local_id(0);
    int bin_num = get_group_id(0); // each workgroup of size=warp is assinged to 1 bin
    int2 bin_bounds;
//    bin_bounds = (int2) *(col_ptr+bin_num);  // cool stuff!
    bin_bounds.x = col_ptr[bin_num];
    bin_bounds.y = col_ptr[bin_num+1];
    int bin_size = bin_bounds.y-bin_bounds.x;
	float sum_data = 0.0f;
	float sum_count = 0.0f;
	float cd = 0.0f;
	float cc = 0.0f;
	float t, y;
	const float epsilon = 1e-10f;
	float coef, data;
	int idx, k, j;

	for (j=bin_bounds.x;j<bin_bounds.y;j+=WORKGROUP_SIZE)
	{
		k = j+thread_id_loc;
        if (k < bin_bounds.y)     // I don't like conditionals!!
        {
   			coef = coefs[k];
   			idx = row_ind[k];
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
       } //end if k < bin_bounds.y
   	};//for j
/*
 * parallel reduction
 */

// REMEMBER TO PASS WORKGROUP_SIZE AS A CPP DEF
    __local float super_sum_data[WORKGROUP_SIZE];
    __local float super_sum_data_correction[WORKGROUP_SIZE];
    __local float super_sum_count[WORKGROUP_SIZE];
    __local float super_sum_count_correction[WORKGROUP_SIZE];
    
    float super_sum_temp = 0.0f;
    int index, active_threads = WORKGROUP_SIZE;
    
    if (bin_size < WORKGROUP_SIZE)
    {
        if (thread_id_loc < bin_size)
        {
            super_sum_data_correction[thread_id_loc] = cd;
            super_sum_count_correction[thread_id_loc] = cc;
            super_sum_data[thread_id_loc] = sum_data;
            super_sum_count[thread_id_loc] = sum_count;
        }
        else
        {
            super_sum_data_correction[thread_id_loc] = 0.0f;
            super_sum_count_correction[thread_id_loc] = 0.0f;
            super_sum_data[thread_id_loc] = 0.0f;
            super_sum_count[thread_id_loc] = 0.0f;
        }
    }
    else
    {
        super_sum_data_correction[thread_id_loc] = cd;
        super_sum_count_correction[thread_id_loc] = cc;
        super_sum_data[thread_id_loc] = sum_data;
        super_sum_count[thread_id_loc] = sum_count;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    cd = 0;
    cc = 0;
    
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (thread_id_loc < active_threads)
        {
            index = thread_id_loc+active_threads;
            cd = super_sum_data_correction[thread_id_loc] + super_sum_data_correction[index];
            super_sum_temp = super_sum_data[thread_id_loc];
            y = super_sum_data[index] - cd;
            t = super_sum_temp + y;
            super_sum_data_correction[thread_id_loc] = (t - super_sum_temp) - y;
            super_sum_data[thread_id_loc] = t;
            
            cc = super_sum_count_correction[thread_id_loc] + super_sum_count_correction[index];
            super_sum_temp = super_sum_count[thread_id_loc];
            y = super_sum_count[index] - cc;
            t = super_sum_temp + y;
            super_sum_count_correction[thread_id_loc]  = (t - super_sum_temp) - y;
            super_sum_count[thread_id_loc] = t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_id_loc == 0)
    {
        outData[bin_num] = super_sum_data[0];
        outCount[bin_num] = super_sum_count[0];
        if (outCount[bin_num] > epsilon)
            outMerge[bin_num] =  outData[bin_num] / outCount[bin_num];
        else
            outMerge[bin_num] = dummy;
    }
};//end kernel

/**
 * \brief Performs 1d azimuthal integration with full pixel splitting based on a LUT in CSR form
 *
 * An image intensity value is spread across the bins according to the positions stored in the LUT.
 * The lut is represented by a set of 3 arrays (coefs, row_ind, col_ptr)
 * Values of 0 in the mask are processed and values of 1 ignored as per PyFAI
 *
 * This kernel is ment to be ran with padded data (the span of each bin must be a multiple of the workgroup size)
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coefficient part of the LUT
 * @param row_ind     Integer pointer to global memory holding the corresponding index of the coefficient
 * @param col_ptr     Integer pointer to global memory holding the pointers to the coefs and row_ind for the CSR matrix
 * @param do_dummy    Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy       Float: value for bad pixels
 * @param outData     Float pointer to the output 1D array with the weighted histogram
 * @param outCount    Float pointer to the output 1D array with the unweighted histogram
 * @param outMerged   Float pointer to the output 1D array with the diffractogram
 *
 */
__kernel void
csr_integrate_padded(   const   __global    float   *weights,
                        const   __global    float   *coefs,
                        const   __global    int     *row_ind,
                        const   __global    int     *col_ptr,
                        const               int      do_dummy,
                        const               float    dummy,
                                __global    float   *outData,
                                __global    float   *outCount,
                                __global    float   *outMerge
                    )
{
    int thread_id_loc = get_local_id(0);
    int bin_num = get_group_id(0); // each workgroup of size=warp is assinged to 1 bin
    int2 bin_bounds;
//    bin_bounds = (int2) *(col_ptr+bin_num);  // cool stuff!
    bin_bounds.x = col_ptr[bin_num];
    bin_bounds.y = col_ptr[bin_num+1];
    float sum_data = 0.0f;
    float sum_count = 0.0f;
    float cd = 0.0f;
    float cc = 0.0f;
    float t, y;
    const float epsilon = 1e-10f;
    float coef, data;
    int idx, k, j;

    for (j=bin_bounds.x;j<bin_bounds.y;j+=WORKGROUP_SIZE)
    {
        k = j+thread_id_loc;
           coef = coefs[k];
        idx = row_ind[k];
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
/*
 * parallel reduction
 */

// REMEMBER TO PASS WORKGROUP_SIZE AS A CPP DEF
    __local float super_sum_data[WORKGROUP_SIZE];
    __local float super_sum_data_correction[WORKGROUP_SIZE];
    __local float super_sum_count[WORKGROUP_SIZE];
    __local float super_sum_count_correction[WORKGROUP_SIZE];
    super_sum_data[thread_id_loc] = sum_data;
    super_sum_count[thread_id_loc] = sum_count;
    super_sum_data_correction[thread_id_loc] = cd;
    super_sum_count_correction[thread_id_loc] = cc;
    barrier(CLK_LOCAL_MEM_FENCE);

    float super_sum_temp = 0.0f;
    int index, active_threads = WORKGROUP_SIZE;
    cd = 0;
    cc = 0;
    
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (thread_id_loc < active_threads)
        {
            index = thread_id_loc+active_threads;
            cd = super_sum_data_correction[thread_id_loc] + super_sum_data_correction[index];
            super_sum_temp = super_sum_data[thread_id_loc];
            y = super_sum_data[index] - cd;
            t = super_sum_temp + y;
            super_sum_data_correction[thread_id_loc] = (t - super_sum_temp) - y;
            super_sum_data[thread_id_loc] = t;
            
            cc = super_sum_count_correction[thread_id_loc] + super_sum_count_correction[index];
            super_sum_temp = super_sum_count[thread_id_loc];
            y = super_sum_count[index] - cc;
            t = super_sum_temp + y;
            super_sum_count_correction[thread_id_loc]  = (t - super_sum_temp) - y;
            super_sum_count[thread_id_loc] = t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_id_loc == 0)
    {
        outData[bin_num] = super_sum_data[0];
        outCount[bin_num] = super_sum_count[0];
        if (outCount[bin_num] > epsilon)
            outMerge[bin_num] =  outData[bin_num] / outCount[bin_num];
        else
            outMerge[bin_num] = dummy;
    }
};//end kernel


/**
 * \brief Performs distortion corrections on an image using a LUT in CSR form
 *
 * @param weights     Float pointer to global memory storing the input image.
 * @param coefs       Float pointer to global memory holding the coeficient part of the LUT
 * @param row_ind     Integer pointer to global memory holding the corresponding index of the coeficient
 * @param col_ptr     Integer pointer to global memory holding the pointers to the coefs and row_ind for the CSR matrix
 * @param outData     Float pointer to the output 1D array with the corrected image
 *
 */

__kernel void
csr_integrate_dis(  const   __global    float   *weights,
                    const   __global    float   *coefs,
                    const   __global    int     *row_ind,
                    const   __global    int     *col_ptr,
                    const               int     do_dummy,
                    const               float   dummy,
                            __global    float   *outData
                 )
{
    int thread_id_loc = get_local_id(0);
    int bin_num = get_group_id(0); // each workgroup of size=warp is assinged to 1 bin
    int2 bin_bounds;
//    bin_bounds = (int2) *(col_ptr+bin_num);  // cool stuff!
    bin_bounds.x = col_ptr[bin_num];
    bin_bounds.y = col_ptr[bin_num+1];
    int bin_size = bin_bounds.y-bin_bounds.x;
    float sum_data = 0.0f;
    float cd = 0.0f;
    float t, y;
    float coef, data;
    int idx, k, j;

    for (j=bin_bounds.x;j<bin_bounds.y;j+=WORKGROUP_SIZE)
    {
        k = j+thread_id_loc;
        if (k < bin_bounds.y)     // I don't like conditionals!!
        {
            coef = coefs[k];
            idx = row_ind[k];
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
             };//end if dummy
       } //end if k < bin_bounds.y
    };//for j
/*
 * parallel reduction
 */

// REMEMBER TO PASS WORKGROUP_SIZE AS A CPP DEF
    __local float super_sum_data[WORKGROUP_SIZE];
    __local float super_sum_data_correction[WORKGROUP_SIZE];
    float super_sum_temp = 0.0f;
    int index, active_threads = WORKGROUP_SIZE;
    
    
    
    if (bin_size < WORKGROUP_SIZE)
    {
        if (thread_id_loc < bin_size)
        {
            super_sum_data_correction[thread_id_loc] = cd;
            super_sum_data[thread_id_loc] = sum_data;
        }
        else
        {
            super_sum_data_correction[thread_id_loc] = 0.0f;
            super_sum_data[thread_id_loc] = 0.0f;
        }
    }
    else
    {
        super_sum_data_correction[thread_id_loc] = cd;
        super_sum_data[thread_id_loc] = sum_data;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
   
    cd = 0;
    
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (thread_id_loc < active_threads)
        {
            index = thread_id_loc+active_threads;

            super_sum_temp = super_sum_data[thread_id_loc];
            y = super_sum_data[index] - cd;
            t = super_sum_temp + y;
            cd = (t - super_sum_temp) - y;
            super_sum_data[thread_id_loc] = t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_id_loc == 0)
    {
        outData[bin_num] = super_sum_data[0];
    }


};//end kernel

