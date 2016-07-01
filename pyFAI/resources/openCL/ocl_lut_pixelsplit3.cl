/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Scatter to Gather transformation
 *
 *
 *   Copyright (C) 2014 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: Giannis Ashiotis <giannis.ashiotis@gmail.com>
 *   					J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 20/10/2014
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
#include "for_eclipse.h"

float area4(float a0, float a1, float b0, float b1, float c0, float c1, float d0, float d1)
{
    return 0.5 * fabs(((c0 - a0) * (d1 - b1)) - ((c1 - a1) * (d0 - b0)));
}


float integrate_line( float A0, float B0, float2 AB)
{
    return (A0==B0) ? 0.0 : AB.s0*(B0*B0 - A0*A0)*0.5 + AB.s1*(B0-A0);
}


float getBinNr(float x0, float delta, float pos0_min)
{
    return (x0 - pos0_min) / delta;
}


float min4f(float a, float b, float c, float d)
{
    return fmin(fmin(a,b),fmin(c,d));
}


float max4f(float a, float b, float c, float d)
{
    return fmax(fmax(a,b),fmax(c,d));
}


void AtomicAdd(volatile __global float *source, const float operand) 
{
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
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
 * \brief convert values of an array of int32 into a float output array.
 *
 * @param array_int:  Pointer to global memory with the data in int
 * @param array_float:  Pointer to global memory with the data in float
 */
__kernel void
s32_to_float(   __global int  *array_int,
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
  if(i < BINS)
  {
    array0[i]=0.0f;
    array1[i]=0.0f;
    array2[i]=0.0f;
  }
}


/**
 * \brief Sets the values of 3 float output arrays to zero.
 *
 * Gridsize = size of arrays + padding.
 *
 * @param array0: int Pointer to global memory with the outMax array
 */
__kernel void
memset_out_int(__global int *array0)
{
    int i = get_global_id(0);
    //Global memory guard for padding
    if(i < BINS)
        array0[i]=0;
}





__kernel
void reduce1(__global float2* buffer,
             __const int length,
             __global float4* preresult) {
    
    
    int global_index = get_global_id(0);
    int global_size  = get_global_size(0);
    float4 accumulator;
    accumulator.x = INFINITY;
    accumulator.y = -INFINITY;
    accumulator.z = INFINITY;
    accumulator.w = -INFINITY;
    
    // Loop sequentially over chunks of input vector
    while (global_index < length/2) {
        float2 element = buffer[global_index];
        accumulator.x = (accumulator.x < element.s0) ? accumulator.x : element.s0;
        accumulator.y = (accumulator.y > element.s0) ? accumulator.y : element.s0;
        accumulator.z = (accumulator.z < element.s1) ? accumulator.z : element.s1;
        accumulator.w = (accumulator.w > element.s1) ? accumulator.w : element.s1;
        global_index += global_size;
    }
    
    __local float4 scratch[WORKGROUP_SIZE];

    // Perform parallel reduction
    int local_index = get_local_id(0);
    
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int active_threads = get_local_size(0);
    
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (local_index < active_threads)
        {
            float4 other = scratch[local_index + active_threads];
            float4 mine  = scratch[local_index];
            mine.x = (mine.x < other.x) ? mine.x : other.x;
            mine.y = (mine.y > other.y) ? mine.y : other.y;
            mine.z = (mine.z < other.z) ? mine.z : other.z;
            mine.w = (mine.w > other.w) ? mine.w : other.w;
            /*
            float2 tmp;
            tmp.x = (mine.x < other.x) ? mine.x : other.x;
            tmp.y = (mine.y > other.y) ? mine.y : other.y;
            scratch[local_index] = tmp;
            */
            scratch[local_index] = mine;
       }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        preresult[get_group_id(0)] = scratch[0];
    }
}




__kernel
void reduce2(__global float4* preresult,
             __global float4* result) {
    
    
    __local float4 scratch[WORKGROUP_SIZE];

    int local_index = get_local_id(0);
    
    scratch[local_index] = preresult[local_index];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int active_threads = get_local_size(0);
    
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (local_index < active_threads)
        {
            float4 other = scratch[local_index + active_threads];
            float4 mine  = scratch[local_index];
            mine.x = (mine.x < other.x) ? mine.x : other.x;
            mine.y = (mine.y > other.y) ? mine.y : other.y;
            mine.z = (mine.z < other.z) ? mine.z : other.z;
            mine.w = (mine.w > other.w) ? mine.w : other.w;
            /*
            float2 tmp;
            tmp.x = (mine.x < other.x) ? mine.x : other.x;
            tmp.y = (mine.y > other.y) ? mine.y : other.y;
            scratch[local_index] = tmp;
            */
            scratch[local_index] = mine;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    

    if (local_index == 0) {
        result[0] = scratch[0];
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
 * @param image           Float pointer to global memory storing the input image.
 * @param do_dark         Bool/int: shall dark-current correction be applied ?
 * @param dark            Float pointer to global memory storing the dark image.
 * @param do_flat         Bool/int: shall flat-field correction be applied ?
 * @param flat            Float pointer to global memory storing the flat image.
 * @param do_solidangle   Bool/int: shall flat-field correction be applied ?
 * @param solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * @param do_polarization Bool/int: shall flat-field correction be applied ?
 * @param polarization    Float pointer to global memory storing the polarization of each pixel.
 * @param do_dummy        Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * @param dummy           Float: value for bad pixels
 * @param delta_dummy     Float: precision for bad pixel value
 *
**/
__kernel void
corrections(        __global float  *image,
            const            int    do_dark,
            const   __global float  *dark,
            const            int    do_flat,
            const   __global float  *flat,
            const            int    do_solidangle,
            const   __global float  *solidangle,
            const            int    do_polarization,
            const   __global float  *polarization,
            const            int    do_dummy,
            const            float  dummy,
            const            float  delta_dummy
            )
{
    float data;
    int i= get_global_id(0);
    if(i < NIMAGE)
    {
        data = image[i];
        int dummy_condition = ((!do_dummy) || ((delta_dummy!=0.0f) && (fabs(data-dummy) > delta_dummy)) || ((delta_dummy==0.0f) && (data!=dummy)));
        data -= do_dark         ? dark[i]           : 0;
        data *= do_flat         ? 1/flat[i]         : 1;
        data *= do_solidangle   ? 1/solidangle[i]   : 1;
        data *= do_polarization ? 1/polarization[i] : 1;
        image[i] = dummy_condition ? data : dummy;
    };//end if NIMAGE
};//end kernel




__kernel
void lut1(__global float8* pos,
//             __global int*    mask,
//             __const  int     check_mask,
          __global float4* minmax,
          const    int     length,
//                   float2  pos0Range,
//                   float2  pos1Range,
          __global int*  outMax)
{
    int global_index = get_global_id(0);
    if (global_index < length)
    {
//         float pos0_min = fmax(fmin(pos0Range.x,pos0Range.y),minmax[0].s0);
//         float pos0_max = fmin(fmax(pos0Range.x,pos0Range.y),minmax[0].s1);
        float pos0_min = minmax[0].s0;
        float pos0_maxin = minmax[0].s1;
        float pos0_max = pos0_maxin*( 1 + EPS);
//         pos0_max *= 1.00001;
        
//         if (global_index == 0)
//             printf("%f  %f  %f", pos0_maxin, pos0_max, ( 1 + EPS));
        
        float delta = (pos0_max - pos0_min) / BINS;
        
        int local_index = get_local_id(0);
        
        float8 pixel = pos[global_index];
        
        pixel.s0 = getBinNr(pixel.s0, delta, pos0_min);
        pixel.s2 = getBinNr(pixel.s2, delta, pos0_min);
        pixel.s4 = getBinNr(pixel.s4, delta, pos0_min);
        pixel.s6 = getBinNr(pixel.s6, delta, pos0_min);
        
        float min0 = min4f(pixel.s0, pixel.s2, pixel.s4, pixel.s6);
        float max0 = max4f(pixel.s0, pixel.s2, pixel.s4, pixel.s6);
        
        int bin0_min = floor(min0);
        int bin0_max = floor(max0);
        
        for (int bin=bin0_min; bin < bin0_max+1; bin++)
        {
//             if (bin < BINS)
//             {
//                 atomic_add(&outMax[bin], 1);
//             }else{
//                 printf("fooo %d \n", bin);
//             }
             atomic_add(&outMax[bin], 1);
            
        }
    }
}

// to be run with global_size = local_size
__kernel
void lut2(__global int*  outMax,
          __global int*  idx_ptr,
          __global int*  lutsize)
{
    int local_index = get_local_id(0);
//    int local_size  = get_local_size(0);
    
    if (local_index == 0)
    {
        idx_ptr[0] = 0;
        for (int i=0; i<BINS; i++)
            idx_ptr[i+1] = idx_ptr[i] + outMax[i];
        lutsize[0] = idx_ptr[BINS];
    }
                     
// for future memory access optimizations
//
//      __local int scratch1[WORKGROUP_SIZE];
//      
//      scratch1[local_index] = 0
//      
//     // Loop sequentially over chunks of input vector
//     for (int i=local_index; i < BINS; i += local_size)
//     {
//         scratch1[i] = outMax[i];
//         barrier(CLK_LOCAL_MEM_FENCE);
//         
//         if (local_index == 0)
//         {
//             for (int j=0; j<local_size; j++)
//             {
//                 if ((i+j) < BINS)
//                     outMaxCum[i+j+1] = outMaxCum[i+j] + scratch1[j];
//             }
//         }
//     }
}


__kernel
void lut3(__global float8* pos,
//             __global int*    mask,
//             __const  int     check_mask,
          __global float4* minmax,
          const    int     length,
//                   float2  pos0Range,
//                   float2  pos1Range,
          __global int*    outMax,
          __global int*    idx_ptr,
          __global int*    indices,
          __global float*  data)
{
    int global_index = get_global_id(0);
    if (global_index < length)
    {
//         float pos0_min = fmax(fmin(pos0Range.x,pos0Range.y),minmax[0].s0);
//         float pos0_max = fmin(fmax(pos0Range.x,pos0Range.y),minmax[0].s1);
        float pos0_min = minmax[0].s0;
        float pos0_maxin = minmax[0].s1;
        float pos0_max = pos0_maxin * (1 + EPS);
       // pos0_max *= 1.1;
        
        float delta = (pos0_max - pos0_min) / BINS;
        
        int local_index  = get_local_id(0);
        
        float8 pixel = pos[global_index];
        
        pixel.s0 = getBinNr(pixel.s0, delta, pos0_min);
        pixel.s2 = getBinNr(pixel.s2, delta, pos0_min);
        pixel.s4 = getBinNr(pixel.s4, delta, pos0_min);
        pixel.s6 = getBinNr(pixel.s6, delta, pos0_min);
        
        float min0 = min4f(pixel.s0, pixel.s2, pixel.s4, pixel.s6);
        float max0 = max4f(pixel.s0, pixel.s2, pixel.s4, pixel.s6);
        
        int bin0_min = floor(min0);
        int bin0_max = floor(max0);
        
        float2 AB, BC, CD, DA;
        
        AB.x=(pixel.s3-pixel.s1)/(pixel.s2-pixel.s0);
        AB.y= pixel.s1 - AB.x*pixel.s0;
        BC.x=(pixel.s5-pixel.s3)/(pixel.s4-pixel.s2);
        BC.y= pixel.s3 - BC.x*pixel.s2;
        CD.x=(pixel.s7-pixel.s5)/(pixel.s6-pixel.s4);
        CD.y= pixel.s5 - CD.x*pixel.s4;
        DA.x=(pixel.s1-pixel.s7)/(pixel.s0-pixel.s6);
        DA.y= pixel.s7 - DA.x*pixel.s6;
        
        float A_lim = pixel.s0;
        float B_lim = pixel.s2;
        float C_lim = pixel.s4;
        float D_lim = pixel.s6;
        float pixelArea  = integrate_line(A_lim, B_lim, AB);
        pixelArea += integrate_line(B_lim, C_lim, BC);
        pixelArea += integrate_line(C_lim, D_lim, CD);
        pixelArea += integrate_line(D_lim, A_lim, DA);

        pixelArea = fabs(pixelArea);
        float oneOverPixelArea = 1.0 / pixelArea;
        
        for (int bin=bin0_min; bin < bin0_max+1; bin++)
        {
            float A_lim = (pixel.s0<=bin)*(pixel.s0<=(bin+1))*bin + (pixel.s0>bin)*(pixel.s0<=(bin+1))*pixel.s0 + (pixel.s0>bin)*(pixel.s0>(bin+1))*(bin+1);
            float B_lim = (pixel.s2<=bin)*(pixel.s2<=(bin+1))*bin + (pixel.s2>bin)*(pixel.s2<=(bin+1))*pixel.s2 + (pixel.s2>bin)*(pixel.s2>(bin+1))*(bin+1);
            float C_lim = (pixel.s4<=bin)*(pixel.s4<=(bin+1))*bin + (pixel.s4>bin)*(pixel.s4<=(bin+1))*pixel.s4 + (pixel.s4>bin)*(pixel.s4>(bin+1))*(bin+1);
            float D_lim = (pixel.s6<=bin)*(pixel.s6<=(bin+1))*bin + (pixel.s6>bin)*(pixel.s6<=(bin+1))*pixel.s6 + (pixel.s6>bin)*(pixel.s6>(bin+1))*(bin+1);
            float partialArea  = integrate_line(A_lim, B_lim, AB);
            partialArea += integrate_line(B_lim, C_lim, BC);
            partialArea += integrate_line(C_lim, D_lim, CD);
            partialArea += integrate_line(D_lim, A_lim, DA);
            float tmp = fabs(partialArea) * oneOverPixelArea;
            int k = atomic_add(&outMax[bin],1);
//             if (bin == BINS)
//            printf("%d  %d  %f  %f\n", bin0_min, bin0_max, min0, max0);
//            check_atomics[atomics] = idx_ptr[bin]+k;
            indices[idx_ptr[bin]+k] = global_index;
            data[idx_ptr[bin]+k] = tmp;
        }
    }
//     if (global_index ==0)
//         printf("AAAAAAAAAAAA");
}



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
csr_integrate(  const   __global    float   *weights,
                const   __global    float   *coefs,
                const   __global    int     *row_ind,
                const   __global    int     *col_ptr,
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
        outMerge[bin_num] =  outData[bin_num] / outCount[bin_num];
    }
};//end kernel
