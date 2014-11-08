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
void integrate1(__global float8* pos,
                __global float*  image,
    //             __global int*    mask,
    //             __const  int     check_mask,
                __global float4* minmax,
                const    int     length,
      //                   float2  pos0Range,
      //                   float2  pos1Range,
  //              const    int     do_dummy,
   //             const    float   dummy,
                __global float*  outData,
                __global float*  outCount)
{
    int global_index = get_global_id(0);
    if (global_index < length)
    {
//         float pos0_min = fmax(fmin(pos0Range.x,pos0Range.y),minmax[0].s0);
//         float pos0_max = fmin(fmax(pos0Range.x,pos0Range.y),minmax[0].s1);
        float pos0_min = minmax[0].s0;
        float pos0_max = minmax[0].s1;
        pos0_max *= 1 + EPS;
        
        float delta = (pos0_max - pos0_min) / BINS;
        
        int local_index  = get_local_id(0);
        
        float8 pixel = pos[global_index];
        float  data  = image[global_index];
        
        pixel.s0 = getBinNr(pixel.s0, delta, pos0_min);
        pixel.s2 = getBinNr(pixel.s2, delta, pos0_min);
        pixel.s4 = getBinNr(pixel.s4, delta, pos0_min);
        pixel.s6 = getBinNr(pixel.s6, delta, pos0_min);
        
        float min0 = min4f(pixel.s0, pixel.s2, pixel.s4, pixel.s6);
        float max0 = max4f(pixel.s0, pixel.s2, pixel.s4, pixel.s6);
        
        int bin0_min = floor(min0);
        int bin0_max = floor(max0);
        
        float2 AB, BC, CD, DA;
        
        pixel.s0 -= bin0_min;
        pixel.s2 -= bin0_min;
        pixel.s4 -= bin0_min;
        pixel.s6 -= bin0_min;
        
        AB.x=(pixel.s3-pixel.s1)/(pixel.s2-pixel.s0);
        AB.y= pixel.s1 - AB.x*pixel.s0;
        BC.x=(pixel.s5-pixel.s3)/(pixel.s4-pixel.s2);
        BC.y= pixel.s3 - BC.x*pixel.s2;
        CD.x=(pixel.s7-pixel.s5)/(pixel.s6-pixel.s4);
        CD.y= pixel.s5 - CD.x*pixel.s4;
        DA.x=(pixel.s1-pixel.s7)/(pixel.s0-pixel.s6);
        DA.y= pixel.s7 - DA.x*pixel.s6;
        
        float areaPixel = area4(pixel.s0, pixel.s1, pixel.s2, pixel.s3, pixel.s4, pixel.s5, pixel.s6, pixel.s7);
        float oneOverPixelArea = 1.0 / areaPixel;
        for (int bin=bin0_min; bin < bin0_max+1; bin++)
        {
//             float A_lim = (pixel.s0<=bin)*(pixel.s0<=(bin+1))*bin + (pixel.s0>bin)*(pixel.s0<=(bin+1))*pixel.s0 + (pixel.s0>bin)*(pixel.s0>(bin+1))*(bin+1);
//             float B_lim = (pixel.s2<=bin)*(pixel.s2<=(bin+1))*bin + (pixel.s2>bin)*(pixel.s2<=(bin+1))*pixel.s2 + (pixel.s2>bin)*(pixel.s2>(bin+1))*(bin+1);
//             float C_lim = (pixel.s4<=bin)*(pixel.s4<=(bin+1))*bin + (pixel.s4>bin)*(pixel.s4<=(bin+1))*pixel.s4 + (pixel.s4>bin)*(pixel.s4>(bin+1))*(bin+1);
//             float D_lim = (pixel.s6<=bin)*(pixel.s6<=(bin+1))*bin + (pixel.s6>bin)*(pixel.s6<=(bin+1))*pixel.s6 + (pixel.s6>bin)*(pixel.s6>(bin+1))*(bin+1);
            int bin0 = bin - bin0_min;
            float A_lim = (pixel.s0<=bin0)*(pixel.s0<=(bin0+1))*bin0 + (pixel.s0>bin0)*(pixel.s0<=(bin0+1))*pixel.s0 + (pixel.s0>bin0)*(pixel.s0>(bin0+1))*(bin0+1);
            float B_lim = (pixel.s2<=bin0)*(pixel.s2<=(bin0+1))*bin0 + (pixel.s2>bin0)*(pixel.s2<=(bin0+1))*pixel.s2 + (pixel.s2>bin0)*(pixel.s2>(bin0+1))*(bin0+1);
            float C_lim = (pixel.s4<=bin0)*(pixel.s4<=(bin0+1))*bin0 + (pixel.s4>bin0)*(pixel.s4<=(bin0+1))*pixel.s4 + (pixel.s4>bin0)*(pixel.s4>(bin0+1))*(bin0+1);
            float D_lim = (pixel.s6<=bin0)*(pixel.s6<=(bin0+1))*bin0 + (pixel.s6>bin0)*(pixel.s6<=(bin0+1))*pixel.s6 + (pixel.s6>bin0)*(pixel.s6>(bin0+1))*(bin0+1);
            float partialArea  = integrate_line(A_lim, B_lim, AB);
            partialArea += integrate_line(B_lim, C_lim, BC);
            partialArea += integrate_line(C_lim, D_lim, CD);
            partialArea += integrate_line(D_lim, A_lim, DA);
            float tmp = fabs(partialArea) * oneOverPixelArea;
//            outCount[bin] += tmp;
//            outData[bin]  ++= data*tmp;
             AtomicAdd(&outCount[bin], tmp); 
             AtomicAdd(&outData[bin], data*tmp);
            
        }
    }
}


__kernel
void integrate2(__global float*  outData,
                __global float*  outCount,
                __global float*  outMerge)
{
    int global_index = get_global_id(0);
    if (global_index < BINS)
        outMerge[global_index] = outData[global_index]/outCount[global_index];
}
