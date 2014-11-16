
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
 * \brief Sets the values of 3 float output arrays to zero.
 *
 * Gridsize = size of arrays + padding.
 *
 * @param array0: int Pointer to global memory with the outMax array
 */
__kernel void
memset_outMax(__global int *array0)
{
    int i = get_global_id(0);
    //Global memory guard for padding
    if(i < BINS)
        array0[i]=0;
}

__kernel
void reduce_minmax_1(__global float2* buffer,
             __global float4* preresult) {
    
    
    int global_index = get_global_id(0);
    int global_size  = get_global_size(0);
    float4 accumulator;
    accumulator.x = INFINITY;
    accumulator.y = -INFINITY;
    accumulator.z = INFINITY;
    accumulator.w = -INFINITY;
    
    // Loop sequentially over chunks of input vector
    while (global_index < POS_SIZE/2) {
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
void reduce_minmax_2(__global float4* preresult,
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

__kernel
void lut_1(__global float8* pos,
//             __global int*    mask,
//             __const  int     check_mask,
           __global float4* minmax,
                    float2 pos0Range,
                    float2 pos1Range,
           __global int*  outMax)
{ 
    int global_index = get_global_id(0);
    if (global_index < SIZE)
    {
        int tmp_bool = (pos0Range.x == pos0Range.y); //(== 0)
        float pos0_min   = !tmp_bool*fmin(pos0Range.x,pos0Range.y) + tmp_bool*minmax[0].s0;
        float pos0_maxin = !tmp_bool*fmax(pos0Range.x,pos0Range.y) + tmp_bool*minmax[0].s1;
//        float pos0_min = minmax[0].s0;
//        float pos0_maxin = minmax[0].s1;
        float pos0_max = pos0_maxin*( 1 + EPS);



        
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
        
//        if (bin0_min >= 0) && (bin0_max < BINS) // do i count half pixels
        
        for (int bin=bin0_min; bin < bin0_max+1; bin++)
        {
            atomic_add(&outMax[bin], 1);
        }
    }
}

// to be run with global_size = local_size
__kernel
void lut_2(__global int*  outMax,
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
void lut_3(__global float8* pos,
//             __global int*    mask,
//             __const  int     check_mask,
          __global float4* minmax,
          __global float2* pos0Range,
          __global float2* pos1Range,
          __global int*    outMax,
          __global int*    idx_ptr,
          __global int*    indices,
          __global float*  data)
{
    int global_index = get_global_id(0);
    if (global_index < SIZE)
    {
//         float pos0_min = fmax(fmin(pos0Range.x,pos0Range.y),minmax[0].s0);
//         float pos0_max = fmin(fmax(pos0Range.x,pos0Range.y),minmax[0].s1);
        float pos0_min = minmax[0].s0;
        float pos0_max = minmax[0].s1;
        pos0_max *= 1 + EPS;
        
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
            float bin0 = bin - bin0_min;
            float A_lim = (pixel.s0<=bin0)*(pixel.s0<=(bin0+1))*bin0 + (pixel.s0>bin0)*(pixel.s0<=(bin0+1))*pixel.s0 + (pixel.s0>bin0)*(pixel.s0>(bin0+1))*(bin0+1);
            float B_lim = (pixel.s2<=bin0)*(pixel.s2<=(bin0+1))*bin0 + (pixel.s2>bin0)*(pixel.s2<=(bin0+1))*pixel.s2 + (pixel.s2>bin0)*(pixel.s2>(bin0+1))*(bin0+1);
            float C_lim = (pixel.s4<=bin0)*(pixel.s4<=(bin0+1))*bin0 + (pixel.s4>bin0)*(pixel.s4<=(bin0+1))*pixel.s4 + (pixel.s4>bin0)*(pixel.s4>(bin0+1))*(bin0+1);
            float D_lim = (pixel.s6<=bin0)*(pixel.s6<=(bin0+1))*bin0 + (pixel.s6>bin0)*(pixel.s6<=(bin0+1))*pixel.s6 + (pixel.s6>bin0)*(pixel.s6>(bin0+1))*(bin0+1);
            float partialArea  = integrate_line(A_lim, B_lim, AB);
            partialArea += integrate_line(B_lim, C_lim, BC);
            partialArea += integrate_line(C_lim, D_lim, CD);
            partialArea += integrate_line(D_lim, A_lim, DA);
            float tmp = fabs(partialArea) * oneOverPixelArea;
            int k = atomic_add(&outMax[bin],1);
            indices[idx_ptr[bin]+k] = global_index;
            data[idx_ptr[bin]+k] = (bin0_min==bin0_max) ? 1 : tmp; // The 2 methods of calculating area give slightly different results. This complicated things when pixel splitting doesn't occur
        }
    }
}
