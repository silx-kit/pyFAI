float integrate_line( float A0, float B0, float2 AB)
{
    return (A0==B0) ? 0.0 : AB.s0*(B0*B0 - A0*A0)*0.5 + AB.s1*(B0-A0);
}


float getBinNr(float x0, float pos0_min, float dpos)
{
    return (x0 - pos0_min) / dpos;
}


float min4f(float a, float b, float c, float d)
{
    float tmp1 = a <= b ? a : b;
    float tmp2 = c <= d ? c : d;
    return tmp1 <= tmp2 ? tmp1 : tmp2;
}


float max4f(float a, float b, float c, float d)
{
    float tmp1 = a >= b ? a : b;
    float tmp2 = c >= d ? c : d;
    return tmp1 >= tmp2 ? tmp1 : tmp2;
}


float2 minmax(float a, float b, float c, float d)
{
    float2 tmp0, tmp1, tmp2;
    tmp0.s0 = a <= b ? a : b;
    tmp0.s1 = a >= b ? a : b;
    tmp1.s0 = c <= d ? c : d;
    tmp1.s1 = c >= d ? c : d;
    
    tmp2.s0 = tmp0.s0 <= tmp1.s0 ? tmp0.s0 : tmp1.s0;
    tmp2.s1 = tmp0.s1 <= tmp1.s1 ? tmp0.s1 : tmp1.s1;
    return tmp2;
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
void integrate(__global float8* pos,
               __global float*  image,
  //             __global int*    mask,
  //             __const  int     check_mask,
               __global float4* minmax,
               __const  int     length,
                        float2  pos0Range,
                        float2  pos1Range)
{
    float pos0_min, pos0_max;
    {
        float tmp = fmin(pos0Range.x,pos0Range.y);
        pos0_min  = (tmp > minmax.s0) ? tmp : minmax.s0;
        tmp = fmax(pos0Range.x,pos0Range.y);
        pos0_max  = (tmp < minmax.s1) ? tmp : minmax.s1;
        pos0_max *= 1 + EPS;
    }
    float delta = (pos0_max - pos0_min) / BINS;

    int local_index  = get_local_id(0);
    int global_index = get_global_id(0);

    
}