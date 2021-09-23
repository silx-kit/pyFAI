/*
 *   Project: 2D Diffraction images peak finding and integrating.
 *            OpenCL Kernels  
 *
 *
 *   Copyright (C) 2021-2021 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 23/09/2021
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

/* Constant to be provided at build time:
    - NBINS Number of bins in the radial dimention
 */

#include "for_eclipse.h"

/* 
 * Calculate the position in the local 1D buffer taking into account the local space is 2D.
    :param dim0: local position in dim0, i.e. y
    :param dim1: local position in dim1, i.e. x
    :param hw: half-width of the patch size, usually 1 or 2.
    :return: 1d buffer index for the given position.
*/
inline int pf8_calc_buffer2(int dim0, int dim1, int hw){
    int wsy = get_local_size(0);
    int wsx = get_local_size(1);
    int width = wsx + 2 * hw;
    return (clamp(dim0, -hw, wsy + hw) + hw)*width + clamp(dim1, -hw, wsx + hw) + hw;
}           


/*  
 * For a pixel at position (gpos0, gpos1), calculate the I-<I> and associated uncertainty
 */  
inline float2 correct_pixel2(
        global  float4 *preproc4,  // both input and output, pixel wise array of (signal, variance, norm, cnt)
  const global  float  *radius2d,  // contains the distance to the center for each pixel
  const global  float  *radius1d,  // Radial bin postion 
  const global  float  *average1d, // average intensity in the bin
  const global  float  *std1d,     // associated deviation
  const         float   radius_min,// minimum radius
  const         float   radius_max,// maximum radius 
  const         int     heigth,    // heigth of the image
  const         int     width,     // width of the image
  const         int     gpos0,     // global position 0 
  const         int     gpos1      // global position 1
  ){
    float2 value = (float2)(0.0f, 0.0f);
    float background = INFINITY;
    float uncert = 0.0f;
    if ((gpos0>=0) && (gpos0<heigth) && (gpos0>=0) &&(gpos1<width)){
        int gid = gpos0 * width + gpos1;
        // Read the value and calculate the background
        float radius = radius2d[gid];
        if (isfinite(radius) && (radius>=radius_min) && (radius<radius_max)) {
            float step = radius1d[1] - radius1d[0];
            float pos = clamp((radius - radius1d[0]) / step, 0.0f, (float)NBINS);
            int index = convert_int_rtz(pos);
            float delta = pos - index;
            if (index + 1 < NBINS) {
                background = average1d[index]*(1.0f-delta) + average1d[index+1]*(delta); // linear interpolation: averge
                uncert = std1d[index]*(1.0f-delta) + std1d[index+1]*(delta); // linear interpolation: std               
            }
            else { //Normal bin, using linear interpolation
                background = average1d[index];
                uncert = std1d[index];
            }//Upper most bin: no interpolation
        }// this pixel is between radius_min and max
        float4 raw = preproc4[gid];
        value = (float2)((raw.s2>0.0) ? raw.s0 / raw.s2 - background : 0.0f, uncert);
    }// this pixel is valid
    return value;
}
    
  
            
inline void set_shared2(
                local float2* buffer,    // local storage      
        const         int     lpos0,     // local position 0
        const         int     lpos1,     // local position 1
        const         int     half_patch,// half of the size of the patch, 1 or 2 for 3x3 or 5x5 patch
        const         float2  value      // 
                                         ){
    // Normalize the intensity of the pixel and store it
    buffer[pf8_calc_buffer2(lpos0, lpos1, half_patch)] = value;
}

/* 
 * Retrieve the background subtracted intensity and the associated uncertainty from the local buffer 
 */
inline float2 get_shared2(local float2* buffer, int dim0, int dim1, int half_patch){
    return buffer[pf8_calc_buffer2(dim0, dim1, half_patch)];
}


kernel void peakfinder8(        global  float4 *preproc4, // both input and output, pixel wise array of (signal, variance, norm, cnt) 
                          const global  float  *radius2d, // contains the distance to the center for each pixel
                          const global  float  *radius1d, // Radial bin postion 
                          const global  float  *average1d,// average intensity in the bin
                          const global  float  *std1d,    // associated deviation
                          const         float   radius_min,// minimum radius
                          const         float   radius_max,// maximum radius 
                          const         float   cutoff,    // pick pixel with I>avg+min(cutoff*std, noise)
                          const         float   noise,     // Noise level of the measurement
                          const         int     heigth,    // heigth of the image
                          const         int     width,     // width of the image
                          const         int     half_patch,// half of the size of the patch, 1 or 2 for 3x3 or 5x5 patch
                          const         int     connected, // keep only enough pixels are above threshold in patch
                                global  int    *counter,   // Counter of the number of peaks found
                                global  int    *highidx,   // indexes of the pixels of high intensity
                                global  float  *integrated,// Sum of signal in the patch
                                global  float  *center0,   // Center of mass of the peak along dim0
                                global  float  *center1,   // Center of mass of the peak along dim1
                                local   int    *local_highidx,   // size: wg0*wg1
                                local   float *local_integrated, // size: wg0*wg1
                                local   float *local_center0,    // size: wg0*wg1
                                local   float *local_center1,    // size: wg0*wg1
                                local   float2 *buffer){         // size: (wg0+2*half_patch)*(wg1+2*half_patch) 
    int tid0 = get_local_id(0);
    int tid1 = get_local_id(1);
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int wg0 = get_local_size(0);
    int wg1 = get_local_size(1);
    
    int tid = tid0*wg1 + tid1; 
    bool valid = (gid0<heigth) && (gid1<width); //mind 0=y, 1=x !
    
    // all thread in this WG share this local counter, upgraded at the end
    volatile local int local_counter[2]; //first element MUST be set to zero
    if (tid<2) local_counter[tid] = 0;
    // those buffer should come from the caller !
    //local int local_highidx[WORKGROUP_SIZE]; //This array does not deserve to be initialized
    //local float local_integrated[WORKGROUP_SIZE];  //This array does not deserve to be initialized
    //local float local_center0[WORKGROUP_SIZE];    //This array does not deserve to be initialized
    //local float local_center1[WORKGROUP_SIZE];   //This array does not deserve to be initialized
    //local float buffer[3*WORKGROUP_SIZE+6];     //This array does not deserve to be initialized
    
    // load data into shared memory
    float2 value;
    float normed;
    int p0, p1;
    if (tid0 == 0){ // process first lines
        for (int i = -half_patch; i<0; i++){
            p0 = gid0+i; p1 = gid1;
            value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1);
            set_shared2(buffer, i, tid1, half_patch, value);
        }
    }
    if (tid0 == (wg0-1)){ // process last lines
        for (int i = 0; i<half_patch; i++){
            p0 = gid0+1+i; p1 = gid1;
            value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1);
            set_shared2(buffer, wg0+i, tid1, half_patch, value);
        }
    }
    if (tid1 == 0){ // process first column
        for (int i = -half_patch; i<0; i++){
            p0 = gid0; p1 = gid1+i;
            value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1);
            set_shared2(buffer, tid0, i, half_patch, value);
        }
    }
    if (tid1 == (wg1-1)){ // process last column
        for (int i = 0; i<half_patch; i++){
            p0 = gid0; p1 = gid1+1+i;
            value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1);
            set_shared2(buffer, tid0, wg1+i, half_patch, value);
        }
    }
    // remains the 4 corners !
    if ((tid0 == 0) && (tid1==0)){ // process first corner: top left
        for (int i = -half_patch; i<0; i++){
            for (int j = -half_patch; j<0; j++){
                p0 = gid0+i; p1 = gid1+j;
                value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1);
                set_shared2(buffer, i, j, half_patch, value);
            }
        }
    }
    if ((tid0 == 0) && (tid1 == (wg1-1))){ // process second corner: top right
        for (int i = -half_patch; i<0; i++){
            for (int j = 0; j<half_patch; j++){
                p0 = gid0+i; p1 = gid1+j+1;
                value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1);
                set_shared2(buffer, i, wg1+j, half_patch, value);
            }
        }
    }
    if ((tid0 == (wg0-1)) && (tid1==0)){ // process third corner: bottom left
        for (int i = 0; i<half_patch; i++){
            for (int j = -half_patch; j<0; j++){
                p0 = gid0+i+1; p1 = gid1+j;
                value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1);
                set_shared2(buffer, wg0+i, j, half_patch, value);
            }
        }
    }
    if ((tid0 == (wg0-1)) && (tid1 == (wg1-1))){ // process second corner: top right
        for (int i = 0; i<half_patch; i++){
            for (int j = 0; j<half_patch; j++){
                p0 = gid0+i+1; p1 = gid1+j+1;
                value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1);
                set_shared2(buffer, wg0+i, wg1+j, half_patch, value);
            }
        }
    }
    // Finally the core of the buffer:
    p0 = gid0; p1 = gid1; 
    value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1);
    set_shared2(buffer, tid0, tid1, half_patch, value);
    // All needed data have been copied to the local buffer.
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (valid) {
        int active = 0;
        // value has been already calculated
        float local_value = value.s0; 
        if (value.s0 >= max(noise, cutoff*value.s1)){
            float local_max = 0.0f;
            float sub, sum_int = 0.0f;
            float som0=0.0f, som1=0.0f;    
            for (int i=-half_patch; i<=half_patch; i++){
                for (int j=-half_patch; j<=half_patch; j++){
                    value = get_shared2(buffer, tid0+i, tid1+j);
                    if (value.s0 >= max(noise, cutoff*value.s1)){
                        active+=1;
                    }
                    local_max = max(local_max, value.s0);
                    sub = max(0.0f, value.s0);
                    som0 += i * sub;
                    som1 += j * sub;
                    sum_int += sub;
                }
            }
            if ((local_value == local_max) && (active>=connected)){
                int position = atomic_inc(local_counter);
                local_highidx[position] = gid0*width + gid1;
                local_integrated[position] = sum_int;
                local_center0[position] = som0/sum_int + (float)(gid0);
                local_center1[position] = som1/sum_int + (float)(gid1);
            }// Record pixel
        }//pixel is high
    } // pixel is in image
    
    //Update global memory counter
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_counter[0]){
        if (tid == 0) local_counter[1] = atomic_add(counter, local_counter[0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid<local_counter[0]){
            int write_pos =  tid +local_counter[1];
            highidx[write_pos] = local_highidx[tid];
            integrated[write_pos] = local_integrated[tid];
            center0[write_pos] = local_center0[tid];
            center1[write_pos] = local_center1[tid];        
        } //Thread is active for copying
    } // end update global memory
} // end kernel peakfinder8
