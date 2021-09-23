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


#include "for_eclipse.h"

// Deprecated!
inline int _calc_buffer(int dim0, int dim1){   
    return (clamp(dim0, -1, 1)+1)*(WORKGROUP_SIZE+2) + clamp(dim1, -1, WORKGROUP_SIZE + 1)+1;
}
/* Calculate the position in the local 1D buffer taking into account the local space is 2D.
:param dim0: local position in dim0, i.e. y
:param dim1: local position in dim1, i.e. x
:param hw: half-width of the patch size, usually 1 or 2.
:return: 1d buffer index for the given position.
*/
inline int _calc_buffer2(int dim0, int dim1, int hw){
    int wsy = get_local_size(0);
    int wsx = get_local_size(1);
    int width = wsx + 2 * hw;
    return (clamp(dim0, -hw, wsy + hw) + hw)*width + clamp(dim1, -hw, wsx + hw) + hw

inline void set_shared(local float* buffer, float4 value, int dim0, int dim1){
    // Normalize the intensity of the pixel and store it
    buffer[_calc_buffer(dim0, dim1)] = (value.s2>0.0) ? value.s0 / value.s2 : 0.0f;
}

inline float set_shared2(local float* buffer, int lpos0, int lpos1, int hw,
                        global  float4 *preproc4, 
                        const         int gpos0, 
                        const         int gpos1,
                        const         int     heigth,    // heigth of the image
                        const         int     width,     // width of the image
                        const global float* radius2d,
                        ){
    float value=0.0f;
    float background = INFINITY
    float uncert=0.0f;
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
        }

    }
    
    
    // Normalize the intensity of the pixel and store it
    buffer[_calc_buffer2(ldim0, ldim1, hw)] = max(0.0f, value);
}


inline float get_shared(local float* buffer, int dim0, int dim1){
    return buffer[_calc_buffer(dim0, dim1)];
}


kernel void peakfinder8(  global  float4 *preproc4,       // both input and output, pixel wise array of (signal, variance, norm, cnt) 
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
                          const         int     half_patch,// half of the size of the patch, 1 or 2 
                          const         int     connected, // keep only enough pixels are above threshold in 3x3 patch
                                global  int    *counter,   // Counter of the number of peaks found
                                global  int    *highidx,   // indexes of the pixels of high intensity
                                global  float  *integrated,// Sum of signal in the patch
                                global  float  *center0,   // Center of mass of the peak along dim0
                                global  float  *center1){  // Center of mass of the peak along dim1
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
    local int local_highidx[WORKGROUP_SIZE]; //This array does not deserve to be initialized
    local float local_integrated[WORKGROUP_SIZE];  //This array does not deserve to be initialized
    local float local_center0[WORKGROUP_SIZE];    //This array does not deserve to be initialized
    local float local_center1[WORKGROUP_SIZE];   //This array does not deserve to be initialized
    local float buffer[3*WORKGROUP_SIZE+6];     //This array does not deserve to be initialized
    
    // load data into shared memory
    float4 value;
    float normed;
    value = ((gid - width  >= 0) && (gid - width < NIMAGE))?preproc4[gid-width]:(float4)(0.0f, 0.0f, 0.0f, 0.0f);
    set_shared(buffer, value, -1, tid);
    value = ((gid >= 0) && (gid < NIMAGE))?preproc4[gid]:(float4)(0.0f, 0.0f, 0.0f, 0.0f);
    set_shared(buffer, value, 0, tid);
    value = ((gid + width  >= 0) && (gid + width < NIMAGE))?preproc4[gid+width]:(float4)(0.0f, 0.0f, 0.0f, 0.0f);
    set_shared(buffer, value, 1, tid);
    if (tid == 0){
        int delta;
        delta = -1;
        value = ((gid + delta >= 0) && (gid + delta < NIMAGE)) ? preproc4[gid + delta]: (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        set_shared(buffer, value, 0, delta);

        delta = WORKGROUP_SIZE;
        value = ((gid + delta >= 0) && (gid + delta < NIMAGE)) ? preproc4[gid + delta]: (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        set_shared(buffer, value, 0, delta);

        delta = -width-1;
        value = ((gid + delta >= 0) && (gid + delta < NIMAGE)) ? preproc4[gid + delta]: (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        set_shared(buffer, value, -1, -1);

        delta = -width+WORKGROUP_SIZE;
        value = ((gid + delta >= 0) && (gid + delta < NIMAGE)) ? preproc4[gid + delta]: (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        set_shared(buffer, value, -1, WORKGROUP_SIZE);

        delta = +width-1;
        value = ((gid + delta >= 0) && (gid + delta < NIMAGE)) ? preproc4[gid + delta]: (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        set_shared(buffer, value, 1, -1);

        delta = +width+WORKGROUP_SIZE;
        value = ((gid + delta >= 0) && (gid + delta < NIMAGE)) ? preproc4[gid + delta]: (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        set_shared(buffer, value, 1, WORKGROUP_SIZE);
    }// All needed data have been copied to the local buffer.
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (gid<NIMAGE) {
        int active = 0;
        float background = INFINITY, uncert=0.0f;
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
        }
        
        float local_max = 0.0f; 
        float thres = background + max(noise, cutoff*uncert);
        float sub, sum_int = 0.0f;
        float som0=0.0f, som1=0.0f;
        for (int i0=-1; i0<2; i0++){
            for (int i1=-1; i1<2; i1++){
                normed = get_shared(buffer, i0, i1+tid);
                if (normed  > thres){
                    active +=1;
                }
                local_max = max(local_max, normed);
                sub = max(0.0f, normed - background);
                som0 += i0 * sub;
                som1 += i1 * sub;
                sum_int += sub;
            } //loop over the patch 3x3 dim1
        } //loop over the patch 3x3 dim0
        
        normed = get_shared(buffer, 0, tid);
        if ((normed == local_max) && (active>=connected)){
            int position = atomic_inc(local_counter);
            local_highidx[position] = gid;
            local_integrated[position] = sum_int;
            local_center0[position] = som0/sum_int + (float)(gid/width);
            local_center1[position] = som1/sum_int + (float)(gid%width);
        }//pixel is considered of high intensity: registering it. 
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
    
} // end kernel find_peaks8
