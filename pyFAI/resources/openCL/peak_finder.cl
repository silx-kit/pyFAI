/*
 *   Project: 2D Diffraction images peak finding and integrating.
 *            OpenCL Kernels  
 *
 *
 *   Copyright (C) 2012-2019 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 04/12/2019
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
 *   WORKGROUP_SIZE: number of threads in a workgroup, 1024 is a good bet.
 *   NIMAGE: size of the image
 */

#include "for_eclipse.h"


/* Pixel-wise kernel that calculated the I-bg and counting pixels of interest
 * 
 * For every pixel in the preproc array, the value for the background level 
 * and the std are interpolated.
 * Pixel with (Icor-Bg)>   min(cutoff*std, noise) are maked as peak-pixel, 
 * counted and their index registered in highidx
 * 
 * The kernel uses local memory for keeping track of peak count and positions 
 */
kernel void find_peaks( const global  float4 *preproc4,
                        const global  float  *radius2d,
                        const global  float  *radius1d,
                        const global  float  *average1d,
                        const global  float  *std1d,
                        const global  float  radius_min,
                        const global  float  radius_max,
                        const         float   cutoff,
                        const         float   noise,
                              global  float4 *result4
                              global  int    *counter,
                              global  int    *highidx){
    int tid = get_local_id(0);
    // all thread in this WG share this local counter, upgraded at the end
    local int local_counter[1];
    local int local_highidx[WORKGROUP_SIZE];
    local_highidx[tid] = 0;
    if (tid == 0)
        local_counter[0] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int gid = get_global_id(0);
    if (gid<NIMAGE) {
        float radius = radius2d[gid];
        if ((radius>=radius_min) && (radius<radius_max)) {
            float4 value = preproc4[gid];
            if (value.s2>0.0) {
                value.s1 = value.s0 / value.s2; 
            } // normalization not null -> calculate corrected value
            else {
                value.s0 = 0.0f;
                value.s1 = 0.0f;
            } // empty pixel
            
            float pos = (radius - radius1d[0]) /  (radius1d[1] - radius1d[0]);
            int index = convert_int_rtz(pos);
            float delta = pos - index;
            value.s2 = average1d[index]*(1.0f-delta) + average1d[index+1]*(delta); // bilinear interpolation: averge
            value.s3 = average1d[index]*(1.0f-delta) + average1d[index+1]*(delta); // bilinear interpolation: std
        } //check radius range
        result4[gid] = value;
        if ((value.s1 - value.s2) > max(noise, cutoff*value.s2)){
            local_highidx[atomic_inc(local_counter)] = gid;
        }//pixel is considered of high intensity: registering it. 
    } //pixel in image
     
    //Update global memory counter
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_counter[0]){
        local int to_upgrade[1];    
        if (tid == 0) 
            to_upgrade[0] = atomic_add(counter, local_counter[0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid<to_upgrade[0])
            highidx[tid + to_upgrade[0]] = local_highidx[tid];
    } // end update global memory

} //end kernel find_peaks

// this kernel takes one
kernel void integrate_peaks(){
	
}