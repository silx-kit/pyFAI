/*
 *   Project: 2D Diffraction images peak finding and integrating.
 *            OpenCL Kernels
 *
 *
 *   Copyright (C) 2021-2021 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 07/06/2022
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
    return (clamp(dim0, -hw, wsy + hw - 1) + hw)*width + clamp(dim1, -hw, wsx + hw - 1) + hw;
}


/*
 * For a pixel at position (gpos0, gpos1), calculate the I-<I> and associated uncertainty
 */
inline float2 correct_pixel2(
  const global  float4 *preproc4,  // pixel wise array of (signal, variance, norm, cnt)
  const global  float  *radius2d,  // contains the distance to the center for each pixel
  const global  float  *radius1d,  // Radial bin postion
  const global  float  *average1d, // average intensity in the bin
  const global  float  *std1d,     // associated deviation
  const         float   radius_min,// minimum radius
  const         float   radius_max,// maximum radius
  const         int     heigth,    // heigth of the image
  const         int     width,     // width of the image
  const         int     gpos0,     // global position 0
  const         int     gpos1,     // global position 1
  const         float   noise      // addition noise level
  ){
    float2 value = (float2)(-1.0f, 0.0f);
    float background = INFINITY;
    float uncert = 0.0f;
    if ((gpos0>=0) && (gpos0<heigth) && (gpos1>=0) && (gpos1<width)){
        int gid = gpos0 * width + gpos1;
        // Read the value and calculate the background
        float radius = radius2d[gid];
        if (isfinite(radius) && (radius>=radius_min) && (radius<radius_max)) {
            float step = radius1d[1] - radius1d[0];
            float pos = clamp((radius - radius1d[0]) / step, 0.0f, (float)(NBINS - 1));
            int index = convert_int_rtz(pos);
            if (index + 1 < NBINS) {
                float delta = pos - index;
                background = average1d[index]*(1.0f-delta) + average1d[index+1]*(delta); // linear interpolation: averge
                uncert = std1d[index]*(1.0f-delta) + std1d[index+1]*(delta); // linear interpolation: std
            }
            else { //Normal bin, using linear interpolation
                index = NBINS-1;
                background = average1d[index];
                uncert = std1d[index];
            }//Upper most bin: no interpolation
        }// this pixel is between radius_min and max
        float4 raw = preproc4[gid];
//        printf("%6.3f %6.3f %6.3f\n",uncert, noise, fast_length((float2)(uncert, noise)));
        value = (float2)((raw.s2>0.0) ? raw.s0 / raw.s2 - background : 0.0f,
                          fast_length((float2)(uncert, noise)));
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

/*peakfinder: kernel that pick peaks in an image after the sigma-clipping
 *
 * In this kernel, each thread works on one pixel and uses data from the neighborhood of radius `half_patch` size
 * The sigma-clipping step preceeding provides the mean signal and the associated uncertainty.
 * This uncertainty sees some noise added quadratically.
 *
 * A pixel is kepts if:
 *  - if it a local maximum,
 *  - its intensity is greater than `mean` plus `cutoff`*`std` (including noise)
 *  - more than `connected` pixels are also intense in the direct vinicy defined by a radius of `half_patch` size.
 *
 * For each peak is calculated:
 *  - the local sum of intensity over the vinicy, only accounting intense pixels
 *  - the associated uncertainty (variance propagation).
 *  - the center of mass of the peak
 *
 * The workgroup size should be as square as possible since the kernel first load all information and uses an explicit stencil pattern
 *
 * parameters:
 *  - preproc4: pixel wise array of (signal, variance, norm, cnt)
 *  - radius2d: Contains the distance to the beamcenter for each pixel
 *  - radius1d: Radial bin postion
 *  - average1d: average intensity for pixels at the given distance from beam-center
 *  - std1d: uncertainty associated to average1d
 *  - radius_min: inner-most radius considered (not in bin, but actual radius)
 *  - radius_min: outer-most radius considered (not in bin, but actual radius)
 *  - cutoff: Intense pixels are the onve with I > avg + cutoff*√(std²+noise²)
 *  - noise: noise level of the measurement, added quadratically to the uncertainty
 *  - heigth: heigth of the image
 *  - width: width of the image
 *  - half_patch: radius or half of the size of the patch to consider for the neighborhood definition. Usually 1 or 2 to have a 3x3 or 5x5 patch.
 *  - connected: keep peak if so many intense pixels are found in the patch
 *  - counter: Number of peaks found
 *  - highidx: index of the most intense pixel of each peak
 *  - peaks: array of struct containing the centroid0, centroid1, sum of intensity and associated uncertainty calculated over all intensi pixels in the patch
 */
kernel void peakfinder(   const global  float4 *preproc4, // Pixel wise array of (signal, variance, norm, cnt)
                          const global  float  *radius2d, // Contains the distance to the center for each pixel
                          const global  float  *radius1d, // Radial bin postion
                          const global  float  *average1d,// average intensity in the bin
                          const global  float  *std1d,    // associated deviation
                          const         float   radius_min,// minimum radius
                          const         float   radius_max,// maximum radius
                          const         float   cutoff,    // pick pixel with I>avg+cutoff*√(std²+noise²)
                          const         float   noise,     // Noise level of the measurement
                          const         int     heigth,    // heigth of the image
                          const         int     width,     // width of the image
                          const         int     half_patch,// half of the size of the patch, 1 or 2 for 3x3 or 5x5 patch
                          const         int     connected, // keep only enough pixels are above threshold in patch
//                          const         int    max_pkg,    // size of the peak array in global memory
//                          const         int    max_pkl,    // size of the peak array in local memory
                                global  int    *counter,   // Counter of the number of peaks found
                                global  int    *highidx,   // indexes of the pixels of high intensity
                                global  float4 *peaks,     // Contains center0, center1, integrated, sigma
                                local   int    *local_highidx,   // size: wg0*wg1
                                local   float4 *local_peaks, // size: wg0*wg1
                                local   float2 *buffer){   // size: (wg0+2*half_patch)*(wg1+2*half_patch)
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

    // load data into shared memory
    float2 value;
    float normed;
    int p0, p1;
    if (tid0 == 0){ // process first lines
        for (int i = -half_patch; i<0; i++){
            p0 = gid0+i; p1 = gid1;
            value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1, noise);
            set_shared2(buffer, i, tid1, half_patch, value);
        }
    }
    if (tid0 == (wg0-1)){ // process last lines
        for (int i = 0; i<half_patch; i++){
            p0 = gid0+1+i; p1 = gid1;
            value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1, noise);
            set_shared2(buffer, wg0+i, tid1, half_patch, value);
        }
    }
    if (tid1 == 0){ // process first column
        for (int i = -half_patch; i<0; i++){
            p0 = gid0; p1 = gid1+i;
            value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1, noise);
            set_shared2(buffer, tid0, i, half_patch, value);
        }
    }
    if (tid1 == (wg1-1)){ // process last column
        for (int i = 0; i<half_patch; i++){
            p0 = gid0; p1 = gid1+1+i;
            value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1, noise);
            set_shared2(buffer, tid0, wg1+i, half_patch, value);
        }
    }
    // remains the 4 corners !
    if ((tid0 == 0) && (tid1==0)){ // process first corner: top left
        for (int i = -half_patch; i<0; i++){
            for (int j = -half_patch; j<0; j++){
                p0 = gid0+i; p1 = gid1+j;
                value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1, noise);
                set_shared2(buffer, i, j, half_patch, value);
            }
        }
    }
    if ((tid0 == 0) && (tid1 == (wg1-1))){ // process second corner: top right
        for (int i = -half_patch; i<0; i++){
            for (int j = 0; j<half_patch; j++){
                p0 = gid0+i; p1 = gid1+j+1;
                value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1, noise);
                set_shared2(buffer, i, wg1+j, half_patch, value);
            }
        }
    }
    if ((tid0 == (wg0-1)) && (tid1==0)){ // process third corner: bottom left
        for (int i = 0; i<half_patch; i++){
            for (int j = -half_patch; j<0; j++){
                p0 = gid0+i+1; p1 = gid1+j;
                value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1, noise);
                set_shared2(buffer, wg0+i, j, half_patch, value);
            }
        }
    }
    if ((tid0 == (wg0-1)) && (tid1 == (wg1-1))){ // process second corner: top right
        for (int i = 0; i<half_patch; i++){
            for (int j = 0; j<half_patch; j++){
                p0 = gid0+i+1; p1 = gid1+j+1;
                value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1, noise);
                set_shared2(buffer, wg0+i, wg1+j, half_patch, value);
            }
        }
    }
    // Finally the core of the buffer:
    p0 = gid0; p1 = gid1;
    value = correct_pixel2(preproc4, radius2d, radius1d, average1d, std1d, radius_min, radius_max, heigth, width, p0, p1, noise);
    set_shared2(buffer, tid0, tid1, half_patch, value);
    // All needed data have been copied to the local buffer.

    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid) {
        int active = 0; // set to -1 to indicate an invalid neigbourhood
        // value has been already calculated
        if ((value.s0>0.0f) && (value.s0 >= cutoff * value.s1)){
//            printf("%6.3f %6.3f %6.3f\n", value.s0, value.s1, cutoff);
            float local_max = value.s0;
            float sum_int = 0.0f, sum_var=0.0f;
            float som0=0.0f, som1=0.0f;
            for (int i=-half_patch; i<=half_patch; i++){
                for (int j=-half_patch; j<=half_patch; j++){
                    float2 local_value = get_shared2(buffer, tid0+i, tid1+j, half_patch);
                    if (local_value.s0>0.0f){
                        if (local_value.s0 >= cutoff * local_value.s1)
                            active+=1;
                        local_max = max(local_max, local_value.s0);
                        som0 += i * local_value.s0;
                        som1 += j * local_value.s0;
                        sum_int += local_value.s0;
                        sum_var += local_value.s1*local_value.s1;
                    }// add pixel to intgral
                    else{
                        if (local_value.s1<=0.0f){
                            // Variance is null, the pixel is masked => complete neighborhood becomes invalid
                            active = -1;
                            break;
                        }
                    }
                } // for j
                if (active<0) break;
            } // for i
            if ((value.s0 == local_max) && (active>=connected)){
                int position = atomic_inc(local_counter);
                local_highidx[position] = gid0*width + gid1;
                local_peaks[position] = (float4)(som0/sum_int + (float)(gid0),
                                                 som1/sum_int + (float)(gid1),
                                                 sum_int,
                                                 sqrt(sum_var));
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
            peaks[write_pos] = local_peaks[tid];
        } //Thread is active for copying
    } // end update global memory
} // end kernel peakfinder
