/*
 *   Project: Stencil based peak finder.
 *            OpenCL Kernels  
 *
 *
 *   Copyright (C) 2020-2020 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 03/08/2020
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

//read without caching. Profiling demonstrated this kernel best works without manual cache. 
float inline read_simple(global int *img, 
                     	 int height,
						 int width,
						 int row,
						 int col){
    //This kernel reads the value and returns it without active caching
    float value = NAN;
    
    // Read
    if ((col>=0) && (col<width) && (row>=0) && (row<height)){
        int read_pos = col + row*width;
        value = (float)img[read_pos];
        if (value<0){
            value = NAN;
        }
    }
    return value;
}


/* Stencil kernel that calculated the I-bg and counting pixels of interest
 * 
 * For every pixel (no preproc), the value for the background level 
 * and the std are calculated from a local patch (7x7).
 * Pixel with (Icor-Bg)>   min(cutoff*std, noise) are maked as peak-pixel, 
 * counted and their index registered in highidx
 * 
 * The kernel uses local memory for keeping track of peak count and positions
 * parameters:
 * img:    the image, tested on int (pilatus images)
 * height: the heigth of the image
 * width: the width of the image
 * half_wind_height: window size, i.e. 3 for a 7x7 window
 * half_wind_width: window size, i.e. 3 for a 7x7 window
 * threshold: keep (value-mean)/sigma>threshold
 * radius: keep pixel where centroid of patch is centered at less then this radius (1 pixel)
 * noise: minimal value for a peak, or noise level i.e. absolute threshold
 * cnt_high: returns the number of peaks found
 * high: the list of peaks (only indexes)
 * high_size: size of the allocated array (should not overflows)
 * local_high: some local memory to store the local peaks
 * local_size: size of the local memory 
 *  
 */
// workgroup size of kernel: as big as possible, (32x32) is working well. 1000 points as local peak cache 
kernel void simple_spot_finder(global int *img, 
                               int height,
                               int width,
                               int half_wind_height,
                               int half_wind_width,
                               float threshold,
                               float radius,
							   float noise,
                        global int *cnt_high, //output
                        global int *high,     //output
                               int high_size,
                        local  int *local_high,
                               int local_size){
    //decaration of variables
    int col, row, cnt, i, j, where, tid, blocksize;
    float value, sum, std, centroid_r, centroid_c, dist, mean, M2, delta, delta2, target_value, centroid;
    col = get_global_id(0);
    row = get_global_id(1);
    
    //Initialization of output array in shared
    local int local_cnt_high[2];
    blocksize = get_local_size(0) * get_local_size(1);
    tid = get_local_id(0) + get_local_id(1) * get_local_size(0);
    if (tid < 2){
        local_cnt_high[tid] = 0;
    }
        
    for (i=0; i<local_size; i+=blocksize){
        if ((i+tid)<local_size)
            local_high[i+tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);        
    
    
    //Calculate mean + std + centroids
    mean = 0.0f;
    M2 = 0.0f;
    centroid_r = 0.0f;
    centroid_c = 0.0f;
    cnt = 0;
    
    for (i=-half_wind_height; i<=half_wind_height; i++){
        for (j=-half_wind_width; j<=half_wind_width; j++){
            value = read_simple(img, height, width, row+i, col+j);
            if (isfinite(value)){
                centroid_r += value*i; 
                centroid_c += value*j;
                cnt += 1;
                delta = value - mean;
                mean += delta / cnt;
                delta2 = value - mean;
                M2 += delta * delta2;
            }                
        }
    }
    if (cnt){
        dist = mean*radius*cnt;
        std = sqrt(M2 / cnt);
        target_value = read_simple(img, height, width, row, col);
        centroid = sqrt(centroid_r*centroid_r + centroid_c*centroid_c);
        if (((target_value-mean)>min(threshold*std, noise)) && (centroid_r<dist)){
                where = atomic_inc(local_cnt_high);
                if (where<local_size){
                    local_high[where] = col+width*row;
                }
        } // if intense signal properly centered
    } // if patch not empty            
    
    //Store the results in global memory
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid==0) {
        cnt = local_cnt_high[0];
        if ((cnt>0) && (cnt<local_size)) {
            where = atomic_add(cnt_high, cnt);
            if (where+cnt>high_size){
                cnt = high_size-where; //store what we can
            }
            local_cnt_high[0] = cnt;
            local_cnt_high[1] = where;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //copy the data from local to global memory
    for (i=0; i<local_cnt_high[0]; i+=blocksize){
        high[local_cnt_high[1]+i+tid] = local_high[i+tid];
    }//store results
} //kernel
