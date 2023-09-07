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
float inline read_simple(global float *img,
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
 * cutoff: keep (value-mean)/sigma>cutoff
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
kernel void simple_spot_finder(
                        global float *img,
                               int height,
                               int width,
                               int half_wind_height,
                               int half_wind_width,
                               float cutoff,
                               float radius,
                               float noise,
                        global int *cnt_high, //output
                        global int *high,     //output
                               int high_size,
                        volatile local  int *local_high,
                               int local_size){
    //decaration of variables
    int col, row, cnt, i, j, where, tid, blocksize;
    float value, sum, std, centroid_r, centroid_c, dist, mean, M2, delta, delta2, target_value, centroid;
    col = get_global_id(0);
    row = get_global_id(1);

    //Initialization of output array in shared
    volatile local int local_cnt_high[2];
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

    // basic check if target_value is worth processing ...
    target_value = read_simple(img, height, width, row, col);
    if (isfinite(target_value) && (target_value>noise)){
        //Calculate mean + std + centroids
        mean = target_value;
        M2 = 0.0f;
        centroid_r = 0.0f;
        centroid_c = 0.0f;
        cnt = 1;

        for (i=-half_wind_height; i<=half_wind_height; i++){
            for (j=-half_wind_width; j<=half_wind_width; j++){
                if (i || j){
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
        }
        std = sqrt(M2 / cnt);
        centroid = sqrt(centroid_r*centroid_r + centroid_c*centroid_c)/(mean*cnt);
        if (((target_value-mean) > max(noise, cutoff*std)) && (centroid<radius)){
            //printf("x=%4d y=%4d value=%6.3f mean=%6.3f std=%6.3f radius %6.3f\n",col, row, target_value, mean, std, centroid);
            where = atomic_inc(local_cnt_high);
            if (where<local_size){
                local_high[where] = col+width*row;
            }
        } // if intense signal properly centered
    } // if value worth processing

    //Store the results in global memory
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid==0) {
        cnt = local_cnt_high[0];
        if (cnt) {
            //printf("group %d, %d found %d peaks\n",cnt, (int)get_group_id(0), (int)get_group_id(1), cnt);
            cnt = min(cnt, local_size);
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
        if ((i+tid) < local_cnt_high[0]){
            high[local_cnt_high[1]+i+tid] = local_high[i+tid];
        }
    }//store results
} //kernel




// Simple kernel for resetting a buffer
kernel void
memset_int(global int *array,
                  int pattern,
                  int size){
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < size)
  {
    array[i] = pattern;
  }
}

/**
 * \brief cast values of an array of int8 into a float output array.
 *
 * - array_s8: Pointer to global memory with the input data as signed8 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
kernel void
s8_to_float(global char  *array_int,
                    int height,
                    int width,
                    char do_mask,
            global char *mask,
            global float  *array_float){
    int x, y, pos;
    x = get_global_id(0);
    y = get_global_id(1);
    //Global memory guard for padding
    if ((x<width) && (y<height)){
        pos = x + y*width;
        if (do_mask && mask[pos]){
            array_float[pos] = NAN;
        }
        else{
            array_float[pos] = (float)(array_int[pos]);
        }
    }
}


/**
 * \brief cast values of an array of uint8 into a float output array.
 *
 * - array_u8: Pointer to global memory with the input data as unsigned8 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
kernel void
u8_to_float(global unsigned char  *array_int,
                            int height,
                            int width,
                            char do_mask,
            global char *mask,
            global float *array_float){
    int x, y, pos;
    x = get_global_id(0);
    y = get_global_id(1);
    //Global memory guard for padding
    if ((x<width) && (y<height)){
        pos = x + y*width;
        if (do_mask && mask[pos]){
            array_float[pos] = NAN;
        }
        else{
            array_float[pos] = (float)(array_int[pos]);
        }
    }
}


/**
 * \brief cast values of an array of int16 into a float output array.
 *
 * - array_s16: Pointer to global memory with the input data as signed16 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
kernel void
s16_to_float(global short *array_int,
                    int height,
                    int width,
                    char do_mask,
             global char *mask,
             global float  *array_float){
    int x, y, pos;
    x = get_global_id(0);
    y = get_global_id(1);
    //Global memory guard for padding
    if ((x<width) && (y<height)){
        pos = x + y*width;
        if (do_mask && mask[pos]){
            array_float[pos] = NAN;
        }
        else{
            array_float[pos] = (float)(array_int[pos]);
        }
    }
}



/**
 * \brief cast values of an array of uint16 into a float output array.
 *
 * - array_u16: Pointer to global memory with the input data as unsigned16 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
kernel void
u16_to_float(global unsigned short *array_int,
                    int height,
                    int width,
                    char do_mask,
             global char *mask,
             global float  *array_float){
    int x, y, pos;
    x = get_global_id(0);
    y = get_global_id(1);
    //Global memory guard for padding
    if ((x<width) && (y<height)){
        pos = x + y*width;
        if (do_mask && mask[pos]){
            array_float[pos] = NAN;
        }
        else{
            array_float[pos] = (float)(array_int[pos]);
        }
    }
}

/**
 * \brief cast values of an array of uint32 into a float output array.
 *
 * - array_u32: Pointer to global memory with the input data as unsigned32 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
kernel void
u32_to_float(global unsigned int  *array_int,
                    int height,
                    int width,
                    char do_mask,
             global char *mask,
             global float  *array_float){
    int x, y, pos;
    x = get_global_id(0);
    y = get_global_id(1);
    //Global memory guard for padding
    if ((x<width) && (y<height)){
        pos = x + y*width;
        if (do_mask && mask[pos]){
            array_float[pos] = NAN;
        }
        else{
            array_float[pos] = (float)(array_int[pos]);
        }
    }
}

/**
 * \brief convert values of an array of int32 into a float output array.
 *
 * - array_int:  Pointer to global memory with the data as unsigned32 array
 * - array_float:  Pointer to global memory with the data float array
 */
kernel void
s32_to_float(global int  *array_int,
                    int height,
                    int width,
                    char do_mask,
             global char *mask,
             global float  *array_float){
  int x, y, pos;
  x = get_global_id(0);
  y = get_global_id(1);
  //Global memory guard for padding
  if ((x<width) && (y<height)){
	pos = x + y*width;
	if (do_mask && mask[pos]){
	    array_float[pos] = NAN;
	}
	else{
	    array_float[pos] = (float)(array_int[pos]);
	}
  }
}

/**
 * \brief convert values of an array of int32 into a float output array.
 *
 * - array_int:  Pointer to global memory with the data as unsigned32 array
 * - array_float:  Pointer to global memory with the data float array
 */
kernel void
f32_to_float(global float  *array_inp,
                    int height,
                    int width,
                    char do_mask,
             global char *mask,
             global float *array_float){
  int x, y, pos;
  x = get_global_id(0);
  y = get_global_id(1);
  //Global memory guard for padding
  if ((x<width) && (y<height)){
    pos = x + y*width;
    if (do_mask && mask[pos]){
        array_float[pos] = NAN;
    }
    else{
        array_float[pos] = (array_inp[pos]);
    }
  }
}

// A simple kernel to copy the intensities of the peak

kernel void copy_intense(global int *peak_position,
                      global int *counter,
                      global float *preprocessed,
                      global float *peak_intensity){
    int cnt, tid;
    tid = get_global_id(0);
    cnt = counter[0];
    if (tid<cnt){
        peak_intensity[tid] =  preprocessed[peak_position[tid]];
    }
}
