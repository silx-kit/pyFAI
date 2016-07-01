/*
 * Project: Azimuthal integration
 *       https://github.com/pyFAI/pyFAI
 *
 * Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
 *
 * Principal author:       Jerome Kieffer (Jerome.Kieffer@ESRF.eu)
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
 *
 */

/*
 * Copy the content of a src array into a dst array,
 * padding if needed with the dummy value
*/
__kernel void copy_pad(__global float *src,
                       __global float *dst,
                       uint src_size,
                       uint dst_size,
                       float dummy)
{
  uint gid = get_global_id(0);
  //Global memory guard for padding
  if(gid < dst_size)
  {
    if (gid < src_size)
    {
      dst[gid] = src[gid];
    }
    else
    {
      dst[gid] = dummy;
    }


  }
}
/*

filter_vertical filters out a sorted array, counts the number of non dummy values and
copies the according value at the position depending on the quantile.

@param src: 2D array of floats of size width*height
@param dst: 1D array of floats of size width
@param width:
@param height:
@param dummy: value of the invalid data
@param quantile: between 0 and 1

Each thread works on a complete column, counting the elements and copying the right one
*/
__kernel void filter_vertical(__global float *src,
                              __global float *dst,
                              uint width,
                              uint height,
                              float dummy,
                              float quantile){
  uint gid = get_global_id(0);
  //Global memory guard for padding
  uint cnt = 0, pos=0;
  float data;
  if(gid < width){
	  for (pos=0; pos<height*width; pos+=width){
        data = src[gid+pos];
        if (data!=dummy){
        	cnt++;
        }
	  }
	  if (cnt){
		  pos = round(quantile*cnt);
		  pos = min(pos+height-cnt, height-1);
		  dst[gid] = src[gid + width * pos];
	  }else{
		  dst[gid] = dummy;
	  }
  }
}
/*

filter_horizontal filters out a sorted array, counts the number of non dummy values and
copies the according value at the position depending on the quantile.

@param src: 2D array of floats of size width*height
@param dst: 1D array of floats of size width
@param width:
@param height:
@param dummy: value of the invalid data
@param quantile: between 0 and 1

Each thread works on a complete column, counting the elements and copying the right one
*/
__kernel void filter_horizontal(__global float *src,
                                __global float *dst,
                                uint width,
                                uint height,
                                float dummy,
                                float quantile){
  uint gid = get_global_id(0);
  //Global memory guard for padding
  uint cnt = 0, pos=0, offset=gid*width;
  float data;

  if(gid < height){
	  for (pos=0; pos<width; pos++){
        data = src[offset+pos];
        if (data!=dummy){
        	cnt++;
        }
	  }
	  if (cnt){
		  pos = round(quantile*cnt);
		  pos = min(pos+width-cnt, width-1);
		  dst[gid] = src[offset+pos];
	  }else{
		  dst[gid] = dummy;
	  }
  }
}
