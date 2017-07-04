/*
 * Project: Azimuthal integration
 *       https://github.com/silx-kit/pyFAI
 *
 * Copyright (C) 2015-2017 European Synchrotron Radiation Facility, Grenoble, France
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

// Functions to be called from an actual kernel.

// check for NANs and discard them, count the number of valid values
static float2 is_valid(float value, float count)
{
    if (isfinite(value))
        {
            count += 1.0f;
        }
        else
        {
            value = 0.0f;
        }
    return (float2)(value, count);
}

// sum_vector return sum(x_i), err(sum(x_i)), sum(x_i^2 ), err(sum(x_i^2 ))
static float8 sum_vector(float8 data)
{
    //implements Kahan summation:
    // see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    float8 result;
    float2 tmp;
    float sum1=0.0f, err_sum1=0.0f, sum2=0.0f, err_sum2=0.0f, y1, t1, y2, t2;
    float value;

    tmp = is_valid(data.s0, 0.0f);
    value = tmp.s0;
    sum1 = value;
    sum2 = data.s0 * data.s0;

    tmp = is_valid(data.s1, tmp.s1);
    value = tmp.s0;
    y1 =  value;
    y2 =  value * value;
    t1 = sum1 + y1;
    t2 = sum2 + y2;
    err_sum1 = (t1 - sum1) - y1;
    err_sum2 = (t2 - sum2) - y2;
    sum1 = t1;
    sum2 = t2;

    tmp = is_valid(data.s2, tmp.s1);
    value = tmp.s0;
    y1 =  value - err_sum1;
    y2 =  value*value - err_sum2;
    t1 = sum1 + y1;
    t2 = sum2 + y2;
    err_sum1 = (t1 - sum1) - y1;
    err_sum2 = (t2 - sum2) - y2;
    sum1 = t1;
    sum2 = t2;

    tmp = is_valid(data.s3, tmp.s1);
    value = tmp.s0;
    y1 =  value - err_sum1;
    y2 =  value * value - err_sum2;
    t1 = sum1 + y1;
    t2 = sum2 + y2;
    err_sum1 = (t1 - sum1) - y1;
    err_sum2 = (t2 - sum2) - y2;
    sum1 = t1;
    sum2 = t2;

    tmp = is_valid(data.s4, tmp.s1);
    value = tmp.s0;
    y1 =  value - err_sum1;
    y2 =  value * value - err_sum2;
    t1 = sum1 + y1;
    t2 = sum2 + y2;
    err_sum1 = (t1 - sum1) - y1;
    err_sum2 = (t2 - sum2) - y2;
    sum1 = t1;
    sum2 = t2;

    tmp = is_valid(data.s5, tmp.s1);
    value = tmp.s0;
    y1 =  value - err_sum1;
    y2 =  value * value - err_sum2;
    t1 = sum1 + y1;
    t2 = sum2 + y2;
    err_sum1 = (t1 - sum1) - y1;
    err_sum2 = (t2 - sum2) - y2;
    sum1 = t1;
    sum2 = t2;

    tmp = is_valid(data.s6, tmp.s1);
    value = tmp.s0;
    y1 =  value - err_sum1;
    y2 =  value * value - err_sum2;
    t1 = sum1 + y1;
    t2 = sum2 + y2;
    err_sum1 = (t1 - sum1) - y1;
    err_sum2 = (t2 - sum2) - y2;
    sum1 = t1;
    sum2 = t2;

    tmp = is_valid(data.s7, tmp.s1);
    value = tmp.s0;
    y1 =  value - err_sum1;
    y2 =  value * value - err_sum2;
    t1 = sum1 + y1;
    t2 = sum2 + y2;
    err_sum1 = (t1 - sum1) - y1;
    err_sum2 = (t2 - sum2) - y2;
    sum1 = t1;
    sum2 = t2;

    return (float8)(sum1, err_sum1, sum2, err_sum2, tmp.s1, 0.0f, 0.0f, 0.0f);
}

// calculate the mean and the standard deviation sigma with reductions.
static float2 mean_and_deviation(uint local_id,
                                 uint local_size,
                                 float8 input,
                                 local float *l_data)
{
    // inspired from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    float8 map = sum_vector(input);
    l_data[local_id * 5 ] = map.s0;
    l_data[local_id * 5 + 1] = map.s1;
    l_data[local_id * 5 + 2] = map.s2;
    l_data[local_id * 5 + 3] = map.s3;
    l_data[local_id * 5 + 4] = map.s4;

    uint stride_size = local_size / 2;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Start parallel reduction
    while (stride_size > 0)
    {
        if (local_id < stride_size)
        {
            float sum1, err_sum1, sum2, err_sum2, y1, t1, y2, t2;
            int local_pos, remote_pos;
            local_pos = 5 * local_id;
            remote_pos = 5 * (local_id + stride_size);

            sum1 = l_data[local_pos];
            err_sum1 = l_data[local_pos + 1] + l_data[remote_pos + 1];
            sum2 = l_data[local_pos + 2];
            err_sum2 = l_data[local_pos + 3] + l_data[remote_pos + 3];

            y1 =  l_data[remote_pos] - err_sum1;
            y2 =  l_data[remote_pos + 2] - err_sum2;
            t1 = sum1 + y1;
            t2 = sum2 + y2;
            err_sum1 = (t1 - sum1) - y1;
            err_sum2 = (t2 - sum2) - y2;
            sum1 = t1;
            sum2 = t2;

            l_data[local_pos] = sum1;
            l_data[local_pos + 1] = err_sum1;
            l_data[local_pos + 2] = sum2;
            l_data[local_pos + 3] = err_sum2;
            l_data[local_pos + 4] += l_data[remote_pos + 4];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        stride_size /=2;
    }

    //Here we perform the Kahan summation for the variance
    float std, mean, sum1, err_sum1, sum2, err_sum2, n, y, t, sum, err;
    sum1 = l_data[0];
    err_sum1 = l_data[1];
    sum2 = l_data[2];
    err_sum1 = l_data[3];
    n = l_data[4];
    mean = sum1 / n;

    // here we perform the Kahan summation
    // sigma2 = (sum_x2 - (sum_x)**2/n )/n

    y = - sum1*sum1/n - err_sum2;
    t = sum2 + y;
    err = (t - sum2) - y;
    sum = t;

    y = -2.0*sum1*err_sum1/n - err;
    t = sum + y;
    err = (t - sum) - y;
    sum = t;

    y = -err_sum1*err_sum1/n - err;
    t = sum + y;
    err = (t - sum) - y;
    sum = t;

    std = sqrt(sum/n);
    return (float2) (mean, std);
}

/*

mean_std_vertical calculate the mean and the standard deviation along a column,
 vertical line.

:param src: 2D array of floats of size width*height
:param mean: 1D array of floats of size height
:param std: 1D array of floats of size height
:param width:
:param height:
:param dummy: value of the invalid data

Each workgroup works on a complete column, using subsequent reductions (sum) for
mean calculation and standard deviation

dim0 = y: wg=number_of_element/8
dim1 = x: wg=1
*/


kernel void mean_std_vertical(global float *src,
                              global float *mean,
                              global float *std,
                              float dummy,
                              local float *l_data
                              )
{
    // we need to read 8 float position along the vertical axis
    float8 input;
    float2 result;
    uint id, global_start, padding;
    float value;

    // Find global address
    padding = get_global_size(1);
    id = get_local_id(0) * 8 * padding + get_global_id(1);
    global_start = get_group_id(0) * get_local_size(0) * 8 * padding + id;

    value = src[global_start            ];
    input.s0 = value=dummy?NAN:value;
    value = src[global_start + padding  ];
    input.s1 = value=dummy?NAN:value;
    value = src[global_start + 2*padding];
    input.s2 = value=dummy?NAN:value;
    value = src[global_start + 3*padding];
    input.s3 = value=dummy?NAN:value;
    value = src[global_start + 4*padding];
    input.s4 = value=dummy?NAN:value;
    value = src[global_start + 5*padding];
    input.s5 = value=dummy?NAN:value;
    value = src[global_start + 6*padding];
    input.s6 = value=dummy?NAN:value;
    value = src[global_start + 7*padding];
    input.s7 = value=dummy?NAN:value;

      result = mean_and_deviation(get_local_id(0), get_local_size(0),
                                  input, l_data);
      if (get_local_id(0) == 0)
      {
          mean[get_global_id(1)] = result.s0;
          std[get_global_id(1)] = result.s1;
      }
}

/*

mean_std_horizontal calculate the mean and the standard deviation along a row,
horizontal line.

:param src: 2D array of floats of size width*height
:param mean: 1D array of floats of size height
:param std: 1D array of floats of size height
:param width:
:param height:
:param dummy: value of the invalid data

Each workgroup works on a complete row, using subsequent reductions (sum) for
mean calculation and standard deviation

dim0 = y: wg=1
dim1 = x: wg=number_of_element/8
*/
kernel void mean_std_horizontal(global float *src,
                                 global float *mean,
                                 global float *std,
                                 float dummy,
                                 local float *l_data)
{
    float8 input, output;
    float2 result;
    uint id, global_start, offset;

    // Find global address
    offset = get_global_size(1)*get_global_id(0)*8;
    id = get_local_id(1) * 8;
    global_start = offset + get_group_id(1) * get_local_size(1) * 8 + id;

    input = (float8)(src[global_start    ],
                     src[global_start + 1],
                     src[global_start + 2],
                     src[global_start + 3],
                     src[global_start + 4],
                     src[global_start + 5],
                     src[global_start + 6],
                     src[global_start + 7]);

    result = mean_and_deviation(get_local_id(1), get_local_size(1),
                                input, l_data);
    if (get_local_id(1) == 0)
    {
        mean[get_global_id(0)] = result.s0;
        std[get_global_id(0)] = result.s1;
    }
}

/*

sigma_clip_vertical reject iteratively all point at n sigma from the mean along
a vertical line.

:param src: 2D array of floats of size width*height
:param dst: 1D array of floats of size width
:param width:
:param height:
:param dummy: value of the invalid data
:param sigma_lo: lower cut-of for <I> - I  > sigma_lo * sigma
:param sigma_hi: higher cut-of for I - <I> > sigma_hi * sigma
:param max_iter: Max number of iteration

Each workgroup works on a complete column, using subsequent reductions (sum) for
mean calculation and standard deviation
*/
kernel void sigma_clip_vertical(global float *src,
                                global float *dst,
                                uint width,
                                uint height,
                                float dummy,
                                float quantile)
{
  //TODO
    uint gid = get_global_id(0);
  //Global memory guard for padding
  uint cnt = 0, pos=0;
  float data;
  if(gid < width){
	  for (pos=0; pos<height*width; pos+=width)
	  {
        data = src[gid+pos];
        if (data!=dummy)
        {
        	cnt++;
        }
	  }
	  if (cnt)
	  {
		  pos = round(quantile*cnt);
		  pos = min(pos+height-cnt, height-1);
		  dst[gid] = src[gid + width * pos];
	  }
	  else
	  {
		  dst[gid] = dummy;
	  }
  }
}
/*

sigma_clip_horizontal reject iteratively all point at n sigma from the mean along
a horizontal line.

:param src: 2D array of floats of size width*height
:param dst: 1D array of floats of size width
:param width:
:param height:
:param dummy: value of the invalid data
:param quantile: between 0 and 1

:param src: 2D array of floats of size width*height
:param dst: 1D array of floats of size width
:param width:
:param height:
:param dummy: value of the invalid data
:param sigma_lo: lower cut-of for <I> - I  > sigma_lo * sigma
:param sigma_hi: higher cut-of for I - <I> > sigma_hi * sigma
:param max_iter: Max number of iteration

Each workgroup works on a complete column, using subsequent reductions (sum) for
mean calculation and standard deviation
*/
kernel void sigma_clip_horizontal(global float *src,
                                  global float *dst,
                                  uint width,
                                  uint height,
                                  float dummy,
                                  float quantile)
{
  uint gid = get_global_id(0);
  //Global memory guard for padding
  uint cnt = 0, pos=0, offset=gid*width;
  float data;

  if(gid < height)
  {
	  for (pos=0; pos<width; pos++)
	  {
        data = src[offset+pos];
        if (data!=dummy)
        {
        	cnt++;
        }
	  }
	  if (cnt)
	  {
		  pos = round(quantile*cnt);
		  pos = min(pos+width-cnt, width-1);
		  dst[gid] = src[offset+pos];
	  }
	  else
	  {
		  dst[gid] = dummy;
	  }
  }
}

