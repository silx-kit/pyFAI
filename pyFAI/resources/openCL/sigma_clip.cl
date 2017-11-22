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
static inline float2 is_valid(float value, float count)
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
    // implements Kahan summation:
    // see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    float2 tmp, sum1, sum2;
    float value;

    tmp = is_valid(data.s0, 0.0f);
    value = tmp.s0;
    sum1 = (float2)(value, 0.0f);
    sum2 = (float2)(value * value, 0.0f);

    tmp = is_valid(data.s1, tmp.s1);
    value = tmp.s0;
    sum1 = kahan_sum(sum1, value);
    sum2 = kahan_sum(sum2, value*value);

    tmp = is_valid(data.s2, tmp.s1);
    value = tmp.s0;
    value = tmp.s0;
    sum1 = kahan_sum(sum1, value);
    sum2 = kahan_sum(sum2, value*value);

    tmp = is_valid(data.s3, tmp.s1);
    value = tmp.s0;
    value = tmp.s0;
    sum1 = kahan_sum(sum1, value);
    sum2 = kahan_sum(sum2, value*value);

    tmp = is_valid(data.s4, tmp.s1);
    value = tmp.s0;
    value = tmp.s0;
    sum1 = kahan_sum(sum1, value);
    sum2 = kahan_sum(sum2, value*value);

    tmp = is_valid(data.s5, tmp.s1);
    value = tmp.s0;
    value = tmp.s0;
    sum1 = kahan_sum(sum1, value);
    sum2 = kahan_sum(sum2, value*value);

    tmp = is_valid(data.s6, tmp.s1);
    value = tmp.s0;
    value = tmp.s0;
    sum1 = kahan_sum(sum1, value);
    sum2 = kahan_sum(sum2, value*value);

    tmp = is_valid(data.s7, tmp.s1);
    value = tmp.s0;
    value = tmp.s0;
    sum1 = kahan_sum(sum1, value);
    sum2 = kahan_sum(sum2, value*value);

    return (float8)(sum1.s0, sum1.s1, sum2.s0, sum2.s1, tmp.s1, 0.0f, 0.0f, 0.0f);
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
            float2 sum1, sum2;
            int local_pos, remote_pos;
            local_pos = 5 * local_id;
            remote_pos = 5 * (local_id + stride_size);

            sum1 = compensated_sum((float2)(l_data[local_pos], l_data[local_pos+1]),
                                   (float2)(l_data[remote_pos], l_data[remote_pos + 1]));
            sum2 = compensated_sum((float2)(l_data[local_pos + 2], l_data[local_pos+3]),
                                   (float2)(l_data[remote_pos + 2], l_data[remote_pos + 3]));

            l_data[local_pos] = sum1.s0;
            l_data[local_pos + 1] = sum1.s1;
            l_data[local_pos + 2] = sum2.s0;
            l_data[local_pos + 3] = sum2.s1;
            l_data[local_pos + 4] += l_data[remote_pos + 4];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        stride_size /=2;
    }

    // Here we perform the Kahan summation for the variance
    float std, mean, n;
    n = l_data[4];
    // if (local_id==0)  printf("%.1f %.1f %.1f %.1f %.1f\n",l_data[0], l_data[1],l_data[2],l_data[3],l_data[4]);
    // if (local_id==0)  printf("(%d, %d) %.1f %.1f %.1f %.1f %.1f\n",get_global_id(0), get_global_id(1), l_data[0], l_data[1], l_data[2], l_data[3], l_data[4]);

    if (fabs(n) < 0.5f)
    {
        mean = NAN;
        std = NAN;
    }
    else
    {
        mean = l_data[0] / n;

        float2 sum1, sum2, sum;
        sum1 = (float2)(l_data[0], l_data[1]);
        sum2 = (float2)(l_data[2], l_data[3]);

        // here we perform the Kahan summation
        // sigma**2 = (sum_x2 - (sum_x)**2/n )/n

        sum = kahan_sum(sum2, -sum1.s0*sum1.s0/n);
        sum = kahan_sum(sum, -sum1.s0*sum1.s1/n);
        sum = kahan_sum(sum, -sum1.s1*sum1.s1/n);

        std = sqrt(sum.s0/n);
    }
    return (float2) (mean, std);
}

static inline float8 clip8(float8 input, float2 mean_std,
                           float sigma_lo, float sigma_hi,
                           local int* discarded)
{
    union
    {
        float  array[8];
        float8 vector;
    } elements;

    elements.vector = input;
    for (int i=0; i<8; i++)
    {
        if (!isfinite(elements.array[i]) || mean_std.s1 == 0.0f)
        {
            elements.array[i] = NAN;
        }
        else
        {
            float ratio = (elements.array[i] - mean_std.s0) / mean_std.s1;
            if (ratio > sigma_hi)
            {
                elements.array[i] = NAN;
                atomic_inc(discarded);

            }

            else if (-ratio > sigma_lo)
            {
                elements.array[i] = NAN;
                atomic_inc(discarded);
            }
        }
    }
    return elements.vector;
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
Shared memory: requires 5 floats (20 bytes) of shared memory per work-item
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

    value = src[global_start];
    input.s0 = (value==dummy)?NAN:value;
    value = src[global_start + padding];
    input.s1 = (value==dummy)?NAN:value;
    value = src[global_start + 2*padding];
    input.s2 = (value==dummy)?NAN:value;
    value = src[global_start + 3*padding];
    input.s3 = (value==dummy)?NAN:value;
    value = src[global_start + 4*padding];
    input.s4 = (value==dummy)?NAN:value;
    value = src[global_start + 5*padding];
    input.s5 = (value==dummy)?NAN:value;
    value = src[global_start + 6*padding];
    input.s6 = (value==dummy)?NAN:value;
    value = src[global_start + 7*padding];
    input.s7 = (value==dummy)?NAN:value;

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
Shared memory: requires 5 float (20 bytes) of shared memory per work-item
*/
kernel void mean_std_horizontal(global float *src,
                                 global float *mean,
                                 global float *std,
                                 float dummy,
                                 local float *l_data)
{
    float8 input;
    float2 result;
    float value;
    uint global_start, offset;

    // Find global address
    offset = get_global_size(1) * get_global_id(0) * 8;
    global_start = offset + get_group_id(1) * get_local_size(1) * 8 + get_local_id(1) * 8;

    value = src[global_start];
    input.s0 = (value==dummy)?NAN:value;
    value = src[global_start + 1];
    input.s1 = (value==dummy)?NAN:value;
    value = src[global_start + 2];
    input.s2 = (value==dummy)?NAN:value;
    value = src[global_start + 3];
    input.s3 = (value==dummy)?NAN:value;
    value = src[global_start + 4];
    input.s4 = (value==dummy)?NAN:value;
    value = src[global_start + 5];
    input.s5 = (value==dummy)?NAN:value;
    value = src[global_start + 6];
    input.s6 = (value==dummy)?NAN:value;
    value = src[global_start + 7];
    input.s7 = (value==dummy)?NAN:value;

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
:param mean: 1D array of floats of size width containing the mean of the array
             along the vertical direction
:param mean: 1D array of floats of size width containing the standard deviation
             of the array along the vertical direction
:param dummy: value of the invalid data
:param sigma_lo: lower cut-of for <I> - I  > sigma_lo * sigma
:param sigma_hi: higher cut-of for I - <I> > sigma_hi * sigma
:param max_iter: Max number of iteration

Each workgroup works on a complete column, using subsequent reductions (sum) for
mean calculation and standard deviation

dim0 = y: wg=number_of_element/8
dim1 = x: wg=1
Shared memory: requires 5 floats (20 bytes) of shared memory per work-item
*/
kernel void sigma_clip_vertical(global float *src,
                                global float *mean,
                                global float *std,
                                float dummy,
                                float sigma_lo,
                                float sigma_hi,
                                int max_iter,
                                local float *l_data)
{
    // we need to read 8 float position along the vertical axis
    float8 input;
    float2 result;
    uint id, global_start, padding, i;
    float value;
    local int discarded[1];

    // Find global address
    padding = get_global_size(1);
    id = get_local_id(0) * 8 * padding + get_global_id(1);
    global_start = get_group_id(0) * get_local_size(0) * 8 * padding + id;

    value = src[global_start];
    input.s0 = (value==dummy)?NAN:value;
    value = src[global_start + padding];
    input.s1 = (value==dummy)?NAN:value;
    value = src[global_start + 2*padding];
    input.s2 = (value==dummy)?NAN:value;
    value = src[global_start + 3*padding];
    input.s3 = (value==dummy)?NAN:value;
    value = src[global_start + 4*padding];
    input.s4 = (value==dummy)?NAN:value;
    value = src[global_start + 5*padding];
    input.s5 = (value==dummy)?NAN:value;
    value = src[global_start + 6*padding];
    input.s6 = (value==dummy)?NAN:value;
    value = src[global_start + 7*padding];
    input.s7 = (value==dummy)?NAN:value;

    result = mean_and_deviation(get_local_id(0), get_local_size(0),
                                input, l_data);
    for (i=0; i<max_iter; i++)
    {

        if (get_local_id(0) == 0)
        {
            discarded[0] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        input = clip8(input, result, sigma_lo, sigma_hi, discarded);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (discarded[0] == 0){
            break;
        }
        else
        {
            result = mean_and_deviation(get_local_id(0), get_local_size(0),
                                        input, l_data);
        }

    }
    if (get_local_id(0) == 0)
    {
        //printf("Discarded %d %d %d\n", get_global_id(1), i,  discarded[0]);
        mean[get_global_id(1)] = result.s0;
        std[get_global_id(1)] = result.s1;
    }
}


/*

sigma_clip_horizontal reject iteratively all point at n sigma from the mean along
a horizontal line.

:param src: 2D array of floats of size width*height
:param mean: 1D array of floats of size width containing the mean of the array
             along the horizontal direction
:param mean: 1D array of floats of size width containing the standard deviation
             of the array along the horizontal direction
:param dummy: value of the invalid data
:param sigma_lo: lower cut-of for <I> - I  > sigma_lo * sigma
:param sigma_hi: higher cut-of for I - <I> > sigma_hi * sigma
:param max_iter: Max number of iteration

Each workgroup works on a complete column, using subsequent reductions (sum) for
mean calculation and standard deviation

dim0 = y: wg=1
dim1 = x: wg=number_of_element/8
Shared memory: requires 5 float (20 bytes) of shared memory per work-item
*/
kernel void sigma_clip_horizontal(global float *src,
                                  global float *mean,
                                  global float *std,
                                  float dummy,
                                  float sigma_lo,
                                  float sigma_hi,
                                  int max_iter,
                                  local float *l_data)
{
    // we need to read 8 float position along the vertical axis
    float8 input;
    float2 result;
    float value;
    uint global_start, offset, i;
    local int discarded[1];

    // Find global address
    offset = get_global_size(1) * get_global_id(0) * 8;
    global_start = offset + get_group_id(1) * get_local_size(1) * 8 + get_local_id(1) * 8;

    value = src[global_start];
    input.s0 = (value==dummy)?NAN:value;
    value = src[global_start + 1];
    input.s1 = (value==dummy)?NAN:value;
    value = src[global_start + 2];
    input.s2 = (value==dummy)?NAN:value;
    value = src[global_start + 3];
    input.s3 = (value==dummy)?NAN:value;
    value = src[global_start + 4];
    input.s4 = (value==dummy)?NAN:value;
    value = src[global_start + 5];
    input.s5 = (value==dummy)?NAN:value;
    value = src[global_start + 6];
    input.s6 = (value==dummy)?NAN:value;
    value = src[global_start + 7];
    input.s7 = (value==dummy)?NAN:value;

    result = mean_and_deviation(get_local_id(1), get_local_size(1),
                                input, l_data);
    for (i=0; i<max_iter; i++)
    {

        if (get_local_id(1) == 0)
        {
            discarded[0] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        input = clip8(input, result, sigma_lo, sigma_hi, discarded);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (discarded[0] == 0){
            break;
        }
        else
        {
            result = mean_and_deviation(get_local_id(1), get_local_size(1),
                                        input, l_data);
        }

    }
    if (get_local_id(1) == 0)
    {
        // printf("Discarded (%d,%d) %d %d\n", get_global_id(0), get_global_id(1), i,  discarded[0]);
        mean[get_global_id(0)] = result.s0;
        std[get_global_id(0)] = result.s1;
    }
}

