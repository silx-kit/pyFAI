/*
 *   Project: Azimuthal  integration for PyFAI.
 *            Reduction Kernels
 *
 *
 *   Copyright (C) 2014 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: Giannis Ashiotis <giannis.ashiotis@gmail.com>
 *   					J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 20/10/2014
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#include "for_eclipse.h"
 
__kernel
void reduce1(__global float* buffer,
             __const int length,
             __global float2* preresult) {
    
    
    int global_index = get_global_id(0);
    int global_size  = get_global_size(0);
    float2 accumulator;
    accumulator.x = INFINITY;
    accumulator.y = -INFINITY;
    // Loop sequentially over chunks of input vector
    while (global_index < length) {
        float element = buffer[global_index];
        accumulator.x = (accumulator.x < element) ? accumulator.x : element;
        accumulator.y = (accumulator.y > element) ? accumulator.y : element;
        global_index += global_size;
    }
    
    __local float2 scratch[WORKGROUP_SIZE];

    // Perform parallel reduction
    int local_index = get_local_id(0);
    
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int active_threads = get_local_size(0);
    
    while (active_threads != 2)
    {
        active_threads /= 2;
        if (thread_id_loc < active_threads)
        {
            float2 other = scratch[local_index + active_threads];
            float2 mine  = scratch[local_index];
            mine.x = (mine.x < other.x) ? mine.x : other.x;
            mine.y = (mine.y > other.y) ? mine.y : other.y;
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
void reduce2(__global float2* preresult,
             __global float4* result) {
    
    
    __local float2 scratch[WORKGROUP_SIZE];

    int local_index = get_local_id(0);
    
    scratch[local_index] = preresult[local_index];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int active_threads = get_local_size(0);
    
    while (active_threads != 2)
    {
        active_threads /= 2;
        if (thread_id_loc < active_threads)
        {
            float2 other = scratch[local_index + active_threads];
            float2 mine  = scratch[local_index];
            mine.x = (mine.x < other.x) ? mine.x : other.x;
            mine.y = (mine.y > other.y) ? mine.y : other.y;
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
        result[0] = vload4(0,scratch);
    }
}
