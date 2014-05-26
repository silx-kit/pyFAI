__kernel
void reduce1(__global float2* buffer,
             __const int length,
             __global float4* preresult) {
    
    
    int global_index = get_global_id(0);
    int global_size  = get_global_size(0);
    float4 accumulator;
    accumulator.x = INFINITY;
    accumulator.y = -INFINITY;
    accumulator.z = INFINITY;
    accumulator.w = -INFINITY;
    
    // Loop sequentially over chunks of input vector
    while (global_index < length/2) {
        float2 element = buffer[global_index];
        accumulator.x = (accumulator.x < element.s0) ? accumulator.x : element.s0;
        accumulator.y = (accumulator.y > element.s0) ? accumulator.y : element.s0;
        accumulator.z = (accumulator.z < element.s1) ? accumulator.z : element.s1;
        accumulator.w = (accumulator.w > element.s1) ? accumulator.w : element.s1;
        global_index += global_size;
    }
    
    __local float4 scratch[WORKGROUP_SIZE];

    // Perform parallel reduction
    int local_index = get_local_id(0);
    
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int active_threads = get_local_size(0);
    
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (local_index < active_threads)
        {
            float4 other = scratch[local_index + active_threads];
            float4 mine  = scratch[local_index];
            mine.x = (mine.x < other.x) ? mine.x : other.x;
            mine.y = (mine.y > other.y) ? mine.y : other.y;
            mine.z = (mine.z < other.z) ? mine.z : other.z;
            mine.w = (mine.w > other.w) ? mine.w : other.w;
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
void reduce2(__global float4* preresult,
             __global float4* result) {
    
    
    __local float4 scratch[WORKGROUP_SIZE];

    int local_index = get_local_id(0);
    
    scratch[local_index] = preresult[local_index];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int active_threads = get_local_size(0);
    
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (local_index < active_threads)
        {
            float4 other = scratch[local_index + active_threads];
            float4 mine  = scratch[local_index];
            mine.x = (mine.x < other.x) ? mine.x : other.x;
            mine.y = (mine.y > other.y) ? mine.y : other.y;
            mine.z = (mine.z < other.z) ? mine.z : other.z;
            mine.w = (mine.w > other.w) ? mine.w : other.w;
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
        result[0] = scratch[0];
    }
}
