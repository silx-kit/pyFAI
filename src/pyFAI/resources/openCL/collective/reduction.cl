
/* sum all elements in a shared memory, same size as the workgroup size 0
 *
 * Return the same sum-value in all threads.
 */

int inline sum_int_reduction(local int* shared)
{
    int wg = get_local_size(0) * get_local_size(1);
    int tid = get_local_id(0) + get_local_size(0)*get_local_id(1);

    // local reduction based implementation
    for (int stride=wg>>1; stride>0; stride>>=1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid<stride) && ((tid+stride)<wg))
            shared[tid] += shared[tid+stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int res = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return res;
}

/* sum all elements in a shared memory, same size as the workgroup size 0
 *
 * Return the same sum-value in all threads.
 */
int inline sum_int_atomic(local int* shared)
{
    int wg = get_local_size(0);
    int tid = get_local_id(0);

    barrier(CLK_LOCAL_MEM_FENCE);
    int value = shared[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid)
        atomic_add(shared, value);
    barrier(CLK_LOCAL_MEM_FENCE);
    int res = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return res;
}

/*
 * Test kernel for group wise functions
 *
 * */

// all arrays have the same shape as as the workgroup
kernel void test_sum_int_reduction(global int* input,
                                   global int* output,
                                   local  int* shared)
{
    int gid = get_global_id(0);
    int tid = get_local_id(0);
    shared[tid] = input[gid];
    output[gid] = sum_int_reduction(shared);
}

// all arrays have the same shape as as the workgroup
kernel void test_sum_int_atomic(global int* input,
                                global int* output,
                                local  int* shared)
{
    int gid = get_global_id(0);
    int tid = get_local_id(0);
    shared[tid] = input[gid];
    output[gid] = sum_int_atomic(shared);
}
