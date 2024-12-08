
/*
 * Cumsum all elements in a shared memory along dim 0
 *
 * Nota: the first wg-size elements, are used.
 * The shared buffer needs to be twice this size of the workgroup
 *
 * Implements Hillis and Steele algorithm
 * https://en.wikipedia.org/wiki/Prefix_sum#cite_ref-hs1986_9-0
 *
 */

#define SWAP_LOCAL_FLOAT(a,b) {__local float *tmp=a;a=b;b=tmp;}

void static inline cumsum_scan_float(local float* shared)
{
    int wg = get_local_size(0);
    int tid = get_local_id(0);
    // Split the input buffer in two parts
    local float* buf1 = shared;
    local float* buf2 = &shared[wg];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i=1; i<wg; i<<=1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid>=i)
            buf2[tid] = buf1[tid] + buf1[tid-i];
        else
            buf2[tid] = buf1[tid];
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP_LOCAL_FLOAT(buf1, buf2);
    }
    // Finally copy all values to both halfs of the
    barrier(CLK_LOCAL_MEM_FENCE);
    buf2[tid] = buf1[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
}


/*
 * Kernel to test the cumsum_scan_float collective function
 *
 * Note the shared buffer needs to be twice the size of the workgroup !
 */

kernel void test_cumsum(global float* input,
                        global float* output,
                        local  float* shared)
{
    int gid = get_global_id(0);
    int wg = get_local_size(0);
    int tid = get_local_id(0);

    shared[tid] = input[gid];
    cumsum_scan_float(shared);
    output[gid] = shared[tid];
}

/*
 * Exclusive prefix sum in a shared memory
 *
 * Implements Blelloch algorithm
 * https://en.wikipedia.org/wiki/Prefix_sum#cite_ref-offman_10-0
 *
 * One workgroup calculates the cumsum in an array of twice its size!
 */


void static inline blelloch_scan_float(local float *shared)
{
    int ws = get_local_size(0);
    int lid = get_local_id(0);
    int dp = 1;
    int w;

    for(int s = ws; s > 0; s >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < s)
        {
            int i = dp*(2*lid+1)-1;
            int j = i + dp;
            shared[j] += shared[i];
        }
        dp <<= 1;
    }

    dp >>= 1;
    for(int s = 1; s < ws; s=((s+1)<<1)-1)
    {
        w = dp;
        dp >>= 1;

        barrier(CLK_LOCAL_MEM_FENCE);

        if(lid < s) {
            int i = (lid+1)*w - 1;
            int j = i + dp;
            shared[j] += shared[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void test_blelloch_scan(global float *input,
                               global float *output,
                               local float *shared)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    shared[2*lid] = input[2*gid];
    shared[2*lid+1] = input[2*gid+1];

    blelloch_scan_float(shared);

    output[2*gid] = shared[2*lid];
    output[2*gid+1] = shared[2*lid+1];
}

kernel void test_blelloch_multi(global float *input,
                                global float *output,
                                       int size,
                                local float *shared)
{
    int lid = get_local_id(0);
    int ws = get_local_size(0);
    float sum = 0.0f;
    int target = (size + ws - 1) & ~ (ws - 1);
    for (int i=lid; i<target; i+=2*ws)
        {
            // Load
            shared[lid] = (i<size)?input[i]:0.0f;
            shared[lid+ws] = (i+ws<size)?input[i+ws]:0.0f;

            blelloch_scan_float(shared);

            // Store
            if (i<size)
                output[i] = sum + shared[lid];
            if (i+ws<size)
                output[i+ws] = sum + shared[lid+ws];
            sum += shared[2*ws-1];
        }
}
