/************  Management of the initial step size *********************/

int inline next_step(int step, float ratio)
{
    return convert_int_rtp((float)step*ratio);
}

int inline previous_step(int step, float ratio)
{
    return convert_int_rtn((float)step/ratio);
}

// smallest step smaller than the size ... iterative version.
// Returns 0 when there is nothing to sort: with size<2 the second loop would
// never end, since previous_step(0, ratio) is 0 and 0>=0 stays true.
int inline first_step(int step, int size, float ratio)
{
    if (size<2)
        return 0;

    while (step<size)
        step=next_step(step, ratio);

    while (step>=size)
        step=previous_step(step, ratio);
    return step;
}

// returns 1 if swapped, else 0
int compare_and_swap(global volatile float* elements, int i, int j)
{
    float vi = elements[i];
    float vj = elements[j];
    if (vi>vj)
    {
        elements[i] = vj;
        elements[j] = vi;
        return 1;
    }
    else
        return 0;
}

// returns 1 if swapped, else 0
int compare_and_swap_float4(global volatile float4* elements, int i, int j)
{
    float4 vi = elements[i];
    float4 vj = elements[j];
    if (vi.s0>vj.s0)
    {
        elements[i] = vj;
        elements[j] = vi;
        return 1;
    }
    else
        return 0;
}



// returns the number of swap performed
int passe(global volatile float* elements,
          int size,
          int step,
          local int* shared)
{
    int wg = get_local_size(0);
    int tid = get_local_id(0);
    int cnt = 0;
    int i, j, k;
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (2*step>=size)
    {
        for (i=tid;i<size-step;i+=wg)
            cnt += compare_and_swap(elements, i, i+step);
    }
    else if (step == 1)
    {
        for (i=2*tid; i<size-step; i+=2*wg)
            cnt+=compare_and_swap(elements, i, i+step);
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (i=2*tid+1; i<size-step; i+=2*wg)
            cnt+=compare_and_swap(elements, i, i+step);
    }
    else
    {   // The `step` comparisons of a block are independent of each other, so
        // spread them over the whole workgroup: mapping one thread per block
        // left `size/(2*step)` threads busy and the rest idle, each walking a
        // contiguous run, which is neither parallel nor coalesced. Here thread
        // `i` handles the offset `i%step` of the block `i/step`, phase by
        // phase as before. Same set of comparisons, same two phases.
        for (i=tid; i<size; i+=wg)
        {
            j = 2*step*(i/step) + i%step;
            k = j + step;
            if (k<size)
                cnt += compare_and_swap(elements, j, k);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (i=tid; i<size; i+=wg)
        {
            j = 2*step*(i/step) + i%step + step;
            k = j + step;
            if (k<size)
                cnt += compare_and_swap(elements, j, k);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (step==1)
    {
        shared[tid] = cnt;
        return sum_int_reduction(shared);
    }
    else
        return 0;
}



// returns the number of swap performed
int passe_float4(global volatile float4* elements,
                 int size,
                 int step,
                 local int* shared)
{
    int wg = get_local_size(0);
    int tid = get_local_id(0);
    int cnt = 0;
    int i, j, k;

    if (2*step>=size)
    {
        for (i=tid;i<size-step;i+=wg)
            cnt += compare_and_swap_float4(elements, i, i+step);
    }
    else if (step == 1)
    {
        for (i=2*tid; i<size-step; i+=2*wg)
            cnt+=compare_and_swap_float4(elements, i, i+step);
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (i=2*tid+1; i<size-step; i+=2*wg)
            cnt+=compare_and_swap_float4(elements, i, i+step);
    }
    else
    {   // The `step` comparisons of a block are independent of each other, so
        // spread them over the whole workgroup: mapping one thread per block
        // left `size/(2*step)` threads busy and the rest idle, each walking a
        // contiguous run, which is neither parallel nor coalesced. Here thread
        // `i` handles the offset `i%step` of the block `i/step`, phase by
        // phase as before. Same set of comparisons, same two phases.
        for (i=tid; i<size; i+=wg)
        {
            j = 2*step*(i/step) + i%step;
            k = j + step;
            if (k<size)
                cnt += compare_and_swap_float4(elements, j, k);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (i=tid; i<size; i+=wg)
        {
            j = 2*step*(i/step) + i%step + step;
            k = j + step;
            if (k<size)
                cnt += compare_and_swap_float4(elements, j, k);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (step==1)
    {
        shared[tid] = cnt;
        return sum_int_reduction(shared);
    }
    else
        return 0;
}

/********* Comb sort of (key, payload) pairs stored as float2 ******************
 *
 * Sorting a float4 moves 16 bytes per element, half of which the comparison never
 * looks at. When the payload can be rebuilt from an index, a float2 holding the key
 * and that index halves the traffic of every pass.
 */

// returns 1 if swapped, else 0
int compare_and_swap_float2(global volatile float2* elements, int i, int j)
{
    float2 vi = elements[i];
    float2 vj = elements[j];
    if (vi.s0>vj.s0)
    {
        elements[i] = vj;
        elements[j] = vi;
        return 1;
    }
    else
        return 0;
}

// returns the number of swap performed
int passe_float2(global volatile float2* elements,
                 int size,
                 int step,
                 local int* shared)
{
    int wg = get_local_size(0);
    int tid = get_local_id(0);
    int cnt = 0;
    int i, j, k;

    if (2*step>=size)
    {
        for (i=tid;i<size-step;i+=wg)
            cnt += compare_and_swap_float2(elements, i, i+step);
    }
    else if (step == 1)
    {
        for (i=2*tid; i<size-step; i+=2*wg)
            cnt += compare_and_swap_float2(elements, i, i+step);
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (i=2*tid+1; i<size-step; i+=2*wg)
            cnt += compare_and_swap_float2(elements, i, i+step);
    }
    else
    {
        for (i=tid; i<size; i+=wg)
        {
            j = 2*step*(i/step) + i%step;
            k = j + step;
            if (k<size)
                cnt += compare_and_swap_float2(elements, j, k);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (i=tid; i<size; i+=wg)
        {
            j = 2*step*(i/step) + i%step + step;
            k = j + step;
            if (k<size)
                cnt += compare_and_swap_float2(elements, j, k);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (step==1)
    {
        shared[tid] = cnt;
        return sum_int_reduction(shared);
    }
    else
        return 0;
}

/********* Comb sort of (key, position) pairs held in local memory **************
 *
 * Same algorithm as above, but the array lives in local memory: only the keys and
 * the positions travel, 8 bytes per element instead of the 16 of a float4, and the
 * passes never reach global memory. The caller is left with the permutation in
 * `positions` and applies it to whatever payload it carries.
 */

// returns 1 if swapped, else 0
int compare_and_swap_local(local volatile float* keys,
                           local volatile int* positions,
                           int i, int j)
{
    float ki = keys[i];
    float kj = keys[j];
    if (ki>kj)
    {
        keys[i] = kj;
        keys[j] = ki;
        int pi = positions[i];
        positions[i] = positions[j];
        positions[j] = pi;
        return 1;
    }
    else
        return 0;
}

// returns the number of swap performed
int passe_local(local volatile float* keys,
                local volatile int* positions,
                int size,
                int step,
                local int* shared)
{
    int wg = get_local_size(0);
    int tid = get_local_id(0);
    int cnt = 0;
    int i, j, k;

    if (2*step>=size)
    {
        for (i=tid;i<size-step;i+=wg)
            cnt += compare_and_swap_local(keys, positions, i, i+step);
    }
    else if (step == 1)
    {
        for (i=2*tid; i<size-step; i+=2*wg)
            cnt += compare_and_swap_local(keys, positions, i, i+step);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (i=2*tid+1; i<size-step; i+=2*wg)
            cnt += compare_and_swap_local(keys, positions, i, i+step);
    }
    else
    {
        for (i=tid; i<size; i+=wg)
        {
            j = 2*step*(i/step) + i%step;
            k = j + step;
            if (k<size)
                cnt += compare_and_swap_local(keys, positions, j, k);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (i=tid; i<size; i+=wg)
        {
            j = 2*step*(i/step) + i%step + step;
            k = j + step;
            if (k<size)
                cnt += compare_and_swap_local(keys, positions, j, k);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (step==1)
    {
        shared[tid] = cnt;
        return sum_int_reduction(shared);
    }
    else
        return 0;
}

// Sorts `size` elements of `keys`/`positions` already loaded in local memory.
void combsort_local(local volatile float* keys,
                    local volatile int* positions,
                    int size,
                    local int* shared)
{
    int step = 11;     // magic value
    float ratio=1.3f;  // magic value
    int cnt = 0;

    step = first_step(step, size, ratio);

    for (step=step; step>0; step=previous_step(step, ratio))
        cnt = passe_local(keys, positions, size, step, shared);

    while (cnt)
        cnt = passe_local(keys, positions, size, 1, shared);
}

// workgroup: (wg, 1)
// grid:      (wg, nb_lines)
// shared: wg*sizeof(int)
kernel void test_combsort_float(global volatile float* elements,
                                global int* positions,
                                local  int* shared)
{
    int gid = get_group_id(1);
    int step = 11;     // magic value
    float ratio=1.3f;  // magic value
    int cnt = 0;

    int start, stop, size;
    start = (gid)?positions[gid-1]:0;
    stop = positions[gid];
    size = stop-start;

    step = first_step(step, size, ratio);

    for (step=step; step>0; step=previous_step(step, ratio))
    {
        cnt = passe(&elements[start], size, step, shared);
    }
    step = 1;
    while (cnt){
        cnt = passe(&elements[start], size, step, shared);
    }


}

// workgroup: (wg, 1)
// grid:      (wg, nb_lines)
// shared: wg*sizeof(int)
kernel void test_combsort_float4(global volatile float4* elements,
                                 global int* positions,
                                 local  int* shared)
{
    int gid = get_group_id(1);
    int step = 11;     // magic value
    float ratio=1.3f;  // magic value
    int cnt = 0;

    int start, stop, size;
    start = (gid)?positions[gid-1]:0;
    stop = positions[gid];
    size = stop-start;

    step = first_step(step, size, ratio);

    for (step=step; step>0; step=previous_step(step, ratio))
        cnt = passe_float4(&elements[start], size, step, shared);

    step = 1;
    while (cnt)
        cnt = passe_float4(&elements[start], size, step, shared);
}

// workgroup: (wg, 1)
// grid:      (wg, nb_lines)
// shared: wg*sizeof(int), keys: capacity*sizeof(float), positions: capacity*sizeof(int)
// `capacity` must be at least as large as the longest line.
kernel void test_combsort_local(global float* elements,
                                global int* positions,
                                local  int* shared,
                                local  float* keys,
                                local  int* order)
{
    int gid = get_group_id(1);
    int wg = get_local_size(0);
    int tid = get_local_id(0);

    int start, stop, size;
    start = (gid)?positions[gid-1]:0;
    stop = positions[gid];
    size = stop-start;

    for (int i=tid; i<size; i+=wg)
    {
        keys[i] = elements[start+i];
        order[i] = i;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    combsort_local(keys, order, size, shared);

    for (int i=tid; i<size; i+=wg)
        elements[start+i] = keys[i];
}

// workgroup: (wg, 1)
// grid:      (wg, nb_lines)
// shared: wg*sizeof(int)
kernel void test_combsort_float2(global volatile float2* elements,
                                 global int* positions,
                                 local  int* shared)
{
    int gid = get_group_id(1);
    int step = 11;     // magic value
    float ratio=1.3f;  // magic value
    int cnt = 0;

    int start, stop, size;
    start = (gid)?positions[gid-1]:0;
    stop = positions[gid];
    size = stop-start;

    step = first_step(step, size, ratio);

    for (step=step; step>0; step=previous_step(step, ratio))
        cnt = passe_float2(&elements[start], size, step, shared);

    while (cnt)
        cnt = passe_float2(&elements[start], size, 1, shared);
}
