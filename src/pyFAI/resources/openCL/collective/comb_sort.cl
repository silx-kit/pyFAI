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
int inline first_step(int step, int size, float ratio)
{
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
    {
        for (i=tid*2*step; i<size-step; i+=2*step*wg)
        {
            for (j=i; j<i+step; j++)
            {
                k  = j + step;
                if (k<size)
                    cnt += compare_and_swap(elements, j, k);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (i=tid*2*step+step; i<size-step; i+=2*step*wg)
        {
            for (j=i; j<i+step; j++)
            {
                k  = j + step;
                if (k<size)
                    cnt += compare_and_swap(elements, j, k);
            }
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
    {
        for (i=tid*2*step; i<size-step; i+=2*step*wg)
        {
            for (j=i; j<i+step; j++)
            {
                k  = j + step;
                if (k<size)
                    cnt += compare_and_swap_float4(elements, j, k);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (i=tid*2*step+step; i<size-step; i+=2*step*wg)
        {
            for (j=i; j<i+step; j++)
            {
                k  = j + step;
                if (k<size)
                    cnt += compare_and_swap_float4(elements, j, k);
            }
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
    int cnt;

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
    int cnt;

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
