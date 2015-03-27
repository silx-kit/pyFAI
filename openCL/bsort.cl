# Taken from "OpenCL in Action" by Matthew Scarpino. License: "public domain"
/* Sort elements within a vector */
#define VECTOR_SORT(input, dir)                                   \
   comp = input < shuffle(input, mask2) ^ dir;                    \
   input = shuffle(input, as_uint4(comp * 2 + add2));             \
   comp = input < shuffle(input, mask1) ^ dir;                    \
   input = shuffle(input, as_uint4(comp + add1));                 \

#define VECTOR_SWAP(input1, input2, dir)                          \
   temp = input1;                                                 \
   comp = (input1 < input2 ^ dir) * 4 + add3;                     \
   input1 = shuffle2(input1, input2, as_uint4(comp));             \
   input2 = shuffle2(input2, temp, as_uint4(comp));               \

/* Perform initial sort */
__kernel void bsort_init(__global float4 *g_data, __local float4 *l_data) {

   int dir;
   uint id, global_start, size, stride;
   float4 input1, input2, temp;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(1, 2, 2, 3);

   id = get_local_id(0) * 2;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   input1 = g_data[global_start];
   input2 = g_data[global_start+1];

   /* Sort input 1 - ascending */
   comp = input1 < shuffle(input1, mask1);
   input1 = shuffle(input1, as_uint4(comp + add1));
   comp = input1 < shuffle(input1, mask2);
   input1 = shuffle(input1, as_uint4(comp * 2 + add2));
   comp = input1 < shuffle(input1, mask3);
   input1 = shuffle(input1, as_uint4(comp + add3));

   /* Sort input 2 - descending */
   comp = input2 > shuffle(input2, mask1);
   input2 = shuffle(input2, as_uint4(comp + add1));
   comp = input2 > shuffle(input2, mask2);
   input2 = shuffle(input2, as_uint4(comp * 2 + add2));
   comp = input2 > shuffle(input2, mask3);
   input2 = shuffle(input2, as_uint4(comp + add3));

   /* Swap corresponding elements of input 1 and 2 */
   add3 = (int4)(4, 5, 6, 7);
   dir = get_local_id(0) % 2 * -1;
   temp = input1;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));

   /* Sort data and store in local memory */
   VECTOR_SORT(input1, dir);
   VECTOR_SORT(input2, dir);
   l_data[id] = input1;
   l_data[id+1] = input2;

   /* Create bitonic set */
   for(size = 2; size < get_local_size(0); size <<= 1) {
      dir = (get_local_id(0)/size & 1) * -1;

      for(stride = size; stride > 1; stride >>= 1) {
         barrier(CLK_LOCAL_MEM_FENCE);
         id = get_local_id(0) + (get_local_id(0)/stride)*stride;
         VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
      }

      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) * 2;
      input1 = l_data[id]; input2 = l_data[id+1];
      temp = input1;
      comp = (input1 < input2 ^ dir) * 4 + add3;
      input1 = shuffle2(input1, input2, as_uint4(comp));
      input2 = shuffle2(input2, temp, as_uint4(comp));
      VECTOR_SORT(input1, dir);
      VECTOR_SORT(input2, dir);
      l_data[id] = input1;
      l_data[id+1] = input2;
   }

   /* Perform bitonic merge */
   dir = (get_group_id(0) % 2) * -1;
   for(stride = get_local_size(0); stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   temp = input1;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   VECTOR_SORT(input1, dir);
   VECTOR_SORT(input2, dir);
   g_data[global_start] = input1;
   g_data[global_start+1] = input2;
}

/* Perform lowest stage of the bitonic sort */
__kernel void bsort_stage_0(__global float4 *g_data, __local float4 *l_data,
                            uint high_stage) {

   int dir;
   uint id, global_start, stride;
   float4 input1, input2, temp;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   /* Determine data location in global memory */
   id = get_local_id(0);
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   /* Perform initial swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + get_local_size(0)];
   comp = (input1 < input2 ^ dir) * 4 + add3;
   l_data[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

   /* Perform bitonic merge */
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   temp = input1;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   VECTOR_SORT(input1, dir);
   VECTOR_SORT(input2, dir);

   /* Store output in global memory */
   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;
}

/* Perform successive stages of the bitonic sort */
__kernel void bsort_stage_n(__global float4 *g_data, __local float4 *l_data,
                            uint stage, uint high_stage) {

   int dir;
   float4 input1, input2;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   /* Perform swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + global_offset];
   comp = (input1 < input2 ^ dir) * 4 + add;
   g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

/* Sort the bitonic set */
__kernel void bsort_merge(__global float4 *g_data, __local float4 *l_data, uint stage, int dir) {

   float4 input1, input2;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   /* Perform swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + global_offset];
   comp = (input1 < input2 ^ dir) * 4 + add;
   g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

/* Perform final step of the bitonic merge */
__kernel void bsort_merge_last(__global float4 *g_data, __local float4 *l_data, int dir) {

   uint id, global_start, stride;
   float4 input1, input2, temp;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   id = get_local_id(0);
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   /* Perform initial swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + get_local_size(0)];
   comp = (input1 < input2 ^ dir) * 4 + add3;
   l_data[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

   /* Perform bitonic merge */
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   temp = input1;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   VECTOR_SORT(input1, dir);
   VECTOR_SORT(input2, dir);

   /* Store the result to global memory */
   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;
}
