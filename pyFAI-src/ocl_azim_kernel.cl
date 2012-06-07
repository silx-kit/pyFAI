/*
 *   Project: Azimuthal regroupping OpenCL kernel for pyFAI.
 *
 *
 *   Copyright (C) 2011 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: D. Karkoulis (karkouli@esrf.fr)
 *   Last revision: 03/06/2011
 *   
 *   Contributing author: Jerome Kieffer (jerome.kieffer@esrf.eu)
 *    
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published
 *   by the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   and the GNU Lesser General Public License  along with this program.
 *   If not, see <http://www.gnu.org/licenses/>.
 */

//OpenCL extensions are silently defined by opencl compiler at compile-time:
#ifdef cl_amd_printf
  #pragma OPENCL EXTENSION cl_amd_printf : enable
#elif defined(cl_intel_printf)
  #pragma OPENCL EXTENSION cl_intel_printf : enable
#else
  #define printf(...)
#endif

#define UINT_ACC 1000

// #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void
create_histo_binarray(__global float *tth, __global int *bin_array, __global float* tth_min_max){

  uint tid,gid;
  int cbin;
  uint x,y;
  float tth_min,tth_max;
  uint segment_offset;
  __local int l_b[BINS];

  tid=get_local_id(0);
  gid=get_global_id(0);

  //Load tth min and max from slow global to fast register cache
  tth_min = tth_min_max[0];
  tth_max = tth_min_max[1];

  //Normalize tth to the [0,1] range, taking into account for the normalisation that the min value
  // can be positive bigger than 0.
  cbin = (int)( ( (tth[gid] - tth_min)*(BINS-1) ) / (tth_max - tth_min) );
  if(cbin>=2047 || cbin<1){
    printf("tid %u gid %u tth %f tth_min_max %f %f cbin %d\n",tid,gid,tth[gid],tth_min_max[0],tth_min_max[1],cbin);
    printf("tthmax - tthmin %f\n",tth_min_max[1] - tth_min_max[0]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  segment_offset =  (gid/BLOCK_SIZE)*BINS;

  //Initialise the local memory buffer of the bins array.
  for(uint i=0;i<(BINS/BLOCK_SIZE);i++){
    l_b[tid + i*BLOCK_SIZE]=0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  atom_inc(&l_b[cbin]);

  barrier(CLK_LOCAL_MEM_FENCE);
  for(uint i=0;i<(BINS/BLOCK_SIZE);i++){
    bin_array[tid + i*BLOCK_SIZE + segment_offset]=l_b[tid + i*BLOCK_SIZE];
    barrier(CLK_LOCAL_MEM_FENCE);

  }

}


__kernel void
create_histo_intensity(__global float *tth, __global float *SolidAngle, __global float *intensity, \
                         __global float *histogram, __global int* bin_array, __global float* tth_min_max, __global float *int_min_max){

  uint tid,gid;
  int cbin;
  uint x,y;
  float tth_min,tth_max;
  float int_max;
  uint segment_offset;
  uint intensity_add;
  float intensity_correction;
  __local unsigned int l_b[BINS];

  tid=get_local_id(0);
  gid=get_global_id(0);

  //Load min and max from slow global memory to fast register cache
  tth_min = tth_min_max[0];
  tth_max = tth_min_max[1];
  int_max = int_min_max[1];

  //Normalize tth to the [0,1] range and assign to a bin, taking into account that the min value
  // can be positive bigger than 0.
  cbin = (int)( ( (tth[gid] - tth_min)*(BINS-1) ) / (tth_max - tth_min) );

  //When running on cpu, quick and dirty check for boundaries
  if(cbin>=2047 || cbin<1){
    printf("tid %u gid %u tth %f tth_min_max %f %f cbin %d\n",tid,gid,tth[gid],tth_min_max[0],tth_min_max[1],cbin);
    printf("tthmax - tthmin %f\n",tth_min_max[1] - tth_min_max[0]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  segment_offset =  (gid/BLOCK_SIZE)*BINS;

  //Initialise the local memory buffer of the histogram array.
  for(uint i=0;i<(BINS/BLOCK_SIZE);i++){
    l_b[tid + i*BLOCK_SIZE]=0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  intensity_correction =  (intensity[gid] /SolidAngle[gid]);
  intensity_add = (uint)((intensity_correction)*UINT_ACC/int_max);
  atom_add(&l_b[cbin],intensity_add);

  barrier(CLK_LOCAL_MEM_FENCE);
  for(uint i=0;i<(BINS/BLOCK_SIZE);i++){
    histogram[tid + i*BLOCK_SIZE + segment_offset]=((float)(l_b[tid + i*BLOCK_SIZE]))/UINT_ACC*int_max;
    barrier(CLK_LOCAL_MEM_FENCE);

  }

}

//create_histo creates (Nx / BLOCK_SIZE) partial histograms which need to be reduced to a single histogram.
// This is a straightforward naive approach and not so fast. The more optimised approach of reduction is used in
// get_max_intensity kernel. Eventually reduce_histo will be converted in the same way.
__kernel void reduce_binarray(__global int *bin_array){
  
  uint bsum=0;
  uint gid=get_global_id(0);

  for(uint i=0;i<BLOCKS;i++){
    bsum+=bin_array[gid + BINS*i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  bin_array[gid]=bsum;
 }

__kernel void reduce_histogram(__global float *histogram){
  
  float hsum=0;
  uint gid=get_global_id(0);

  for(uint i=0;i<BLOCKS;i++){
    hsum+=histogram[gid + BINS*i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  histogram[gid]=hsum;
 }
//This is a parallel reduction fuction. The workgroup size changes on every call.
// the local_size changes only on the last call.
__kernel void
get_min_max(__global float *input,__global float * sort_buffer,__global float *result){

  uint tid,gid;
  uint blockId;
  uint global_size;
  uint local_size;
  __local float l_i[BLOCK_SIZE_M*2];

  tid = get_local_id(0);
  gid = get_global_id(0);
  blockId = gid/BLOCK_SIZE_M;

  //global_size is HALF of the current reduction width.
  global_size = get_global_size(0);
  local_size  = get_local_size(0); 

  if(local_size == BLOCK_SIZE_M){
    l_i[tid] = input[gid + BLOCK_SIZE_M*blockId];
    l_i[tid + BLOCK_SIZE_M] = input[gid + BLOCK_SIZE_M + BLOCK_SIZE_M*blockId];
    barrier(CLK_LOCAL_MEM_FENCE);
  
    for(uint s=BLOCK_SIZE/2; s>0; s>>=1){
      if(tid<s) {
        l_i[tid] = max(l_i[tid],l_i[tid+s]);
      } else {
        l_i[tid + BLOCK_SIZE_M] = max(l_i[tid + BLOCK_SIZE_M],l_i[tid + BLOCK_SIZE_M + s]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  
    barrier(CLK_LOCAL_MEM_FENCE);
    //Reduced 2 results are compared by the first thread of every block. Non elegant operation but it reduces
    // the result array from size BLOCKS*2 to size BLOCKS. Where BLOCKS = global_size/BLOCK_SIZE
    if(tid==0){
      sort_buffer[blockId]=max(l_i[0],l_i[BLOCK_SIZE_M]);
/*      printf("Block %u,max_intensity %.7f\n",blockId,sort_buffer[blockId]);*/
    }
  } else {
    l_i[tid] = input[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s=local_size/2; s>0; s>>=1){
      if(tid<s) {
        l_i[tid] = max(l_i[tid],l_i[tid+s]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  
    barrier(CLK_LOCAL_MEM_FENCE);    
    if(tid==0){
      result[blockId]=l_i[0];
//       printf("Block %u,max_intensity %.7f\n",blockId,max_intensity[blockId]);

    }      
  }
//   if(tid==1)max_intensity[blockId + blocks]=l_i[BLOCK_SIZE];

}

__kernel void
get_max_partial(__global float *input,__global float * sort_buffer,uint first_run){

  uint tid,gid;
  uint blockId;
  uint global_size;
  uint local_size;
  __local float l_i[BLOCK_SIZE_M*2];

  tid = get_local_id(0);
  gid = get_global_id(0);
  blockId = gid/BLOCK_SIZE_M;

  //global_size is HALF of the current reduction width.
  global_size = get_global_size(0);
  local_size  = get_local_size(0); 

  if(first_run){
    l_i[tid] = input[gid + BLOCK_SIZE_M*blockId];
    l_i[tid + BLOCK_SIZE_M] = input[gid + BLOCK_SIZE_M + BLOCK_SIZE_M*blockId];
  }else{
    l_i[tid] = sort_buffer[gid + BLOCK_SIZE_M*blockId];
    l_i[tid + BLOCK_SIZE_M] = sort_buffer[gid + BLOCK_SIZE_M + BLOCK_SIZE_M*blockId];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  //This can be enhanced if BLOCK_SIZE_M-num of threads are progressively reduced.
  //the rate of convergence will remain the same but thread utilization can increase.
  for(uint s=BLOCK_SIZE_M; s>0; s>>=1){
    if(tid<s) {
      l_i[tid] = max(l_i[tid],l_i[tid+s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  //Reduced 2 results are compared by the first thread of every block. Non elegant operation but it reduces
  // the result array from size BLOCKS*2 to size BLOCKS. Where BLOCKS = global_size/BLOCK_SIZE
  if(tid==0){
    sort_buffer[blockId]=l_i[0];
/*    printf("Block %u,max_intensity %.7f\n",blockId,sort_buffer[blockId]);*/
  }
}

/*The last call to get_max_partial will return BLOCKS results where <= BLOCK_SIZE_M.
    The actual workgroup and block size will become then BLOCKS/2. We Use a separate kernel,
    because now BLOCK_SIZE_M is found by local_size. But since compiler does not know local_size
    at compile time, it cannot optimize loops as effectively. */
__kernel void
get_max_final(__global float * sort_buffer,__global float *result){

  uint tid,gid;
  uint blockId;
  uint global_size;
  uint local_size;

  //the needs in shared memory are guaranteed to be <= to BLOCK_SIZE_M*2
  __local float l_i[BLOCK_SIZE_M*2];

  tid = get_local_id(0);
  gid = get_global_id(0);
  global_size = get_global_size(0);
  local_size  = get_local_size(0);
  blockId = gid/local_size;

  l_i[tid] = sort_buffer[gid + local_size*blockId];
  l_i[tid + local_size] = sort_buffer[gid + local_size + local_size*blockId];

  barrier(CLK_LOCAL_MEM_FENCE);

  for(uint s=local_size; s>0; s>>=1){
    if(tid<s) {
      l_i[tid] = max(l_i[tid],l_i[tid+s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if(tid==0){
    result[1]=l_i[0]; //In all min max arrays, 0 is min, 1 is max
/*    printf("Block %u,max_intensity %.7f\n",blockId,max_intensity[blockId]);*/
  }
}


// /////
// __kernel void
// reduce_histo_f(){
// }
// __kernel void oclMemsetBin(__global int *bin_array,__global float *histogram){
// 
//   uint gid=get_global_id(0);
// 
//   histogram[gid]=0;
//   bin_array[gid]=0;
// }