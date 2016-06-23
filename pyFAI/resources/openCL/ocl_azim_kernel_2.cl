/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Kernel with full pixel-split
 *
 *
 *   Copyright (C) 2012 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: D. Karkoulis (karkouli@esrf.fr)
 *   Last revision: 11/05/2012
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

/**
 * \file
 * \brief OpenCL kernels for 1D azimuthal integration
 */


#ifdef ENABLE_FP64
//  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
  #define UACC  100000000
  #define UACCf 100000000.0f
  typedef unsigned long UINTType;
#else
//  #pragma OPENCL EXTENSION cl_khr_fp64 : disable
  #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
  #define UACC  100
  #define UACCf 100.0f
  typedef unsigned int UINTType;
#endif

#define GROUP_SIZE BLOCK_SIZE

#include "for_eclipse.h"


/**
 * \brief Sets the values of two unsigned integer input arrays to zero.
 *
 * Gridsize = size of arrays + padding.
 * UINTType is determined upon compilation. If double precision is enabled
 * Unsigned long is used, unsigned int otherwise.
 *
 * @param array0 UINTType Pointer to global memory with the uhistogram or uweights arrays
 * @param array1 UINTType Pointer to global memory with the uhistogram or uweights arrays
 */
__kernel void
uimemset2(__global UINTType *array0,
          __global UINTType *array1
)
{
  uint gid = get_global_id(0);
  //Global memory guard for padding
  if(gid < BINS)
  {
    array0[get_global_id(0)]=0;
    array1[get_global_id(0)]=0;
  }
}

__kernel void
imemset(__global int *iarray)
{
  uint gid = get_global_id(0);
  if(gid < NN)
  {
    iarray[gid] = 0;
  }
}

/**
 * \brief Converts the value of two UINTType arrays to float
 *
 * This is done by rescaling by UACCf and saved to the
 * float output arrays. uarray0's result is saved to farray0 etc.
 *
 * @param uarray0 UINTType Pointer to global memory with the uhistogram or uweights arrays
 * @param uarray1 UINTType Pointer to global memory with the uhistogram or uweights arrays
 * @param farray0 float Pointer to global memory with the histogram or weights arrays
 * @param farray1 float Pointer to global memory with the histogram or weights arrays
 */
__kernel void
ui2f2(const __global UINTType *uarray0,
      const __global UINTType *uarray1,
            __global float    *farray0,
            __global float    *farray1
)
{
  uint gid = get_global_id(0);
  float histval, binval;

  if(gid < BINS)
  {
    binval = (float)uarray0[gid]/UACCf;
    histval = (float)uarray1[gid]/UACCf;

    //barrier(CLK_LOCAL_MEM_FENCE);  //does not really matter.Breaks CPU OCLs
    farray0[gid] = binval;
    if(binval) farray1[gid] = histval / binval;
    else farray1[gid] = 0.0f;
  }
}

/**
 * \brief Retrieves the distance between 2 point-corners
 *
 * They then get converted to "bin-sizes" and saved tin span_range
 *
 * @param tth         Float pointer to global memory storing the 2th  data.
 * @param dtth        Float pointer to global memory storing the d2th data.
 * @param tth_range   Float pointer to global memory of size 2 (vector) storing the
 *                     min and max values for 2th +- d2th (default) OR a user defined.
                       range set by setRange()
 * @param span_range  Float pointer to global memory where to store the results.
 */
__kernel void
get_spans(const __global float *tth,
          const __global float *dtth,
          const __global float *tth_range,
                __global float *span_range
)
{
  uint gid = get_global_id(0);
  float tth_min, tth_max;
  float tthb;
  float value;

  if(gid < NN)
  {
    value = tth[gid];

    tth_max = tth_range[1];
    tth_min = tth_range[0];
    tthb = (tth_max - tth_min) / BINS;

    if(value < tth_min || value > tth_max)
      span_range[gid] = 0.0f;
    else
      span_range[gid] = (2 * dtth[gid])/tthb;
  }
}

/**
 * \brief Groups the bin-spans calculated by get_spans in groups
 *
 * Thee size is defined by GROUP_SIZE. For each group the local maximum bin-span is
 * found and saved in the first slot of each group's slice of span_range array.
 * The algorithms is a local parallel reduce.
 *
 * @param span_range Float pointer to global memory used to read the spans and save the max
 *                   for each group
 */
__kernel void
group_spans(__global float *span_range)
{
  uint gid = get_global_id(0);
  uint tid = get_local_id(0);
  uint blockId = get_group_id(0);

  uint Ngroups, GroupSize;
  __local float loc_max[BLOCK_SIZE];

  GroupSize = GROUP_SIZE;
  Ngroups = NN / GroupSize;

  loc_max[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

  if(gid < NN)
  {
    loc_max[tid] = span_range[gid];
  }//Broke the if here as it is not really needed further on (except that it made blockID<Ngroups obsolete)
   //Unfortunately CPU OCLs break with barriers in such if clauses. NV works ok
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s=BLOCK_SIZE/2; s>0; s>>=1){
      if(tid<s) {
        loc_max[tid] = max(loc_max[tid],loc_max[tid+s]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //Reduced 2 results are compared by the first thread of every block. Non elegant operation but it reduces
    // the result array from size BLOCKS*2 to size BLOCKS. Where BLOCKS = global_size/BLOCK_SIZE
    if(tid==0 && blockId < Ngroups){
      span_range[blockId]=loc_max[0];
    }
}

/**
 * \brief Applies solid angle correction to an image
 *
 * Applies the solid angle correction by dividing the image intensity by the solidangle correction
 * factor for each pixel as PyFAI.
 *
 * @param intensity Float pointer to global memory where the input image resides
 * @param solidangle Const float pointer to global memory with the solidangle array
 */
__kernel void
solidangle_correction(      __global float *intensity,
                      const __global float *solidangle
)
{
  uint gid = get_global_id(0);
  if(gid < NN)
  {
    intensity[gid] /= solidangle[gid];
  }
}

/**
 * \brief Replaces a dummy value found in an image by 0
 *
 * @param intensity Float pointer to global memory where the input image resides
 * @param solidangle Const float pointer to global memory with the dummy value (single value)
 */
__kernel void
dummyval_correction(      __global float *intensity,
                    const __global float *dummyval,
                    const __global float *deltadummyval
)
{
  uint gid = get_global_id(0);
  float img_val;

  if(gid < NN)
  {
    img_val = intensity[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(fabs(img_val - dummyval[0])<=deltadummyval[0])
    {
      intensity[gid]=0.0f;
    }
  }
}

/**
 * \brief Performs 1d azimuthal integration with full pixel splitting
 *
 * An image instensity value is spread across the bins the 2th +- d2th spans.
 * Note that tth_range will have the values of tth_min_max if the use of a tth range is not enabled.
 * When tth range is enabled, integration is performed only in the 2th +- d2th values that reside
 * COMPLETELY inside the tth_range interval.
 * Values of 0 in the mask are processed and values of 1 ignored as per PyFAI
 *
 * @param tth         Float pointer to global memory storing the 2th data.
 * @param dtth        Float pointer to global memory storing the d2th data.
 * @param binarray    UINTType Pointer to global memory with the uweights array.
 * @param tth_min_max Float pointer to global memory of size 2 (vector) storing the min and max values
 *                     for 2th +- d2th.
 * @param intensity   Float pointer to global memory where the input image resides.
 * @param histogram   UINTType Pointer to global memory with the uhistogram array.
 * @param span_range  Float pointer to global memory with the max values of spans per group.
 * @param mask        Int pointer to global memory with the mask to be used.
 * @param tth_range   Float pointer to global memory of size 2 (vector) storing the min and max for integration.
 *                     If tth range is not specified the this array points to tth_min_max.
 */
__kernel void
create_histo_binarray(const __global float    *tth,
                      const __global float    *dtth,
                            __global UINTType *binarray,
                      const __global float    *tth_min_max,
                      const __global float    *intensity,
                            __global UINTType *histogram,
                      const __global float    *span_range,
                      const __global int      *mask,
                      const __global float    *tth_range
)
{

  uint gid;
  UINTType convert0, convert1;
//  float tth_min, tth_max;
  float tth_rmin, tth_rmax;
  float fbin0_min, fbin0_max;
  int    bin0_min, bin0_max;
  int    cbin;
  float  fbin;
  float  a0, b0, center;
  int spread;

  float x_interp;
  float I_interp;
  float fbinsize;
  gid=get_global_id(0);

  //Load tth min and max from slow global to fast register cache
//  tth_min = tth_min_max[0];
//  tth_max = tth_min_max[1];
  tth_rmin= tth_range[0];
  tth_rmax= tth_range[1];

  if(gid < NN)
  {
    if(!mask[gid])
    {
      center = tth[gid];

      fbinsize = (tth_rmax - tth_rmin)/BINS;

      a0=center + dtth[gid];
      b0=center - dtth[gid];


      if(center  >= tth_rmin && center <= tth_rmax )
      {
        if(b0 < tth_rmin) b0 = center;
        if(a0 > tth_rmax) a0 = center;
        //As of 20/06/12 The range problem is expected to be handled at input level
        fbin0_min=(b0 - tth_rmin) * (BINS) / (tth_rmax - tth_rmin);
        fbin0_max=(a0 - tth_rmin) * (BINS) / (tth_rmax - tth_rmin);
        bin0_min = (int)fbin0_min;
        bin0_max = (int)fbin0_max;

        spread = round(span_range[gid/GROUP_SIZE]);

        I_interp = (fbin0_max - fbin0_min)/fbinsize;
        I_interp = I_interp * (I_interp < 1.0f) + 1.0f * (I_interp >= 1.0f);
        //barrier(CLK_LOCAL_MEM_FENCE); //does not really matter, but breaks CPU OCLs
        for(int spreadloop=0;spreadloop<spread+1;spreadloop++)
        {
          fbin = fbin0_min + spreadloop;
          cbin = (int)fbin;

          x_interp = ( ( 1.0f - (fbin - cbin) )* (cbin == bin0_min) ) +
                      ( ( fbin - cbin ) * ( cbin == bin0_max)       ) +
                      (1.0f * ( (cbin > bin0_min)&&(cbin < bin0_max) ) );

          convert0 = (UINTType)((x_interp * I_interp)*UACC);
          convert1 = (UINTType)((x_interp * I_interp * intensity[gid])*UACC);
          //barrier(CLK_LOCAL_MEM_FENCE); //CPU OCLs
          if((cbin<=bin0_max) && (cbin < BINS)){
            atom_add(&binarray[cbin],convert0);
            atom_add(&histogram[cbin],convert1);
          }
        }
      }
    }//mask
  }//gid guard
}
