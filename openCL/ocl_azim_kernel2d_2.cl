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


#ifdef ENABLE_FP64
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
  #define UACC  100000000
  #define UACCf 100000000.0f
  #define PI M_PI
  typedef unsigned long UINTType;
#else
  #pragma OPENCL EXTENSION cl_khr_fp64 : disable
  #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
  #define UACC  10000
  #define UACCf 10000.0f
  #define PI M_PI_F
  typedef unsigned int UINTType;
#endif

#define GROUP_SIZE BLOCK_SIZE

#include "for_eclipse.h"


/* uimemset2
 * Sets the values of two unsigned integer input arrays
 * to zero.
 * Recommended Gridsize = size of arrays, currently without padding only
 */
__kernel void
uimemset2(__global UINTType *array0,__global UINTType *array1)
{
  uint gid = get_global_id(0);

  if(gid < BINS)
  {
    array0[gid]=0;
    array1[gid]=0;
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

/* ui2f2
 * Converts the value of the unsigned int input arrays to
 * float by rescaling based on UINT_ACC and saved to the
 * float output arrays
 */
__kernel void
ui2f2(const __global UINTType *uarray0, const __global UINTType *uarray1,
      __global float *farray0,__global float *farray1)
{
  uint gid = get_global_id(0);
  float histval, binval;

  if(gid < BINS)
  {
    binval = (float)uarray0[gid]/UACCf;
    histval = (float)uarray1[gid]/UACCf;

    //if(gid== 2003 + 2048*1779)printf("uarr %u farr %f\n",uarray0[gid],binval);
    //if(binval >100 ) printf("binval %f gid %d bint %d binc %d\n",binval,gid,gid%BINSTTH,gid/BINSTTH);
    barrier(CLK_LOCAL_MEM_FENCE);
    farray0[gid] = binval;
    //if (binval)
      farray1[gid] = histval / binval;
    //else farray1[gid] = 0.0f;
  }
}

/* get_spans
 * Retrieves the distance between 2 point-corners, converts them to bin-spans
 * and saves in span_range
 */
__kernel void
get_spans(const __global float *tth,
          const __global float *dtth,
          const __global float *tth_min_max,
          const __global float *chi,
          const __global float *dchi,
          const __global float *chi_min_max,
                __global float *span_range)
{
  uint gid = get_global_id(0);
  float tth_min, tth_max;
  float chi_min, chi_max;
  float tthb;
  float chib;
  if(gid < NN)
  {
    tth_max = tth_min_max[1];
    tth_min = tth_min_max[0];
    chi_max = chi_min_max[1];
    chi_min = chi_min_max[0];

    tthb = (tth_max - tth_min) / BINSTTH;
    chib = (chi_max - chi_min) / BINSCHI;

    span_range[gid] = (2 * dtth[gid])/tthb;
    span_range[gid + NN] = (2 * dchi[gid])/chib;
    //if(gid<2)printf("Spanrange %f %f\n",span_range[gid],span_range[gid+NN]);
  }
}

/* group_spans
 * Groups the bin-spans calculated by get_spans in groups
 * whose size is defined by GROUP_SIZE. For each group
 * the local maximum bin-span is found and saved in the
 * first slot of each group's slice of span_range array
 */
__kernel void
group_spans(__global float *span_range)
{
  uint gid = get_global_id(0);
  uint tid = get_local_id(0);
  uint blockId = get_group_id(0);

  int Ngroups, GroupSize;
  __local float loc_max[BLOCK_SIZE*2];

  GroupSize = GROUP_SIZE;
  Ngroups = NN / GroupSize;

  if(gid < NN)
  {
    loc_max[tid] = span_range[gid];
    loc_max[tid + BLOCK_SIZE] = span_range[gid + NN];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s=BLOCK_SIZE/2; s>0; s>>=1){
      if(tid<s)
      {
        loc_max[tid] = max(loc_max[tid],loc_max[tid+s]);
      }else if(tid >= s)
      {
        loc_max[tid - s + BLOCK_SIZE] = max(loc_max[tid - s + BLOCK_SIZE],loc_max[tid + BLOCK_SIZE]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //Reduced 2 results are compared by the first thread of every block. Non elegant operation but it reduces
    // the result array from size BLOCKS*2 to size BLOCKS. Where BLOCKS = global_size/BLOCK_SIZE
    if(tid==0)
    {
      span_range[blockId]=loc_max[0];
    }else if (tid==32)
    {
      span_range[blockId + NN] = loc_max[BLOCK_SIZE];
    }
  }
}

/* apply_solidangle_correction
 * Self-explanatory, applies the solid angle correction by dividing the image intensity
 * by the solidangle correction factor for each pixel
 */
__kernel void
solidangle_correction(__global float *intensity, const __global float *solidangle)
{
  uint gid = get_global_id(0);
  if(gid < NN)
  {
    intensity[gid] /= solidangle[gid];
  }
}

__kernel void
dummyval_correction(__global float *intensity, const __global float *dummyval)
{
  uint gid = get_global_id(0);
  float epsilon = 1e-6f;
  float img_val;

  if(gid < NN)
  {
    img_val= intensity[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(fabs(img_val - dummyval[0])<=epsilon)
    {
      intensity[gid]=0.0f;
    }
  }
}

__kernel void
create_histo_binarray(const __global float    *tth,
                      const __global float    *dtth,
                      const __global float    *chi,
                      const __global float    *dchi,
                            __global UINTType *binarray,
                      const __global float    *tth_min_max,
                      const __global float    *chi_min_max,
                      const __global float    *intensity,
                            __global UINTType *histogram,
                      const __global float    *span_range,
                      const __global int      *mask,
                      const __global float    *tth_range,
                      const __global float    *chi_range
)
{

  uint tid,gid;
  UINTType convert0,convert1;
  float2 tthmm, tthrmm;
  float2 chimm, chirmm;
  float2 f0mm, f1mm;
  int2   b0mm, b1mm;
  int    cbint, cbinc;
  float  fbint, fbinc;
  float2  a0,a1;
  float epsilon = 1e-6f;
  int2 spread;

  float2 x_interp;
  float2 I_interp;
  float2 fbinsize;
  float2 inrange;

  float chi_start_from_zero;

  tid=get_local_id(0);
  gid=get_global_id(0);

  if(gid < NN)
  {
    if(!mask[gid])
    {

    //Load tth min and max from slow global to fast register cache
    tthrmm.x = tth_range[0];
    tthrmm.y = tth_range[1];

    chirmm.x = chi_range[0];
    chirmm.y = chi_range[1];

    //Cutoff the histogram at the discontiniuity.
    if(chirmm.x < -PI) chirmm.x = -PI;
    if(chirmm.y > PI ) chirmm.y =  PI;

    chi_start_from_zero = - ( chirmm.x * (chirmm.x < 0.0f) ) ;
    //chi_start_from_zero = PI ;
    chirmm.x = chirmm.x + chi_start_from_zero;
    chirmm.y = chirmm.y + chi_start_from_zero;

    fbinsize.x = (tthrmm.y - tthrmm.x)/BINSTTH;
    fbinsize.y = (chirmm.y - chirmm.x)/BINSCHI;

    a0.x=tth[gid] - dtth[gid];
    a0.y=tth[gid] + dtth[gid];
    a1.x=chi[gid] - dchi[gid] + chi_start_from_zero;
    a1.y=chi[gid] + dchi[gid] + chi_start_from_zero;

    spread.x = round ( span_range[gid/GROUP_SIZE] );
    spread.y = round ( span_range[gid/GROUP_SIZE + NN] );
    barrier(CLK_LOCAL_MEM_FENCE);


    f0mm.x=(a0.x - epsilon - tthrmm.x) * (BINSTTH) / (tthrmm.y - tthrmm.x);
    f0mm.y=(a0.y - epsilon - tthrmm.x) * (BINSTTH) / (tthrmm.y - tthrmm.x);

    f1mm.x=(a1.x - 1e-2*chirmm.x - chirmm.x) * (BINSCHI) / (chirmm.y - chirmm.x);
    f1mm.y=(a1.y - 1e-2*chirmm.x - chirmm.x) * (BINSCHI) / (chirmm.y - chirmm.x);

    b0mm.x = (int)f0mm.x;
    b0mm.y = (int)f0mm.y;

    b1mm.x = (int)f1mm.x;
    b1mm.y = (int)f1mm.y;

    I_interp.x = (f0mm.y - f0mm.x)/fbinsize.x;
    I_interp.x = I_interp.x * (I_interp.x < 1.0f) + 1.0f * (I_interp.x >= 1.0f);

    I_interp.y = (f1mm.y - f1mm.x)/fbinsize.y;
    I_interp.y = I_interp.y * (I_interp.y < 1.0f) + 1.0f * (I_interp.y >= 1.0f);

    for(int spreadloopx=0;spreadloopx<spread.x+1;spreadloopx++)
    {
      fbint = f0mm.x + spreadloopx;
      cbint = (int)fbint;
      inrange.x = (cbint<=b0mm.y);

      x_interp.x = ( ( 1.0f - (fbint - cbint) )* (cbint == b0mm.x) )  +
                    ( ( fbint - cbint ) * ( cbint == b0mm.y)       )   +
                    (1.0f * ( (cbint > b0mm.x)&&(cbint < b0mm.y) ) );

      for(int spreadloopy=0;spreadloopy<spread.y+1;spreadloopy++)
      {
        fbinc = f1mm.x + spreadloopy;
        cbinc = (int)fbinc;
        inrange.y = (cbinc<=b1mm.y);

        x_interp.y = ( ( 1.0f - (fbinc - cbinc) )* (cbinc == b1mm.x) )  +
                    ( ( fbinc - cbinc ) * ( cbinc == b1mm.y)       )   +
                    (1.0f * ( (cbinc > b1mm.x)&&(cbinc < b1mm.y) ) );


        convert0 = (UINTType)((x_interp.x * x_interp.y * I_interp.x * I_interp.y)*UACC);
        convert1 = (UINTType)((x_interp.x * x_interp.y * I_interp.x * I_interp.y * intensity[gid])*UACC);

      //barrier(CLK_LOCAL_MEM_FENCE);
        if(inrange.x && inrange.y && (cbint + BINSTTH*cbinc) < BINS && cbinc >=0 ){
          atom_add(&binarray[cbint + BINSTTH*(cbinc)],convert0);
          atom_add(&histogram[cbint + BINSTTH*(cbinc)],convert1);
        }
      }
    }
    }
  }
}
