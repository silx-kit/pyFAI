/*
 *   Project: OpenCL Framework and kernels for pyFAI.
 *
 *   List of files: ocl_base.cpp
 *                  ocl_base.hpp
 *                  ocl_xrpd1d_fullsplit.cpp
 *                  ocl_xrpd2d_fullsplit.cpp
 *                  ocl_xrpd1d.hpp
 *                  ocl_xrpd2d.hpp
 *                  ocl_tools.h
 *                  ocl_ckerr.h
 *                  ocl_azim_kernel_2.cl
 *                  ocl_azim_kernel2D_2.cl
 *                  ocl_xrpd1d.i
 *                  ocl_xrpd2d.i
 *                  setup_xrpd1d.py
 *                  setup_xrpd2d.py
 *
 *   Copyright (C) 2011-12 European Synchrotron Radiation Facility
 *                             Grenoble, France
 *
 *   Principal authors: D. Karkoulis (karkouli@esrf.fr)
 *   Last revision: 11/05/2012
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocl_xrpd2d.hpp"

#define CE CL_CHECK_ERR_PR
#define C  CL_CHECK_PR

#define CEN CL_CHECK_ERR_PRN
#define CN  CL_CHECK_PRN

#define CER CL_CHECK_ERR_PR_RET
#define CR  CL_CHECK_PR_RET

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#endif

//#define silent
#ifdef _SILENT
  #define fprintf(stream,...)
#endif

#define BLOCK_SIZE 128

enum NAMED_CL_BUFFERS
{
  CLMEM_TTH,
  CLMEM_CHI,
  CLMEM_IMAGE,
  CLMEM_SOLIDANGLE,
  CLMEM_HISTOGRAM,
  CLMEM_UHISTOGRAM,
  CLMEM_WEIGHTS,
  CLMEM_UWEIGHTS,
  CLMEM_SPAN_RANGES,  
  CLMEM_TTH_MIN_MAX,
  CLMEM_CHI_MIN_MAX,
  CLMEM_TTH_DELTA,
  CLMEM_CHI_DELTA,
  CLMEM_TTH_RANGE,
  CLMEM_CHI_RANGE,
  CLMEM_MASK,
  CLMEM_DUMMYVAL
} ;

enum NAMED_CL_KERNELS
{
  CLKERN_INTEGRATE,
  CLKERN_UIMEMSET2,
  CLKERN_IMEMSET,
  CLKERN_UI2F2,
  CLKERN_GET_SPANS,
  CLKERN_GROUP_SPANS,
  CLKERN_SOLIDANGLE_CORRECTION,
  CLKERN_DUMMYVAL_CORRECTION
} ;


ocl_xrpd2D_fullsplit::ocl_xrpd2D_fullsplit():ocl()
{
  setDocstring("OpenCL 2d Azimuthal integrator. Check the readme file for more details\n","ocl_xrpd2d_fullsplit.readme");
}

ocl_xrpd2D_fullsplit::ocl_xrpd2D_fullsplit(const char* fname):ocl(fname)
{
  setDocstring("OpenCL 2d Azimuthal integrator. Check the readme file for more details\n","ocl_xrpd2d_fullsplit.readme");
}

ocl_xrpd2D_fullsplit::~ocl_xrpd2D_fullsplit()
{
  clean();
}


/* Get a copy of the arguments and cast them to appropriate internal types */
int ocl_xrpd2D_fullsplit::getConfiguration(const int Nx,const int Nimage,const int NbinsTth, const int NbinsChi,const bool usefp64)
{


  if(Nx < 1 || Nimage < 1 || NbinsTth < 1 || NbinsChi < 1){
    fprintf(stderr,"get_azim_args() parameters make no sense {%d %d %d %d}\n",Nx,Nimage,NbinsTth,NbinsChi);
    return -2;
  }
  if(!(this->sgs)){
    ocl_errmsg("Fatal error in get_azim_args(). Cannot allocate argument structure",__FILE__,__LINE__);
    return -1;
  } else {
    this->sgs->Nimage = Nimage;
    this->sgs->Nx = Nx;
    this->sgs->Nbinsc = NbinsChi;
    this->sgs->Nbinst = NbinsTth;
    this->sgs->Nbins  = NbinsTth * NbinsChi;
    this->sgs->usefp64 = (int)usefp64;
  }

return 0;
}

/* prepare queue, allocate static memory, compile kernel, configure static kernel execution param*/
int ocl_xrpd2D_fullsplit::configure()
{

  //using namespace ocl_xrpd2D_fullsplit;
  if(!sgs->Nx || !sgs->Nimage || !sgs->Nbinst || !sgs->Nbinsc){
    fprintf(stderr,"You may not call config() at this point. Image and histogram parameters not set. (Hint: run get_azim_args())\n");
    return -2;
  }
  if(!hasActiveContext){
    fprintf(stderr,"You may not call config() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }

  //If configure is recalled, force cleanup of OpenCL resources to avoid accidental leaks
  clean(1);

  cl_int err=0;

  //Next step after the creation of a context is to create a command queue. After this step we can enqueue command to the device
  // such as memory copies, arguments, kernels etc.
  oclconfig->oclcmdqueue = clCreateCommandQueue(oclconfig->oclcontext,oclconfig->ocldevice,CL_QUEUE_PROFILING_ENABLE,&err);
  if(err){fprintf(stderr,"clCreateKernel error, %s\n",ocl_perrc(err));return -1;};
  hasQueue =1;

  //Allocate device memory
  if(allocate_CL_buffers())return -1;
  hasBuffers=1;

  //Compile kernels in "ocl_azim_kernel.cl"
  char optional[1000];
  //Dynamic compilation allows to define these arguments. Defined constant arguments can make code faster,
  // especially when present in a loop statement, since the compiler by knowing the exact size of the loop
  // can unroll more efficiently.
  if(sgs->usefp64==0)
    sprintf(optional,"-D BINS=%u -D BINSTTH=%u -D BINSCHI=%u -D NX=%u -D NN=%u ",sgs->Nbins,sgs->Nbinst,sgs->Nbinsc,sgs->Nx,sgs->Nimage);
  else
    sprintf(optional,"-D BINS=%u -D BINSTTH=%u -D BINSCHI=%u -D NX=%u -D NN=%u -D ENABLE_FP64",sgs->Nbins,sgs->Nbinst,sgs->Nbinsc,sgs->Nx,sgs->Nimage);

  //The blocksize itself is set by the compiler function explicitly and then appends the string "optional"
  char kern_ver[100];
  sprintf(kern_ver,"ocl_azim_kernel2d_%d.cl",2);
  fprintf(stream,"Will use kernel %s\n",kern_ver);
  if(ocl_compiler(oclconfig,kern_ver,BLOCK_SIZE,optional,stream))return -1;
  hasProgram=1;

  oclconfig->oclkernels = (cl_kernel*)malloc(8*sizeof(cl_kernel));
  if(!oclconfig->oclkernels){
    ocl_errmsg("Fatal error in ocl_config. Cannot allocate kernels",__FILE__,__LINE__);
    return -2;
  }

  int i=0;
  oclconfig->oclkernels[CLKERN_INTEGRATE] = clCreateKernel(oclconfig->oclprogram,"create_histo_binarray",&err);
  if(err){fprintf(stderr,"clCreateKernel error, %s\n",ocl_perrc(err));return -1;};i++;

  oclconfig->oclkernels[CLKERN_UIMEMSET2] = clCreateKernel(oclconfig->oclprogram,"uimemset2",&err);
  if(err){fprintf(stderr,"clCreateKernel error, %s\n",ocl_perrc(err));return -1;};i++;

  oclconfig->oclkernels[CLKERN_IMEMSET] = clCreateKernel(oclconfig->oclprogram,"imemset",&err);
  if(err){fprintf(stderr,"clCreateKernel error, %s\n",ocl_perrc(err));return -1;};i++;
  
  oclconfig->oclkernels[CLKERN_UI2F2] = clCreateKernel(oclconfig->oclprogram,"ui2f2",&err);
  if(err){fprintf(stderr,"clCreateKernel error, %s\n",ocl_perrc(err));return -1;};i++;

  oclconfig->oclkernels[CLKERN_GET_SPANS] = clCreateKernel(oclconfig->oclprogram,"get_spans",&err);
  if(err){fprintf(stderr,"clCreateKernel error, %s\n",ocl_perrc(err));return -1;};i++;

  oclconfig->oclkernels[CLKERN_GROUP_SPANS] = clCreateKernel(oclconfig->oclprogram,"group_spans",&err);
  if(err){fprintf(stderr,"clCreateKernel error, %s\n",ocl_perrc(err));return -1;};i++;

  oclconfig->oclkernels[CLKERN_SOLIDANGLE_CORRECTION] = clCreateKernel(oclconfig->oclprogram,"solidangle_correction",&err);
  if(err){fprintf(stderr,"clCreateKernel error, %s\n",ocl_perrc(err));return -1;};i++;

  oclconfig->oclkernels[CLKERN_DUMMYVAL_CORRECTION] = clCreateKernel(oclconfig->oclprogram,"dummyval_correction",&err);
  if(err){fprintf(stderr,"clCreateKernel error, %s\n",ocl_perrc(err));return -1;};i++;

  oclconfig->Nkernels=i;
  hasKernels = 1;

  isConfigured=1; //At this point the device is able to execute kernels (kernels compiled and set)
  
  //Set kernel function arguments
  if(set_kernel_arguments())return -1;

  //We need to initialise the Mask to 0
  size_t wdim[] = { (sgs->Nimage/BLOCK_SIZE) * BLOCK_SIZE + (sgs->Nimage%BLOCK_SIZE) * BLOCK_SIZE, 1, 1};
  size_t tdim[] = {BLOCK_SIZE, 1, 1};
  CR(
    clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_IMEMSET],1,0,wdim,tdim,0,0,&oclconfig->t_s[0]) );

  execTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],  "Initialise Mask to 0");
  clReleaseEvent(oclconfig->t_s[0]);  

return 0;
}

int ocl_xrpd2D_fullsplit::loadTth(float* tth, float* dtth, float tth_min, float tth_max)
{

  fprintf(stream,"Loading Tth\n");
  float tthmm[2];
  tthmm[0]=tth_min;
  tthmm[1]=tth_max;

  if(!oclconfig->Nbuffers || !isConfigured){
    fprintf(stderr,"You may not call loadTth() at this point, OpenCL is not configured (Hint: run config())\n");
    return -2;
  }

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_TTH],CL_TRUE,0,sgs->Nimage*sizeof(cl_float),(void*)tth,0,0,&oclconfig->t_s[0]) );

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_TTH_DELTA],CL_TRUE,0,sgs->Nimage*sizeof(cl_float),(void*)dtth,0,0,&oclconfig->t_s[1]) );

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_TTH_MIN_MAX],CL_TRUE,0,2*sizeof(cl_float),(void*)tthmm,0,0,&oclconfig->t_s[2]) );

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[2],"Load Tth",stream);
  clReleaseEvent(oclconfig->t_s[0]);
  clReleaseEvent(oclconfig->t_s[1]);
  clReleaseEvent(oclconfig->t_s[2]);
  hasTthLoaded=1;
  return 0;
}

int ocl_xrpd2D_fullsplit::loadChi(float* chi, float* dchi, float chi_min, float chi_max)
{

  fprintf(stream,"Loading Chi\n");
  float chimm[2];
  chimm[0]=chi_min;
  chimm[1]=chi_max;

  if(!hasActiveContext){
    fprintf(stderr,"You may not call loadChi() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }  
  
  if(!oclconfig->Nbuffers || !isConfigured){
    fprintf(stderr,"You may not call loadChi() at this point, OpenCL is not configured (Hint: run configure())\n");
    return -2;
  }

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_CHI],CL_TRUE,0,sgs->Nimage*sizeof(cl_float),(void*)chi,0,0,&oclconfig->t_s[0]) );

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_CHI_DELTA],CL_TRUE,0,sgs->Nimage*sizeof(cl_float),(void*)dchi,0,0,&oclconfig->t_s[1]) );

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_CHI_MIN_MAX],CL_TRUE,0,2*sizeof(cl_float),(void*)chimm,0,0,&oclconfig->t_s[2]) );

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[2],"Load Chi",stream);
  clReleaseEvent(oclconfig->t_s[0]);
  clReleaseEvent(oclconfig->t_s[1]);
  clReleaseEvent(oclconfig->t_s[2]);
  hasChiLoaded=1;
  return 0;
}

int ocl_xrpd2D_fullsplit::setSolidAngle(float *SolidAngle)
{

  fprintf(stream,"Setting SolidAngle\n");

  if(!oclconfig->Nbuffers || !isConfigured){
    fprintf(stderr,"You may not call setSolidAngle() at this point, the required buffers are not allocated (Hint: run config())\n");
    return -2;
  }

  if(!hasActiveContext){
    fprintf(stderr,"You may not call setSolidAngle() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_SOLIDANGLE],CL_TRUE,0,sgs->Nimage*sizeof(cl_float),(void*)SolidAngle,0,0,&oclconfig->t_s[0]) );

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"Load SolidAngle");
  clReleaseEvent(oclconfig->t_s[0]);

  useSolidAngle=1;
  return 0;
}

int ocl_xrpd2D_fullsplit::unsetSolidAngle()
{
  fprintf(stream,"Unsetting SolidAngle\n");

  if(useSolidAngle)
  {
    useSolidAngle=0;
    return 0;
  }
  else return -2;
}

int ocl_xrpd2D_fullsplit::setMask(int* Mask)
{
  fprintf(stream,"Setting Mask\n");

  if(!oclconfig->Nbuffers || !isConfigured){
    fprintf(stderr,"You may not call setMask() at this point, the required buffers are not allocated (Hint: run config())\n");
    return -2;
  }

  if(!hasActiveContext){
    fprintf(stderr,"You may not call setMask() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_MASK],CL_TRUE,0,sgs->Nimage*sizeof(cl_int),(void*)Mask,0,0,&oclconfig->t_s[0]) );

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"Load Mask");
  clReleaseEvent(oclconfig->t_s[0]);

  useMask=1;
  return 0;

}

int ocl_xrpd2D_fullsplit::unsetMask()
{
  fprintf(stream,"Unsetting Mask\n");

  if(useMask)
  {
    size_t wdim[] = { (sgs->Nimage/BLOCK_SIZE) * BLOCK_SIZE + (sgs->Nimage%BLOCK_SIZE) * BLOCK_SIZE, 1, 1};
    size_t tdim[] = {BLOCK_SIZE, 1, 1};    
    CR(
      clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_IMEMSET],1,0,wdim,tdim,0,0,&oclconfig->t_s[0]) );

    memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"Reset Mask to 0");
    clReleaseEvent(oclconfig->t_s[0]);
    
    useMask=0;
    return 0;
  }
  else return -2;
}



int ocl_xrpd2D_fullsplit::setDummyValue(float dummyVal)
{
  fprintf(stream,"Setting Dummy Value\n");

  if(!oclconfig->Nbuffers || !isConfigured){
    fprintf(stderr,"You may not call setMask() at this point, the required buffers are not allocated (Hint: run config())\n");
    return -2;
  }

  if(!hasActiveContext){
    fprintf(stderr,"You may not call setMask() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_DUMMYVAL],CL_TRUE,0,sizeof(cl_float),(void*)&dummyVal,0,0,&oclconfig->t_s[0]) );

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"Load Dummy Value");
  clReleaseEvent(oclconfig->t_s[0]);

  useDummyVal=1;
  return 0;

}

int ocl_xrpd2D_fullsplit::unsetDummyValue()
{
  fprintf(stream,"Unsetting Dummy Value\n");

  if(useDummyVal)
  {
    useDummyVal=0;
    return 0;
  }
  else return -2;

}

/**
 * \brief Instructs the program to use a user-defined range for 2th values
 *
 * setRange is optional. By default the integration will use the tth_min and tth_max given by loadTth() as integration
 * range. When setRange is called it sets a new integration range without affecting the 2th array. All values outside that
 * range will then be discarded when interpolating.
 * Currently, if the interval of 2th (2th +- d2th) is not all inside the range specified, it is discarded. The bins of the
 * histogram are RESCALED to the defined range and not the original tth_max-tth_min range.
 *
 * setRange can be called at any point and as many times required after a valid configuration is created.
 *
 * @param lowerBound A float value for the lower bound of the integration range
 * @param upperBound A float value for the upper bound of the integration range
 */
int ocl_xrpd2D_fullsplit::setTthRange(float lowerBound, float upperBound)
{
  fprintf(stream,"Setting 2th Range\n");
  float tthrmm[2];
  tthrmm[0]=lowerBound;
  tthrmm[1]=upperBound;

  if(!oclconfig->Nbuffers || !isConfigured){
    fprintf(stderr,"You may not call setMask() at this point, the required buffers are not allocated (Hint: run config())\n");
    return -2;
  }

  if(!hasActiveContext){
    fprintf(stderr,"You may not call setMask() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_TTH_RANGE],CL_TRUE,0,2*sizeof(cl_float),(void*)&tthrmm,0,0,&oclconfig->t_s[0]) );

  //Set the tth_range argument of the kernels to point to tthRange instead of tth_min_max
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],11,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_RANGE]) ); //TTH range user values


  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"Load 2th Range",stream);
  clReleaseEvent(oclconfig->t_s[0]);

  useTthRange=1;
  return 0;
}

/**
 * \brief Disable the use of a user-defined 2th range and revert to tth_min,tth_max range
 *
 * unsetRange instructs the program to revert to its default integration range. If the method is called when
 * no user-defined range had been previously specified, no action will be performed (return -2)
 */
int ocl_xrpd2D_fullsplit::unsetTthRange()
{
  fprintf(stream,"Unsetting 2th Range\n");

  if(useTthRange)
  {
    //Reset the tth_range argument of the kernels to point to tth_min_max (the default)
    CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],11,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]) ); //Reset to default value

    useTthRange=0;
    return 0;
  }
  else return -2;
}

/**
 * \brief Instructs the program to use a user-defined range for 2th values
 *
 * setRange is optional. By default the integration will use the tth_min and tth_max given by loadTth() as integration
 * range. When setRange is called it sets a new integration range without affecting the 2th array. All values outside that
 * range will then be discarded when interpolating.
 * Currently, if the interval of 2th (2th +- d2th) is not all inside the range specified, it is discarded. The bins of the
 * histogram are RESCALED to the defined range and not the original tth_max-tth_min range.
 *
 * setRange can be called at any point and as many times required after a valid configuration is created.
 *
 * @param lowerBound A float value for the lower bound of the integration range
 * @param upperBound A float value for the upper bound of the integration range
 */
int ocl_xrpd2D_fullsplit::setChiRange(float lowerBound, float upperBound)
{
  fprintf(stream,"Setting 2th Range\n");
  float tthrmm[2];
  tthrmm[0]=lowerBound;
  tthrmm[1]=upperBound;

  if(!oclconfig->Nbuffers || !isConfigured){
    fprintf(stderr,"You may not call setMask() at this point, the required buffers are not allocated (Hint: run config())\n");
    return -2;
  }

  if(!hasActiveContext){
    fprintf(stderr,"You may not call setMask() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }

  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_CHI_RANGE],CL_TRUE,0,2*sizeof(cl_float),(void*)&tthrmm,0,0,&oclconfig->t_s[0]) );

  //Set the tth_range argument of the kernels to point to tthRange instead of tth_min_max
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],12,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_CHI_RANGE]) ); //TTH range user values

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"Load 2th Range",stream);
  clReleaseEvent(oclconfig->t_s[0]);

  useChiRange=1;
  return 0;
}

/**
 * \brief Disable the use of a user-defined 2th range and revert to tth_min,tth_max range
 *
 * unsetRange instructs the program to revert to its default integration range. If the method is called when
 * no user-defined range had been previously specified, no action will be performed (return -2)
 */
int ocl_xrpd2D_fullsplit::unsetChiRange()
{
  fprintf(stream,"Unsetting 2th Range\n");

  if(useChiRange)
  {
    //Reset the tth_range argument of the kernels to point to tth_min_max (the default)
    CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],12,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_CHI_MIN_MAX]) ); //Reset to default value

    useChiRange=0;
    return 0;
  }
  else return -2;
}



/* kernel calls 1D */
int ocl_xrpd2D_fullsplit::execute(float *im_inten,float *histogram,float *bins)
{

  if(!isConfigured){
    fprintf(stderr,"You may not call execute() at this point, kernels are not configured (Hint: run configure())\n");
    return -2;
  }

  if(!hasActiveContext){
    fprintf(stderr,"You may not call execute() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }

  if(!hasTthLoaded || !hasChiLoaded){
    fprintf(stderr,"You may not call execute() at this point. There is no 2th or chi array loaded. (Hint: run loadTth() or loadChi()))\n");
    return -2;
  }

  //Setup the kernel execution parameters, grid,blocks and threads.
  // Notice that in CUDA, a grid is measured in blocks, while in OpenCL is measured in threads.


  size_t wdim_partialh[] = { (sgs->Nimage/BLOCK_SIZE) * BLOCK_SIZE + (sgs->Nimage%BLOCK_SIZE) * BLOCK_SIZE, 1, 1};
  size_t tdim_partialh[] = {BLOCK_SIZE, 1, 1};
  size_t wdim_reduceh[] = { (sgs->Nbins/BLOCK_SIZE) * BLOCK_SIZE + (sgs->Nbins%BLOCK_SIZE) * BLOCK_SIZE, 1, 1};
  size_t tdim_reduceh[] = {BLOCK_SIZE, 1, 1};

  fprintf(stream,"--Histo / Spans workdim %lu %lu %lu\n",(lui)wdim_partialh[0],(lui)wdim_partialh[1],(lui)wdim_partialh[2]);
  fprintf(stream,"--Histo / Spans threadim %lu %lu %lu -- Blocks:%lu\n",(lui)tdim_partialh[0],(lui)tdim_partialh[1],(lui)tdim_partialh[2],\
                                                                  (lui)wdim_partialh[0]/(lui)tdim_partialh[0]);

  fprintf(stream,"--Memset / Convert workdim %lu %lu %lu\n",(lui)wdim_reduceh[0],(lui)wdim_reduceh[1],(lui)wdim_reduceh[2]);
  fprintf(stream,"--Memset / Convert threadim %lu %lu %lu -- Blocks:%lu\n",(lui)tdim_reduceh[0],(lui)tdim_reduceh[1],(lui)tdim_reduceh[2],\
                                                                  (lui)wdim_reduceh[0]/(lui)tdim_reduceh[0]);


  //Copy the new image
  CR(
    clEnqueueWriteBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_IMAGE],CL_TRUE,0,sgs->Nimage*sizeof(cl_float),(void*)im_inten,0,0,&oclconfig->t_s[0]) );

  //Memset
  CR(
  clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_UIMEMSET2],1,0,wdim_reduceh,tdim_reduceh,0,0,&oclconfig->t_s[1]) );
  
  //Get 2th span ranges
  CR(
  clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_GET_SPANS],2,0,wdim_partialh,tdim_partialh,0,0,&oclconfig->t_s[2]) );
  
  //Group 2th span ranges
  CR(
  clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_GROUP_SPANS],1,0,wdim_partialh,tdim_partialh
  ,0,0,&oclconfig->t_s[3]) );

  //Apply dummyval_correction if needed
  if(useDummyVal)
  {
    CR(
    clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_DUMMYVAL_CORRECTION],1,0,wdim_partialh,tdim_partialh,0,0,&oclconfig->t_s[9]) );
  }
  
  //Apply Solidangle correction if needed
  if(useSolidAngle)
  {
    CR(
    clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_SOLIDANGLE_CORRECTION],1,0,wdim_partialh,tdim_partialh,0,0,&oclconfig->t_s[8]) );
  }

  //Histogram

  CR(
  clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_INTEGRATE],1,0,wdim_partialh,tdim_partialh,0,0,&oclconfig->t_s[4]) );

  //Convert to float
  CR(
  clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_UI2F2],1,0,wdim_reduceh,tdim_reduceh,0,0,&oclconfig->t_s[5]) );

  //Copy the results back
  CR(
    clEnqueueReadBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_WEIGHTS],CL_TRUE,0,sgs->Nbins*sizeof(cl_float),(void*)bins,0,0,&oclconfig->t_s[6]) );//bins
  CR(
    clEnqueueReadBuffer(oclconfig->oclcmdqueue,oclconfig->oclmemref[CLMEM_HISTOGRAM],CL_TRUE,0,sgs->Nbins*sizeof(cl_float),(void*)histogram,0,0,&oclconfig->t_s[7]) );

  fprintf(stream,"--Waiting for the command queue to finish\n");
  CR(clFinish(oclconfig->oclcmdqueue));

  //Get execution time from first memory copy to last memory copy.

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"copyIn    ");
  execTime_ms += ocl_get_profT(&oclconfig->t_s[1], &oclconfig->t_s[1],  "memset    ");
  execTime_ms += ocl_get_profT(&oclconfig->t_s[2], &oclconfig->t_s[2],  "getSpa    ");
  execTime_ms += ocl_get_profT(&oclconfig->t_s[3], &oclconfig->t_s[3],  "groupS    ");
  execTime_ms += ocl_get_profT(&oclconfig->t_s[4], &oclconfig->t_s[4],  "Azim GPU  ");
  execTime_ms += ocl_get_profT(&oclconfig->t_s[5], &oclconfig->t_s[5],  "convert   ");

  if(useSolidAngle)
    execTime_ms += ocl_get_profT(&oclconfig->t_s[8], &oclconfig->t_s[8],"Solidan   ");
  if(useDummyVal)
    execTime_ms += ocl_get_profT(&oclconfig->t_s[9], &oclconfig->t_s[9],"dummyva   ");

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[6], &oclconfig->t_s[7],"copyOut   ");

  for(int ievent=0;ievent<8 + useSolidAngle + useDummyVal;ievent++)clReleaseEvent(oclconfig->t_s[ievent]);
  return 0;
}

/* Create a list of cl_mem objects and allocate the appropriate amount of device memory in them*/
int ocl_xrpd2D_fullsplit::allocate_CL_buffers()
{

  cl_int err;
  oclconfig->oclmemref   = (cl_mem*)malloc(16*sizeof(cl_mem));
  if(!oclconfig->oclmemref){
    fprintf(stderr,"Fatal error in ocl_azim_clbuffers. Cannot allocate memrefs\n");
    return -2;
  }

  if(sgs->Nimage < BLOCK_SIZE){
    fprintf(stderr,"Fatal error in ocl_azim_clbuffers. Nimage (%d) must be >= BLOCK_SIZE (%d)\n",sgs->Nimage,BLOCK_SIZE);
    return -2;
  }

  cl_ulong ualloc=0;
  ualloc += (sgs->Nimage*sizeof(cl_float)) * 9;
  ualloc += (sgs->Nbins * sizeof(cl_float)) *2;
  if(sgs->usefp64)
    ualloc += (sgs->Nbins * sizeof(cl_ulong)) *2;
  else
    ualloc += (sgs->Nbins * sizeof(cl_uint)) *2;

  ualloc += 2*sizeof(cl_float) * 4 + sizeof(cl_float);

  if(ualloc >= oclconfig->dev_mem && oclconfig->dev_mem != 0){
    fprintf(stderr,"Fatal error in ocl_azim_clbuffers. Not enough device memory for buffers (%lu requested, %lu available)\n",\
               (lui)ualloc,(lui)oclconfig->dev_mem);
    return -1;
  } else {
    if(oclconfig->dev_mem == 0){
      fprintf(stream,"Caution: Device did not return the available memory size (%lu requested)\n",(lui)ualloc);
      if(stream!=stdout) fprintf(stderr,"Caution: Device did not return the available memory size (%lu requested)\n",(lui)ualloc);
    }
  }

//allocate GPU memory buffers
  int i=0;
  oclconfig->oclmemref[CLMEM_TTH]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//tth array corners -0
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_CHI]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//chi array corners -1
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;  

  oclconfig->oclmemref[CLMEM_IMAGE]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//Image intensity -2
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_SOLIDANGLE]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//Solid Angle -3
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_HISTOGRAM]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_float)),0,&err);//Histogram -4
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  if(sgs->usefp64)
  {
    oclconfig->oclmemref[CLMEM_UHISTOGRAM]=
      clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_ulong)),0,&err);//ulHistogram -5
    if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;
  }else
  {
    oclconfig->oclmemref[CLMEM_UHISTOGRAM]=
      clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_uint)),0,&err);//ulHistogram -5
    if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;
  }

  oclconfig->oclmemref[CLMEM_WEIGHTS]=
   clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_float)),0,&err);//Bin array -6
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  if(sgs->usefp64)
  {
    oclconfig->oclmemref[CLMEM_UWEIGHTS]=
      clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_ulong)),0,&err);//uBinarray -7
    if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;
  }else
  {
    oclconfig->oclmemref[CLMEM_UWEIGHTS]=
      clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_uint)),0,&err);//uBinarray -7
    if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;
  }

  oclconfig->oclmemref[CLMEM_SPAN_RANGES]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)((sgs->Nimage * 2)*sizeof(cl_float)),0,&err);//span_ranges buffer -8
  if(err){fprintf(stderr,"clCreateKernel error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(2*sizeof(cl_float)),0,&err);//Min,Max values for tth -9
  if(err){fprintf(stderr,"clCreateKernel error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_CHI_MIN_MAX]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(2*sizeof(cl_float)),0,&err);//Min,Max values for chi -10
  if(err){fprintf(stderr,"clCreateKernel error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_TTH_DELTA]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//tth array min corners -12
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_CHI_DELTA]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//tth array max corners -13
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_TTH_RANGE]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(2*sizeof(cl_float)),0,&err);//Min,Max values for tth -9
  if(err){fprintf(stderr,"clCreateKernel error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_CHI_RANGE]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(2*sizeof(cl_float)),0,&err);//Min,Max values for chi -10
  if(err){fprintf(stderr,"clCreateKernel error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;  

  oclconfig->oclmemref[CLMEM_MASK]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nimage*sizeof(cl_int)),0,&err);//Mask -14
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_DUMMYVAL]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sizeof(cl_float)),0,&err);//Dummy Value -15
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;

  fprintf(stream,"Allocated %d buffers (%.3f Mb) on device\n",i,(float)ualloc/1024./1024.);
  oclconfig->Nbuffers = i;
return 0;
}

/* Sets the kernel arguments */
int ocl_xrpd2D_fullsplit::set_kernel_arguments()
{

  //The second argument of clSetKernelArg is directly related to the position of the argument on the kernel (starts from 0).
  // I.e. : __kernel void dummy(int arg0,int arg1,...., __global argN-1)
  int i=0;

  i=0;
  //------------------------bin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH]) ); //tth_array corners
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_DELTA]) ); //tth_array corners
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_CHI]) ); //tth_array corners
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_CHI_DELTA]) ); //tth_array corners
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UWEIGHTS]) ); //uBin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]) ); //tth_min_max
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_CHI_MIN_MAX]) ); //tth_min_max
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_IMAGE]) ); //Image
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UHISTOGRAM]) ); //uHistogram
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_SPAN_RANGES]) ); //span_range
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_MASK]) ); //mask
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]) ); //TTH range default
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_CHI_MIN_MAX]) ); //Chi range default

  i=0;
  //------------------------uiMemset2
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UIMEMSET2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UWEIGHTS]) ); //uBin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UIMEMSET2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UHISTOGRAM]) ); //uHistogram

  i=0;
  //------------------------iMemset
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_IMEMSET],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_MASK]) ); //Mask array

  i=0;
  //-----------------------Ulong2Float kernel
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UI2F2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UWEIGHTS]) ); //uBin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UI2F2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UHISTOGRAM]) ); //uHistogram
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UI2F2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_WEIGHTS]) ); //Bin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UI2F2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_HISTOGRAM]) ); //Histogram

  i=0;
  //-----------------------get_spans kernel
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH]) ); //tth_array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_DELTA]) ); //tth_array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]) ); //tth_min_max
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_CHI]) ); //chi_array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_CHI_DELTA]) ); //chi_array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_CHI_MIN_MAX]) ); //chi_min_max
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_SPAN_RANGES]) ); //span_range

  i=0;
  //-----------------------group_spans kernel
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GROUP_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_SPAN_RANGES]) ); //span_range

  i=0;
  //-----------------------solidangle_correction
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_SOLIDANGLE_CORRECTION],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_IMAGE]) ); //Image
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_SOLIDANGLE_CORRECTION],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_SOLIDANGLE]) ); //SolidAngle

  i=0;
  //-----------------------dummyval_correction
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_DUMMYVAL_CORRECTION],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_IMAGE]) ); //Image
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_DUMMYVAL_CORRECTION],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_DUMMYVAL]) ); //Dummy

return 0;
}

int ocl_xrpd2D_fullsplit::clean(int preserve_context)
{

  if(hasBuffers)
  {
    clean_clbuffers(oclconfig->Nbuffers);
    fprintf(stream,"--released OpenCL buffers\n");
    hasBuffers    = 0;
    hasTthLoaded  = 0;
    hasChiLoaded  = 0;
    useSolidAngle = 0;
    useMask       = 0;
    useDummyVal   = 0;
    useTthRange   = 0;
    useChiRange   = 0;
  }

  if(hasKernels)
  {
    clean_clkernels(oclconfig->Nkernels);
    fprintf(stream,"--released OpenCL kernels\n");
    hasKernels=0;
  }

  if(hasProgram)
  {
    CR(clReleaseProgram(oclconfig->oclprogram));
    fprintf(stream,"--released OpenCL program\n");
    hasProgram=0;
  }

  if(hasQueue)
  {
    CR(clReleaseCommandQueue(oclconfig->oclcmdqueue));
    fprintf(stream,"--released OpenCL queue\n");
    hasQueue=0;
  }

  isConfigured = 0;

  reset_time();
  
  if(!preserve_context)
  {
    if(oclconfig->oclmemref)
    {
      free(oclconfig->oclmemref);
      oclconfig->oclmemref=NULL;
      fprintf(stream,"--released OpenCL memory references\n");
    }
    if(oclconfig->oclkernels)
    {
      free(oclconfig->oclkernels);
      oclconfig->oclkernels=NULL;
      fprintf(stream,"--released OpenCL kernel references\n");
    }
    reset_time();
    if(hasActiveContext){
      ocl_destroy_context(oclconfig->oclcontext);
      hasActiveContext=0;
      fprintf(stream,"--released OpenCL context\n");
      return 0;
    }
  }
  return 0;
}
