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
#include <iostream>

#include "ocl_xrpd1d.hpp"

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

//Silence unsigned long printf warnings
typedef unsigned long lui;

/**
 * \brief Enumerate OpenCL buffers.
 *
 * Since OpenCL buffers are directly referenced via ocl_tools, enum is useful
 * to name the references in order to avoid confusion
 */
enum NAMED_CL_BUFFERS
{
  CLMEM_TTH,
  CLMEM_IMAGE,
  CLMEM_SOLIDANGLE,
  CLMEM_HISTOGRAM,
  CLMEM_UHISTOGRAM,
  CLMEM_WEIGHTS,
  CLMEM_UWEIGHTS,
  CLMEM_SPAN_RANGES,
  CLMEM_TTH_MIN_MAX,
  CLMEM_TTH_DELTA,
  CLMEM_MASK,
  CLMEM_DUMMYVAL,
  CLMEM_TTH_RANGE
} ;

/**
 * \brief Enumerate OpenCL kernels.
 *
 * Since OpenCL kernels are directly referenced via ocl_tools, enum is useful
 * to name the references in order to avoid confusion
 */
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

/**
 * \brief Default constructor for xrpd1d.
 * 
 * Output is set to stdout and the docstring is set
 *
 */
ocl_xrpd1D_fullsplit::ocl_xrpd1D_fullsplit():ocl()
{
  setDocstring("OpenCL 1d Azimuthal integrator. Check the readme file for more details\n","ocl_xrpd1d_fullsplit.readme");
}

/**
 * \brief Overloaded constructor for xrpd1d.
 *
 * Output is set to filename "fname" and the docstring is set
 *
 * @param fname A const C-string with the name of the textfile to use as output
 *
 */
ocl_xrpd1D_fullsplit::ocl_xrpd1D_fullsplit(const char* fname):ocl(fname)
{
  setDocstring("OpenCL 1d Azimuthal integrator. Check the readme file for more details\n","ocl_xrpd1d_fullsplit.readme");
}

/**
 * \brief xrpd1d destructor.
 *
 * The destructor only calls clean() which handles the proper release of
 * OpenCL and system resources
 */
ocl_xrpd1D_fullsplit::~ocl_xrpd1D_fullsplit()
{
  clean();
}

/**
 * \brief getConfiguration gets the description of the integrations to be performed and keeps an internal copy
 *
 * All the parameters passed to getConfiguration are crucial for the configuration of the OpenCL buffers
 * and kernels. As a result, any parameters by passed by getConfiguration require a call to configure()
 * for them to take effect
 *
 * @param Nx      An integer with the stride of the image array (i.e the size of x dimension)
 * @param Nimage  The total size of the image in pixels
 * @param Nbins   The number of bins for the integrations
 * @param usefp64 A Boolean stating if the integration should be performed in double-True or single precision-False (default = True).
 *                The suggested value is True. False is much faster but quite unsafe so one must make sure that for a specific configuration
 *                the result is correct before performing other integrations with single precision.
 */
int ocl_xrpd1D_fullsplit::getConfiguration(const int Nx,const int Nimage,const int Nbins,const bool usefp64)
{


  if(Nx < 1 || Nimage <1 || Nbins<1){
    fprintf(stderr,"get_azim_args() parameters make no sense {%d %d %d}\n",Nx,Nimage,Nbins);
    return -2;
  }
  if(!(this->sgs)){
    ocl_errmsg("Fatal error in get_azim_args(). Cannot allocate argument structure",__FILE__,__LINE__);
    return -1;
  } else {
    this->sgs->Nimage = Nimage;
    this->sgs->Nx = Nx;
    this->sgs->Nbins = Nbins;
    this->sgs->usefp64 = (int)usefp64;
  }

return 0;
}

/**
 * \brief The method configure() allocates the OpenCL resources required and compiled the OpenCL kernels.
 *
 * An active context must exist before a call to configure() and getConfiguration() must have been
 * called at least once. Since the compiled OpenCL kernels carry some information on the integration
 * parameters, a change to any of the parameters of getConfiguration() requires a subsequent call to
 * configure() for them to take effect.
 *
 * If a configuration exists and configure() is called, the configuration is cleaned up first to avoid
 * OpenCL memory leaks
 *
 * kernel_path is the path to the actual kernel
 */
int ocl_xrpd1D_fullsplit::configure(const char* kernel_path)
{

  if(!sgs->Nx || !sgs->Nimage || !sgs->Nbins){
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
    sprintf(optional," -D BINS=%d -D NX=%u -D NN=%u ",sgs->Nbins,sgs->Nx,sgs->Nimage);
  else
    sprintf(optional," -D BINS=%d -D NX=%u -D NN=%u -D ENABLE_FP64",sgs->Nbins,sgs->Nx,sgs->Nimage);

  //The blocksize itself is set by the compiler function explicitly and then appends the string "optional"
  fprintf(stream,"Will use kernel %s\n",kernel_path);
  if(ocl_compiler(oclconfig,kernel_path,BLOCK_SIZE,optional,stream))return -1;
  hasProgram=1;
  
  oclconfig->oclkernels = (cl_kernel*)malloc(8*sizeof(cl_kernel));
  if(!oclconfig->oclkernels){
    ocl_errmsg("Fatal error in ocl_config. Cannot allocate kernels",__FILE__,__LINE__);
    return -2;
  }

  //Create the OpenCL kernels found in the compile OpenCL program
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

/**
 * \brief Upload the 2th array to OpenCL memory
 *
 * loadTth is required to be called at least once after a new configuration.
 * It requires an active context and a configuration.
 * The 2th and d2th arrays may be updated whenever fit. E.g. set 2th, call execute() 20 times, update 2th,
 * call execute() again.
 *
 * @param tth     A float pointer to the N 2th data.
 * @param dtth    A float pointer to the N d2th data.
 * @param tth_min The minimum value in 2th +- d2th data.
 * @param tth_max The maximum value in 2th +- d2th data.
 */
int ocl_xrpd1D_fullsplit::loadTth(float* tth, float* dtth, float tth_min, float tth_max)
{

  fprintf(stream,"Loading Tth\n");
  float tthmm[2];
  tthmm[0]=tth_min;
  tthmm[1]=tth_max;

  if(!hasActiveContext){
    fprintf(stderr,"You may not call loadTth() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }  
  
  if(!oclconfig->Nbuffers || !isConfigured){
    fprintf(stderr,"You may not call loadTth() at this point, OpenCL is not configured (Hint: run configure())\n");
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

/**
 * \brief Instructs the program to use solidangle correction using the input array
 *
 * setSolidAngle is optional. The default behaviour of the program is to not perform
 * solid angle correction when integrating. When setSolidAngle is called an option
 * is enabled internally to always perform solid angle correction on the input image.
 *
 * Solid angle correction can be called at any point and as many times required
 * after a valid configuration is created.
 *
 * To disable solid angle correction unsetSolidAngle() can be used
 *
 * @param SolidAngle A float pointer to the array of size N with coefficients for the correction
 */
int ocl_xrpd1D_fullsplit::setSolidAngle(float *SolidAngle)
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

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"Load SolidAngle",stream);
  clReleaseEvent(oclconfig->t_s[0]);

  useSolidAngle=1;
  return 0;
}

/**
 * \brief Disable solidangle correction
 *
 * unsetSolidAngle instructs the program to disable solidangle correction. If the method
 * is called when solid angle corrections is not set, the method will not perform any action
 * (return -2)
 */
int ocl_xrpd1D_fullsplit::unsetSolidAngle()
{
  fprintf(stream,"Unsetting SolidAngle\n");

  if(useSolidAngle)
  {
    useSolidAngle=0;
    return 0;
  }
  else return -2;
}

/**
 * \brief Instructs the program to apply the input mask during integration
 *
 * setMask is optional. By default the integration will not use any mask. If setMask is called
 * it will enable the use of a mask when integrating. The mask must be PyFAI compatible (0-in 1-out)
 *
 * setMask can be called at any point and as many times required after a valid configuration is created.
 *
 * @param Mask An integer pointer to the Mask
 */
int ocl_xrpd1D_fullsplit::setMask(int* Mask)
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

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"Load Mask",stream);
  clReleaseEvent(oclconfig->t_s[0]);

  useMask=1;
  return 0;  
  
}

/**
 * \brief Disable the use of mask
 *
 * unsetMask instructs the program to disable masking funtionality. If the method is called when
 * use of mask is not set, no action will be performed (return -2)
 */
int ocl_xrpd1D_fullsplit::unsetMask()
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

/**
 * \brief Instructs the program to set a value as a dummy
 *
 * setDummyValue is optional. By default the integration will not assign any values to dummy.
 * If setDummyValue is called it will set the input value as a dummy. Whenever this dummy value is
 * encountered in the image it will set the value of the image to 0.
 *
 * setDummyValue can be called at any point and as many times required after a valid configuration is created.
 *
 * @param dummyVal A float with the value to be set as a dummy
 */
int ocl_xrpd1D_fullsplit::setDummyValue(float dummyVal)
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

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"Load Dummy Value",stream);
  clReleaseEvent(oclconfig->t_s[0]);

  useDummyVal=1;
  return 0;

}

/**
 * \brief Disable the use of a dummy value
 *
 * unsetDummyValue instructs the program to disable use of dummy values. If the method is called when
 * use of dummy values is not set, no action will be performed (return -2)
 */
int ocl_xrpd1D_fullsplit::unsetDummyValue()
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
int ocl_xrpd1D_fullsplit::setRange(float lowerBound, float upperBound)
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
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],8,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_RANGE]) ); //TTH range user values
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],2,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_RANGE]) ); //TTH range user values

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
int ocl_xrpd1D_fullsplit::unsetRange()
{
  fprintf(stream,"Unsetting 2th Range\n");

  if(useTthRange)
  {
    //Reset the tth_range argument of the kernels to point to tth_min_max (the default)
    CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],8,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]) ); //Reset to default value
    CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],2,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]) ); //Reset to default value

    useTthRange=0;
    return 0;
  }
  else return -2;
}

/**
 * \brief Perform a 1D azimuthal integration
 *
 * execute() may be called only after an OpenCL device is configured and a Tth array has been loaded (at least once)
 * It takes the input image and based on the configuration provided earlier it performs the 1D integration.
 * Notice that if the provided image is bigger than N then only N points will be taked into account, while
 * if the image is smaller than N the result may be catastrophic.
 * set/unset and loadTth methods have a direct impact on the execute() method.
 * All the rest of the methods will require at least a new configuration via configure().
 *
 * @param im_inten  Float pointer to the input image with size N to integrate
 * @param histogram Float pointer to store the output integrated intensity
 * @param bins      Float pointer to store the output weights
 */
int ocl_xrpd1D_fullsplit::execute(float *im_inten,float *histogram,float *bins)
{

  if(!isConfigured){
    fprintf(stderr,"You may not call execute() at this point, kernels are not configured (Hint: run config())\n");
    return -2;
  }

  if(!hasActiveContext){
    fprintf(stderr,"You may not call execute() at this point. There is no Active context. (Hint: run init())\n");
    return -2;
  }

  if(!hasTthLoaded){
    fprintf(stderr,"You may not call execute() at this point. There is no 2th array loaded. (Hint: run loadTth())\n");
    return -2;
  }
  
  //Setup the kernel execution parameters, grid,blocks and threads.
  // Notice that in CUDA, a grid is measured in blocks, while in OpenCL is measured in threads.

  fprintf(stream,"\n--Integration nr. %d\n",get_exec_count() + 1);
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
  clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_GET_SPANS],1,0,wdim_partialh,tdim_partialh,0,0,&oclconfig->t_s[2]) );
  
  //Group 2th span ranges
  CR(
  clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_GROUP_SPANS],1,0,wdim_partialh,tdim_partialh,0,0,&oclconfig->t_s[3]) );

  //Apply Solidangle correction if needed
  if(useSolidAngle)
  {
    CR(
    clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_SOLIDANGLE_CORRECTION],1,0,wdim_partialh,tdim_partialh,0,0,&oclconfig->t_s[8]) );
  }

  //Apply dummyval_correction if needed
  if(useDummyVal)
  {
    CR(
    clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_DUMMYVAL_CORRECTION],1,0,wdim_partialh,tdim_partialh,0,0,&oclconfig->t_s[9]) );
  }
  
  //Perform the integration
  CR(
  clEnqueueNDRangeKernel(oclconfig->oclcmdqueue,oclconfig->oclkernels[CLKERN_INTEGRATE],
                                            1,0,wdim_partialh,tdim_partialh,0,0,&oclconfig->t_s[4]) );

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

  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[0], &oclconfig->t_s[0],"copyIn   ",stream);
  execTime_ms += ocl_get_profT(&oclconfig->t_s[1], &oclconfig->t_s[1],  "memset   ",stream);
  execTime_ms += ocl_get_profT(&oclconfig->t_s[2], &oclconfig->t_s[2],  "getSpa   ",stream);
  execTime_ms += ocl_get_profT(&oclconfig->t_s[3], &oclconfig->t_s[3],  "groupS   ",stream);
  execTime_ms += ocl_get_profT(&oclconfig->t_s[4], &oclconfig->t_s[4],  "Azim GPU ",stream);
  execTime_ms += ocl_get_profT(&oclconfig->t_s[5], &oclconfig->t_s[5],  "convert  ",stream);
  
  if(useSolidAngle)
    execTime_ms += ocl_get_profT(&oclconfig->t_s[8], &oclconfig->t_s[8],"Solidan  ",stream);
  if(useDummyVal)
    execTime_ms += ocl_get_profT(&oclconfig->t_s[9], &oclconfig->t_s[9],"dummyva  ",stream);
  
  memCpyTime_ms += ocl_get_profT(&oclconfig->t_s[6], &oclconfig->t_s[7],"copyOut  ",stream);

  execCount++;

  //This is very important. OpenCL Events are inherently retained. If not explicitly released after their use they cause memory leaks
  for(int ievent=0;ievent<8 + useSolidAngle + useDummyVal;ievent++)clReleaseEvent(oclconfig->t_s[ievent]);
  
  return 0;
}

/**
 * \brief Allocate OpenCL buffers required for a specific configuration
 *
 * allocate_CL_buffers() is a private method and is called by configure().
 * Given the size of the image and the number of the bins, all the required OpenCL buffers
 * are allocated.
 * The method performs a basic check to see if the memory required by the configuration is
 * smaller than the total global memory of the device. However, there is no built-in way in OpenCL
 * to check the real available memory.
 * In the case allocate_CL_buffers fails while allocating buffers, it will automatically deallocate
 * the buffers that did not fail and leave the flag hasBuffers to 0.
 *
 * Note that an OpenCL context also requires some memory, as well as Event and other OpenCL functionalities which cannot and
 * are not taken into account here.
 * The memory required by a context varies depending on the device. Typical for GTX580 is 65Mb but for a 9300m is ~15Mb
 * In addition, a GPU will always have at least 3-5Mb of memory in use.
 * Unfortunately, OpenCL does NOT have a built-in way to check the actual free memory on a device, only the total memory.
 */
int ocl_xrpd1D_fullsplit::allocate_CL_buffers()
{

  cl_int err;
  oclconfig->oclmemref   = (cl_mem*)malloc(13*sizeof(cl_mem));
  if(!oclconfig->oclmemref){
    fprintf(stderr,"Fatal error in allocate_CL_buffers. Cannot allocate memrefs\n");
    return -2;
  }

  if(sgs->Nimage < BLOCK_SIZE){
    fprintf(stderr,"Fatal error in allocate_CL_buffers. Nimage (%d) must be >= BLOCK_SIZE (%d)\n",sgs->Nimage,BLOCK_SIZE);
    return -2;
  }
    
  cl_ulong ualloc=0;
  ualloc += (sgs->Nimage*sizeof(cl_float)) * 6;
  ualloc += (sgs->Nbins * sizeof(cl_float)) *2;
  if(sgs->usefp64)
    ualloc += (sgs->Nbins * sizeof(cl_ulong)) *2;
  else
    ualloc += (sgs->Nbins * sizeof(cl_uint)) *2;
    
  ualloc += 5*sizeof(cl_float);

  /*
   * Note that an OpenCL context also requires some memory, as well as Event and other OpenCL functionalities.
   * The memory required by a context varies depending on the device. Typical for GTX580 is 65Mb but for a 9300m is ~15Mb
   * In addition, a GPU may already have memory in use and even if not it will always have 3-5Mb in use.
   * Unfortunately, OpenCL does NOT have a built-in way to check the actual free memory on a device, only the total memory.
  */
  
  if(ualloc >= oclconfig->dev_mem && oclconfig->dev_mem != 0){
    fprintf(stderr,"Fatal error in allocate_CL_buffers. Not enough device memory for buffers (%lu requested, %lu available)\n",\
               (lui)ualloc,(lui)oclconfig->dev_mem);
    return -1;
  } else {
    if(oclconfig->dev_mem == 0){
      fprintf(stream,"Caution: Device did not return the available memory size (%lu requested)\n",(lui)ualloc);
    }
  }

//allocate GPU memory buffers. Notice the clean_clbuffers(i-1), If a failure occures before completing all the allocations
// all the successfull allocations will be released.
  int i=0;
  oclconfig->oclmemref[CLMEM_TTH]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//tth array -0
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_IMAGE]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//Image intensity -1
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_SOLIDANGLE]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//Solid Angle -2
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_HISTOGRAM]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_float)),0,&err);//Histogram -3
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  if(sgs->usefp64)
  {
    oclconfig->oclmemref[CLMEM_UHISTOGRAM]=
      clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_ulong)),0,&err);//ulHistogram -4
    if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;
  }else
  {
    oclconfig->oclmemref[CLMEM_UHISTOGRAM]=
      clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_uint)),0,&err);//ulHistogram -4
    if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;    
  }
  
  oclconfig->oclmemref[CLMEM_WEIGHTS]=
   clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_float)),0,&err);//Bin array -5
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  if(sgs->usefp64)
  {
    oclconfig->oclmemref[CLMEM_UWEIGHTS]=
      clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_ulong)),0,&err);//uBinarray -6
    if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;
  }else
  {
    oclconfig->oclmemref[CLMEM_UWEIGHTS]=
      clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)(sgs->Nbins*sizeof(cl_uint)),0,&err);//uBinarray -6
    if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;
  }

  oclconfig->oclmemref[CLMEM_SPAN_RANGES]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_WRITE,(size_t)((sgs->Nimage)*sizeof(cl_float)),0,&err);//span_ranges buffer -7
  if(err){fprintf(stderr,"clCreateKernel error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(2*sizeof(cl_float)),0,&err);//Min,Max values for tth -8
  if(err){fprintf(stderr,"clCreateKernel error, %s (@%d)\n",ocl_perrc(err),i-1);clean_clbuffers(i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_TTH_DELTA]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sgs->Nimage*sizeof(cl_float)),0,&err);//tth delta -9
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_MASK]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sgs->Nimage*sizeof(cl_int)),0,&err);//Mask -10
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_DUMMYVAL]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(sizeof(cl_float)),0,&err);//Dummy Value -11
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;

  oclconfig->oclmemref[CLMEM_TTH_RANGE]=
    clCreateBuffer(oclconfig->oclcontext,CL_MEM_READ_ONLY,(size_t)(2*sizeof(cl_float)),0,&err);//TTH Range -12
  if(err){fprintf(stderr,"clCreateBuffer error, %s (@%d)\n",ocl_perrc(err),i-1);return -1;};i++;
  
  fprintf(stream,"Allocated %d buffers (%.3f Mb) on device\n",i,(float)ualloc/1024./1024.);
  oclconfig->Nbuffers = i;
return 0;
}

/**
 * \brief Tie arguments of OpenCL kernel-functions to the actual kernels
 *
 * set_kernel_arguments() is a private method, called by configure(). It uses
 * clSetKernelArg() of the OpenCL API to tie kernel arguments to the kernels.
 * Note that by default, since TthRange is disabled, the integration kernels have tth_min_max tied to the tthRange argument slot.
 * When setRange is called it replaces that argument with tthRange low and upper bounds. When unsetRange is called, the argument slot
 * is reset to tth_min_max.
 */
int ocl_xrpd1D_fullsplit::set_kernel_arguments()
{

  //The second argument of clSetKernelArg is directly related to the position of the argument on the kernel (starts from 0).
  // I.e. : __kernel void dummy(int arg0,int arg1,...., __global argN-1)
  int i=0;

  i=0;
  //------------------------bin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH]) ); //tth
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_DELTA]) ); //stth
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UWEIGHTS]) ); //uBin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]) ); //tth_min_max
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_IMAGE]) ); //Image
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UHISTOGRAM]) ); //uHistogram
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_SPAN_RANGES]) ); //span_range
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_MASK]) ); //span_range
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_INTEGRATE],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]) ); //TTH range default

  i=0;
  //------------------------uiMemset2
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UIMEMSET2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UWEIGHTS]) ); //uBin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UIMEMSET2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UHISTOGRAM]) ); //uHistogram

  i=0;
  //------------------------uiMemset2
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_IMEMSET],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_MASK]) ); //uBin arrays
  
  i=0;
  //-----------------------Ulong2Float kernel
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UI2F2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UWEIGHTS]) ); //uBin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UI2F2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_UHISTOGRAM]) ); //uHistogram
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UI2F2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_WEIGHTS]) ); //Bin array
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_UI2F2],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_HISTOGRAM]) ); //Histogram

  i=0;
  //-----------------------get_spans kernel
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH]) ); //tth
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_DELTA]) ); //dtth
  CR( clSetKernelArg(oclconfig->oclkernels[CLKERN_GET_SPANS],i++,sizeof(cl_mem),&oclconfig->oclmemref[CLMEM_TTH_MIN_MAX]) ); //TTH range default
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

/**
 * \brief Free OpenCL related resources allocated by the library.
 *
 * clean() is used to reinitiate the library back in a vanilla state.
 * It may be asked to preserve the context created by init or completely clean up OpenCL.
 * Guard/Status flags that are set will be reset. All the Operation flags are also reset
 *
 * In the case the context is preserved, steps required to be able to perform an integration are
 * getConfiguration, configure, loadTth and then execute may be called.
 * In the case the context is not preserved, invocation of init is required before the above steps. 
 * The total execution time is also reinitialised via reset_time().
 * 
 * @param preserve_context An integer flag to preserve(1) or release(default-0) the active OpenCL context
 */
int ocl_xrpd1D_fullsplit::clean(int preserve_context)
{

  if(hasBuffers)
  {
    clean_clbuffers(oclconfig->Nbuffers);
    fprintf(stream,"--released OpenCL buffers\n");
    hasBuffers = 0;
    hasTthLoaded = 0;
    useSolidAngle = 0;
    useMask = 0;
    useDummyVal = 0;
    useTthRange = 0;    
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
    if(hasActiveContext){
      ocl_destroy_context(oclconfig->oclcontext);
      hasActiveContext=0;
      fprintf(stream,"--released OpenCL context\n");
    }
  }
  return 0;
}

