/**
 * \file
 * \brief  OpenCL tools API
 *
 *  OpenCL tools for device probe, selection, deletion, error notification
 *  and vector type conversion. This source is the low-level layer of our
 *  OpenCL Toolbox (ocl_init_exec.cpp). However, it can be used directly
 *  as an API
 */

/*
 *   Project: OpenCL tools for device probe, selection, deletion, error notification
 *              and vector type conversion. This source is the low-level layer of our
 *              OpenCL Toolbox (ocl_init_context.cpp). However, it can be used directly
 *              as an API
 *
 *   Copyright (C) 2011 - 2012 European Synchrotron Radiation Facility
 *                                 Grenoble, France
 *
 *   Principal authors: D. Karkoulis (karkouli@esrf.fr)
 *   Last revision: 21/06/2012
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
#include "ocl_tools.h"

/*This is required for OpenCL callbacks in windows*/
#ifdef _WIN32
	#ifndef _CRT_SECURE_NO_WARNINGS
		#define _CRT_SECURE_NO_WARNINGS
	#endif
	#pragma warning(disable : 4996)
  //This is required for OpenCL callbacks in windows.
  //This can also be achieved by setting cl.exe flags.
  // flag /Gz uses the __stdcall calling convention
  #define __call_compat __stdcall
#else
  #define __call_compat 
#endif

//#define silent
#ifdef _SILENT
  #define fprintf(stream,...)
#endif

#define CE CL_CHECK_ERR_PR ///Short for CL_CHECK_ERR_PR
#define C CL_CHECK_PR      ///Short for CL_CHECK_PR
#define CL CL_CHECK_PR_RET ///short for CL_CHECK_PR_RET

/**
 * Sets some data members of oclconfig to a default value.
 * Calls ocl_platform_info_init() and ocl_device_info_init()
 *
 * @param oclconfig The OpenCL configuration to be initialised
 * @return void
 */
void ocl_tools_initialise(ocl_config_type *oclconfig)
{
  oclconfig->platfid = -1;
  oclconfig->devid = -1;
  oclconfig->Nbuffers=0;
  oclconfig->Nkernels=0;
  oclconfig->oclmemref =NULL;
  oclconfig->oclkernels=NULL;
  ocl_platform_info_init(oclconfig->platform_info);
  ocl_device_info_init(oclconfig->device_info);

  return;
}

/**
 * Calls ocl_platform_info_del() and ocl_device_info_del().
 * Other data members are cleaned internally when appropriate by
 * other calls.
 *
 * @param oclconfig The OpenCL configuration to be initialised
 * @return void
 */
void ocl_tools_destroy(ocl_config_type *oclconfig)
{
  //Deallocate memory
  ocl_platform_info_del(oclconfig->platform_info);
  ocl_device_info_del(oclconfig->device_info);

  return;
}

/**
 * Check for devices that match the cl_device_type specification.
 * Devices that meet the criteria are checked to be available for computing.
 * If a device is probed successfully, it's id is stored internally.
 *
 * @param oclconfig The OpenCL configuration. It is used only as output here
 *         and the following fields are updated:
 *         ocldevice, oclplatform, dev_mem, compiler_options.
 * @param ocldevtype A cl_device_type bit-field, designated the type of device to
 *         look for:
 *         CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_DEFAULT,
 *         CL_DEVICE_TYPE_ACCELERATOR and CL_DEVICE_TYPE_ALL.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *         to stdout. If another stream is used, it must be open.
 *         Error messages are always displayed to stderr.
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 *        -2: Other kind of error
 */
int ocl_probe(ocl_config_type *oclconfig,cl_device_type ocldevtype,int usefp64,FILE *stream){

  //OpenCL API types
  cl_platform_id *oclplatforms;
  cl_device_id *ocldevices;

  //clGetDeviceInfo returns datatype depending on requested info.
  cl_int check_ocl=0;
  int device_found=0;
  cl_uint num_platforms,num_devices;
  cl_bool param_value_b;
  size_t param_value_size;
  char param_value_s[1024];

  oclplatforms = (cl_platform_id*)malloc(OCL_MAX_PLATFORMS*sizeof(cl_platform_id));
  ocldevices   = (cl_device_id*)malloc(OCL_MAX_DEVICES*sizeof(cl_device_id));

  //Get OpenCL-capable Platforms. A platform is not the GPU or CPU etc, but it is related to vendor and OpenCL version.
  // I.e. If an Nvidia card is present and the ATI SDK is installed (gives access to CPU) we will see 2 platforms!
  // One platform will have OpenCL v1.0 and vendor NVIDIA and the other OpenCL v1.1 and vendor AMD
  CL_CHECK_PRN( clGetPlatformIDs(OCL_MAX_PLATFORMS,oclplatforms,&num_platforms) );

  if(num_platforms==0){
    ocl_errmsg("No OpenCL platforms detected", __FILE__, __LINE__);
    free(oclplatforms);
    free(ocldevices);  
    return -1;
  }
  for(cl_uint i=0;i<num_platforms;i++){
    //After we get the platforms, check them all for devices that meet our criteria (ocldevtype, Availability):
    check_ocl = clGetDeviceIDs(oclplatforms[i],ocldevtype,OCL_MAX_DEVICES,ocldevices,&num_devices) ;
    if(check_ocl != CL_SUCCESS || (num_devices<1)) continue;
    else{
      for(cl_uint j=0;j<num_devices;j++){
        CL_CHECK_PRN( clGetDeviceInfo(ocldevices[j],CL_DEVICE_AVAILABLE,sizeof(cl_device_info),(void*)&param_value_b,&param_value_size));
        
        if( param_value_b==CL_TRUE ){
          if(usefp64)
          {
            if(ocl_eval_FP64(ocldevices[j])) continue;
          }
          oclconfig->ocldevice  = ocldevices[j]; oclconfig->devid = j;
          oclconfig->oclplatform= oclplatforms[i]; oclconfig->platfid = i;
          CL_CHECK(clGetDeviceInfo(ocldevices[j],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),&oclconfig->dev_mem,NULL));
          CL_CHECK(clGetDeviceInfo(ocldevices[j],CL_DEVICE_NAME,sizeof(param_value_s),&param_value_s,NULL));
          fprintf(stream,"Selected device: (%d.%d) %s. \n",i,j,param_value_s);
          CL_CHECK(clGetDeviceInfo(ocldevices[j],CL_DEVICE_VENDOR,sizeof(param_value_s),&param_value_s,NULL));
          //Since kernel compilation is on the fly for OpenCL we can get device information and create customised
          // compilation parameters. Here we request that nvidia compiler gives a verbose output.
          if(!strcmp("NVIDIA Corporation",param_value_s)){
            fprintf(stream,"Device is NVIDIA, adding nv extensions\n");
            sprintf(oclconfig->compiler_options,"-cl-nv-verbose");
          }else sprintf(oclconfig->compiler_options," ");
          device_found = 1;
          break;
        }
      }
    }if(device_found)break;
  }
  free(oclplatforms);
  free(ocldevices);
  if(!device_found){ocl_errmsg("No device found / No device matching the criteria found", __FILE__, __LINE__);return -1;}
  else return 0;
 
}

/**
 * Check for a specific device on a specific platform with a specific id. Such a list
 * can be retrieved by using ocl_check_platforms().
 * If such device exists it is checked to be available for computing.
 * If a device is probed successfully, it's id is stored internally.
 *
 * @param oclconfig The OpenCL configuration. It is used only as output here
 *          and the following fields are updated:
 *          ocldevice, oclplatform, dev_mem, compiler_options.
 * @param ocldevtype A cl_device_type bit-field, designated the type of device to
 *          look for:
 *          CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_DEFAULT,
 *          CL_DEVICE_TYPE_ACCELERATOR and CL_DEVICE_TYPE_ALL.
 * @param preset_platform Order of the platform
 *          (i.e first is 0, seconds is 1).
 *          Notice that platforms order is prone to changes. It solely depends on which
 *          OpenCL driver is loaded first by the ICD.
 * @param preset_device Order of the device
 *          (i.e first is 0, seconds is 1).
 *          Unlike platforms, device order does not usually change on a machine.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr.
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 *        -2: Other kind of error
 */
int ocl_probe(ocl_config_type *oclconfig,cl_device_type ocldevtype,int preset_platform, int preset_device,int usefp64,
              FILE *stream){

  cl_int check_ocl=0;
  cl_platform_id *oclplatforms;
  cl_device_id *ocldevices;  
  cl_uint num_platforms,num_devices=0;
  cl_bool param_value_b;
  size_t param_value_size;
  char param_value_s[1024];

  oclplatforms = (cl_platform_id*)malloc(OCL_MAX_PLATFORMS*sizeof(cl_platform_id));
  ocldevices   = (cl_device_id*)malloc(OCL_MAX_DEVICES*sizeof(cl_device_id));
  
  check_ocl = clGetPlatformIDs(OCL_MAX_PLATFORMS,oclplatforms,&num_platforms);
  
  if(num_platforms==0 || check_ocl != CL_SUCCESS){
    fprintf(stderr,"No OpenCL platforms detected (%s@%d with error %s)\n", __FILE__, __LINE__, ocl_perrc(check_ocl));
    free(oclplatforms);
    free(ocldevices);
    return -1;    
  }
  if(int(num_platforms-1)<preset_platform){
    fprintf(stderr,"!!Bad choice: preset_platform %d, preset_device %d. Available platforms %d (Take care of C notation)\n",preset_platform,preset_device,num_platforms);
    free(oclplatforms);
    free(ocldevices);    
    return -2;
  }
  if(preset_platform<0 || preset_device<0 || preset_platform >9 || preset_device>9){
    fprintf(stderr,"!!Bad choice: preset_platform %d, preset_device %d\n",preset_platform,preset_device);
    free(oclplatforms);
    free(ocldevices);    
    return -2;
  }
  
  check_ocl =  clGetDeviceIDs(oclplatforms[preset_platform],ocldevtype,OCL_MAX_DEVICES,ocldevices,&num_devices);

  if(int(num_devices-1)<preset_device || check_ocl != CL_SUCCESS){
    fprintf(stderr,"!!Bad choice: preset_platform %d, preset_device %d. Available devices %d.(%s)\n",preset_platform,preset_device,num_devices,ocl_perrc(check_ocl));
    free(oclplatforms);
    free(ocldevices);    
    return -2;
  }
  CL_CHECK(clGetDeviceInfo(ocldevices[preset_device],CL_DEVICE_AVAILABLE,sizeof(cl_device_info),
                                                        (void*)&param_value_b,&param_value_size));
  if(param_value_b==CL_TRUE){
    if(usefp64)
    {
      if(ocl_eval_FP64(ocldevices[preset_device]))
      {
        free(oclplatforms);
        free(ocldevices);
        fprintf(stderr,"Preset device is not FP64 capable\n");
        return -1;
      }
    }    
    oclconfig->ocldevice  = ocldevices[preset_device]; oclconfig->devid = preset_device;
    oclconfig->oclplatform= oclplatforms[preset_platform]; oclconfig->platfid = preset_platform;
    CL_CHECK(clGetDeviceInfo(ocldevices[preset_device],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),&oclconfig->dev_mem,NULL));
    CL_CHECK(clGetDeviceInfo(ocldevices[preset_device],CL_DEVICE_NAME,sizeof(param_value_s),&param_value_s,NULL));
    fprintf(stream,"Selected device: (%d.%d) %s. \n",preset_platform,preset_device,param_value_s);
    CL_CHECK(clGetDeviceInfo(ocldevices[preset_device],CL_DEVICE_VENDOR,sizeof(param_value_s),&param_value_s,NULL));
    if(!strcmp("NVIDIA Corporation",param_value_s)){
      fprintf(stream,"Device is NVIDIA, adding nv extensions\n");
      sprintf(oclconfig->compiler_options,"-cl-nv-verbose");
    }else sprintf(oclconfig->compiler_options," ");
    free(oclplatforms);
    free(ocldevices);       
    return 0;
  } else {
    fprintf(stderr,"Preset device not available for computations\n");
    free(oclplatforms);
    free(ocldevices);       
    return -1;
  }
  
  return -2;
}

/**
 * Check for a specific device using directly internals OpenCL types instead of the order.
 * The internal OpenCL types can be acquired directly through the OpenCL API or by using
 * ocl_find_devicetype().
 * If such device exists it is checked to be available for computing.
 * If a device is probed successfully, it's id is stored internally.
 *
 * @param oclconfig The OpenCL configuration. It is used only as output here
 *          and the following fields are updated:
 *          ocldevice, oclplatform, dev_mem, compiler_options.
 * @param platform The OpenCL cl_platform_id for a given platform. This can be acquired
 *          on runtime via the OpenCL API.
 * @param device The OpenCL cl_device_id for a given platform. This can be acquired on
 *          on runtime via the OpenCL API.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr.
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 *        -2: Other kind of error
 */
int ocl_probe(ocl_config_type *oclconfig,cl_platform_id platform,cl_device_id device,int usefp64,FILE *stream){

  cl_bool param_value_b;
  size_t param_value_size;
  char param_value_s[1024];
  cl_uint num_platforms,num_devices;
  int finished = 0;
  
  CL_CHECK_PRN(clGetDeviceInfo(device,CL_DEVICE_AVAILABLE,sizeof(cl_device_info),
                                                        (void*)&param_value_b,&param_value_size));
  if(param_value_b==CL_TRUE){
    if(usefp64)
    {
      if(ocl_eval_FP64(device))
      {
        fprintf(stderr,"Preset device is not FP64 capable\n");
        return -1;
      }
    }        
    oclconfig->ocldevice  = device;
    oclconfig->oclplatform= platform;

    cl_platform_id oclplatforms[OCL_MAX_PLATFORMS];
    cl_device_id ocldevices[OCL_MAX_DEVICES];
    CL_CHECK_PRN( clGetPlatformIDs(OCL_MAX_PLATFORMS,oclplatforms,&num_platforms) );
    
    for(cl_uint i =0; i<num_platforms;i++)
    {
      if (oclplatforms[i] == platform)
      {
        oclconfig->platfid = i;
        clGetDeviceIDs(oclplatforms[i],CL_DEVICE_TYPE_ALL,OCL_MAX_DEVICES,ocldevices,&num_devices) ;
        for(cl_uint j = 0; j<num_devices;j++)
        {
          if(ocldevices[j] == device)
          {
            oclconfig->devid = j;
            finished=1;
            break;
          }
        }
      }
      if(finished)break;
    }
    
    CL_CHECK(clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),&oclconfig->dev_mem,NULL));
    CL_CHECK(clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(param_value_s),&param_value_s,NULL));
    fprintf(stream,"Selected device: %s. \n",param_value_s);
    CL_CHECK(clGetDeviceInfo(device,CL_DEVICE_VENDOR,sizeof(param_value_s),&param_value_s,NULL));
    if(!strcmp("NVIDIA Corporation",param_value_s)){
      fprintf(stream,"Device is NVIDIA, adding nv extensions\n");
      sprintf(oclconfig->compiler_options,"-cl-nv-verbose");
    }else sprintf(oclconfig->compiler_options," ");
    return 0;
  } else {
    fprintf(stderr,"Preset device not available for computations\n");
    return -1;
  }

  return -2;
}

/* Simple check for a "device_type" device. Returns the first occurance only
 * Typically, the first occurance of a GPU device on a GPU platform is the best one
 */

/**
 * Finds and saves in internal OpenCL types the first occurence of a specific OpenCL device type.
 * In the case of NVIDIA OpenCL implementation, the first device is typically the most
 * capable.
 *
 * @param device_type A cl_device_type bit-field, designated the type of device to
 *         look for:
 *         CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_DEFAULT,
 *         CL_DEVICE_TYPE_ACCELERATOR and CL_DEVICE_TYPE_ALL.
 * @param &platform The cl_platform_id variable to save the first platform encountered
 *         which is a device_type device.
 * @param &devid The cl_device_id variable to save the first device encountered which
 *         is a device_type device.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr.
 * 
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 */
int ocl_find_devicetype(cl_device_type device_type, cl_platform_id &platform, cl_device_id &devid){

  cl_platform_id *oclplatforms;
  cl_device_id *ocldevices;
  cl_uint num_platforms,num_devices;
  num_devices = 0;

  oclplatforms = (cl_platform_id*)malloc(OCL_MAX_PLATFORMS*sizeof(cl_platform_id));
  ocldevices   = (cl_device_id*)malloc(OCL_MAX_DEVICES*sizeof(cl_device_id));

  CL_CHECK_PRN (clGetPlatformIDs(OCL_MAX_PLATFORMS,oclplatforms,&num_platforms));
  if(num_platforms){
    for(cl_uint i=0;i<num_platforms;i++){
      clGetDeviceIDs(oclplatforms[i],device_type,OCL_MAX_DEVICES,ocldevices,&num_devices);
      if(num_devices){
        devid = ocldevices[0];
        platform = oclplatforms[i];
        free(oclplatforms);
        free(ocldevices);        
        return 0;
      }
    }
  }
  
  free(oclplatforms);
  free(ocldevices);
  return -1;
};

/**Constrains search to FP64 capable devices.
 * Finds and saves in internal OpenCL types the first occurence of a specific OpenCL device type.
 * In the case of NVIDIA OpenCL implementation, the first device is typically the most
 * capable.
 *
 * @param device_type A cl_device_type bit-field, designated the type of device to
 *         look for:
 *         CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_DEFAULT,
 *         CL_DEVICE_TYPE_ACCELERATOR and CL_DEVICE_TYPE_ALL.
 * @param &platform The cl_platform_id variable to save the first platform encountered
 *         which is a device_type device.
 * @param &devid The cl_device_id variable to save the first device encountered which
 *         is a device_type device.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr.
 * 
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 */
int ocl_find_devicetype_FP64(cl_device_type device_type, cl_platform_id &platform, cl_device_id &devid){

  cl_platform_id *oclplatforms;
  cl_device_id *ocldevices;
  cl_uint num_platforms,num_devices;
  num_devices = 0;

  oclplatforms = (cl_platform_id*)malloc(OCL_MAX_PLATFORMS*sizeof(cl_platform_id));
  ocldevices   = (cl_device_id*)malloc(OCL_MAX_DEVICES*sizeof(cl_device_id));

  CL_CHECK_PRN (clGetPlatformIDs(OCL_MAX_PLATFORMS,oclplatforms,&num_platforms));
  if(num_platforms){
    for(cl_uint i=0;i<num_platforms;i++){
      clGetDeviceIDs(oclplatforms[i],device_type,OCL_MAX_DEVICES,ocldevices,&num_devices);
      if(num_devices){
				for(cl_uint idev=0;idev<num_devices; idev++){
					if(!ocl_eval_FP64(ocldevices[idev])){
						devid = ocldevices[idev];
						platform = oclplatforms[i];
						free(oclplatforms);
						free(ocldevices);        
						return 0;
					}
				}
      }
    }
  }
  
  free(oclplatforms);
  free(ocldevices);
  return -1;
};

/**
 * Enumerates OCL_MAX_PLATFORMS OpenCL platforms and OCL_MAX_DEVICES OpenCL devices
 * and prints their information.
 * @param device_type A cl_device_type bit-field, designated the type of device to
 *         look for:
 *         CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_DEFAULT,
 *         CL_DEVICE_TYPE_ACCELERATOR and CL_DEVICE_TYPE_ALL.
 * @param &platform The cl_platform_id variable to save the first platform encountered
 *         which is a device_type device.
 * @param &devid The cl_device_id variable to save the first device encountered which
 *         is a device_type device.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr.
 * 
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 */
int ocl_check_platforms(FILE *stream){

  cl_platform_id *oclplatforms;
  cl_device_id *ocldevices;
  cl_uint num_platforms,num_devices;
  char param_value[10000];
  size_t param_value_size=sizeof(param_value);

  oclplatforms = (cl_platform_id*)malloc(OCL_MAX_PLATFORMS*sizeof(cl_platform_id));
  ocldevices   = (cl_device_id*)malloc(OCL_MAX_DEVICES*sizeof(cl_device_id));

  CL_CHECK_PRN (clGetPlatformIDs(OCL_MAX_PLATFORMS,oclplatforms,&num_platforms));
  if(num_platforms==0) {
    fprintf(stderr,"No OpenCL compatible platform found\n");
    free(oclplatforms);
    free(ocldevices);
    return -1;
  }else{
    fprintf(stream,"%d OpenCL platform(s) found\n",num_platforms);
    for(cl_uint i=0;i<num_platforms;i++){
      fprintf(stream," Platform info:\n");
//       clGetPlatformInfo(oclplatforms[i],CL_PLATFORM_PROFILE,param_value_size,&param_value,NULL);
//       printf(" %s\n",param_value);
//       CL_CHECK_PRN(clGetPlatformInfo(oclplatforms[i],CL_PLATFORM_VERSION,param_value_size,&param_value,NULL));
//       printf(" %s\n",param_value);
      CL_CHECK_PRN(clGetPlatformInfo(oclplatforms[i],CL_PLATFORM_NAME,param_value_size,&param_value,NULL));
      fprintf(stream,"  %s\n",param_value);
//       clGetPlatformInfo(oclplatforms[i],CL_PLATFORM_VENDOR,param_value_size,&param_value,NULL);
//       printf(" %s\n",param_value);      
//       clGetPlatformInfo(oclplatforms[i],CL_PLATFORM_EXTENSIONS,param_value_size,&param_value,NULL);
//       printf(" %s\n",param_value);
      CL_CHECK_PRN( clGetDeviceIDs(oclplatforms[i],CL_DEVICE_TYPE_ALL,OCL_MAX_DEVICES,ocldevices,&num_devices) );
      fprintf(stream,"  %d Device(s) found:\n",num_devices);
      for(cl_uint j=0;j<num_devices;j++){
          CL_CHECK_PRN(clGetDeviceInfo(ocldevices[j],CL_DEVICE_NAME,param_value_size,&param_value,NULL));
          fprintf(stream,"   (%d.%d): %s\n",i,j,param_value);
      }    
    }
  }
  free(oclplatforms);
  free(ocldevices);
  return 0;

}

/**
 * Releases an OpenCL context while performing error checking.
 *
 * @param oclcontext The OpenCL context to be Released.
 * 
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 */
int ocl_destroy_context(cl_context oclcontext){

  CL_CHECK_PR_RET(clReleaseContext(oclcontext));

return 0;
}

/* Needs a string with the type of device: GPU,CPU,ACC,ALL,DEF. Runs ocl_probe and creates the context,
    adding it on the appropriate ocl_config_type field*/
/**
 * Creates an OpenCL context, by first invoking ocl_probe() with the desired device type
 *
 * @param oclconfig oclconfig will be used to keep the resulting OpenCL configuration.
 * @param device_type A string with the device type to be used. Accepted values:
 *          "GPU","CPU","ACC","ALL","DEF".
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 *        -2: Other kind of error
 */
int ocl_init_context(ocl_config_type *oclconfig,const char *device_type, int usefp64,FILE *stream){

  cl_device_type ocldevtype;
  cl_int err;

  if(!oclconfig){
    ocl_errmsg("Fatal error in ocl_init_context. oclconfig does not exist",__FILE__,__LINE__);
    return -2;
  }
  if( ocl_string_to_cldevtype(device_type,ocldevtype) )return -2;

  if( ocl_probe(oclconfig,ocldevtype,usefp64,stream) )return -1;
  
  //Create the context for the chosen device
  cl_context_properties akProperties[] ={CL_CONTEXT_PLATFORM, (cl_context_properties)oclconfig->oclplatform,0};
  
  oclconfig->oclcontext = clCreateContext(akProperties,1,&oclconfig->ocldevice,NULL/*&pfn_notify*/,NULL,&err);
  if(err){fprintf(stderr,"Context failed: %s (%d)\n",ocl_perrc(err),err);return -1; }

  ocl_current_device_info(oclconfig);
  ocl_current_platform_info(oclconfig);

return 0;  
}

/**
 * Creates an OpenCL context, by first invoking ocl_probe() with the desired device type,
 * platform and device order
 *
 * @param oclconfig oclconfig will be used to keep the resulting OpenCL configuration.
 * @param device_type A string with the device type to be used. Accepted values:
 *          "GPU","CPU","ACC","ALL","DEF".
 * @param preset_platform Explicit platform to use when probing for device_type
 * @param devid Explicit device number to use when probing for device_type
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 *        -2: Other kind of error
 */
int ocl_init_context(ocl_config_type *oclconfig,const char *device_type,int preset_platform,int devid, int usefp64,
                     FILE *stream){

  cl_device_type ocldevtype;
  cl_int err;

  if(!oclconfig){
    ocl_errmsg("Fatal error in ocl_init_context. oclconfig does not exist",__FILE__,__LINE__);
    return -1;
  }
  if( ocl_string_to_cldevtype(device_type,ocldevtype) )return -2;

  if( ocl_probe(oclconfig,ocldevtype,preset_platform,devid,usefp64,stream) )return -1;
  //Create the context for the chosen device
  cl_context_properties akProperties[] ={CL_CONTEXT_PLATFORM, (cl_context_properties)oclconfig->oclplatform,0};
  oclconfig->oclcontext = clCreateContext(akProperties,1,&oclconfig->ocldevice,NULL/*&pfn_notify*/,NULL,&err);
  if(err){fprintf(stderr,"Context failed: %s (%d)\n",ocl_perrc(err),err);return -1; }

  ocl_current_device_info(oclconfig);
  ocl_current_platform_info(oclconfig);

return 0;  
}

/**
 * Creates an OpenCL context directly with an OpenCL device that is described OpenCL internal types
 * \todo To probe a cl_device_id, cl_platform_id is not required. But it is required to create the context.
 *       In a future revision this will be fixed so that the cl_platform_id is retrieved by the cl_device_id
 *
 * @param oclconfig oclconfig will be used to keep the resulting OpenCL configuration.
 * @param platform cl_platform_id value for the device
 * @param device cl_device_id value of the device
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 *        -2: Other kind of error
 */
int ocl_init_context(ocl_config_type *oclconfig,cl_platform_id platform,cl_device_id device, int usefp64,
                     FILE *stream){

  cl_int err;

  if(!oclconfig){
    ocl_errmsg("Fatal error in ocl_init_context. oclconfig does not exist",__FILE__,__LINE__);
    return -1;
  }

  if( ocl_probe(oclconfig,platform,device,usefp64,stream) )return -1;
  //Create the context for the chosen device
  cl_context_properties akProperties[] ={CL_CONTEXT_PLATFORM, (cl_context_properties)oclconfig->oclplatform,0};
  oclconfig->oclcontext = clCreateContext(akProperties,1,&oclconfig->ocldevice,NULL/*&pfn_notify*/,NULL,&err);
  if(err){fprintf(stderr,"Context failed: %s (%d)\n",ocl_perrc(err),err);return -1; }

  ocl_current_device_info(oclconfig);
  ocl_current_platform_info(oclconfig);

return 0;
};

/* OpenCL Compiler for dynamic kernel creation. It will always report success or failure of the build.*/
/**
 * OpenCL Just-In-Time compiler, to create OpenCL kernel program on runtime.
 * This implementation expects a file with the OpenCL kernel source code. This file
 * or a symlink must be on the execution path. The status of the build is always reported, wether
 * successful or failed.
 *
 * @param oclconfig The OpenCL configuration that will hold the compiled program and kernels.
 * @param kernelfilename The filename of the file containing the OpenCL source code.
 * @param BLOCK_SIZE The blockSize. This value will be defined in the compiled program.
 * @param optional A string containing additional compilation options. These options along will
 *          be appended to the compilation string included in oclconfig and the BLOCK_SIZE.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 *        -2: Other kind of error
 */
int ocl_compiler(ocl_config_type *oclconfig,const char *kernelfilename,int BLOCK_SIZE,const char *optional, FILE *stream){

  
  FILE *kernel;
  cl_int err;    

  oclconfig->kernelstring_lens = (size_t*)malloc(1*sizeof(size_t));
  oclconfig->kernelstrings = (char**)malloc(1*sizeof(char));

  //Load the kernel in a string
  kernel=fopen(kernelfilename,"r");
  if(!kernel){fprintf(stderr,"INVALID OpenCL file %s\n",kernelfilename);return -1;} //Fallback
  
  fseek(kernel,0,SEEK_END);
  oclconfig->kernelstring_lens[0]=ftell(kernel);
  rewind(kernel);

  oclconfig->kernelstrings[0] = (char*)malloc(oclconfig->kernelstring_lens[0]);
  fread ( oclconfig->kernelstrings[0], sizeof(char), oclconfig->kernelstring_lens[0]/sizeof(char), kernel );
  fprintf(stream,"Loaded kernel in string\n");
  fclose(kernel);
  
  //Create Program from kernel string
  oclconfig->oclprogram  = \
    clCreateProgramWithSource(oclconfig->oclcontext,1,\
    (const char**)&oclconfig->kernelstrings[0],(const size_t*)&oclconfig->kernelstring_lens[0],&err);
  if(err)fprintf(stderr,"%s\n",ocl_perrc(err));
  
  //Compile the program
  char *compiler_options_flush;
	char *compiler_options_temp;
	compiler_options_flush = (char*)malloc(10000*sizeof(char));
	compiler_options_temp  = (char*)malloc(10000*sizeof(char));
	sprintf(compiler_options_temp,"-I. -D BLOCK_SIZE=%d %s",BLOCK_SIZE,oclconfig->compiler_options);	
  if(optional) sprintf(compiler_options_flush,"%s %s",compiler_options_temp,optional);
	else sprintf(compiler_options_flush,"%s",compiler_options_temp);
  fprintf(stream,"Compiler options: %s\n",compiler_options_flush);

  err = clBuildProgram(oclconfig->oclprogram, 1, &oclconfig->ocldevice, compiler_options_flush, 0, 0);
	free(compiler_options_flush);

  if(err){
    char buildinfo[100000];
    fprintf(stderr,"clBuildProgram has failed: %s\n",ocl_perrc(err));
    CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,sizeof(buildinfo),&buildinfo,NULL));
    fprintf(stderr,"%s\n",buildinfo);
    return -1; //Fallback
  } else{
    char buildinfo[100000];
    cl_build_status   build_status;        
    CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_STATUS,sizeof(build_status),&build_status,NULL));\
    fprintf(stream,"clBuildProgram was successful: \n");
    fprintf(stream,"------------------------------------------------------------\n");
    if(build_status==CL_BUILD_SUCCESS)fprintf(stream,"CL_BUILD_SUCCESS\n");
    else if (build_status==CL_BUILD_NONE)fprintf(stream,"CL_BUILD_NONEd\n");
    else if (build_status==CL_BUILD_IN_PROGRESS)fprintf(stream,"CL_BUILD_IN_PROGRESS\n");
    CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,sizeof(buildinfo),&buildinfo,NULL));
    fprintf(stream,"%s\n",buildinfo);
    fprintf(stream,"------------------------------------------------------------\n");
    
  }
return 0;
}

/* OpenCL Compiler for dynamic kernel creation. It will always report success or failure of the build.*/
/* Here it can take multiple cl files*/
/**
 * OpenCL Just-In-Time compiler, to create OpenCL kernel program on runtime.
 * This implementation expects multiple with the OpenCL kernel source code. The files
 * or their symlinks must be on the execution path. The status of the build is always reported, wether
 * successful or failed.
 *
 * @param oclconfig The OpenCL configuration that will hold the compiled program and kernels.
 * @param clList An array of filenames of the files containing the OpenCL source code.
 * @param clNum The number of filenames to be parsed.
 * @param BLOCK_SIZE The blockSize. This value will be defined in the compiled program.
 * @param optional A string containing additional compilation options. These options along will
 *          be appended to the compilation string included in oclconfig and the BLOCK_SIZE.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 *        -2: Other kind of error
 */
int ocl_compiler(ocl_config_type *oclconfig,const char **clList,int clNum,int BLOCK_SIZE,const char *optional, FILE *stream){


  FILE *kernel;
  cl_int err;

  char compiler_options_flush[10000];
  char compiler_options_temp[10000];

  
  if(BLOCK_SIZE > 9999){
    fprintf(stderr,"Blocksize too big");
    return -2;
  }
  
  size_t optlen=0;
  size_t complen=strlen(oclconfig->compiler_options) +2;
  size_t deflen=strlen("-I. -D BLOCK_SIZE=9999 ") + 2;
  
  if(optional) optlen = strlen(optional) + 2;
  if(optlen + complen + deflen > 9999){
    fprintf(stderr,"Compile string is too long\n");
    return -2;
  }

  sprintf(compiler_options_temp,"-I. -D BLOCK_SIZE=%d %s",BLOCK_SIZE,oclconfig->compiler_options);
  if(optional) sprintf(compiler_options_flush,"%s %s",compiler_options_temp,optional);
  else sprintf(compiler_options_flush,"%s",compiler_options_temp);
  fprintf(stream,"Compiler options: %s\n",compiler_options_flush);
    
  oclconfig->nprgs=clNum;
  oclconfig->prgs = new ocl_program_type [clNum];
  for(int clfile=0;clfile<clNum;clfile++)
  {
    oclconfig->prgs[clfile].kernelstring_lens = (size_t*)malloc(1*sizeof(size_t));
    oclconfig->prgs[clfile].kernelstrings = (char**)malloc(1*sizeof(char*));

    //Load the kernel in a string
    kernel=fopen(clList[clfile],"r");
    if(!kernel){fprintf(stderr,"INVALID OpenCL file %s\n",clList[clfile]);return -1;} //Fallback

    fseek(kernel,0,SEEK_END);
    oclconfig->prgs[clfile].kernelstring_lens[0]=ftell(kernel);
    rewind(kernel);

    oclconfig->prgs[clfile].kernelstrings[0] = (char*)malloc(oclconfig->prgs[clfile].kernelstring_lens[0]);
    fread ( oclconfig->prgs[clfile].kernelstrings[0], sizeof(char), oclconfig->prgs[clfile].kernelstring_lens[0]/sizeof(char), kernel );
    fprintf(stream,"Loaded kernel \"%s\" in string\n",clList[clfile]);
    fclose(kernel);

    //Create Program from kernel string
    oclconfig->prgs[clfile].oclprogram  = \
      clCreateProgramWithSource(oclconfig->oclcontext,1,\
      (const char**)&(oclconfig->prgs[clfile].kernelstrings[0]),(const size_t*)&(oclconfig->prgs[clfile].kernelstring_lens[0]),&err);
    if(err)fprintf(stderr,"%s\n",ocl_perrc(err));

    //Compile the program
    err = clBuildProgram(oclconfig->prgs[clfile].oclprogram, 1, &oclconfig->ocldevice, compiler_options_flush, 0, 0);
    char    *buildinfo;
    size_t   binf_size;
    
    if(err){
      fprintf(stderr,"clBuildProgram has failed: %s\n",ocl_perrc(err));
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,0,NULL,&binf_size) );
      buildinfo = (char *)malloc((binf_size+1)*sizeof(char));
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,binf_size,buildinfo,NULL));
      buildinfo[binf_size] = '\0';
      fprintf(stderr,"%s\n",buildinfo);
      free(buildinfo);
      return -1; //Fallback
    } else{
      cl_build_status   build_status;
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_STATUS,sizeof(build_status),&build_status,NULL));\
      fprintf(stream,"clBuildProgram was successful: \n");
      fprintf(stream,"------------------------------------------------------------\n");
      if(build_status==CL_BUILD_SUCCESS)fprintf(stream,"CL_BUILD_SUCCESS\n");
      else if (build_status==CL_BUILD_NONE)fprintf(stream,"CL_BUILD_NONEd\n");
      else if (build_status==CL_BUILD_IN_PROGRESS)fprintf(stream,"CL_BUILD_IN_PROGRESS\n");
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,0,NULL,&binf_size));
      buildinfo = (char *)malloc((binf_size+1)*sizeof(char));
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,binf_size,buildinfo,NULL));
      buildinfo[binf_size] = '\0';      
      fprintf(stream,"%s\n",buildinfo);
      free(buildinfo);      
      fprintf(stream,"------------------------------------------------------------\n");

    }
  }
return 0;
}

/*This variant takes a list of strings and their lengths instead of a list of files*/
/**
 * OpenCL Just-In-Time compiler, to create OpenCL kernel program on runtime.
 * This implementation expects an array of strings with the OpenCL kernel source code and thus
 * does not have a special requirement concerning the path of the source.
 * The status of the build is always reported, wether successful or failed.
 *
 * @param oclconfig The OpenCL configuration that will hold the compiled program and kernels.
 * @param clList An array of strings containing the OpenCL source code.
 * @param clLen An array of integers containing the length of each string.
 * @param clNum The number of strings to be parsed. 
 * @param BLOCK_SIZE The blockSize. This value will be defined in the compiled program.
 * @param optional A string containing additional compilation options. These options along will
 *          be appended to the compilation string included in oclconfig and the BLOCK_SIZE.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr
 *
 * @return An integer that represends the ocl_tools error_code:
 *         0: Success
 *        -1: OpenCL API related error
 *        -2: Other kind of error
 */
int ocl_compiler(ocl_config_type *oclconfig,unsigned char **clList,unsigned int *clLen,int clNum,int BLOCK_SIZE,const char *optional, FILE *stream){


  cl_int err;

  char compiler_options_flush[10000];
  char compiler_options_temp[10000];


  if(BLOCK_SIZE > 9999){
    fprintf(stderr,"Blocksize too big");
    return -2;
  }

  size_t optlen=0;
  size_t complen=strlen(oclconfig->compiler_options) +2;
  size_t deflen=strlen("-I. -D BLOCK_SIZE=9999 ") + 2;

  if(optional) optlen = strlen(optional) + 2;
  if(optlen + complen + deflen > 9999){
    fprintf(stderr,"Compile string is too long\n");
    return -2;
  }

  sprintf(compiler_options_temp,"-I. -D BLOCK_SIZE=%d %s",BLOCK_SIZE,oclconfig->compiler_options);
  if(optional) sprintf(compiler_options_flush,"%s %s",compiler_options_temp,optional);
  else sprintf(compiler_options_flush,"%s",compiler_options_temp);
  fprintf(stream,"Compiler options: %s\n",compiler_options_flush);

  oclconfig->nprgs=clNum;
  oclconfig->prgs = new ocl_program_type [clNum];
  for(int clfile=0;clfile<clNum;clfile++)
  {
    oclconfig->prgs[clfile].kernelstring_lens = (size_t*)malloc(1*sizeof(size_t));
    oclconfig->prgs[clfile].kernelstrings = (char**)malloc(1*sizeof(char*));

    oclconfig->prgs[clfile].kernelstring_lens[0] = clLen[clfile]; //Get the length
    oclconfig->prgs[clfile].kernelstrings[0]  = (char*) clList[clfile]; //And the reference to the string

    //Create Program from kernel string
    oclconfig->prgs[clfile].oclprogram  = \
      clCreateProgramWithSource(oclconfig->oclcontext,1,\
      (const char**)&(oclconfig->prgs[clfile].kernelstrings[0]),(const size_t*)&(oclconfig->prgs[clfile].kernelstring_lens[0]),&err);
    if(err)fprintf(stderr,"%s\n",ocl_perrc(err));

    //Compile the program
    err = clBuildProgram(oclconfig->prgs[clfile].oclprogram, 1, &oclconfig->ocldevice, compiler_options_flush, 0, 0);
    char    *buildinfo;
    size_t   binf_size;

    if(err){
      fprintf(stderr,"clBuildProgram has failed: %s\n",ocl_perrc(err));
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,0,NULL,&binf_size) );
      buildinfo = (char *)malloc((binf_size+1)*sizeof(char));
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,binf_size,buildinfo,NULL));
      buildinfo[binf_size] = '\0';
      fprintf(stderr,"%s\n",buildinfo);
      free(buildinfo);
      return -1; //Fallback
    } else{
      cl_build_status   build_status;
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_STATUS,sizeof(build_status),&build_status,NULL));\
      fprintf(stream,"clBuildProgram was successful: \n");
      fprintf(stream,"------------------------------------------------------------\n");
      if(build_status==CL_BUILD_SUCCESS)fprintf(stream,"CL_BUILD_SUCCESS\n");
      else if (build_status==CL_BUILD_NONE)fprintf(stream,"CL_BUILD_NONEd\n");
      else if (build_status==CL_BUILD_IN_PROGRESS)fprintf(stream,"CL_BUILD_IN_PROGRESS\n");
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,0,NULL,&binf_size));
      buildinfo = (char *)malloc((binf_size+1)*sizeof(char));
      CL_CHECK_PRN (clGetProgramBuildInfo (oclconfig->prgs[clfile].oclprogram,oclconfig->ocldevice,CL_PROGRAM_BUILD_LOG,binf_size,buildinfo,NULL));
      buildinfo[binf_size] = '\0';
      fprintf(stream,"%s\n",buildinfo);
      free(buildinfo);
      fprintf(stream,"------------------------------------------------------------\n");

    }
  }
return 0;
}

/* Queries device capabilities to figure if it meets the minimum requirement for double precision*/
/* Returns 0 on successful FP64 evaluation and -1 if only FP32 */
/**
 * This implementation uses the definition of OpenCL 1.1 and 1.2 of which bit-field of CL_DEVICE_DOUBLE_FP_CONFIG
 * should be active for a device to have minimum support of double precision.
 *
 * @param oclconfig The OpenCL configuration containing the device to be checked. The fp64 field
 *           is updated with 1 if double precision is possible and 0 otherwise.
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open.
 *          Error messages are always displayed to stderr
 *
 * @return An integer that represends the query status:
 *         0: The device supports an opencl_*_fp64 extension
 *        -1: The device only supports single precision
 */
int ocl_eval_FP64(ocl_config_type *oclconfig, FILE *stream){

  cl_device_fp_config clfp64;

/* From the OpenCL 1.2 documentation:
 * Double precision is an optional feature so the mandated minimum double precision floating-point capability is 0.
 * If double precision is supported by the device, then the minimum double precision floatingpoint capability must be:
 * CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM
 */  
  CL_CHECK_PRN(clGetDeviceInfo(oclconfig->ocldevice,CL_DEVICE_DOUBLE_FP_CONFIG,sizeof(clfp64),&clfp64,NULL));
  if( (CL_FP_FMA & clfp64) && (CL_FP_ROUND_TO_NEAREST & clfp64) && (CL_FP_ROUND_TO_ZERO & clfp64) &&
    (CL_FP_ROUND_TO_INF & clfp64) && (CL_FP_INF_NAN & clfp64) && (CL_FP_DENORM & clfp64)
  ){
    oclconfig->fp64 = 1;
    fprintf(stream,"Device supports double precision \n");
    return 0;
  }
  else{
    oclconfig->fp64 = 0;
    fprintf(stderr,"Device does not support double precision \n");
    return -1;
  }
  return -2;
}

/* Same as above but directly query a device (as not set in ocl_config_type)
 * It is designed to be used while probing for devices so it does not print anything
 * neither it sets the fp64 field
 */
/**
 * This implementation uses the definition of OpenCL 1.1 and 1.2 of which bit-field of CL_DEVICE_DOUBLE_FP_CONFIG
 * should be active for a device to have minimum support of double precision.
 * A device can be directly queried without having been probed first (stored internally in an
 * ocl_config_type data structure).
 *
 * @param cl_device_id The OpenCL internal device id representation
 *
 * @return An integer that represends the query status:
 *         0: The device supports an opencl_*_fp64 extension
 *        -1: The device only supports single precision
 */
int ocl_eval_FP64(cl_device_id devid){

  cl_device_fp_config clfp64;

/* From the OpenCL 1.2 documentation:
 * Double precision is an optional feature so the mandated minimum double precision floating-point capability is 0.
 * If double precision is supported by the device, then the minimum double precision floatingpoint capability must be:
 * CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM
 */
  CL_CHECK_PRN(clGetDeviceInfo(devid,CL_DEVICE_DOUBLE_FP_CONFIG,sizeof(clfp64),&clfp64,NULL));
  if( (CL_FP_FMA & clfp64) && (CL_FP_ROUND_TO_NEAREST & clfp64) && (CL_FP_ROUND_TO_ZERO & clfp64) &&
    (CL_FP_ROUND_TO_INF & clfp64) && (CL_FP_INF_NAN & clfp64) && (CL_FP_DENORM & clfp64)
  )return 0;
  else return -1;
}

/**
 * This function releases OpenCL memory buffers by their reference
 * in oclconfig structure. Errors are ommited
 * 
 * @param oclconfig The OCL toolkbox configuration structure holding the references
 * @param level How many buffers to release. Max is oclconfig.Nbuffers
 */
void ocl_relNbuffers_byref(ocl_config_type *oclconfig,int level){
  //If level=n we need to clean n-1,n-2,...,0
  if(level<=oclconfig->Nbuffers){
    do{
      level--;
      clReleaseMemObject(oclconfig->oclmemref[level]);
      oclconfig->Nbuffers--;
    }while(level);
  }
return;
}

/**
 * This function releases OpenCL kernels by their reference
 * in oclconfig structure. Errors are ommited
 *
 * @param oclconfig The OCL toolkbox configuration structure holding the references
 * @param level How many kernels to release. Max is oclconfig.Nkernels
 */
void ocl_relNkernels_byref(ocl_config_type *oclconfig,int level){
  //If level=n we need to clean n-1,n-2,...,0
  if(level<=oclconfig->Nkernels){
    do{
      level--;
      clReleaseKernel(oclconfig->oclkernels[level]);
      oclconfig->Nkernels--;
    }while(level);
  }
return;
}

/* A simple function to get OpenCL profiler information*/
/**
 * clEvents enabled profiling. CL_QUEUE_PROFILING_ENABLE flag must be set when the
 * OpenCL command queue is created for clEvent profiling to work.
 *
 * @param start A clEvent that was assigned to a profilable OpenCL call and is to be used as
 *          starting point of the profiling.
 * @param stop A clEvent that was assigned to a profilable OpenCL call and is to be used as
 *          ending point of the profiling. If only one call is to be profiled, start and stop
 *          maybe be the same clEvent.
 * @param message Optional string to be appended on the diplayed info
 * @param stream C File stream to direct the output to. Optional argument which defaults
 *          to stdout. If another stream is used, it must be open and ocl_get_profT will report
 *          the profiling information to both stdout and this stream.
 *
 * @return Returns directly a float variable containing the profiling result in milliseconds. 
 */
float ocl_get_profT(cl_event* start, cl_event* stop, const char* message, FILE* stream)
{

  cl_ulong ts,te;
  CL( clWaitForEvents(1,start) );
  CL( clWaitForEvents(1,stop) );
  CL( clGetEventProfilingInfo(*stop,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&te,NULL) );
  CL( clGetEventProfilingInfo(*start,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&ts,NULL) );
  fprintf(stream,"%s: t %f(ms), t %f(s) \n",message,(te-ts)/(1e6),(te-ts)/(1e9));

  return (te-ts)/(1e6f);
}

/**
 * Silent clEvents enabled profiling. CL_QUEUE_PROFILING_ENABLE flag must be set when the
 * OpenCL command queue is created for clEvent profiling to work.
 *
 * @param start A clEvent that was assigned to a profilable OpenCL call and is to be used as
 *          starting point of the profiling.
 * @param stop A clEvent that was assigned to a profilable OpenCL call and is to be used as
 *          ending point of the profiling. If only one call is to be profiled, start and stop
 *          maybe be the same clEvent.
 *
 * @return Returns directly a float variable containing the profiling result in milliseconds.
 */
float ocl_get_profT(cl_event *start, cl_event *stop){

  cl_ulong ts,te;
  float tms=0.0f;
  CL( clWaitForEvents(1,start) );
  CL( clWaitForEvents(1,stop));
  CL( clGetEventProfilingInfo(*stop,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&te,NULL));
  CL( clGetEventProfilingInfo(*start,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&ts,NULL));

  tms = (te-ts)/(1e6f);
  return tms;
}

/**
 * \brief Initialises the data members of an ocl_plat_t struct.
 *
 * Proper initialisation is essential because all the data members
 * are strings whose size may need to be changed via a realloc.
 * A device context must exist for this to work.
 *
 * @param platinfo An ocl_plat_t structure that holds various informations
 *          on the current platform
 * @return Void 
 */
void ocl_platform_info_init(ocl_plat_t &platinfo)
{
  platinfo.name       = NULL;
  platinfo.vendor     = NULL;
  platinfo.version    = NULL;
  platinfo.extensions = NULL;
return;  
}

/**
 * \brief Frees the resources used inside an ocl_plat_t struct.
 *
 * Because the data members get allocated inside ocl tools,
 * ocl_tools also provides the function to free them.
 *
 * @param platinfo An ocl_plat_t structure that holds various informations
 *          on the current platform
 * @return Void
 */
void ocl_platform_info_del(ocl_plat_t &platinfo)
{
  if(platinfo.name)       free (platinfo.name)      ;
  if(platinfo.vendor)     free (platinfo.vendor)    ;
  if(platinfo.version)    free (platinfo.version)   ;
  if(platinfo.extensions) free (platinfo.extensions);
return;
}

/**
 * \brief Initialises the data members of an ocl_dev_t struct.
 *
 * Proper initialisation is essential because all the data members
 * are strings whose size may need to be changed via a realloc.
 * A device context must exist for this to work.
 *
 * @param devinfo An ocl_dev_t structure that holds various informations
 *          on the current platform
 * @return Void
 */
void ocl_device_info_init(ocl_dev_t &devinfo)
{
  devinfo.name           = NULL;
  devinfo.version        = NULL;
  devinfo.driver_version = NULL;
  devinfo.extensions     = NULL;
  devinfo.global_mem     = 0;
return;  
}

/**
 * \brief Frees the resources used inside an ocl_dev_t struct.
 *
 * Because the data members get allocated inside ocl tools,
 * ocl_tools also provides the function to free them.
 *
 * @param devinfo An ocl_dev_t structure that holds various informations
 *          on the current device
 * @return Void
 */
void ocl_device_info_del(ocl_dev_t &devinfo)
{
  if(devinfo.name)free (devinfo.name);
  if(devinfo.version)free (devinfo.version);
  if(devinfo.driver_version)free (devinfo.driver_version);
  if(devinfo.extensions)free (devinfo.extensions);
  devinfo.global_mem = 0;
return;
}

/**
 * \brief Updates an ocl_plat_t struct with the informations of the current platform
 *
 * List of info returned: CL_PLATFORM_NAME
 *                        CL_PLATFORM_VERSION
 *                        CL_PLATFORM_VENDOR
 *                        CL_PLATFORM_EXTENSIONS
 * 
 * @param oclconfig The OCL toolbox configuration structure holding the references
 * @param platinfo An ocl_plat_t structure that holds various informations
 *          on the current platform
 * @return Error code. 0 for success, -1 for OpenCL error and -2 for other error
 */
int ocl_current_platform_info(ocl_config_type *oclconfig)
{
  size_t pinf_size;
  
  CL( clGetPlatformInfo(oclconfig->oclplatform, CL_PLATFORM_NAME, 0 , NULL , &pinf_size) );
  oclconfig->platform_info.name = (char *)realloc(oclconfig->platform_info.name, (pinf_size + 1));
  if(!oclconfig->platform_info.name) {
    ocl_errmsg("Failed to allocate platform.name",__FILE__,__LINE__);
    return -2;
  }
  CL( clGetPlatformInfo(oclconfig->oclplatform, CL_PLATFORM_NAME, pinf_size , oclconfig->platform_info.name , 0) );

  CL( clGetPlatformInfo(oclconfig->oclplatform, CL_PLATFORM_VERSION, 0 , NULL , &pinf_size) );
  oclconfig->platform_info.version = (char *)realloc(oclconfig->platform_info.version, (pinf_size + 1));
  if(!oclconfig->platform_info.version) {
    ocl_errmsg("Failed to allocate platform.version",__FILE__,__LINE__);
    return -2;
  }
  CL( clGetPlatformInfo(oclconfig->oclplatform, CL_PLATFORM_VERSION, pinf_size , oclconfig->platform_info.version , 0) );

  CL( clGetPlatformInfo(oclconfig->oclplatform, CL_PLATFORM_VENDOR, 0 , NULL , &pinf_size) );
  oclconfig->platform_info.vendor = (char *)realloc(oclconfig->platform_info.vendor, (pinf_size + 1));
  if(!oclconfig->platform_info.vendor) {
    ocl_errmsg("Failed to allocate platform.vendor",__FILE__,__LINE__);
    return -2;
  }
  CL( clGetPlatformInfo(oclconfig->oclplatform, CL_PLATFORM_VENDOR, pinf_size , oclconfig->platform_info.vendor , 0) );

  CL( clGetPlatformInfo(oclconfig->oclplatform, CL_PLATFORM_EXTENSIONS, 0 , NULL , &pinf_size) );
  oclconfig->platform_info.extensions = (char *)realloc(oclconfig->platform_info.extensions, (pinf_size + 1));
  if(!oclconfig->platform_info.extensions) {
    ocl_errmsg("Failed to allocate platform.extensions",__FILE__,__LINE__);
    return -2;
  }
  CL( clGetPlatformInfo(oclconfig->oclplatform, CL_PLATFORM_EXTENSIONS, pinf_size , oclconfig->platform_info.extensions , 0) );  
  
  return 0;
}

/**
 * \brief Updates an ocl_dev_t struct with the informations of the current device
 *
 * List of info returned: CL_DEVICE_NAME
 *                        CL_DEVICE_TYPE
 *                        CL_DEVICE_VERSION
 *                        CL_DRIVER_VERSION
 *                        CL_DEVICE_EXTENSIONS
 *                        CL_DEVICE_GLOBAL_MEM_SIZE
 * 
 * @param oclconfig The OCL toolbox configuration structure holding the references
 * @param devinfo An ocl_dev_t structure that holds various informations
 *          on the current device
 * @return Error code. 0 for success, -1 for OpenCL error and -2 for other error
 */
int ocl_current_device_info(ocl_config_type *oclconfig)
{
  size_t pinf_size;
  cl_device_type devtype;

  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DEVICE_NAME, 0 , NULL , &pinf_size) );
  oclconfig->device_info.name = (char *)realloc(oclconfig->device_info.name, (pinf_size + 1));
  if(!oclconfig->device_info.name) {
    ocl_errmsg("Failed to allocate device_info.name",__FILE__,__LINE__);
    return -2;
  }
  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DEVICE_NAME, pinf_size , oclconfig->device_info.name , 0) );

  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DEVICE_TYPE, sizeof(cl_device_type) , &devtype , 0) );
  if(devtype == CL_DEVICE_TYPE_GPU) strcpy(oclconfig->device_info.type,"GPU");
  else if (devtype == CL_DEVICE_TYPE_CPU) strcpy(oclconfig->device_info.type,"CPU");
  else if (devtype == CL_DEVICE_TYPE_ACCELERATOR) strcpy(oclconfig->device_info.type,"ACC");
  else strcpy(oclconfig->device_info.type,"DEF");

  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DEVICE_VERSION, 0 , NULL , &pinf_size) );
  oclconfig->device_info.version = (char *)realloc(oclconfig->device_info.version, (pinf_size + 1));
  if(!oclconfig->device_info.version) {
    ocl_errmsg("Failed to allocate device_info.version",__FILE__,__LINE__);
    return -2;
  }
  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DEVICE_VERSION, pinf_size , oclconfig->device_info.version , 0) );

  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DRIVER_VERSION, 0 , NULL , &pinf_size) );
  oclconfig->device_info.driver_version = (char *)realloc(oclconfig->device_info.driver_version, (pinf_size + 1));
  if(!oclconfig->device_info.driver_version) {
    ocl_errmsg("Failed to allocate device_info.driver_version",__FILE__,__LINE__);
    return -2;
  }
  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DRIVER_VERSION, pinf_size , oclconfig->device_info.driver_version , 0) );    

  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DEVICE_EXTENSIONS, 0 , NULL , &pinf_size) );
  oclconfig->device_info.extensions = (char *)realloc(oclconfig->device_info.extensions, (pinf_size + 1));
  if(!oclconfig->device_info.extensions) {
    ocl_errmsg("Failed to allocate device_info.extensions",__FILE__,__LINE__);
    return -2;
  }
  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DEVICE_EXTENSIONS, pinf_size , oclconfig->device_info.extensions , 0) );

  CL( clGetDeviceInfo(oclconfig->ocldevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong) , &(oclconfig->device_info.global_mem) , 0) );
  
  return 0;
}

/**
 * For readability and ease of use, OCL toolbox allows the use of simplified strings on it's interface
 * to describe a device. Example: "GPU" will be converted to CL_DEVICE_TYPE_GPU.
 * Acceptable values are:
 *   "GPU", "gpu", "CPU", "cpu", "ACC", "acc", "ALL", "all", "DEF", "def", Null
 *
 * @param devicetype The input string to convert to an opencl internal device type representation
 * @param ocldevtype The OpenCL internal representation of a device type
 * @return Returns 0 on success and -2 on failure to find a suitable representation
 */
int ocl_string_to_cldevtype(const char *device_type, cl_device_type &ocldevtype){

  if(!device_type){
    ocldevtype = CL_DEVICE_TYPE_DEFAULT;
  } else{
    //OpenCL defines the following macros for devicetypes: CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_ACCELERATOR,
    // CL_DEVICE_TYPE_ALL,CL_DEVICE_TYPE_DEFAULT
    if(strcmp(device_type,"GPU")==0 || strcmp(device_type,"gpu")==0 ) ocldevtype=CL_DEVICE_TYPE_GPU;
    else if(strcmp(device_type,"CPU")==0 || strcmp(device_type,"cpu")==0 ) ocldevtype=CL_DEVICE_TYPE_CPU;
    else if(strcmp(device_type,"ACC")==0 || strcmp(device_type,"acc")==0)
      ocldevtype=CL_DEVICE_TYPE_ACCELERATOR; //CELL Processor or FPGAs
    else if(strcmp(device_type,"ALL")==0 || strcmp(device_type,"all")==0) ocldevtype=CL_DEVICE_TYPE_ALL;
    else if(strcmp(device_type,"DEF")==0 || strcmp(device_type,"def")==0) ocldevtype=CL_DEVICE_TYPE_DEFAULT;
    else {
      fprintf(stderr,"Failed to recognize device type '%s'. Occured on %s:%d\n",device_type,__FILE__,__LINE__);
      return -2;
    }
  }
  return 0;
};

/* This function get OpenCL error codes and returns the appropriate string with the error name. It is
    REQUIRED by the error handling macros*/
/**
 * @param err cl_int OpenCL error code to translate to the name of it's macro
 *
 * @return A string containing the name of the macro that resembles the cl_int err var.
 */
/*inline*/ const char *ocl_perrc(cl_int err){

/* Error Codes */
if(err==CL_SUCCESS) return "CL_SUCCESS" ;
if(err==CL_DEVICE_NOT_FOUND) return "CL_DEVICE_NOT_FOUND" ;
if(err==CL_DEVICE_NOT_AVAILABLE) return "CL_DEVICE_NOT_AVAILABLE";
if(err==CL_COMPILER_NOT_AVAILABLE) return "CL_COMPILER_NOT_AVAILABLE";
if(err==CL_MEM_OBJECT_ALLOCATION_FAILURE) return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
if(err==CL_OUT_OF_RESOURCES) return "CL_OUT_OF_RESOURCES";
if(err==CL_OUT_OF_HOST_MEMORY) return "CL_OUT_OF_HOST_MEMORY";
if(err==CL_PROFILING_INFO_NOT_AVAILABLE) return "CL_PROFILING_INFO_NOT_AVAILABLE";
if(err==CL_MEM_COPY_OVERLAP) return "CL_MEM_COPY_OVERLAP";
if(err==CL_IMAGE_FORMAT_MISMATCH) return "CL_IMAGE_FORMAT_MISMATCH";
if(err==CL_IMAGE_FORMAT_NOT_SUPPORTED) return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
if(err==CL_BUILD_PROGRAM_FAILURE) return "CL_BUILD_PROGRAM_FAILURE";
if(err==CL_MAP_FAILURE) return "CL_MAP_FAILURE";
if(err==CL_MISALIGNED_SUB_BUFFER_OFFSET) return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
if(err==CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";

if(err==CL_INVALID_VALUE) return "CL_INVALID_VALUE";
if(err==CL_INVALID_DEVICE_TYPE) return "CL_INVALID_DEVICE_TYPE";
if(err==CL_INVALID_PLATFORM) return "CL_INVALID_PLATFORM";
if(err==CL_INVALID_DEVICE) return "CL_INVALID_DEVICE";
if(err==CL_INVALID_CONTEXT) return "CL_INVALID_CONTEXT";
if(err==CL_INVALID_QUEUE_PROPERTIES) return "CL_INVALID_QUEUE_PROPERTIES";
if(err==CL_INVALID_COMMAND_QUEUE) return "CL_INVALID_COMMAND_QUEUE";
if(err==CL_INVALID_HOST_PTR) return "CL_INVALID_HOST_PTR";
if(err==CL_INVALID_MEM_OBJECT) return "CL_INVALID_MEM_OBJECT";
if(err==CL_INVALID_IMAGE_FORMAT_DESCRIPTOR) return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
if(err==CL_INVALID_IMAGE_SIZE) return "CL_INVALID_IMAGE_SIZE";
if(err==CL_INVALID_SAMPLER) return "CL_INVALID_SAMPLER";
if(err==CL_INVALID_BINARY) return "CL_INVALID_BINARY";
if(err==CL_INVALID_BUILD_OPTIONS) return "CL_INVALID_BUILD_OPTIONS";
if(err==CL_INVALID_PROGRAM) return "CL_INVALID_PROGRAM";
if(err==CL_INVALID_PROGRAM_EXECUTABLE) return "CL_INVALID_PROGRAM_EXECUTABLE";
if(err==CL_INVALID_KERNEL_NAME) return "CL_INVALID_KERNEL_NAME";
if(err==CL_INVALID_KERNEL_DEFINITION) return "CL_INVALID_KERNEL_DEFINITION";
if(err==CL_INVALID_KERNEL) return "CL_INVALID_KERNEL";
if(err==CL_INVALID_ARG_INDEX) return "CL_INVALID_ARG_INDEX";
if(err==CL_INVALID_ARG_VALUE) return "CL_INVALID_ARG_VALUE";
if(err==CL_INVALID_ARG_SIZE) return "CL_INVALID_ARG_SIZE";
if(err==CL_INVALID_KERNEL_ARGS) return "CL_INVALID_KERNEL_ARGS";
if(err==CL_INVALID_WORK_DIMENSION) return "CL_INVALID_WORK_DIMENSION";
if(err==CL_INVALID_WORK_GROUP_SIZE) return "CL_INVALID_WORK_GROUP_SIZE";
if(err==CL_INVALID_WORK_ITEM_SIZE) return "CL_INVALID_WORK_ITEM_SIZE";
if(err==CL_INVALID_GLOBAL_OFFSET) return "CL_INVALID_GLOBAL_OFFSET";
if(err==CL_INVALID_EVENT_WAIT_LIST) return "CL_INVALID_EVENT_WAIT_LIST";
if(err==CL_INVALID_EVENT) return "CL_INVALID_EVENT";
if(err==CL_INVALID_OPERATION) return "CL_INVALID_OPERATION";
if(err==CL_INVALID_GL_OBJECT) return "CL_INVALID_GL_OBJECT";
if(err==CL_INVALID_BUFFER_SIZE) return "CL_INVALID_BUFFER_SIZE";
if(err==CL_INVALID_MIP_LEVEL) return "CL_INVALID_MIP_LEVEL";
if(err==CL_INVALID_GLOBAL_WORK_SIZE) return "CL_INVALID_GLOBAL_WORK_SIZE";
//this is special. This error is on the OpenCL 1.1 header not 1.0 (Nvidia).
//we keep it a number to avoid missing definition
if(err==-64) return "CL_INVALID_PROPERTY";

return "Unknown Error";
}

/* Opencl error function. Some Opencl functions allow pfn_notify to report errors, by passing it as pointer.
      Consult the OpenCL reference card for these functions. */
void __call_compat pfn_notify(const char *errinfo)
{
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
  return;
}

/* Basic function to handle error messages */
void ocl_errmsg(const char *userstring, const char *file, const int line){
  fprintf(stderr,"ocl_errmsg: %s (%s:%d)\n",userstring,file,line);
  return;
}


#ifdef CL_HAS_NAMED_VECTOR_FIELDS

 void make_int2(int x,int y, cl_int2 &conv){

  conv.x = x;
  conv.y = y;

return;
};

 cl_int2 make_int2(int x,int y){

  cl_int2 conv;
  conv.x = x;
  conv.y = y;

return conv;
}

 void make_uint2(unsigned int x,unsigned int y, cl_uint2 &conv){

  conv.x = x;
  conv.y = y;

return;
};

cl_uint2 make_uint2(unsigned int x,unsigned int y){

  cl_uint2 conv;
  conv.x = x;
  conv.y = y;

return conv;
};

void make_float2(float x,float y, cl_float2 &conv){
  
  conv.x = x;
  conv.y = y;

return;  
};

 cl_float2 make_float2(float x,float y){
   
  cl_float2 conv;
  conv.x = x;
  conv.y = y;

return conv;  
};
 void make_double2(double x,double y, cl_double2 &conv){

  conv.x = x;
  conv.y = y;

return;
};

 cl_double2 make_double2(double x,double y){

  cl_double2 conv;
  conv.x = x;
  conv.y = y;

return conv;
};

 void make_uint4(unsigned int x,unsigned int y,unsigned int z,unsigned int w,cl_uint4 &conv){

  conv.x = x;
  conv.y = y;
  conv.z = z;
  conv.w = w;

return;
};

 cl_uint4 make_uint4(unsigned int x,unsigned int y,unsigned int z,unsigned int w){

  cl_uint4 conv;
  conv.x = x;
  conv.y = y;
  conv.z = z;
  conv.w = w;

return conv;
};
#endif
