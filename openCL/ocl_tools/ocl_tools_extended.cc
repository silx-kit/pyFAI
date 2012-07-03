/**
 * \file
 * \brief  OpenCL tools API extended functions
 *
 *  OpenCL tools for device probe, selection, deletion, error notification
 *  and vector type conversion. This source is the low-level layer of our
 *  OpenCL Toolbox (ocl_init_exec.cpp). However, it can be used directly
 *  as an API
 */

/*
 *   Project: OpenCL tools for device probe, selection, deletion, error notification
 *              and vector type conversion.
 *
 *   Copyright (C) 2012 Dimitrios Karkoulis
 *
 *   Principal authors: D. Karkoulis (dimitris.karkoulis@gmail.com)
 *   Last revision: 02/07/2012
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
#include "ocl_tools_extended.h"
#include <assert.h>

/* Queries device capabilities to figure if it meets the minimum requirement for double precision*/
/* Returns 0 on successful FP64 evaluation and -1 if only FP32 */
/**
 * This implementation uses the definition of OpenCL 1.1 and 1.2 of which bit-field of CL_DEVICE_DOUBLE_FP_CONFIG
 * should be active for a device to have minimum support of double precision.
 *
 * @param oclconfig The OpenCL configuration containing the device to be checked. The fp64 field
 *           is updated with 1 if double precision is possible and 0 otherwise.
 * @param hLog handle to a cLogger configuration.
 *
 * @return An integer that represends the query status:
 *         0: The device supports an opencl_*_fp64 extension
 *        -1: The device only supports single precision
 */
int ocl_eval_FP64(ocl_config_type *oclconfig)
{

  cl_device_fp_config clfp64;

/* From the OpenCL 1.2 documentation:
 * Double precision is an optional feature so the mandated minimum double precision floating-point capability is 0.
 * If double precision is supported by the device, then the minimum double precision floatingpoint capability must be:
 * CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM
 */  
  CL_CHECK_PRN(clGetDeviceInfo(oclconfig->ocldevice,CL_DEVICE_DOUBLE_FP_CONFIG,sizeof(clfp64),&clfp64,NULL),
    oclconfig->hLog);

  if( (CL_FP_FMA & clfp64) && (CL_FP_ROUND_TO_NEAREST & clfp64) && (CL_FP_ROUND_TO_ZERO & clfp64) &&
    (CL_FP_ROUND_TO_INF & clfp64) && (CL_FP_INF_NAN & clfp64) && (CL_FP_DENORM & clfp64)
  ){
    oclconfig->fp64 = 1;
    cLog_debug(oclconfig->hLog,"Device supports double precision \n");
    return 0;
  }
  else{
    oclconfig->fp64 = 0;
    cLog_debug(oclconfig->hLog,"Device does not support double precision \n");
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
 * @param hLog handle to a cLogger configuration (In case of error only)
 * 
 * @return An integer that represends the query status:
 *         0: The device supports an opencl_*_fp64 extension
 *        -1: The device only supports single precision
 */
int ocl_eval_FP64(cl_device_id devid, logger_t *hLog){

  cl_device_fp_config clfp64;

/* From the OpenCL 1.2 documentation:
 * Double precision is an optional feature so the mandated minimum double precision floating-point capability is 0.
 * If double precision is supported by the device, then the minimum double precision floatingpoint capability must be:
 * CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM
 */
  CL_CHECK_PRN(clGetDeviceInfo(devid,CL_DEVICE_DOUBLE_FP_CONFIG,sizeof(clfp64),&clfp64,NULL), hLog);
  if( (CL_FP_FMA & clfp64) && (CL_FP_ROUND_TO_NEAREST & clfp64) && (CL_FP_ROUND_TO_ZERO & clfp64) &&
    (CL_FP_ROUND_TO_INF & clfp64) && (CL_FP_INF_NAN & clfp64) && (CL_FP_DENORM & clfp64)
  )return 0;
  else return -1;
}

/**
 * \brief Returns info and pairs for all platforms and their devices
 *
 * List of info returned: pair and info for each platform - device combination
 * This function can be called without an active context.
 *
 * @param Ninfo Pointer to ocl_gen_info_t. Allocated by this function. Holds all the information
 * @return Allocated pointer to Ninfo populated with OpenCL platform and device information
 **/
ocl_gen_info_t *ocl_get_all_device_info(ocl_gen_info_t *Ninfo)
{

  //Clear stalle data
  if(Ninfo)
  {
    ocl_clr_all_device_info(Ninfo);
  }else
  {
    Ninfo = (ocl_gen_info_t *)malloc(sizeof(ocl_gen_info_t));
    assert(Ninfo != NULL);
  }
  cl_platform_id *clplatforms = NULL;
  cl_device_id   *cldevices   = NULL;
  
  unsigned int num_platforms = 0;
  unsigned int num_devices   = 0;
  
  clGetPlatformIDs(NULL,NULL,&num_platforms);
  
  Ninfo->Nplatforms = num_platforms;
  
  if(num_platforms >0)
  {
    clplatforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id) );
    assert (clplatforms != NULL);

    clGetPlatformIDs(num_platforms,clplatforms,NULL);    

    Ninfo->platform = (ocl_gen_dev_info_t *)malloc(num_platforms * sizeof(ocl_gen_dev_info_t));
    assert ( Ninfo->platform != NULL );

    Ninfo->platform_ids = (unsigned int *)malloc(num_platforms * sizeof(unsigned int));
    assert( Ninfo->platform_ids != NULL );

    for(unsigned int iplat = 0; iplat < num_platforms; iplat++)
    {
      Ninfo->platform_ids[iplat] = iplat;
      ocl_platform_info_init( Ninfo->platform[iplat].platform_info );
      ocl_current_platform_info( &clplatforms[iplat], &Ninfo->platform[iplat].platform_info, NULL);
      clGetDeviceIDs(clplatforms[iplat],CL_DEVICE_TYPE_ALL,NULL,NULL,&num_devices);
      Ninfo->platform[iplat].Ndevices = num_devices;
      if(num_devices > 0)
      {
        cldevices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id) );
        assert( cldevices != NULL );

        clGetDeviceIDs(clplatforms[iplat],CL_DEVICE_TYPE_ALL,num_devices,cldevices,NULL);

        Ninfo->platform[iplat].device_info = (ocl_dev_t *)malloc(num_devices * sizeof(ocl_dev_t) );
        assert( Ninfo->platform[iplat].device_info != NULL );

        Ninfo->platform[iplat].device_ids = (unsigned int *)malloc(num_devices * sizeof(unsigned int));   
        assert( Ninfo->platform[iplat].device_ids != NULL );

        for( unsigned int idev = 0 ; idev < num_devices; idev++)
        {
          Ninfo->platform[iplat].device_ids[idev] = idev;
          ocl_device_info_init(Ninfo->platform[iplat].device_info[idev]);
          ocl_current_device_info( &cldevices[idev], &Ninfo->platform[iplat].device_info[idev], NULL);
        }
        free(cldevices);
      }else
      {
        Ninfo->platform[iplat].device_info = NULL;
        Ninfo->platform[iplat].device_ids  = NULL;
      }
    } //for(unsigned int iplat = 0; iplat < num_platforms; iplat++)
    free(clplatforms);
  }else
  {
    Ninfo->platform     = NULL;
    Ninfo->platform_ids = NULL;
  }

  return Ninfo;
}

void ocl_clr_all_device_info(ocl_gen_info_t *Ninfo)
{
  if(Ninfo)
  {
    if( Ninfo->platform_ids )
    {
      free(Ninfo->platform_ids);
      Ninfo->platform_ids = NULL;
    }
    for(unsigned int iplat = 0; iplat < Ninfo->Nplatforms; iplat++)
    {
      ocl_platform_info_del(Ninfo->platform[iplat].platform_info);

      if( Ninfo->platform[iplat].device_ids )
      {
        free(Ninfo->platform[iplat].device_ids);
        Ninfo->platform[iplat].device_ids = NULL;
      }
      for(unsigned int idev = 0; idev < Ninfo->platform[iplat].Ndevices; idev++)
      {
        ocl_device_info_del(Ninfo->platform[iplat].device_info[idev]);
      }

      if( Ninfo->platform[iplat].device_info )
      {
        free(Ninfo->platform[iplat].device_info);
        Ninfo->platform[iplat].device_info = NULL;
      }
      Ninfo->platform[iplat].Ndevices = 0;
    }

    if( Ninfo->platform )
    {
      free(Ninfo->platform);
      Ninfo->platform = NULL;
    }
    Ninfo->Nplatforms = 0;
  }
return;
}
