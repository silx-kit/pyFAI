/**
 * \file
 * \brief  OpenCL tools API generic structures
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


/* Header for OpenCL Datatypes */
#ifndef OCLTOOLS_STRUCTS_H
#define OCLTOOLS_STRUCTS_H

#include <CL/opencl.h>
#include "ocl_clogger_ckerr.h"
#include "cLogger/cLogger.h"

/**
 * \brief Maximum number of OpenCL platforms to scan
 */
#define OCL_MAX_PLATFORMS 10

/**
 * \brief Maximum number of OpenCL devices-per platform- to scan
 */
#define OCL_MAX_DEVICES 10

/**
 * \brief OpenCL tools cl_program structure
 * 
 * Substruct of ocl_configuration_parameters.
 * It is used when multiple cl programs must be
 * built, i.e. from multiple OpenCL sources.
 */
typedef struct{

  cl_program        oclprogram;
  size_t            *kernelstring_lens;
  char              **kernelstrings;

}ocl_program_type;

/**
 * \brief OpenCL tools platform information struct
 *
 * It can be passed to ocl_platform_info to
 * retrieve and save platform information
 * for the currect context
 */
typedef struct
{
  char *name;
  char *vendor;
  char *version;
  char *extensions;
}ocl_plat_t;

/**
 * \brief OpenCL tools platform information struct
 *
 * It can be passed to ocl_device_info to
 * retrieve and save device information
 * for the currect context
 */
typedef struct
{
  char *name;
  char type[4];
  char *version;
  char *driver_version;
  char *extensions;
  unsigned long global_mem;
}ocl_dev_t;

/**
 * \brief OpenCL tools active device information struct
 *
 * Used by OCL tools to hold all the information
 * of the platform, device and pair on the
 * current context
 */
typedef struct
{
  int platformid;
  int deviceid;
  ocl_plat_t platform_info;
  ocl_dev_t  device_info;
}ocl_info_t;

/**
 * \brief OpenCL tools configuration parameters
 * 
 * OpenCL configuration structure.
 * This version supports single device, but multiple
 *   memory buffers, kernels and sources.
 */
typedef struct ocl_configuration_parameters{
  cl_context        oclcontext;
  cl_device_id      ocldevice;
  cl_platform_id    oclplatform;
  cl_command_queue  oclcmdqueue;
  cl_mem            *oclmemref;

//Active device and platform info
//   int devid;
//   int platfid;
//   ocl_plat_t        platform_info;
//   ocl_dev_t         device_info;

  ocl_info_t        active_dev_info;
  
  //If single .cl file:
  cl_program        oclprogram;
  size_t            *kernelstring_lens;
  char              **kernelstrings;

  //if multiple .cl files:
  ocl_program_type  *prgs;
  int               nprgs;

  char              compiler_options[1000];
  cl_kernel         *oclkernels;
  int               fp64;
  size_t            work_dim[3];
  size_t            thread_dim[3];
  cl_event          t_s[20];
  cl_int            event_counter;
  cl_ulong          dev_mem;
  cl_int            Nbuffers;
  cl_int            Nkernels;

  //Logging
  logger_t          *hLog; //Keep track of logger
  int               external_cLogger; //If external, do not touch the handle at finish
}ocl_config_type;

#endif