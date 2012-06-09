/**
 * \file
 * \brief  OpenCL tools header
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
 *   Last revision: 26/04/2012
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


/* Header for OpenCL Utilities */
#ifndef OCLTOOLS_H
#define OCLTOOLS_H

#include <CL/opencl.h>
#include "ocl_ckerr.h"

/*This is required for OpenCL callbacks in windows*/
#ifdef _WIN32
  #define __call_compat __stdcall
#else
  #define __call_compat 
#endif

/**
 * \brief Maximum number of OpenCL platforms to scan
 */
#define OCL_MAX_PLATFORMS 5

/**
 * \brief Maximum number of OpenCL devices-per platform- to scan
 */
#define OCL_MAX_DEVICES 5

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

}ocl_config_type;

/* All production functions return 0 on success, -1 on OpenCL error and -2 on other errors.
    when an error is encountered internally, it will print the message to stderr and fallback.
    It is important for the user to decide how to continue.*/

/**
 * \brief Simple check all platforms and devices and print information
 */
int ocl_check_platforms(FILE *stream=stdout);

/**
 * \brief Simple check for a "device_type" device. Returns the first occurance only
 */
int ocl_find_devicetype(cl_device_type device_type, cl_platform_id &platform, cl_device_id &devid,FILE *stream=stdout);

/**
 * \brief Simple check for a "device_type" device. Returns the first occurance that supports double precision only
 */
int ocl_find_devicetype_FP64(cl_device_type device_type, cl_platform_id &platform, cl_device_id &devid,FILE *stream=stdout);

/**
 * \brief Probes OpenCL platforms & devices of a given cl_device_type. Keeps the selected device in oclconfig
 */
int ocl_probe(ocl_config_type *oclconfig,cl_device_type ocldevtype, int usefp64, FILE *stream=stdout);

/**
 * \brief Probes OpenCL device of a specific cl_device_type, platform and device number. Keeps the selected device in oclconfig
 */
int ocl_probe(ocl_config_type *oclconfig,cl_device_type ocldevtype,int preset_platform,int preset_device, int usefp64,
              FILE *stream=stdout);

/**
 * \brief Probes OpenCL device of a specific cl_platform_id and cl_device_id. Keeps the selected device in oclconfig
 */
int ocl_probe(ocl_config_type *oclconfig,cl_platform_id platform,cl_device_id device, int usefp64,
              FILE *stream=stdout);

/**
 * \brief Create an OpenCL context by device type
 */
/* Needs a string with the type of device: GPU,CPU,ACC,ALL,DEF. Runs ocl_probe and creates the context,
    adding it on the appropriate ocl_config_type field*/
int ocl_init_context(ocl_config_type *oclconfig,const char *device_type, int usefp64, FILE *stream=stdout);

/**
 * \brief Create an OpenCL context by device type, platform and device number
 */
int ocl_init_context(ocl_config_type *oclconfig,const char *device_type,int preset_platform,int devid, int usefp64,
                     FILE *stream=stdout);

/**
 * \brief Create an OpenCL context by cl_platform_id and cl_device_id
 */
int ocl_init_context(ocl_config_type *oclconfig,cl_platform_id platform,cl_device_id device, int usefp64,
                     FILE *stream=stdout);

/**
 * \brief Destroy an OpenCL context
 */
int ocl_destroy_context(cl_context oclcontext,FILE *stream=stdout);

/**
 * \brief deprecated eval_Fp64. Use ocl_eval_FP64 instead
 */
/*WARNING this is a deprecated function as this way may not always fail under different OpenCL compilers*/
/* Use the new ocl_eval_FP64 instead*/
/* Used fixed minimal kernels to check if FP64 is supported. Returns 0 on successful FP64 evaluation and -1 if only FP32. Exits on failure*/
int _deprec_ocl_eval_FP64(ocl_config_type *oclconfig,int *eval_res,FILE *stream=stdout);

/**
 * \brief Queries the fp64 capability of an OpenCL device that has been selected by ocl_probe
 */
/* Queries device capabilities to figure if it meets the minimum requirement for double precision*/
/* Returns 0 on successful FP64 evaluation and -1 if only FP32 */
int ocl_eval_FP64(ocl_config_type *oclconfig,int *eval_res, FILE *stream);

/**
 * \brief Queries the fp64 capability of an OpenCL device directly via the cl_device_id
 */
/* Same as above but directly query a device (as not set in ocl_config_type)
 * It is designed to be used while probing for devices so it does not print anything
 * neither it sets the fp64 field */
int ocl_eval_FP64(cl_device_id devid);

/**
 * \brief Release N buffers referenced by oclconfig
 */
void ocl_relNbuffers_byref(ocl_config_type *oclconfig,int level);

/**
 * \brief Release N kernels referenced by oclconfig
 */
void ocl_relNkernels_byref(ocl_config_type *oclconfig,int level);

/**
 * \brief OpenCL sources compiler for a .cl file
 */
/* OpenCL Compiler for dynamic kernel creation. It will always report success or failure of the build.*/
int ocl_compiler(ocl_config_type *oclconfig,const char *kernelfilename,int BLOCK_SIZE,const char *optional=NULL, FILE *stream=stdout);

/**
 * \brief OpenCL sources compiler for multiple .cl files
 */
int ocl_compiler(ocl_config_type *oclconfig,const char **clList,int clNum,int BLOCK_SIZE,const char *optional, FILE *stream);

/**
 * \brief OpenCL sources compiler for cl string
 */
int ocl_compiler(ocl_config_type *oclconfig,unsigned char **clList,unsigned int *clLen,int clNum,int BLOCK_SIZE,const char *optional, FILE *stream);

/**
 * \brief Profiler function based on OpenCL events, for display
 */
/* A simple function to get OpenCL profiler information*/
float ocl_get_profT(cl_event *start, cl_event *stop, const char *message,FILE *stream=stdout);

/**
 * \brief Profiler function based on OpenCL events, only return value
 */
float ocl_get_profT(cl_event *start, cl_event *stop);

/**
 * \brief Convert simple string to cl_device_type
 */
int ocl_string_to_cldevtype(const char *devicetype, cl_device_type &ocldevtype);

/**
 * \brief Translate error code to error message
 */
/* This function get OpenCL error codes and returns the appropriate string with the error name. It is
    REQUIRED by the error handling macros*/
/*inline*/ const char *ocl_perrc(cl_int err);

/**
 * \brief OpenCL callback function
 */
/* Opencl error function. Some Opencl functions allow pfn_notify to report errors, by passing it as pointer.
      Consult the OpenCL reference card for these functions. */
void __call_compat pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data);

/* Basic function to handle error messages */
void ocl_errmsg(const char *userstring, const char *file, const int line);


/* Vector creation functions. OpenCL allows reintepretation and conversion, but no direct clean vector initialisation like CUDA*/
#ifdef CL_HAS_NAMED_VECTOR_FIELDS

void make_int2(int x,int y, cl_int2 &conv);
cl_int2 make_int2(int x,int y);
void make_uint2(unsigned int x,unsigned int y, cl_uint2 &conv);
cl_uint2 make_uint2(unsigned int x,unsigned int y);
void make_float2(float x,float y, cl_float2 &conv);
cl_float2 make_float2(float x,float y);
void make_double2(double x,double y, cl_double2 &conv);
cl_double2 make_double2(double x,double y);
void make_uint4(unsigned int x,unsigned int y,unsigned int z,unsigned int w,cl_uint4 &conv);
cl_uint4 make_uint4(unsigned int x,unsigned int y,unsigned int z,unsigned int w);

#endif

#endif