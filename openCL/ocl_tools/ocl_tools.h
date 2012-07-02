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
 *              and vector type conversion.
 *
 *   Copyright (C) 2011 - 2012 European Synchrotron Radiation Facility
 *                                 Grenoble, France
 *
 *   Principal authors: D. Karkoulis (karkouli@esrf.fr)
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


/* Header for OpenCL Utilities */
#ifndef OCLTOOLS_H
#define OCLTOOLS_H

#include <CL/opencl.h>
#include "ocl_tools_datatypes.h"
#include "ocl_tools_extended.h"
#include "ocl_clogger_ckerr.h"
#include "cLogger/cLogger.h"

/*This is required for OpenCL callbacks in windows*/
#ifdef _WIN32
	#ifndef _CRT_SECURE_NO_WARNINGS
		#define _CRT_SECURE_NO_WARNINGS
	#endif
	#pragma warning(disable : 4996)
#endif

/* All production functions return 0 on success, -1 on OpenCL error and -2 on other errors.
    when an error is encountered internally, it will fallback.
    It is important for the user to decide how to continue.*/

/**
 * \brief Initialises the internals of an ocl_config_type and sets logger to defaults
 */
void ocl_tools_initialise(ocl_config_type *oclconfig);

/**
 * \brief Initialises the internals of an ocl_config_type. Logger settings depend on input
 */
void ocl_tools_initialise(ocl_config_type *oclconfig,logger_t *hLogIN);

/**
 * \brief Initialises the internals of an ocl_config_type. Logger settings are predefined
 */
logger_t *ocl_tools_initialise(ocl_config_type *oclconfig, FILE *stream, const char *fname, 
                          int severity, enum_LOGTYPE type, enum_LOGDEPTH depth, int perf, 
                          int timestamps);

/**
 * \brief Deallocations inside ocl_config_type
 */
void ocl_tools_destroy(ocl_config_type *oclconfig);

/**
 * \brief Simple check all platforms and devices and print information
 */
int ocl_check_platforms(logger_t *hLog);

/**
 * \brief Simple check for a "device_type" device. Returns the first occurance only
 */
int ocl_find_devicetype(cl_device_type device_type, cl_platform_id &platform, cl_device_id &devid, logger_t *hLog);

/**
 * \brief Simple check for a "device_type" device. Returns the first occurance that supports double precision only
 */
int ocl_find_devicetype_FP64(cl_device_type device_type, cl_platform_id& platform, cl_device_id& devid, logger_t *hLog);

/**
 * \brief Probes OpenCL platforms & devices of a given cl_device_type. Keeps the selected device in oclconfig
 */
int ocl_probe(ocl_config_type *oclconfig,cl_device_type ocldevtype, int usefp64);

/**
 * \brief Probes OpenCL device of a specific cl_device_type, platform and device number. Keeps the selected device in oclconfig
 */
int ocl_probe(ocl_config_type *oclconfig,cl_device_type ocldevtype,int preset_platform,int preset_device, int usefp64);

/**
 * \brief Probes OpenCL device of a specific cl_platform_id and cl_device_id. Keeps the selected device in oclconfig
 */
int ocl_probe(ocl_config_type *oclconfig,cl_platform_id platform,cl_device_id device, int usefp64);

/**
 * \brief Create an OpenCL context by device type
 */
/* Needs a string with the type of device: GPU,CPU,ACC,ALL,DEF. Runs ocl_probe and creates the context,
    adding it on the appropriate ocl_config_type field*/
int ocl_init_context(ocl_config_type *oclconfig,const char *device_type, int usefp64);

/**
 * \brief Create an OpenCL context by device type, platform and device number
 */
int ocl_init_context(ocl_config_type *oclconfig,const char *device_type,int preset_platform,int devid, int usefp64);

/**
 * \brief Create an OpenCL context by cl_platform_id and cl_device_id
 */
int ocl_init_context(ocl_config_type *oclconfig,cl_platform_id platform,cl_device_id device, int usefp64);

/**
 * \brief Destroy an OpenCL context
 */
int ocl_destroy_context(cl_context oclcontext, logger_t *hLog);

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
int ocl_compiler(ocl_config_type *oclconfig,const char *kernelfilename,int BLOCK_SIZE,const char *optional=NULL);

/**
 * \brief OpenCL sources compiler for multiple .cl files
 */
int ocl_compiler(ocl_config_type *oclconfig,const char **clList,int clNum,int BLOCK_SIZE,const char *optional);

/**
 * \brief OpenCL sources compiler for cl string
 */
int ocl_compiler(ocl_config_type *oclconfig,unsigned char **clList,unsigned int *clLen,int clNum,int BLOCK_SIZE,const char *optional);

/**
 * \brief Profiler function based on OpenCL events, for display
 */
/* A simple function to get OpenCL profiler information*/
float ocl_get_profT(cl_event *start, cl_event *stop, const char *message, logger_t *hLog);

/**
 * \brief Profiler function based on OpenCL events, only return value
 */
float ocl_get_profT(cl_event *start, cl_event *stop, logger_t *hLog);

/**
 * \brief Convert simple string to cl_device_type
 */
int ocl_string_to_cldevtype(const char *devicetype, cl_device_type &ocldevtype, logger_t *hLog);

/**
 * \brief Initialise an ocl_plat_t struct
 */
void ocl_platform_info_init(ocl_plat_t &platinfo);

/**
 * \brief Release the memory held by the strings inside an ocl_plat_t struct
 */
void ocl_platform_info_del(ocl_plat_t &platinfo);

/**
 * \brief Initialise an ocl_dev_t struct
 */
void ocl_device_info_init(ocl_dev_t &devinfo);

/**
 * \brief Release the memory held by the strings inside an ocl_dev_t struct
 */
void ocl_device_info_del(ocl_dev_t &devinfo);

/**
 * \brief Populates an ocl_plat_t struct
 */
int ocl_current_platform_info(cl_platform_id *oclplatform, ocl_plat_t *platform_info, logger_t *hLog);

/**
 * \brief Populates an ocl_dev_t struct
 */
int ocl_current_device_info(cl_device_id *ocldevice, ocl_dev_t *device_info, logger_t *hLog);

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
void CL_CALLBACK pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data);

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