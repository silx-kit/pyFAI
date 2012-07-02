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

/* Header for OpenCL Extended Utilities */
#ifndef OCLTOOLS_EXT_H
#define OCLTOOLS_EXT_H

#include <CL/opencl.h>
#include "ocl_tools_datatypes.h"
#include "ocl_tools.h"
#include "ocl_clogger_ckerr.h"
#include "cLogger/cLogger.h"

/*This is required for OpenCL callbacks in windows*/
#ifdef _WIN32
	#ifndef _CRT_SECURE_NO_WARNINGS
		#define _CRT_SECURE_NO_WARNINGS
	#endif
	#pragma warning(disable : 4996)
#endif

/**
 *
 * TODO
 */
typedef struct
{
  unsigned int platformid;
  unsigned int deviceid;
  char         devtype[4];
}ocl_pair;

/**
 *
 * TODO
 */
typedef struct
{
  cl_platform_id platformid;
  cl_device_id   deviceid;
  cl_device_type devtype;
}ocl_pair_opencl;

/**
 * \brief OpenCL tools generic device information struct
 *
 * Used to provide information for all the OpenCL devices
 */
typedef struct
{
  unsigned int Ndevices;
  unsigned int *device_ids;
  ocl_plat_t platform_info;
  ocl_dev_t  *device_info;
}ocl_gen_dev_info_t;

/**
 * \brief OpenCL tools generic platform information struct
 *
 * Chains information for each pair
 */
typedef struct
{
  unsigned int Nplatforms;
  unsigned int *platform_ids;
  unsigned int **ids; //TODO
  ocl_gen_dev_info_t *platform;

}ocl_gen_info_t;



/**
 * \brief Queries the fp64 capability of an OpenCL device that has been selected by ocl_probe
 */
/* Queries device capabilities to figure if it meets the minimum requirement for double precision*/
/* Returns 0 on successful FP64 evaluation and -1 if only FP32 */
int ocl_eval_FP64(ocl_config_type *oclconfig);

/**
 * \brief Queries the fp64 capability of an OpenCL device directly via the cl_device_id
 */
/* Same as above but directly query a device (as not set in ocl_config_type)
 * It is designed to be used while probing for devices so it does not print anything
 * neither it sets the fp64 field */
int ocl_eval_FP64(cl_device_id devid, logger_t *hLog);

/**
 * \brief Returns info and pairs for all platforms and their devices
 **/
ocl_gen_info_t *ocl_get_all_device_info(ocl_gen_info_t *Ninfo);

/**
 * \brief Releases all memory in Ninfo
 **/
void ocl_clr_all_device_info(ocl_gen_info_t *Ninfo);

#endif