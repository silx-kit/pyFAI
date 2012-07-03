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
 *   Principal authors: D. Karkoulis (dimitris.karkoulis@gmail.com)
 *   Last revision: 03/07/2012
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


/* All the functionsreturn an integer error code. -1 for OpenCL error and -2 for other.
    Errors wont kill the program, but print on stderr and fallback.
    It is up to the user to decide how to handle those errors.*/

/**
 * \file
 * \brief Header for the base functionality of OpenCL XRPD
  */

#ifndef OCL_BASE_H
#define OCL_BASE_H

#ifdef _WIN32
	#ifndef _CRT_SECURE_NO_WARNINGS
		#define _CRT_SECURE_NO_WARNINGS
	#endif
	#pragma warning(disable : 4996)
#endif

#include <iostream>
#include <sstream>
#include <CL/opencl.h>

#include "ocl_tools/ocl_tools.h"
#include "ocl_tools/cLogger/cLogger.h"

/**
 * \brief Holds the integration configuration parameters
 * 
 */
typedef class azimuthal_configuration{

public:
  cl_uint Nimage;
  cl_uint Nx;
  cl_uint Nbins;
	cl_uint Nbinst;
	cl_uint Nbinsc;
	cl_uint usefp64;

}az_argtype;

/**
 * \brief Base class. Provides basic funcionality to the integration classes
 * 
 * Most functionality is  protected and accessible only through the appropriate derived class.
 * In principal, one may only use the ocl class in order to check for OpenCL devices and get information for them
 */
class ocl{
public:

  explicit ocl(FILE *stream, const char *fname, int safe, int depth, int perf_time, int timestamp, const char *identity=NULL);
  explicit ocl(const char *fname, const char *identity=NULL);
  ocl();
  virtual ~ocl();

  /* Changes settings of logger configuration in hLog*/
  void update_logger(FILE *stream, const char *fname, int safe, int depth, int perf_time, int timestamp);

/*
 * Initial configuration: Choose a device and initiate a context. Devicetypes can be GPU,gpu,CPU,cpu,DEF,ACC,ALL.
 * Suggested are GPU,CPU. For each setting to work there must be such an OpenCL device and properly installed.
 * E.g.: If Nvidia driver is installed, GPU will succeed but CPU will fail. The AMD SDK kit is required for CPU via OpenCL.
 */
  virtual int init(const bool useFp64=true);
  virtual int init(const char *devicetype,const bool useFp64=true);
  virtual int init(const char *devicetype,int platformid,int devid,const bool useFp64=true);

/*
 * Frees all the memory alocated by this library, for the device and system
 */
  virtual int clean(int preserve_context=0);

/*
 * Forcibly and recklessly destroy an OpenCL context. Can lead to memory leaks.
 * This method will be removed in the future
 */  
  void kill_context();

/*
 * Prints a list of OpenCL capable devices, their platforms and their ids
 */  
  void show_devices(int ignoreStream=1);

/*
 * Prints some basic information about the device in use
 */  
  void show_device_details(int ignoreStream=1);

/*
 * Returns a structure with information for all the present OpenCL devices
 */  
  void get_all_device_details();

/*
 * Prints some basic information about all the OpenCL devices present
 */  
  void show_all_device_details(int ignoreStream=1);

/*
 * Provide help message for interactive environments
 */  
  virtual void help();

/*
 * Resets the internal profiling timers to 0
 */  
  void  reset_time();

/*
 * Returns the internal profiling timer for the kernel executions
 */  
  float get_exec_time();

/*
 * Returns how many integrations have been performed
 */  
  unsigned int get_exec_count();

/*
 * Returns the time spent on memory copies
 */
  float get_memCpy_time();

  /*
   * Get the status of the intergator as an integer
   * bit 0: has context
   * bit 1: size are set
   * bit 2: is configured (kernel compiled)
   * bit 3: pos0/delta_pos0 arrays are loaded (radial angle)
   * bit 4: pos1/delta_pos1 arrays are loaded (azimuthal angle)
   * bit 5: solid angle correction is set
   * bit 6: mask is set
   * bit 7: use dummy value
   */
  int get_status();

 /*
  *Returns the pair ID (platform.device) of the active device
  */
  void get_contexed_Ids(int &platform, int &device);

 /*
  *Returns the -C++- pair ID (platform.device) of the active device
  */
  std::pair<int,int> get_contexed_Ids()  ;

/**
 * \brief Holds the documentation message
 */  
  char *docstr;

  ocl_plat_t platform_info;
  ocl_dev_t device_info; 
  ocl_gen_info_t *Ninfo;
protected:

  /**
   * \defgroup opers Operation flags
   * @{
   */  
  int usesStdout; //!< Set by constructor
  
  /**@}*/
  
  FILE *stream; //!< Deprecated, replaced by hLog. Set but not used
  
  logger_t hLog; //!< Logger configuration
  const char *exec_identity; //!< Caller name or user defined string (e.g. argv[0] in C/C++)

  /**
   * \defgroup guards Status flags/guards
   * \brief These flags keep track of the configuration stages needed for execution
   *
   * This set of flags are used to protect from calling the methods in a wrong order and prevent memory leaks.
   * This is due to the fact that this library is implemented with interactive environments in mind.
   * With the use of these flags the library can give hints in case the wrong method is called
   * and to keep track of OpenCL key-stages. As an example, if the library
   * is configured and the user tries to perform a configuration again without cleaning up,
   * the library will track what relative OpenCL resources are already allocated and release them
   * and then only continue with the new configuration.
   * @{
   */
  int hasActiveContext; //!< Set by init()  
  int isConfigured;     //!< Set by configure()
  int hasQueue;         //!< Set by configure()
  int hasBuffers;       //!< Set by configure()
  int hasProgram;       //!< Set by configure()
  int hasKernels;       //!< Set by configure()
  int hasTthLoaded;     //!< Set by loadTth()
  int hasChiLoaded;     //!< Set by loadChi()
  /**@}*/
  
  /**@}*/
  
  /**
   * \ingroup opers
   * @{
   */
  int useSolidAngle; //!< Set by setSolidAngle(), reset by unsetSolidAngle()
  int useDark;  	 //!< Set by setDark(), reset by unsetDark()
  int useMask;       //!< Set by setMask(), reset by unsetMask()
  int useDummyVal;   //!< Set by setDummyVal(), reset by unsetDummyVal()
  int useTthRange;   //!< Set by setTthRange(), reset by unsetTthRange()
  int useChiRange;   //!< Set by setChiRange(), reset by unsetChiRange()

  /**@}*/
  
  /**
   * \defgroup counters Profiling counters
   * @{
   */
  float execTime_ms;
  unsigned int  execCount;
  float memCpyTime_ms;
  
  /**@}*/
  
  /**
   * \brief Holds the integration configuration
   */
  az_argtype *sgs;

  /**
   * \brief OpenCL configuration for the OpenCL toolbox (ocl_tools)
   *
   * is kept private for now. I.e. user does not have access to GPU memory buffers or the command queue
   */
  ocl_config_type *oclconfig;

  /**
   * \brief Common initialisations for all the constructors
   */
  virtual void ContructorInit();
  
 /**
  * \brief OpenCL memory deallocation
  */
  void clean_clbuffers(int level);
  
 /**
  * \brief OpenCL kernel deallocation
  */
  void clean_clkernels(int level);

/**
 * \brief Creates a documentation string to aid in interactive environments
 *
 */  
  void setDocstring(const char *default_text, const char *filename);

  void promote_device_details();
};

#endif
