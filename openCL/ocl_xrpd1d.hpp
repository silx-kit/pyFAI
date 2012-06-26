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
 *   Last revision: 26/06/2012
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

#ifndef OCL_XRPD1D_H
#define OCL_XRPD1D_H
 
#include <iostream>
#include <CL/opencl.h>

#include "ocl_tools/ocl_tools.h"
#include "ocl_base.hpp"

/**
 *  \brief OpenCL-based 1D full pixelsplit azimuthal integrator for PyFAI.
 *
 * Extends the class ocl which defines the basic behaviour and tools
 * All the methods return an integer error code. -1 for OpenCL error and -2 for other.
 * Such errors will not kill the library, but print on stderr and fallback.
 * It is up to the user to decide how to handle those errors as the library can
 * recover from most OpenCL errors.
 */
class ocl_xrpd1D_fullsplit: public ocl{
public:

  //Default constructor - Prints messages on stdout with highest logging level
  ocl_xrpd1D_fullsplit();

  //cLogger is set to fname with highest logging level
  explicit ocl_xrpd1D_fullsplit(const char* fname, const char *identity=NULL);

  //Complete logging functionality
  explicit ocl_xrpd1D_fullsplit(FILE *stream, const char *fname, int safe, int depth, int perf_time, int timestamp, const char *identity=NULL);
  ~ocl_xrpd1D_fullsplit();


  /* getConfiguration gets the description of the integrations to be performed and keeps an internal copy
   */
  int getConfiguration(const int Nx,const int Nimage,const int Nbins,const bool usefp64=false);

  /* configure is possibly the most crucial method of the class.
   * It is responsible of allocating the required memory and compile the OpenCL kernels
   * based on the configuration of the integration.
   * It also "ties" the OpenCL memory to the kernel arguments.
   *
   * If ANY of the arguments of getConfiguration needs to be changed, configure must
   * be called again for them to take effect
   *
   * kernel_path is the path to the file containing the kernel
   */
  int configure(const char* kernel_path);

  /*
   * Load the 2th arrays along with the min and max value.
   * loadTth maybe be recalled at any time of the execution in order to update
   * the 2th arrays.
   *
   * loadTth is required and must be called at least once after a configure()
   */ 
  int loadTth(float *tth,float *dtth, float tth_min,float tth_max);

  /*
   * Enables SolidAngle correction and uploads the suitable array to the OpenCL device.
   * By default the program will assume no solidangle correction unless setSolidAngle() is called.
   * From then on, all integrations will be corrected via the SolidAngle array.
   *
   * If the SolidAngle array needs to be changes, one may just call setSolidAngle() again
   * with that array
   */
  int setSolidAngle(float *SolidAngle);

  /*
   * Instructs the program to not perform solidangle correction from now on.
   * SolidAngle correction may be turned back on at any point
   *
   */
  int unsetSolidAngle();

  /*
   * Enables the use of a Mask during integration. The Mask can be updated by
   * recalling setMask at any point.
   *
   * The Mask must be a PyFAI Mask
   */
  int setMask(int *Mask);

  /*
   * Disables the use of a Mask from that point. It may be reenabled at any point
   * via setMask
   */
  int unsetMask();

  /*
   * Enables dummy value functionality and uploads the value to the OpenCL device.
   * Image values that are similar to the dummy value are set to 0.
   */
  int setDummyValue(float dummyVal, float deltaDummyVal);

  /*
   * Disable a dummy value. May be reenabled at any time by setDummyValue
   */
  int unsetDummyValue();

  /*
   * Sets the active range to integrate on. By default the range is set to tth_min and tth_max
   * By calling this functions, one may change to different bounds
   */
  int setRange(float lowerBound, float upperBound);

  /*
   * Resets the 2th integration range back to tth_min, tth_max
   */
  int unsetRange();

  /*
   * Take an image, integrate and return the histogram and weights
   * set/unset and loadTth methods have a direct impact on the execute() method.
   * All the rest of the methods will require at least a new configuration via configure()
   */
  int execute(float *im_inten,float *histogram,float *bins);

  /*
   * Free OpenCL related resources.
   * It may be asked to preserve the context created by init or completely clean up OpenCL.
   *
   * Guard/Status flags that are set will be reset. All the Operation flags are also reset
   */
  int clean(int preserve_context=0);
  
protected:  
  virtual int set_kernel_arguments();
  virtual int allocate_CL_buffers();

};


#endif
