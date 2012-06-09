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

#ifndef OCL_XRPD2D_H
#define OCL_XRPD2D_H

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include <iostream>
#include <CL/opencl.h>
#include "ocl_ckerr.h"

#include "ocl_tools.h"
#include "ocl_base.hpp"

typedef unsigned long lui;

class ocl_xrpd2D_fullsplit: public ocl{
public:

  ocl_xrpd2D_fullsplit();
  explicit ocl_xrpd2D_fullsplit(const char* fname);
  ~ocl_xrpd2D_fullsplit();

  int getConfiguration(const int Nx,const int Nimage,const int NbinsTth, const int NbinsChi,const bool usefp64=false);
  int configure();
  int loadTth(float *tth, float *dtth, float tth_min,float tth_max);
  int loadChi(float *chi, float *dchi, float chi_min,float chi_max);
  int setSolidAngle(float *SolidAngle);
  int unsetSolidAngle();
  int setMask(int *Mask);
  int unsetMask();
  int setDummyValue(float dummyVal);
  int unsetDummyValue();
  int setTthRange(float lowerBound, float upperBound);
  int unsetTthRange();
  int setChiRange(float lowerBound, float upperBound);
  int unsetChiRange();
  int execute(float *im_inten,float *histogram,float *bins);
  int clean(int preserve_context=0);

private:
  int hasChiLoaded;
  int useChiRange;
  int set_kernel_arguments();
  int allocate_CL_buffers();
};

#endif