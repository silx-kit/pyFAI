/*##########################################################################
#
# Copyright (C) 2019 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
#ifndef __PYFAI_OPENMP_H__
#define __PYFAI_OPENMP_H__

#ifdef _OPENMP
#include "omp.h"

static const int COMPILED_WITH_OPENMP = 1;

#else

static const int COMPILED_WITH_OPENMP = 0;

#define omp_set_num_threads(a)
#define omp_get_num_threads() (1)
#define omp_get_max_threads()(1)
#define omp_get_thread_num() (0)
#define omp_get_num_procs() (1)

#endif /*_OPENMP*/

#endif /*__PYFAI_OPENMP_H__*/
