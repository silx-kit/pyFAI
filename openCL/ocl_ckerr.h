/*
 *   Project: Macros for OpenCL API error handling. Requires ocl_tools
 *
 *   Copyright (C) 2011 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: D. Karkoulis (karkouli@esrf.fr)
 *   Last revision: 27/05/2011
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

#ifdef _WIN32
#define typeof(_expr) typeid(_expr)
#endif

#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL: '%s' returned %d!\n", #_expr, (int)_err);   \
     exit(1);                                                                   \
   } while (0)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL: '%s' returned %d!\n", #_expr, (int)_err); \
       exit(1);                                                                 \
     }                                                                          \
     _ret;                                                                      \
   })

#define CL_CHECK_PR(_expr)                                                          \
   do {                                                                             \
     cl_int _err = _expr;                                                           \
     if (_err == CL_SUCCESS)                                                        \
       break;                                                                       \
     fprintf(stderr, "OpenCL: '%s:%d' returned %s!\n",__FILE__,__LINE__, ocl_perrc(_err)); \
     exit(1);                                                                       \
   } while (0)

#define CL_CHECK_PRN(_expr)                                                         \
   do {                                                                             \
     cl_int _err = _expr;                                                           \
     if (_err == CL_SUCCESS)                                                        \
       break;                                                                       \
     fprintf(stderr, "OpenCL: '%s:%d' returned %s!\n",__FILE__,__LINE__, ocl_perrc(_err)); \
       break; \
   } while (0)

#define CL_CHECK_PR_RET(_expr)                                                         \
   do {                                                                             \
     cl_int _err = _expr;                                                           \
     if (_err == CL_SUCCESS)                                                        \
       break;                                                                       \
     fprintf(stderr, "OpenCL: '%s:%d' returned %s!\n",__FILE__,__LINE__, ocl_perrc(_err)); \
       return -1; \
   } while (0)

#define CL_CHECK_ERR_PR(_expr)                                                        \
   ({                                                                                 \
     cl_int err = CL_INVALID_VALUE;                                                  \
     typeof(_expr) _ret = _expr;                                                      \
     if (err != CL_SUCCESS) {                                                        \
       fprintf(stderr, "OpenCL: '%s:%d' returned %s!\n",__FILE__,__LINE__, ocl_perrc(err)); \
       exit(1);                                                                       \
     }                                                                                \
     _ret;                                                                            \
   })

#define CL_CHECK_ERR_PRN(_expr)                                                       \
   ({                                                                                 \
     cl_int err = CL_INVALID_VALUE;                                                   \
     typeof(_expr) _ret = _expr;                                                      \
     if (err != CL_SUCCESS) {                                                         \
       fprintf(stderr, "OpenCL: '%s:%d' returned %s!\n",__FILE__,__LINE__, ocl_perrc(err));  \
     }                                                                                \
     _ret;                                                                            \
   })

#define CL_CHECK_ERR_PR_RET(_expr)                                                        \
   ({                                                                                 \
     cl_int err = CL_INVALID_VALUE;                                                  \
     typeof(_expr) _ret = _expr;                                                      \
     if (err != CL_SUCCESS) {                                                        \
       fprintf(stderr, "OpenCL: '%s:%d' returned %s!\n",__FILE__,__LINE__, ocl_perrc(err)); \
       return -1;                                                                       \
     }                                                                                \
     _ret;                                                                            \
   })
