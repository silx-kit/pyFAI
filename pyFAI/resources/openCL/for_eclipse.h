/*
 * Project: Azimuthal integration
 *       https://github.com/pyFAI/pyFAI
 *
 * Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
 *
 * Principal author:       Jerome Kieffer (Jerome.Kieffer@ESRF.eu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef __OPENCL_VERSION__
//This is for Eclipse to stop seeing errors everywhere ...
#define __kernel
#define __global
#define __constant
#define __local

typedef struct float2 {
  float x, y;
} float2;
typedef struct float3 {
  float x, y, z;
  float2 xy, xz, yx, yz, zx, zy;
} float3;
typedef struct float4 {
  float x, y, z, w;
  float2 xy, yx;
  float3 xyz, xzy, yxz, yzx, zxy, zyx;
} float4;
#endif

