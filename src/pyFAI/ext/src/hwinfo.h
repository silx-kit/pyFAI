/*
MIT License

Source: https://github.com/lfreist/hwinfo
Copyright (c) 2022 Leon Freist

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Footer
*/

#pragma once

#if defined(unix) || defined(__unix) || defined(__unix__)
#define HWINFO_UNIX
#endif
#if defined(__APPLE__)
#define HWINFO_APPLE
#endif
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
#define HWINFO_WINDOWS
#endif

#if defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) || defined(_M_X64)
#define HWINFO_CPU_X86_64
#define HWINFO_CPU_X86 64
#elif defined(__i386__) || defined(_M_IX86)
#define HWINFO_CPU_X86_32
#define HWINFO_CPU_X86 32
#endif

#if defined(__ARM_ARCH)
#define HWINFO_CPU_ARM __ARM_ARCH
#endif


#if defined(__ARM_ARCH)
#define HWINFO_CPU_ARM __ARM_ARCH
#endif

#if defined(__PPC__)
#if defined(__PPC64__)
#define HWINFO_CPU_PPC 64
#else
#define HWINFO_CPU_PPC 32
#endif

#if defined(__ALTIVEC__)
#define HWINFO_CPU_PPC_ALTIVEC __ALTIVEC__
#else
#define HWINFO_CPU_PPC_ALTIVEC 0
#endif
#endif
