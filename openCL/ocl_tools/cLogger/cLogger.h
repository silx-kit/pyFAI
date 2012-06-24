
/*
 *   Project: Minimal logger suite. Logging with no depedencies
 *
 *   Copyright (C) 2012 Dimitrios Karkoulis
 *
 *   Principal authors: D. Karkoulis (dimitris.karkoulis@gmail.com)
 *   Last revision: 24/06/2012
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

#ifndef CLOGGER_H
#define CLOGGER_H

#include <stdarg.h>

#ifdef _WIN32
  #ifdef _DLLIMPORT
    #define DllInterface   __declspec( dllimport )
  #else
    #define DllInterface   __declspec( dllexport )
  #endif
#else
  #define DllInterface 
#endif

typedef struct
{
  FILE * stream;
  int severity;
  int type;
  int depth;
  int perf;
  int timestamps;
  int status;
  char *fname;

} logger_t;

typedef enum
{
  LOGTFAST,
  LOGTSAFE
}enum_LOGTYPE;

typedef enum
{
  LOGDNONE,
  LOGDONLYERRORS,
  LOGDBASIC,
  LOGDEXTENDED,
  LOGDDEBUG
}enum_LOGDEPTH;

#ifdef __cplusplus
extern "C"{
#endif

DllInterface void cLog_basic(logger_t *hLog, const char * format, ...);
DllInterface void cLog_extended(logger_t *hLog, const char* format, ...);
DllInterface void cLog_debug(logger_t *hLog, const char * format, ...);
DllInterface void cLog_critical(logger_t *hLog, const char * format, ...);
DllInterface void cLog_bench(logger_t *hLog, const char* format, ...);
DllInterface void cLog_date(logger_t *hLog, enum_LOGDEPTH depth);
DllInterface void cLog_date_text(logger_t *hLog, enum_LOGDEPTH depth, const char *format, ...);
DllInterface void cLog_init(logger_t *hLog, FILE *stream, const char *fname, int severity, enum_LOGTYPE type, enum_LOGDEPTH depth, int perf, int timestamps);
DllInterface void cLog_fin(logger_t *hLog);

#ifdef __cplusplus
}
#endif

#endif