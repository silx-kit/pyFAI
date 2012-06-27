/**
 * \file
 * \brief  Minimal logger
 *
 *  Provides logger functions with basic functionality while trying to remain
 *  fast and minimal in depedency requirements
 */

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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdarg.h>

#include "cLogger.h"

#ifdef _MSC_VER
#define STATINL static __inline
#else
#define STATINL static inline
#endif

//#ifdef _WIN32
//#define _BUFLIMIT 1
//#define fprintf(...)                                           \
//  do{                                                                \
//    if( ((*hLog).safebuf > _BUFLIMIT) && ((*hLog).type != LOGTSAFE)) \
//      {                                                              \
//        fflush((*hLog).stream);                                      \
//        (*hLog).safebuf = 0;                                         \
//      }                                                              \
//      fprintf(__VA_ARGS__);                                          \
//      (*hLog).safebuf++;                                             \
//    }while(0)
//
//
//#define vfprintf(...)                                           \
//  do{                                                                \
//    if( ((*hLog).safebuf > _BUFLIMIT) && ((*hLog).type != LOGTSAFE)) \
//      {                                                              \
//        fflush((*hLog).stream);                                      \
//        (*hLog).safebuf = 0;                                         \
//      }                                                              \
//      vfprintf(__VA_ARGS__);                                          \
//      (*hLog).safebuf++;                                             \
//    }while(0)
//
//#else
//#define fprintf fprintf
//#define vfprintf vfprintf
//#endif

const char *get_date()
{
  static char date[50];
  time_t t = time(0);

  sprintf(date, "%s" , asctime(localtime(&t)));

  return date;
}

STATINL const char *get_timestamp()
{
  static char date[13];
  time_t t = time(0);

  strftime(date, sizeof(date), "(%H:%M:%S)", localtime(&t));

  return date;
}

void cLog_init(logger_t *hLog, FILE *stream, const char *fname, int severity, enum_LOGTYPE type, enum_LOGDEPTH depth, int perf, int timestamps)
{
  (*hLog).status = 0;
  (*hLog).fname = NULL;
  (*hLog).severity = 0;
  (*hLog).type = 0;
  (*hLog).depth = 0;
  (*hLog).perf = 0;
  (*hLog).timestamps = 0;

  if( stream == NULL && fname == NULL ) stream = stdout;
  if (stream == stdout || stream == stderr )
  {
    (*hLog).stream=stdout;
    (*hLog).fname = (char *)malloc(6*sizeof(char));
    strcpy((*hLog).fname,"NULL");
  }else if( stream != NULL && fname == NULL )
  {
    //Very dangerous. Not Allowed!
    fprintf(stderr, "\n"
          "/-------------cLog---------------------------------------\\\n"
          "| You are trying to use cLog with a stream that is not   |\n"
          "|  NULL, stdout or stderr. This is not allowed when a    |\n"
          "|  a filename is not set and cLog will be disabled.      |\n"
          "|  All messages through cLog will be directed to stdout  |\n"
          "|  or stderr (for critical messages).                    |\n"
         "\\--------------------------------------------------------/\n"
          "\n"
          );
          
    (*hLog).status = 0;
    return;
  }else
  {
    (*hLog).stream = fopen(fname,"w");
    (*hLog).fname = (char *)malloc(strlen(fname) + 1);
    strcpy((*hLog).fname,fname);
  }
  (*hLog).severity = severity;
  (*hLog).type = type;
  (*hLog).depth = depth;
  (*hLog).perf = perf;
  (*hLog).timestamps = timestamps;
  (*hLog).status = 1;

  return;
}

void cLog_fin(logger_t *hLog)
{
  if((*hLog).status)
  {
    if( (*hLog).stream && (*hLog).stream != stdout && (*hLog).stream != stderr) 
      fclose((*hLog).stream);
    (*hLog).stream = NULL;
    if((*hLog).fname)free((*hLog).fname);
    (*hLog).status = 0;

  }
  return;
}

void cLog_date(logger_t *hLog, enum_LOGDEPTH depth)
{
  if( ((*hLog).depth >= depth) &&  ((*hLog).status == 1) )
  {
    switch((*hLog).type)
    {
    case LOGTFAST:
      fprintf((*hLog).stream,"%s",get_date());
      break;
    case LOGTSAFE:
      fflush((*hLog).stream);
      fprintf((*hLog).stream,"%s",get_date());
      fflush((*hLog).stream);
      break;
    }
  }
}

void cLog_date_text(logger_t *hLog, enum_LOGDEPTH depth, const char *format, ...)
{
  va_list argp;
  if( ((*hLog).depth >= depth) &&  ((*hLog).status == 1) )
  {
    va_start(argp,format);
    switch((*hLog).type)
    {
    case LOGTFAST:
      fprintf((*hLog).stream,"%s",get_date());
      vfprintf((*hLog).stream,format,argp);
      break;
    case LOGTSAFE:
      fflush((*hLog).stream);
      fprintf((*hLog).stream,"%s",get_date());
      vfprintf((*hLog).stream,format,argp);
      fflush((*hLog).stream);
      break;
    }
    va_end(argp);
  } else if ( (*hLog).status == 0 )
  {
    va_start(argp,format);
    vprintf(format,argp);
    va_end(argp);
  }
}

void cLog_basic(logger_t *hLog, const char * format, ...)
{
  va_list argp;

  if( ((*hLog).depth >=  LOGDBASIC) && (*hLog).status)
  {
    if((*hLog).timestamps)fprintf((*hLog).stream,"%s ",get_timestamp());
    va_start(argp,format);
    switch((*hLog).type)
    {
    case LOGTFAST:
      vfprintf((*hLog).stream,format,argp);
      break;
    case LOGTSAFE:
      fflush((*hLog).stream);
      vfprintf((*hLog).stream,format,argp);
      fflush((*hLog).stream);
      break;
    }
    va_end(argp);
  } else if ( (*hLog).status == 0 )
  {
    va_start(argp,format);
    vprintf(format,argp);
    va_end(argp);
  }
  return;
}

void cLog_extended(logger_t *hLog, const char * format, ...)
{
  va_list argp;

  if( ((*hLog).depth >= LOGDEXTENDED) && ((*hLog).status == 1) )
  {
    if((*hLog).timestamps)fprintf((*hLog).stream,"%s ",get_timestamp());
    va_start(argp,format);

    switch((*hLog).type)
    {
    case LOGTFAST:
      vfprintf((*hLog).stream,format,argp);
      break;
    case LOGTSAFE:
      fflush((*hLog).stream);
      vfprintf((*hLog).stream,format,argp);
      fflush((*hLog).stream);
      break;
    }
    va_end(argp);
  } else if ( (*hLog).status == 0 )
  {
    va_start(argp,format);
    vprintf(format,argp);
    va_end(argp);
  }

  return;
}

void cLog_debug(logger_t *hLog, const char * format, ...)
{
  va_list argp;

  if((*hLog).depth >= LOGDDEBUG && (*hLog).status)
  {
    if((*hLog).timestamps)fprintf((*hLog).stream,"%s ",get_timestamp());
    va_start(argp,format);

    switch((*hLog).type)
    {
    case LOGTFAST:
      vfprintf((*hLog).stream,format,argp);
      break;
    case LOGTSAFE:
      fflush((*hLog).stream);
      vfprintf((*hLog).stream,format,argp);
      fflush((*hLog).stream);
      break;
    }
    va_end(argp);
  } else if ( (*hLog).status == 0 )
  {
    va_start(argp,format);
    vprintf(format,argp);
    va_end(argp);
  }

  return;
}

void cLog_critical(logger_t *hLog, const char * format, ...)
{
  va_list argp;

  if( ((*hLog).depth >= LOGDONLYERRORS) && ((*hLog).status == 1) )
  {
    va_start(argp,format);  
    if((*hLog).timestamps)fprintf((*hLog).stream,"%s ",get_timestamp());

    fflush((*hLog).stream);
    vfprintf((*hLog).stream,format,argp);
    fflush((*hLog).stream);
    va_end(argp);
  }


  if( ((*hLog).stream != stderr) || ((*hLog).status == 0) )
  {
    va_start(argp,format);
    fflush(stderr);
    vfprintf(stderr,format,argp);
    fflush(stderr);
    va_end(argp);
  }

  return;
}

void cLog(logger_t *hLog, enum_LOGDEPTH depth, const char * format, ...)
{
  va_list argp;

  if( ((*hLog).depth >= depth) && ((*hLog).status == 1) )
  {
    va_start(argp,format);  
    if((*hLog).timestamps)fprintf((*hLog).stream,"%s ",get_timestamp());

    fflush((*hLog).stream);
    vfprintf((*hLog).stream,format,argp);
    fflush((*hLog).stream);
    va_end(argp);
  }
  return;
}

void cLog_bench(logger_t *hLog, const char * format, ...)
{
  va_list argp;

  if( ((*hLog).perf == 1) && ((*hLog).status) )
  {
    if((*hLog).timestamps)fprintf((*hLog).stream,"%s ",get_timestamp());
    va_start(argp,format);

    switch((*hLog).type)
    {
    case LOGTFAST:
      vfprintf((*hLog).stream,format,argp);
      break;
    case LOGTSAFE:
      fflush((*hLog).stream);
      vfprintf((*hLog).stream,format,argp);
      fflush((*hLog).stream);
      break;
    }
    va_end(argp);
  } else if ( (*hLog).status == 0 )
  {
    va_start(argp,format);
    vprintf(format,argp);
    va_end(argp);
  }

  return;
}

void cLog_report_configuration(logger_t *hLog)
{

  printf("\ncLogger Configuration\n");
  if( (hLog) )
  {
    if( (*hLog).status == 0 ) printf("- Disabled \n");
    else
    {
      if( (*hLog).stream == stdout )printf("- Stream: stdout \n");
      else if( (*hLog).stream == stderr )printf("- Stream: stderr \n");
      else printf("- Stream: %s \n", (*hLog).fname);

      printf("- Speed: %s\n",((*hLog).type)?"SAFE":"FAST");
      printf("- Depth: ");
      switch ( (*hLog).depth )
      {
      case LOGDNONE:
        printf("Nothing \n");
        break;
      case LOGDONLYERRORS:
        printf("Errors \n");
        break;
      case LOGDBASIC:
        printf("Basic \n");
        break;
      case LOGDEXTENDED:
        printf("Extended \n");
        break;
      case LOGDDEBUG:
        printf("Debug \n");
        break;
      default:
        printf("Undefined \n");
        break;
      }
      printf("- Profiling Information: %s\n",((*hLog).perf)?"YES":"NO");
      printf("- Timestamps: %s\n",((*hLog).timestamps)?"YES":"NO");
    } 
  }else printf("- Not Initialised \n");

  printf("\n");
  return;
}

void cLog_log_configuration(logger_t *hLog)
{

  if( (!hLog) || (*hLog).status == 0 ) return;
  else
  {
    cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"\n");
    cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"cLogger Configuration\n");
    if( (*hLog).stream == stdout )cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Stream: stdout \n");
    else if( (*hLog).stream == stderr )cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Stream: stderr \n");
    else cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Stream: %s \n", (*hLog).fname);

    cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Speed: %s\n",((*hLog).type)?"SAFE":"FAST");
    switch ( (enum_LOGDEPTH)(*hLog).depth )
    {
    case LOGDNONE:
      cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Depth: Nothing \n");
      break;
    case LOGDONLYERRORS:
      cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Depth: Errors \n");
      break;
    case LOGDBASIC:
      cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Depth: Basic \n");
      break;
    case LOGDEXTENDED:
      cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Depth: Extended \n");
      break;
    case LOGDDEBUG:
      cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Depth: Debug \n");
      break;
    default:
      cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Depth: Undefined \n");
      break;
    }
    cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Profiling Information: %s\n",((*hLog).perf)?"YES":"NO");
    cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"- Timestamps: %s\n",((*hLog).timestamps)?"YES":"NO");

    cLog(hLog, (enum_LOGDEPTH)(*hLog).depth,"\n");
  }
  return;
}
