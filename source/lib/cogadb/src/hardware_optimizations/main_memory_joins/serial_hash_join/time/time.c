/**
 * @file
 *
 * Our poor man's replacements for POSIX real-time functionality.
 * Mainly intended to make our code compile and run on Mac OS X
 * systems (@em not for productive use or experiments!).
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * (c) 2010-2013 ETH Zurich, Systems Group
 * (c) 2013 TU Dortmund
 *
 * $Id$
 */

#include "config.h"

#include <stdlib.h>

#include <sys/time.h>

#ifndef HAVE_LIBRT

int my_gettime(struct timespec *ts) {
  struct timeval t;
  gettimeofday(&t, NULL);
  ts->tv_sec = t.tv_sec;
  ts->tv_nsec = t.tv_usec * 1000;
  return 0;
}

void my_nanosleep(struct timespec *ts) {
  struct timeval t;

  do {
    gettimeofday(&t, NULL);
  } while (t.tv_sec * 1000000000 + t.tv_usec * 1000 <
           ts->tv_sec * 1000000000 + ts->tv_nsec);
}

#endif
