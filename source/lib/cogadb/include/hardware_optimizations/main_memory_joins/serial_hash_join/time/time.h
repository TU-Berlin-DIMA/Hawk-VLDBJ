/**
 * @file
 *
 * Wrapper for time-related stuff.  This is mainly because Mac OS X
 * does not support POSIX real-time functionality.  Here we provide
 * an interface that will support OS functionality as available.
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * (c) 2010-2013 ETH Zurich, Systems Group
 * (c) 2013 TU Dortmund
 *
 * $Id$
 */

#ifndef TIME_H
#define TIME_H

#ifdef HAVE_LIBRT

#include <time.h>

#define my_gettime(tm) clock_gettime(CLOCK_REALTIME, (tm))

#define my_nanosleep(ts) \
  clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, (ts), NULL)

#else

#include <sys/time.h>

int my_gettime(struct timespec *ts);

void my_nanosleep(struct timespec *ts);

#endif

#endif /* TIME_H */
