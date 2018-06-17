/**
 * @file
 *
 * Partitioning stage of partitioned hash join
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#include "config.h"

#include "join/partition.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "hashtable/hashtable.h"

/**
 * number of tuples to put into one buffer entry (when
 * using software-managed buffers
 */
#define BUFFER_SIZE 8

/**
 * Mask to implement modulo in a cheaper way.
 */
#define BUFFER_SIZE_MASK 7

#define WITH_SOFTWARE_MANAGED_BUFFERS 1

/**
 * Worker for partition().
 */
static void partition_worker(const unsigned long num_tuples,
                             const unsigned int from_bit,
                             const unsigned int to_bit, const tuple_t *src,
                             tuple_t **dst) {
  const unsigned long num_partitions = 1 << (to_bit - from_bit + 1);
  const unsigned long mask = num_partitions - 1;
  const unsigned int shift_by = sizeof(unsigned long) * 8 - to_bit;
  unsigned long *histogram;
  unsigned long *pos;

#if WITH_SOFTWARE_MANAGED_BUFFERS
  tuple_t *buf;
#endif

  histogram = malloc(num_partitions * sizeof(*histogram));
  assert(histogram);
  pos = malloc(num_partitions * sizeof(*histogram));
  assert(histogram);

#if WITH_SOFTWARE_MANAGED_BUFFERS
  buf = malloc(BUFFER_SIZE * num_partitions * sizeof(*buf));
  assert(buf);
#endif

  /* initialize histogram */
  for (unsigned long i = 0; i < num_partitions; i++) {
    histogram[i] = 0;
    pos[i] = 0;
  }

  /* populate histogram */
  for (unsigned long i = 0; i < num_tuples; i++) {
    unsigned long hash = (HASH(src[i].key) >> shift_by) & mask;

    histogram[hash]++;
  }

  /* determine beginning of each partition with help of prefix sum */
  assert(dst[0]);
  for (unsigned long i = 1; i < num_partitions; i++)
    dst[i] = dst[i - 1] + histogram[i - 1];

  /* finally: partition */
  for (unsigned long i = 0; i < num_tuples; i++) {
    unsigned long hash = (HASH(src[i].key) >> shift_by) & mask;

#if WITH_SOFTWARE_MANAGED_BUFFERS

    /*
    printf ("copying %3u|%10u to buf[%3lu] (hash: %lu).\n",
            src[i].key, src[i].value,
            hash * BUFFER_SIZE + (pos[hash] & BUFFER_SIZE_MASK), hash);
    */

    buf[hash * BUFFER_SIZE + (pos[hash] & BUFFER_SIZE_MASK)] = src[i];
    pos[hash]++;

    /* flush out data at cache line boundaries */
    if (((unsigned long)(dst[hash] + pos[hash]) & 63) == 0) {
      /*
      printf ("flushing buffer with hash value %lu.\n", hash);
      */

      for (unsigned int j = (pos[hash] > BUFFER_SIZE) ? pos[hash] - BUFFER_SIZE
                                                      : 0;
           j < pos[hash]; j++) {
        /*
        printf ("copying from buf[%3lu] to dst[%3lu][%3u].\n",
                hash * BUFFER_SIZE + (j & BUFFER_SIZE_MASK),
                hash, j);
        */

        dst[hash][j] = buf[hash * BUFFER_SIZE + (j & BUFFER_SIZE_MASK)];
      }
    }

#else
    dst[hash][pos[hash]] = src[i];
    pos[hash]++;
#endif
  }

#if WITH_SOFTWARE_MANAGED_BUFFERS

  /* flush out remaining data from buf[] */
  for (unsigned long i = 0; i < num_partitions; i++) {
    if (((unsigned long)(dst[i] + pos[i]) & 63) != 0) {
      /*
      printf ("flushing buffer with hash value %lu.\n", i);
      */

      for (unsigned int j = (pos[i] > BUFFER_SIZE) ? pos[i] - BUFFER_SIZE : 0;
           j < pos[i]; j++) {
        /*
        printf ("copying from buf[%3lu] to dst[%3lu][%3u].\n",
                i * BUFFER_SIZE + (j & BUFFER_SIZE_MASK),
                i, j);
        */

        dst[i][j] = buf[i * BUFFER_SIZE + (j & BUFFER_SIZE_MASK)];
      }
    }
  }

#endif

  free(histogram);
  free(pos);
}

/**
 * Takes a partitioned relation and partitions it further.
 */
partitioned_relation_t partition(const partitioned_relation_t rel,
                                 const unsigned int from_bit,
                                 const unsigned int to_bit) {
  partitioned_relation_t ret;
  unsigned long fanout = (1 << (to_bit - from_bit + 1));

  ret.num_tuples = rel.num_tuples;
  ret.num_part = rel.num_part * fanout;
  ret.startaddrs = malloc((ret.num_part + 1) * sizeof(*ret.startaddrs));

  if (!ret.startaddrs) {
    fprintf(stderr,
            "Could not allocate memory for .startaddrs in partition().\n");
    exit(EXIT_FAILURE);
  }

  ret.startaddrs[0] = malloc(rel.num_tuples * sizeof(**ret.startaddrs));

  if (!ret.startaddrs) {
    fprintf(stderr,
            "Could not allocate memory for .startaddrs[0] "
            "in partition().\n");
    exit(EXIT_FAILURE);
  }

  for (unsigned long i = 1; i <= rel.num_part; i++)
    ret.startaddrs[i * fanout] =
        ret.startaddrs[0] + (rel.startaddrs[i] - rel.startaddrs[0]);

  for (unsigned int i = 0; i < rel.num_part; i++) {
    partition_worker(rel.startaddrs[i + 1] - rel.startaddrs[i], from_bit,
                     to_bit, rel.startaddrs[i], ret.startaddrs + i * fanout);
  }

  return ret;
}
