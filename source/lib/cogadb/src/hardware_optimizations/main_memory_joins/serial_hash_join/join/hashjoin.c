/**
 * @file
 *
 * Basic hash join implementation.
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#include "config.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "hashtable/hashtable.h"
#include "join/hashjoin.h"
#include "time/time.h"

#include <inttypes.h>

//#define REPORT_NO_MATCH 1

/**
 * Basic hash join; called by hashjoin() which also includes
 * functionality for partitioning.
 */
unsigned long hashjoin_impl(const relation_t R, const relation_t S) {
  hashtable_t *hashtable;
  unsigned long num_result = 0;

  /* Build phase */
  hashtable = hash_new(R.count);

  for (unsigned long r = 0; r < R.count; r++) hash_put(hashtable, R.tuples[r]);

  /* Join phase */
  for (unsigned long s = 0; s < S.count; s++) {
    unsigned long hash = HASH(S.tuples[s].key) & hashtable->mask;
    hash_bucket_t *bucket = &hashtable->buckets[hash];

#if REPORT_NO_MATCH
    unsigned long old_num_result = num_result;
#endif

    while (bucket) {
      for (unsigned int i = 0; i < bucket->count; i++)
        if (bucket->tuples[i].key == S.tuples[s].key) {
#if ENABLE_PRINT_RESULTS
          printf(" %10u | %10u | %10u | %10u \n", bucket->tuples[i].key,
                 bucket->tuples[i].value, S.tuples[s].key, S.tuples[s].value);
#endif
          num_result++;
        }

      bucket = bucket->next;
    }

#if REPORT_NO_MATCH
    if (old_num_result == num_result)
      fprintf(stderr, "No match found for key %u.\n", S.tuples[s].key);
#endif
  }

  hash_release(hashtable);

  return num_result;
}

/* For debugging only. */
static void print_relation(relation_t rel, char *name) __attribute__((unused));

static void print_relation(relation_t rel, char *name) {
  printf("Relation %s:\n\n", name);
  printf("      key    |   value\n");
  printf(" ------------+------------\n");

  //    for (unsigned long i = 0; i < rel.count; i++)
  //        printf ("  %10u | %10u\n", rel.tuples[i].key, rel.tuples[i].value);

  for (unsigned long i = 0; i < rel.count; i++)
    printf("  %10" PRIu64 " | %10" PRIu64 "\n", rel.tuples[i].key,
           rel.tuples[i].value);

  printf("\n");
}

/**
 * (Partitioned) hash join.
 */
unsigned long hashjoin(const relation_t R, const relation_t S,
                       const part_bits_t *part_bits) {
  const part_bits_t *bits = part_bits;
  partitioned_relation_t part_R, part_S;
  unsigned long ret = 0;

  /*
   * Turn relation representation of R into "partitioned" representation
   * (this is actually just one partition).
   */
  part_R.num_tuples = R.count;
  part_R.num_part = 1;
  part_R.startaddrs = malloc(2 * sizeof(*part_R.startaddrs));

  if (!part_R.startaddrs) {
    fprintf(stderr, "Error allocating memory for partition start address.\n");
    exit(EXIT_FAILURE);
  }

  part_R.startaddrs[0] = R.tuples;
  part_R.startaddrs[1] = R.tuples + R.count;

  /* Same for S */
  part_S.num_tuples = S.count;
  part_S.num_part = 1;
  part_S.startaddrs = malloc(2 * sizeof(*part_S.startaddrs));

  if (!part_S.startaddrs) {
    fprintf(stderr, "Error allocating memory for partition start address.\n");
    exit(EXIT_FAILURE);
  }

  part_S.startaddrs[0] = S.tuples;
  part_S.startaddrs[1] = S.tuples + S.count;

  /* now partition as long as we are instructed to do so */
  while (bits) {
    partitioned_relation_t new_part_rel;
    struct timespec t_start;
    struct timespec t_end;

    my_gettime(&t_start);

    new_part_rel = partition(part_R, bits->from_bit, bits->to_bit);

    free(part_R.startaddrs[0]);
    free(part_R.startaddrs);

    part_R = new_part_rel;

    new_part_rel = partition(part_S, bits->from_bit, bits->to_bit);

    free(part_S.startaddrs[0]);
    free(part_S.startaddrs);

    part_S = new_part_rel;

    my_gettime(&t_end);

    printf("Partitioning took %lu nsec.\n",
           (t_end.tv_sec * 1000000000L + t_end.tv_nsec) -
               (t_start.tv_sec * 1000000000L + t_start.tv_nsec));

    bits = bits->next;
  }

  struct timespec t_start;
  struct timespec t_end;

  my_gettime(&t_start);

  /* for each partition compute the join */
  for (unsigned long i = 0; i < part_R.num_part; i++) {
    relation_t R, S;
    unsigned long num_matches;

    R.count = part_R.startaddrs[i + 1] - part_R.startaddrs[i];
    R.tuples = part_R.startaddrs[i];
    S.count = part_S.startaddrs[i + 1] - part_S.startaddrs[i];
    S.tuples = part_S.startaddrs[i];

    /*
    printf (" -- partition %lu --\n", i);
    print_relation (R, "R");
    print_relation (S, "S");
    */

    num_matches = hashjoin_impl(R, S);

    /*
    printf (" → %lu matches.\n\n", num_matches);
    */

    ret += num_matches;
  }

  my_gettime(&t_end);

  printf("Join took %lu nsec.\n",
         (t_end.tv_sec * 1000000000L + t_end.tv_nsec) -
             (t_start.tv_sec * 1000000000L + t_start.tv_nsec));

  return ret;
}
