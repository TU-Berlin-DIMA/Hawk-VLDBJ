/**
 * @file
 *
 * Cuckoo Hashtable implementation (no support for locks)
 *
 * @author David Broneske
 * @author based on Code by Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#include "config.h"

#include <hardware_optimizations/main_memory_joins/serial_hash_join/hashtable/cuckoo_hashtable.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
inline uint64_t mult_shift(uint64_t x, uint64_t multiplicator, uint32_t
log_ht_size) {
        x *= multiplicator;
        x >>= (64-log_ht_size);
        return x;
}
*/
cuckoo_hashtable_t *cuckoo_hash_new(
    const unsigned int num_tuples,
    /* uint64_t num_buckets,*/ uint8_t num_hash_tables, uint32_t seed) {
  cuckoo_hashtable_t *ret;
  uint64_t i;
  uint64_t j;
  ret = malloc(sizeof(*ret));
  if (!ret) {
    fprintf(stderr, "Error during hash table allocation 1.\n");
    exit(EXIT_FAILURE);
  }

  /* compute number of buckets -- maybe too much?*/
  uint64_t num_buckets =
      (num_tuples * 1.3) / num_hash_tables;  // + HASH_TUPLES_PER_BUCKET - 1) /
                                             // HASH_TUPLES_PER_BUCKET;

  for (i = 0; (1 << i) < num_buckets; i++) /* do nothing */
    ;

  num_buckets = 1 << i;

  /* prepare hash table data structures */
  ret->log_num_buckets = i;
  // ret->mask = num_buckets - 1;
  ret->multiplicators = malloc(num_hash_tables * sizeof(*ret->multiplicators));
  ret->num_hash_tables = num_hash_tables;
  ret->buckets = malloc(num_hash_tables * sizeof(*ret->buckets));
  if (!ret->buckets) {
    fprintf(stderr, "Error during hash table allocation 2.\n");
    exit(EXIT_FAILURE);
  }
  srand(seed);
  for (i = 0; i < num_hash_tables; i++) {
    ret->multiplicators[i] = (((uint64_t)rand()) << 32) | (rand() & 1);
    ret->buckets[i] = malloc(num_buckets * sizeof(**ret->buckets));
    if (!ret->buckets[i]) {
      fprintf(stderr, "Error during hash table allocation 3.\n");
      exit(EXIT_FAILURE);
    }
    for (j = 0; j < num_buckets; j++)
      ret->buckets[i][j] = (cuckoo_hash_bucket_t){.count = 0, .values = NULL};
  }

  return ret;
}

/**
 * Allocate a new hash bucket.
 *
 * @todo Avoid excessive memory allocation from OS.  Instead, get
 *       buckets in chunks from OS, then pass them on to the application
 *       one-by-one.

static hash_bucket_t *
new_bucket (void)
{
    hash_bucket_t *bucket;

    bucket = malloc (sizeof (*bucket));
    if (!bucket)
    {
        fprintf (stderr, "Error allocating bucket!\n");
        exit (EXIT_FAILURE);
    }

    bucket->count = 0;
    bucket->next = NULL;

    return bucket;
}
 */

void cuckoo_hash_put(cuckoo_hashtable_t *hashtable, const tuple_t tuple) {
  unsigned int hash;
  cuckoo_hash_bucket_t to_insert;
  to_insert.count = 1;
  to_insert.key = tuple.key;
  to_insert.values = malloc(sizeof(ht_value_t));
  if (!to_insert.values) {
    fprintf(stderr, "Error during hash table allocation 4.\n");
    exit(EXIT_FAILURE);
  }
  to_insert.values[0] = tuple.value;
  int cur_ht = 0;
  int iterations = 10;
  cuckoo_hash_bucket_t temp;
  uint64_t temp_size = 1;
  while (iterations > 0) {
    hash = (to_insert.key * hashtable->multiplicators[cur_ht]) >>
           (64 - hashtable->log_num_buckets);
    if (hashtable->buckets[cur_ht][hash].count == 0) {
      hashtable->buckets[cur_ht][hash].count += to_insert.count;
      hashtable->buckets[cur_ht][hash].key = to_insert.key;
      hashtable->buckets[cur_ht][hash].values = to_insert.values;

      break;
    } else if (hashtable->buckets[cur_ht][hash].key == to_insert.key) {
      while (temp_size <
             hashtable->buckets[cur_ht][hash].count + to_insert.count)
        temp_size *= 2;

      hashtable->buckets[cur_ht][hash].values =
          realloc(hashtable->buckets[cur_ht][hash].values,
                  temp_size * 2 * sizeof(ht_value_t));

      if (!hashtable->buckets[cur_ht][hash].values) {
        fprintf(stderr, "Error during hash table allocation 5.\n");
        exit(EXIT_FAILURE);
      }

      hashtable->buckets[cur_ht][hash].values = to_insert.values;
      hashtable->buckets[cur_ht][hash].count++;

      free(to_insert.values);
      break;
    } else {
      for (temp_size = 1; temp_size < hashtable->buckets[cur_ht][hash].count;
           temp_size *= 2)
        /* do nothing */;

      temp.count = hashtable->buckets[cur_ht][hash].count;
      temp.key = hashtable->buckets[cur_ht][hash].key;
      temp.values = hashtable->buckets[cur_ht][hash].values;

      hashtable->buckets[cur_ht][hash].key = to_insert.key;
      hashtable->buckets[cur_ht][hash].count = to_insert.count;
      hashtable->buckets[cur_ht][hash].values = to_insert.values;

      to_insert.count = temp.count;
      to_insert.key = temp.key;
      to_insert.values = temp.values;
    }
    --iterations;
    cur_ht = (cur_ht + 1) % hashtable->num_hash_tables;
  }
  if (iterations == 0) {
    fprintf(stderr, "Could not find a bucket.\n");
  }
}

int cuckoo_hash_get(const cuckoo_hashtable_t *hashtable, const ht_key_t key,
                    ht_value_t *val) {
  unsigned int hash;
  for (unsigned int cur_ht = 0; cur_ht < hashtable->num_hash_tables; cur_ht++) {
    hash = (key * hashtable->multiplicators[cur_ht]) >>
           (64 - hashtable->log_num_buckets);
    if (hashtable->buckets[cur_ht][hash].count > 0 &&
        hashtable->buckets[cur_ht][hash].key == key) {
      val = hashtable->buckets[cur_ht][hash].values;
      return hashtable->buckets[cur_ht][hash].count;
    }
  }

  return 0;
}

static void cuckoo_hashtable_release(cuckoo_hash_bucket_t *bucket,
                                     uint64_t log_num_buckets) {
  uint64_t max = (uint64_t)pow(2, log_num_buckets);
  for (uint64_t i = 0; i < max; i++) {
    if (bucket[i].count > 0) free(bucket[i].values);
  }
  free(bucket);
}

void cuckoo_hash_release(cuckoo_hashtable_t *hashtable) {
  /* release dynamically allocated buckets */
  for (unsigned long i = 0; i < hashtable->num_hash_tables; i++)
    cuckoo_hashtable_release(hashtable->buckets[i], hashtable->log_num_buckets);

  /* release all the top-level buckets */
  free(hashtable->buckets);
  free(hashtable);
}
