/**
 * @file
 *
 * Hashtable implementation (no support for locks)
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#include "config.h"

#include "hashtable/hashtable.h"

#include <stdio.h>
#include <stdlib.h>

uint64_t simple_hash_function(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccd;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53;
  x ^= x >> 33;
  return x;
}

hashtable_t *hash_new(const unsigned int num_tuples) {
  hashtable_t *ret;
  unsigned int num_buckets;
  unsigned int i;

  ret = malloc(sizeof(*ret));
  if (!ret) {
    fprintf(stderr, "Error during hash table allocation.\n");
    exit(EXIT_FAILURE);
  }

  /* compute number of buckets */
  num_buckets =
      (num_tuples * 1.3 + HASH_TUPLES_PER_BUCKET - 1) / HASH_TUPLES_PER_BUCKET;

  for (i = 0; (1 << i) < num_buckets; i++) /* do nothing */
    ;

  num_buckets = 1 << i;

  /* prepare hash table data structures */
  ret->num_buckets = num_buckets;
  ret->mask = num_buckets - 1;
  ret->buckets = malloc(num_buckets * sizeof(*ret->buckets));
  if (!ret->buckets) {
    fprintf(stderr, "Error during hash table allocation.\n");
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < num_buckets; i++)
    ret->buckets[i] = (hash_bucket_t){.count = 0, .next = NULL};

  return ret;
}

/**
 * Allocate a new hash bucket.
 *
 * @todo Avoid excessive memory allocation from OS.  Instead, get
 *       buckets in chunks from OS, then pass them on to the application
 *       one-by-one.
 */
static hash_bucket_t *new_bucket(void) {
  hash_bucket_t *bucket;

  bucket = malloc(sizeof(*bucket));
  if (!bucket) {
    fprintf(stderr, "Error allocating bucket!\n");
    exit(EXIT_FAILURE);
  }

  bucket->count = 0;
  bucket->next = NULL;

  return bucket;
}

void hash_put(hashtable_t *hashtable, const tuple_t tuple) {
  unsigned int hash;
  hash_bucket_t *bucket;

  hash = HASH(tuple.key) & hashtable->mask;

  /*
  fprintf (stderr, "putting key value %u into bucket %u.\n",
          tuple.key, hash);
  //*/

  bucket = &hashtable->buckets[hash];

  while (bucket->count == HASH_TUPLES_PER_BUCKET) {
    if (bucket->next)
      bucket = bucket->next;
    else
      bucket = bucket->next = new_bucket();
  }

  bucket->tuples[bucket->count] = tuple;
  bucket->count++;
}

bool hash_get_first(const hashtable_t *hashtable, const ht_key_t key,
                    ht_value_t *val) {
  unsigned int hash;
  hash_bucket_t *bucket;

  hash = HASH(key) & hashtable->mask;

  bucket = &hashtable->buckets[hash];

  while (bucket) {
    for (unsigned int i = 0; i < bucket->count; i++)
      if (bucket->tuples[i].key == key) {
        *val = bucket->tuples[i].value;
        return true;
      }

    bucket = bucket->next;
  }

  return false;
}

static void bucket_release(hash_bucket_t *bucket) {
  if (!bucket)
    return;
  else {
    bucket_release(bucket->next);
    free(bucket);
  }
}

void hash_release(hashtable_t *hashtable) {
  /* release dynamically allocated buckets */
  for (unsigned long i = 0; i < hashtable->num_buckets; i++)
    bucket_release(hashtable->buckets[i].next);

  /* release all the top-level buckets */
  free(hashtable->buckets);
  free(hashtable);
}
