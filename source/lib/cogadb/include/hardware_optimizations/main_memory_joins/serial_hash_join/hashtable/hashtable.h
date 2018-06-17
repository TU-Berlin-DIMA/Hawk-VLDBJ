/**
 * @file
 *
 * Hashtable implementation (no support for locks)
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#ifndef HASHTABLE_H
#define HASHTABLE_H

#ifndef __cplusplus
#include <stdbool.h>
#endif
//#include <stdbool.h>

#include <hardware_optimizations/main_memory_joins/serial_hash_join/schema.h>

//#define HASH_TUPLES_PER_BUCKET 2
#define HASH_TUPLES_PER_BUCKET 20

// test code, took from:
// http://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
// inline unsigned int simple_hash_function(unsigned int x) {
//    x = ((x >> 16) ^ x) * 0x45d9f3b;
//    x = ((x >> 16) ^ x) * 0x45d9f3b;
//    x = ((x >> 16) ^ x);
//    return x;
//}

uint64_t simple_hash_function(uint64_t x);

//#define HASH(x) (x)
#define HASH(x) simple_hash_function(x)

struct hashtable_t {
  unsigned int num_buckets;
  unsigned int mask; /**< @a num_buckets - 1, so we can do
                          AND rather than MODULO */
  struct hash_bucket_t *buckets;
};

typedef struct hashtable_t hashtable_t;

struct hash_bucket_t {
  unsigned int count;
  tuple_t tuples[HASH_TUPLES_PER_BUCKET];
  struct hash_bucket_t *next;
};

typedef struct hash_bucket_t hash_bucket_t;

/**
 * Create a new, empty hash table, where the initial buckets have
 * room for at least @a num_tuples tuples.  Internally, the bucket
 * count might be rounded up, such that the number of buckets is
 * always a power of two.
 *
 * @param num_tuples  Provide initial space for this many tuples.
 */
hashtable_t *hash_new(const unsigned int num_tuples);

void hash_put(hashtable_t *hashtable, const tuple_t tuple);

bool hash_get_first(const hashtable_t *hashtable, const ht_key_t key,
                    ht_value_t *val);

void hash_release(hashtable_t *hashtable);

#endif /* HASHTABLE_H */
