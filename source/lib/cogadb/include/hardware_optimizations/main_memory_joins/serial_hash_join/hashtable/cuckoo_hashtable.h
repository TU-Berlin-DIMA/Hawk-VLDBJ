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

#ifndef CUCKOO_HASHTABLE_H
#define CUCKOO_HASHTABLE_H

#ifndef __cplusplus
#include <stdbool.h>
#endif
//#include <stdbool.h>

#include <hardware_optimizations/main_memory_joins/serial_hash_join/schema.h>

//#define HASH_TUPLES_PER_BUCKET 2
//#define HASH_TUPLES_PER_BUCKET 20

// test code, took from:
// http://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
// inline unsigned int simple_hash_function(unsigned int x) {
//    x = ((x >> 16) ^ x) * 0x45d9f3b;
//    x = ((x >> 16) ^ x) * 0x45d9f3b;
//    x = ((x >> 16) ^ x);
//    return x;
//}

// inline uint64_t mult_shift(uint64_t x, uint64_t multiplicator, uint32_t
// log_ht_size);

//#define HASH(x) (x)
//#define CUCKOO_H(x,mult,ht_size) mult_shift(x,mult,ht_size)

struct cuckoo_hashtable_t {
  uint64_t log_num_buckets;
  uint8_t num_hash_tables;
  uint64_t *multiplicators;
  // unsigned int     mask;            /**< @a num_buckets - 1, so we can do
  //                                       AND rather than MODULO */
  // cockoo_hash_bucket_t[table_num][bucket_num]
  struct cuckoo_hash_bucket_t **buckets;
};

typedef struct cuckoo_hashtable_t cuckoo_hashtable_t;

struct cuckoo_hash_bucket_t {
  unsigned int count;
  ht_key_t key;
  ht_value_t *values;
};

typedef struct cuckoo_hash_bucket_t cuckoo_hash_bucket_t;

/**
 * Create a new, empty hash table, where the initial buckets have
 * room for at least @a num_tuples tuples.  Internally, the bucket
 * count might be rounded up, such that the number of buckets is
 * always a power of two.
 *
 * @param num_tuples  Provide initial space for this many tuples.
 */
cuckoo_hashtable_t *cuckoo_hash_new(
    const unsigned int num_tuples,
    /*uint64_t num_buckets,*/ uint8_t num_hash_tables, uint32_t seed);

void cuckoo_hash_put(cuckoo_hashtable_t *hashtable, const tuple_t tuple);

int cuckoo_hash_get(const cuckoo_hashtable_t *hashtable, const ht_key_t key,
                    ht_value_t *val);

void cuckoo_hash_release(cuckoo_hashtable_t *hashtable);

#endif /* HASHTABLE_H */
