/**
 * @file
 *
 * Hashtable implementation (no support for locks)
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 * @author Sebastian Breß <sebastian.bress@cs.tu-dortmund.de>
 *
 * $Id$
 */

#ifndef HASHTABLE_HPP
#define HASHTABLE_HPP

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <typeinfo>
#include <utility>

#include <boost/make_shared.hpp>

#include <core/global_definitions.hpp>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

//#include <tr1/functional_hash.h>

#define HashValueType uint64_t

//#define DEBUG_TYPED_HASH_TABLE

namespace CoGaDB {

  template <typename KeyType, typename PayloadType>
  struct TypedHashTable {
    /**
     * Types for key and value columns.
     *
     * @note We keep things simple here and assume the same schema for
     *       both input tables.
     */
    //        typedef KeyType ht_key_t;
    //        typedef PayloadType ht_value_t;

    /** A tuple consists of a key and a value */
    struct tuple_t {
      KeyType key; /**< This is the attribute that we join over. */
      PayloadType
          payload; /**< Payload; not actually touched by join algorithms. */
    };
    // typedef struct tuple_t tuple_t;

    /**
     * A relation is an array of tuples.  We represent this as a pointer
     * to the first tuple and a @a count information that tells how many
     * tuples are in the relation.  This representation has the advantage
     * that we can easily also pass parts of a larger relation.
     */
    struct relation_t {
      unsigned long count;
      struct tuple_t *tuples;
    };
// typedef struct relation_t relation_t;

//        #define TYPED_HASHTABLE_TUPLES_PER_BUCKET 2
//        #define TYPED_HASHTABLE_TUPLES_PER_BUCKET 8
//        #define TYPED_HASHTABLE_TUPLES_PER_BUCKET 20
#define TYPED_HASHTABLE_TUPLES_PER_BUCKET 40
    //        #define TYPED_HASHTABLE_TUPLES_PER_BUCKET 80
    //        #define TYPED_HASHTABLE_TUPLES_PER_BUCKET 4

    struct hash_bucket_t {
      size_t count;
      tuple_t tuples[TYPED_HASHTABLE_TUPLES_PER_BUCKET];
      struct hash_bucket_t *next;
      //            pthread_mutex_t mutex;
    };

    typedef struct hash_bucket_t hash_bucket_t;

    struct hashtable_t {
      size_t num_buckets;
      HashValueType mask; /**< @a num_buckets - 1, so we can do
                                     AND rather than MODULO */
      hash_bucket_t *buckets;
    };

    typedef struct hashtable_t hashtable_t;

    //#define HASH(x) (x)
    //#define HASH(x) hash_function(x)

    TypedHashTable(const size_t num_elements);

    ~TypedHashTable();

    hashtable_t *hashtable;

    void put(const tuple_t &tuple);

    void put_unique(const tuple_t &tuple);

    bool hash_get_first(const KeyType key, PayloadType *val);

    hash_bucket_t *getBucket(const KeyType &key);

    void printStatistics() const;

   private:
    hashtable_t *hash_new(const size_t num_tuples);

    /**
     * Allocate a new hash bucket.
     *
     * @todo Avoid excessive memory allocation from OS.  Instead, get
     *       buckets in chunks from OS, then pass them on to the application
     *       one-by-one.
     */
    hash_bucket_t *new_bucket(void);

    void bucket_release(hash_bucket_t *bucket);

    void hash_release(hashtable_t *hashtable);
  };

  template <typename KeyType, typename PayloadType>
  void TypedHashTable<KeyType, PayloadType>::printStatistics() const {
    std::cout << "Typed Hash Table" << std::endl;
    std::cout << "Number of Buckets: " << hashtable->num_buckets << std::endl;

    std::ofstream file("hash_table_stats.csv");

    file << "#BucketID  Number_Of_Entries" << std::endl;

    typedef std::map<uint64_t, uint64_t> OccurencesMap;
    OccurencesMap occurences_map;

    size_t max_number_of_entries = 0;
    size_t sum_number_of_entries = 0;
    //        size_t count_buckets=0;

    for (size_t i = 0; i < hashtable->num_buckets; ++i) {
      hash_bucket_t *bucket = &hashtable->buckets[i];

      size_t num_entries = bucket->count;
      while (bucket) {
        num_entries += bucket->count;
        bucket = bucket->next;
      }
      occurences_map[num_entries]++;
      sum_number_of_entries += num_entries;
      max_number_of_entries = std::max(max_number_of_entries, num_entries);

      //            std::cout << "Entries in Bucket " << i << ": " <<
      //            num_entries << std::endl;
      file << i << "  " << num_entries << std::endl;
    }

    std::cout << "Max Number Of Entries per Bucket: " << max_number_of_entries
              << std::endl;
    std::cout << "Average Number Of Entries per Bucket: "
              << double(sum_number_of_entries) / hashtable->num_buckets
              << std::endl;

    std::cout << "============================================================="
              << std::endl;
    std::cout << "Bucket Distribution Histogram: " << std::endl;

    OccurencesMap::const_iterator cit;
    for (cit = occurences_map.begin(); cit != occurences_map.end(); ++cit) {
      std::cout << "\t" << cit->first << " elements in Bucket: " << cit->second
                << std::endl;
    }
    std::cout << "============================================================="
              << std::endl;
  }

  template <typename KeyType>
  HashValueType hash_function(KeyType x) {
    COGADB_FATAL_ERROR("No hash function available for this type.", "");
  }

  // took from:
  // http://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
  template <>
  inline HashValueType hash_function(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
  }

  template <>
  inline HashValueType hash_function(int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
  }

  /******************************************************************************/
  /* BEGIN: fast hash function*/
  /* The MIT License

     Copyright (C) 2012 Zilong Tan (eric.zltan@gmail.com)

     Permission is hereby granted, free of charge, to any person
     obtaining a copy of this software and associated documentation
     files (the "Software"), to deal in the Software without
     restriction, including without limitation the rights to use, copy,
     modify, merge, publish, distribute, sublicense, and/or sell copies
     of the Software, and to permit persons to whom the Software is
     furnished to do so, subject to the following conditions:

     The above copyright notice and this permission notice shall be
     included in all copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
     NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
     BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
     ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
     CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
     SOFTWARE.
  */

  // Compression function for Merkle-Damgard construction.
  // This function is generated using the framework provided.
  template <typename T>
  inline T mix(T h) {
    h ^= h >> 23;
    h *= 0x2127599bf4325c37ULL;
    h ^= h >> 47;
    return h;
  }

  inline uint64_t fasthash64(const void *buf, size_t len, uint64_t seed) {
    const uint64_t m = 0x880355f21e6d1965ULL;
    const uint64_t *pos = static_cast<const uint64_t *>(buf);
    const uint64_t *end = pos + (len / 8);
    const unsigned char *pos2;
    uint64_t h = seed ^ (len * m);
    uint64_t v;

    while (pos != end) {
      v = *pos++;
      h ^= mix(v);
      h *= m;
    }

    pos2 = reinterpret_cast<const unsigned char *>(pos);
    v = 0;

    switch (len & 7) {
      case 7:
        v ^= static_cast<uint64_t>(pos2[6]) << 48;
      case 6:
        v ^= static_cast<uint64_t>(pos2[5]) << 40;
      case 5:
        v ^= static_cast<uint64_t>(pos2[4]) << 32;
      case 4:
        v ^= static_cast<uint64_t>(pos2[3]) << 24;
      case 3:
        v ^= static_cast<uint64_t>(pos2[2]) << 16;
      case 2:
        v ^= static_cast<uint64_t>(pos2[1]) << 8;
      case 1:
        v ^= static_cast<uint64_t>(pos2[0]);
        h ^= mix(v);
        h *= m;
    }

    return mix(h);
  }
  /* END fast hash function */
  /******************************************************************************/

  // http://stackoverflow.com/questions/5085915/what-is-the-best-hash-function-for-uint64-t-keys-ranging-from-0-to-its-max-value
  template <>
  inline HashValueType hash_function(uint64_t x) {
    return fasthash64(&x, sizeof(uint64_t), 3);
  }

  template <typename KeyType, typename PayloadType>
  TypedHashTable<KeyType, PayloadType>::TypedHashTable(
      const size_t num_elements)
      : hashtable(hash_new(num_elements)) {
    assert(hashtable != NULL);
  }

  template <typename KeyType, typename PayloadType>
  TypedHashTable<KeyType, PayloadType>::~TypedHashTable() {
    hash_release(hashtable);
  }

  template <typename KeyType, typename PayloadType>
  typename TypedHashTable<KeyType, PayloadType>::hashtable_t *
  TypedHashTable<KeyType, PayloadType>::hash_new(const size_t num_tuples) {
    hashtable_t *ret;
    size_t num_buckets;
    size_t i;

    ret = static_cast<hashtable_t *>(malloc(sizeof(hashtable_t)));
    if (!ret) {
      COGADB_FATAL_ERROR("Error during hash table allocation.", "");
    }

    /* compute number of buckets */
    num_buckets = static_cast<size_t>(
        (num_tuples * 1.3 + TYPED_HASHTABLE_TUPLES_PER_BUCKET * 2 - 1) /
        TYPED_HASHTABLE_TUPLES_PER_BUCKET * 2);

    for (i = 0; (1u << i) < num_buckets; ++i)
      ;  // do nothing

    num_buckets = 1 << i;

    /* prepare hash table data structures */
    ret->num_buckets = num_buckets;
    ret->mask = num_buckets - 1;
    ret->buckets = static_cast<hash_bucket_t *>(
        malloc(num_buckets * sizeof(hash_bucket_t)));
    if (!ret->buckets) {
      fprintf(stderr, "Error during hash table allocation.\n");
      exit(EXIT_FAILURE);
    }

    for (i = 0; i < num_buckets; i++) {
      ret->buckets[i].count = 0;
      ret->buckets[i].next = NULL;
    }

    return ret;
  }

  /**
   * Allocate a new hash bucket.
   *
   * @todo Avoid excessive memory allocation from OS.  Instead, get
   *       buckets in chunks from OS, then pass them on to the application
   *       one-by-one.
   */
  template <typename KeyType, typename PayloadType>
  typename TypedHashTable<KeyType, PayloadType>::hash_bucket_t *
  TypedHashTable<KeyType, PayloadType>::new_bucket(void) {
    hash_bucket_t *bucket;

    bucket = static_cast<hash_bucket_t *>(malloc(sizeof(*bucket)));
    if (!bucket) {
      fprintf(stderr, "Error allocating bucket!\n");
      exit(EXIT_FAILURE);
    }

    bucket->count = 0;
    bucket->next = NULL;

    return bucket;
  }

  template <typename KeyType, typename PayloadType>
  void TypedHashTable<KeyType, PayloadType>::put(const tuple_t &tuple) {
    HashValueType hash;
    hash_bucket_t *bucket;

    hash = hash_function(tuple.key) & hashtable->mask;

    bucket = &hashtable->buckets[hash];

    while (bucket->count == TYPED_HASHTABLE_TUPLES_PER_BUCKET) {
      if (bucket->next)
        bucket = bucket->next;
      else
        bucket = bucket->next = new_bucket();
    }

    bucket->tuples[bucket->count] = tuple;
    bucket->count++;
  }

  template <typename KeyType, typename PayloadType>
  void TypedHashTable<KeyType, PayloadType>::put_unique(const tuple_t &tuple) {
    HashValueType hash;
    hash_bucket_t *bucket;

    hash = hash_function(tuple.key) & hashtable->mask;

    bucket = &hashtable->buckets[hash];

    // check whether the element is not in the hash table
    bool found = false;
    while (bucket) {
      for (size_t j = 0; j < bucket->count; j++) {
        if (bucket->tuples[j].key == tuple.key) {
          found = true;
          break;
        }
      }
      bucket = bucket->next;
    }
    // if the tuple does not exist in the hash table, insert it,
    // otherwise, do nothing
    if (!found) {
      hashtable->put(tuple);
    }
  }
  template <typename KeyType, typename PayloadType>
  bool TypedHashTable<KeyType, PayloadType>::hash_get_first(const KeyType key,
                                                            PayloadType *val) {
    HashValueType hash;
    hash_bucket_t *bucket;

    hash = hash_function(key) & hashtable->mask;

    bucket = &hashtable->buckets[hash];

    while (bucket) {
      for (unsigned int i = 0; i < bucket->count; i++)
        if (bucket->tuples[i].key == key) {
          *val = bucket->tuples[i].payload;
          return true;
        }

      bucket = bucket->next;
    }

    return false;
  }

  template <typename KeyType, typename PayloadType>
  typename TypedHashTable<KeyType, PayloadType>::hash_bucket_t *
  TypedHashTable<KeyType, PayloadType>::getBucket(const KeyType &key) {
    HashValueType hash = hash_function(key) & hashtable->mask;
    return &hashtable->buckets[hash];
  }

  template <typename KeyType, typename PayloadType>
  void TypedHashTable<KeyType, PayloadType>::bucket_release(
      hash_bucket_t *bucket) {
    if (!bucket)
      return;
    else {
      bucket_release(bucket->next);
      free(bucket);
    }
  }

  template <typename KeyType, typename PayloadType>
  void TypedHashTable<KeyType, PayloadType>::hash_release(
      hashtable_t *hashtable) {
    /* release dynamically allocated buckets */
    for (unsigned long i = 0; i < hashtable->num_buckets; i++)
      bucket_release(hashtable->buckets[i].next);

    /* release all the top-level buckets */
    free(hashtable->buckets);
    free(hashtable);
  }

}  // end namespace CoGaDB

#endif /* HASHTABLE_HPP */
