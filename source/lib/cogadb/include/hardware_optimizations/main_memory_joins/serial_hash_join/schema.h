/**
 * @file
 *
 * Defines the basic "schema" of the tables that we operate on;
 * i.e., the data types of key and value columns.
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#ifndef SCHEMA_H
#define SCHEMA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * Types for key and value columns.
 *
 * @note We keep things simple here and assume the same schema for
 *       both input tables.
 */
// typedef uint32_t ht_key_t;
// typedef uint32_t ht_value_t;
typedef uint64_t ht_key_t;
typedef uint64_t ht_value_t;

/** A tuple consists of a key and a value */
struct tuple_t {
  ht_key_t key;     /**< This is the attribute that we join over. */
  ht_value_t value; /**< Payload; not actually touched by join algorithms. */
};
typedef struct tuple_t tuple_t;

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
typedef struct relation_t relation_t;

/**
 * Partitioned relation.  The entire array is represented as one
 * contiguous array of @a num_tuples elements, starting at
 * @a startaddrs[0].  The individual pointers in @a startaddrs point
 * to the start addresses of the @a num_part partitions.
 * @a startaddrs contains @a num_part + 1 entries; the last entry
 * points to the first array position @b after the relation.
 */
struct partitioned_relation_t {
  unsigned long num_tuples;
  unsigned long num_part;
  struct tuple_t **startaddrs;
};
typedef struct partitioned_relation_t partitioned_relation_t;

#ifdef __cplusplus
}
#endif

#endif /* SCHEMA_H */
