/**
 * @file
 *
 * Declarations for partitioning stage of partitioned hash join
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#include "config.h"

#ifndef PARTITION_H
#define PARTITION_H

#include "schema.h"

/**
 * Encoding of a partitioning range: bits @a from_bit to @a to_bit
 * are used for partitioning.
 */
struct part_bits_t {
  unsigned int from_bit;
  unsigned int to_bit;
  struct part_bits_t *next;
};
typedef struct part_bits_t part_bits_t;

/**
 * Refine the partitioning of a relation by considering another set of bits.
 */
partitioned_relation_t partition(const partitioned_relation_t rel,
                                 const unsigned int from_bit,
                                 const unsigned int to_bit);

#endif /* PARTITION_H */
