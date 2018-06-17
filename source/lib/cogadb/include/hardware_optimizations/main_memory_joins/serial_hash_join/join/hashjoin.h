/**
 * @file
 *
 * Declarations for hash join implementation
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#include "config.h"

#ifndef HASHJOIN_H
#define HASHJOIN_H

#include "join/partition.h"
#include "schema.h"

unsigned long hashjoin_impl(const relation_t R, const relation_t S);

unsigned long hashjoin(const relation_t R, const relation_t S,
                       const part_bits_t *part_bits);

#endif /* HASHJOIN_H */
