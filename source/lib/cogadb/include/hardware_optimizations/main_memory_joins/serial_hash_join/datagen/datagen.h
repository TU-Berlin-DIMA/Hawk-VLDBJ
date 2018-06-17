/**
 * @file
 *
 * Declarations for data generator
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#include "config.h"

#ifndef DATAGEN_H
#define DATAGEN_H

#include "schema.h"

relation_t gen_unique(unsigned long num_tuples);
relation_t gen_zipf(unsigned long num_tuples, unsigned long alphabet_size,
                    double zipfparam);

#endif /* DATAGEN_H */
