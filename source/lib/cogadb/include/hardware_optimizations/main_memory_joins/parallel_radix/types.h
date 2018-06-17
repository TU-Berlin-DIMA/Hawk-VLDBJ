/**
 * @file    types.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Tue May 22 16:43:30 2012
 * @version $Id: types.h 3017 2012-12-07 10:56:20Z bcagri $
 *
 * @brief  Provides general type definitions used by all join algorithms.
 *
 * The following code is entirely based on the source code package
 * 'multicore-hashjoins-0.1.tar.gz' which is available online from
 * the website http://www.systems.ethz.ch/projects/paralleljoins.
 * The original author is Cagri Balkesen from ETH Zurich, Systems Group.
 *
 */
#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <core/base_column.hpp>

/**
 * @defgroup Types Common Types
 * Common type definitions used by all join implementations.
 * @{
 */

typedef struct tuple_t_pro tuple_t_pro;
typedef struct relation_t_pro relation_t_pro;

// MODIFIED
/** Type definition for a tuple, depending on KEY_8B a tuple can be 16B or 8B */
struct tuple_t_pro {
  int key;
  unsigned int
      payload;  // WARNING: should be TID, but results in slow perfomrance
};

/**
 * Type definition for a relation.
 * It consists of an array of tuples and a size of the relation.
 */
struct relation_t_pro {
  tuple_t_pro* tuples;
  uint32_t num_tuples;
};

/** @} */

#endif /* TYPES_H */
