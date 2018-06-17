/**
 * @file prj_params.h
 *
 * @brief Constant parameters used by Parallel Radix Join implementations.
 *
 * @author Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * (c) 2012, ETH Zurich, Systems Group
 *
 * @author Stefan Noll <stefan.noll@cs.tu-dortmund.de>
 *
 * The following code is entirely based on the source code package
 * 'multicore-hashjoins-0.1.tar.gz' which is available online from
 * the website http://www.systems.ethz.ch/projects/paralleljoins.
 * The original author is Cagri Balkesen from ETH Zurich, Systems Group.
 *
 * Some small adjustments were made to the "PRO: Parallel Radix Join Optimized"
 * algorithm in order to test the join algorithm in CoGaDB. All other algorithms
 * from the source code packages were removed.
 * These changes were made by Stefan Noll, TU Dortmund.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef PRJ_PARAMS_H
#define PRJ_PARAMS_H

/** number of total radix bits used for partitioning. */
#ifndef NUM_RADIX_BITS_PRO
#define NUM_RADIX_BITS_PRO 14
#endif

/** number of passes in multipass partitioning, currently fixed at 2. */
#ifndef NUM_PASSES_PRO
#define NUM_PASSES_PRO 2
#endif

/**
 * Whether to use software write-combining optimized partitioning,
 * see --enable-optimized-part config option
 */
/* #define USE_SWWC_OPTIMIZED_PART 1 */

/** @defgroup SystemParameters System Parameters
 *  Various system specific parameters such as cache/cache-line sizes,
 *  associativity, etc.
 *  @{
 */

/** L1 cache parameters. \note Change as needed for different machines */
#if defined(HAVE_CONFIG_H) && defined(COGADB_L1_CACHELINE_SIZE)
#define CACHE_LINE_SIZE_PRO COGADB_L1_CACHELINE_SIZE
#else
#define CACHE_LINE_SIZE_PRO 64
#endif

/** L1 cache size */
#if defined(HAVE_CONFIG_H) && defined(COGADB_L1_CACHE_SIZE)
#define L1_CACHE_SIZE_PRO COGADB_L1_CACHE_SIZE
#else
#define L1_CACHE_SIZE_PRO 32768
#endif

/** L1 associativity */
#if defined(HAVE_CONFIG_H) && defined(COGADB_L1_CACHE_ASSOCIATIVITY)
#define L1_ASSOCIATIVITY_PRO COGADB_L1_CACHE_ASSOCIATIVITY
#else
#define L1_ASSOCIATIVITY_PRO 8
#endif

/** number of tuples fitting into L1 */
#define L1_CACHE_TUPLES_PRO (L1_CACHE_SIZE_PRO / sizeof(tuple_t_pro))

/** thresholds for skewed partitions in 3-phase parallel join */
#ifndef SKEW_HANDLING_PRO
#define SKEW_HANDLING_PRO 0
#endif
#define THRESHOLD1_PRO(NTHR) (NTHR * L1_CACHE_TUPLES_PRO)
#define THRESHOLD2_PRO(NTHR) (NTHR * NTHR * L1_CACHE_TUPLES_PRO)

/** }*/

/** \internal some padding space is allocated for relations in order to
 *  avoid L1 conflict misses and PADDING_TUPLES is placed between
 *  partitions in pass-1 of partitioning and SMALL_PADDING_TUPLES is placed
 *  between partitions in pass-2 of partitioning. 3 is a magic number.
 */

/* num-parts at pass-1 */
#define FANOUT_PASS1_PRO (1 << (NUM_RADIX_BITS_PRO / NUM_PASSES_PRO))
/* num-parts at pass-1 */
#define FANOUT_PASS2_PRO \
  (1 << (NUM_RADIX_BITS_PRO - (NUM_RADIX_BITS_PRO / NUM_PASSES_PRO)))

/**
 * Put an odd number of cache lines between partitions in pass-2:
 * Here we put 3 cache lines.
 */
#define SMALL_PADDING_TUPLES_PRO (3 * CACHE_LINE_SIZE_PRO / sizeof(tuple_t_pro))
#define PADDING_TUPLES_PRO (SMALL_PADDING_TUPLES_PRO * (FANOUT_PASS2_PRO + 1))

/** @warning This padding must be allocated at the end of relation */
#define RELATION_PADDING_PRO \
  (PADDING_TUPLES_PRO * FANOUT_PASS1_PRO * sizeof(tuple_t_pro))

/** \endinternal */

#endif /* PRJ_PARAMS_H */
