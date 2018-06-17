/**
 * @file parallel_radix_join.h
 *
 * @brief Provides interface for PRO: Parallel Radix Join Optimized.
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

#ifndef PARALLEL_RADIX_JOIN_H
#define PARALLEL_RADIX_JOIN_H

#include "hardware_optimizations/main_memory_joins/parallel_radix/types.h" /* relation_t */

/**
 * PRO: Parallel Radix Join Optimized.
 *
 * The "Parallel Radix Join Optimized" implementation denoted as PRO implements
 * the parallel radix join idea by Kim et al. with further optimizations. Mainly
 * it uses the bucket chaining for the build phase instead of the
 * histogram-based relation re-ordering and performs better than other
 * variations such as PRHO, which applies SIMD and prefetching
 * optimizations.

 * @param relR  input relation R - inner relation
 * @param relS  input relation S - inner relation
 *
 * @return number of result tuples
 */
int64_t PRO(relation_t_pro* relR, relation_t_pro* relS, int nthreads,
            std::vector<CoGaDB::PositionListPairPtr>* join_tid_lists);

#endif /* PARALLEL_RADIX_JOIN_H */
