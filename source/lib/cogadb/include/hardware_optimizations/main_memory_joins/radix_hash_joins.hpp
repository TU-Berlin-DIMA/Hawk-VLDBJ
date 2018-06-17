/**
 * @file radix_hash_joins.hpp
 *
 * @brief several radix hash join implementations
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

#ifndef RADIX_JOIN_HPP
#define RADIX_JOIN_HPP

#include <stdint.h>
#include <core/base_column.hpp>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

namespace CoGaDB {
  namespace CDK {
    namespace main_memory_joins {

/** L1 cache parameters. \note Change as needed for different machines */
#if defined(HAVE_CONFIG_H) && defined(COGADB_L1_CACHELINE_SIZE)
      const unsigned int CACHE_LINE_SIZE__ = COGADB_L1_CACHELINE_SIZE;
#else
      const unsigned int CACHE_LINE_SIZE__ = 64;
#endif
      /** L1 cache size */
      //            const unsigned int L1_CACHE_SIZE__ = 32768;

      /** Type definition for a tuple, containing the join key and the TID */
      template <typename T>
      struct tuple_t__ {
        T key;
        TID payload;
      };

      /**
      * Type definition for a relation.
      * It consists of an array of tuples and a size of the relation.
      */
      template <typename T>
      struct relation_t__ {
        tuple_t__<T>* tuples;
        size_t num_tuples;
      };

      template <typename T>
      class radix_hash_joins {
       public:
        /**
        * Single threaded Radix Hash Join.
        *
        * The "Radix Join" implementation implements
        * the single-threaded original multipass radix cluster join idea
        * by Manegold et al.
        *
        * @param build_relation input
        * @param br_size size of the build_relation
        * @param probe_relation input
        * @param pr_size size of the probe_relation
        *
        * @return PositionLisPairPtr join result TIDs
        */
        static const PositionListPairPtr serial_radix_hash_join(
            T* build_relation, size_t br_size, T* probe_relation,
            size_t pr_size);

       private:
        static inline T hash_bit_modulo(const T K, const T MASK, const T NBITS);

        static void radix_cluster_nopadding(relation_t__<T>* outRel,
                                            relation_t__<T>* inRel,
                                            const unsigned int NBITS,
                                            const unsigned int NUM_BITS_MASK);

        static void bucket_chaining_join(const relation_t__<T>* const R,
                                         const relation_t__<T>* const S,
                                         PositionListPairPtr join_tids,
                                         const int NUMBER_OF_RADIX_BITS);

        static void RJ(relation_t__<T>* relR, relation_t__<T>* relS,
                       PositionListPairPtr join_tids,
                       const unsigned int NUMBER_OF_PASSES,
                       const unsigned int NUMBER_OF_RADIX_BITS);
      };

      const PositionListPairPtr parallel_radix_hash_join(int* build_relation,
                                                         size_t br_size,
                                                         int* probe_relation,
                                                         size_t pr_size);

    }  // end namespace main_memory_joins
  }    // end namespace CDK
}  // end namespace CoGaDB

#endif /* RADIX_JOIN_HPP */
