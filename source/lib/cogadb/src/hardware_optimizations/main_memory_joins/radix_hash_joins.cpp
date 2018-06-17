/**
 * @file radix_hash_joins.cpp
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

#include <stdlib.h> /* malloc, posix_memalign */

#include <backends/cpu/cpu_backend.hpp>
#include <core/base_column.hpp>
#include <core/column.hpp>
#include <hardware_optimizations/main_memory_joins/radix_hash_joins.hpp>
#include <hardware_optimizations/malloc.hpp>

#include <hardware_optimizations/main_memory_joins/parallel_radix/parallel_radix_join.h>
#include <hardware_optimizations/main_memory_joins/parallel_radix/prj_params.h> /* constant parameters */
#include <boost/thread/thread.hpp>

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

inline void* alloc_aligned(size_t size) {
  void* ret;
  int rv;
  rv = posix_memalign((void**)&ret, CACHE_LINE_SIZE_PRO, size);

  if (rv) {
    perror("alloc_aligned() failed: out of memory");
    return 0;
  }

  return ret;
}
#define MALLOC(SZ) \
  alloc_aligned(SZ + RELATION_PADDING_PRO) /*malloc(SZ+RELATION_PADDING)*/

#define FREE(X, SZ) free(X)
inline void delete_relation(relation_t_pro* rel) {
  FREE(rel->tuples, rel->num_tuples * sizeof(tuple_t));
}

/* DEBUG */
//#include <bitset>

/** \internal */

/* #define RADIX_HASH(V)  ((V>>7)^(V>>13)^(V>>21)^V) */

#define HASH_BIT_MODULO(K, MASK, NBITS) (((K)&MASK) >> NBITS)

#ifndef NEXT_POW_2
/**
 *  compute the next number, greater than or equal to 32-bit unsigned v.
 *  taken from "bit twiddling hacks":
 *  http://graphics.stanford.edu/~seander/bithacks.html
 */
#define NEXT_POW_2(V) \
  do {                \
    V--;              \
    V |= V >> 1;      \
    V |= V >> 2;      \
    V |= V >> 4;      \
    V |= V >> 8;      \
    V |= V >> 16;     \
    V++;              \
  } while (0)
#endif

namespace CoGaDB {
namespace CDK {
namespace main_memory_joins {

/**
* used as a hash function
*/
template <class T>
inline T radix_hash_joins<T>::hash_bit_modulo(const T K, const T MASK,
                                              const T NBITS) {
  return (((K)&MASK) >> NBITS);
}

/**
* Radix clustering algorithm which does not put padding in between
* clusters. This is used only by single threaded radix join
* implementation RJ.
*
* @param outRel
* @param inRel
* @param NBITS
* @param NUM_BITS_MASK
*/
template <class T>
void radix_hash_joins<T>::radix_cluster_nopadding(
    relation_t__<T>* outRel, relation_t__<T>* inRel, const unsigned int NBITS,
    const unsigned int NUM_BITS_MASK) {
  typedef tuple_t__<T> tuple_t__;
  typedef relation_t__<T> relation_t__;

  tuple_t__** dst;
  tuple_t__* input;
  /* tuple_t ** dst_end; */
  uint32_t* tuples_per_cluster;
  uint32_t i;
  uint32_t offset;
  const uint32_t MASK = ((1 << NUM_BITS_MASK) - 1) << NBITS;
  const uint32_t FANOUT = 1 << NUM_BITS_MASK;
  const uint32_t NTUPLES = inRel->num_tuples;

  tuples_per_cluster = (uint32_t*)calloc(FANOUT, sizeof(uint32_t));
  /* the following are fixed size when D is same for all the passes,
     and can be re-used from call to call. Allocating in this function
     just in case D differs from call to call. */
  dst = (tuple_t__**)malloc(sizeof(tuple_t__*) * FANOUT);

  input = inRel->tuples;
  /* count tuples per cluster */
  for (i = 0; i < NTUPLES; i++) {
    uint32_t idx = (uint32_t)(hash_bit_modulo(input->key, MASK, NBITS));
    tuples_per_cluster[idx]++;
    input++;
  }

  offset = 0;
  /* determine the start and end of each cluster depending on the counts. */
  for (i = 0; i < FANOUT; i++) {
    dst[i] = outRel->tuples + offset;
    offset += tuples_per_cluster[i];
  }

  input = inRel->tuples;
  /* copy tuples to their corresponding clusters at appropriate offsets */
  for (i = 0; i < NTUPLES; i++) {
    uint32_t idx = (uint32_t)(hash_bit_modulo(input->key, MASK, NBITS));
    *dst[idx] = *input;
    ++dst[idx];
    input++;
  }

  /* clean up temp */
  free(dst);
  free(tuples_per_cluster);
}

/**
 * This algorithm builds the hashtable using the bucket chaining idea and used
 * in PRO implementation. Join between given two relations is evaluated using
 * the "bucket chaining" algorithm proposed by Manegold et al. It is used after
 * the partitioning phase, which is common for all algorithms. Moreover, R and
 * S typically fit into L2 or at least R and |R|*sizeof(int) fits into L2 cache.
 *
 * @param R input relation R
 * @param S input relation S
 * @param join_tids output result join TIDs
 * @param NUMBER_OF_RADIX_BITS input
 */
template <class T>
void radix_hash_joins<T>::bucket_chaining_join(const relation_t__<T>* const R,
                                               const relation_t__<T>* const S,
                                               PositionListPairPtr join_tids,
                                               const int NUMBER_OF_RADIX_BITS) {
  typedef tuple_t__<T> tuple_t__;
  typedef relation_t__<T> relation_t__;

  int *next, *bucket;
  const uint32_t numR = R->num_tuples;
  uint32_t N = numR;

  NEXT_POW_2(N);
  /* N <<= 1; */
  const uint32_t MASK = (N - 1) << (NUMBER_OF_RADIX_BITS);

  next = (int*)malloc(sizeof(int) * numR);
  /* posix_memalign((void**)&next, CACHE_LINE_SIZE, numR * sizeof(int)); */
  bucket = (int*)calloc(N, sizeof(int));

  const tuple_t__* const Rtuples = R->tuples;
  for (uint32_t i = 0; i < numR;) {
    uint32_t idx =
        (uint32_t)hash_bit_modulo(R->tuples[i].key, MASK, NUMBER_OF_RADIX_BITS);
    next[i] = bucket[idx];
    bucket[idx] = ++i; /* we start pos's from 1 instead of 0 */
  }

  const tuple_t__* const Stuples = S->tuples;
  const uint32_t numS = S->num_tuples;

  /* PROBE- LOOP */
  for (uint32_t i = 0; i < numS; i++) {
    uint32_t idx =
        (uint32_t)hash_bit_modulo(Stuples[i].key, MASK, NUMBER_OF_RADIX_BITS);

    for (int hit = bucket[idx]; hit > 0; hit = next[hit - 1]) {
      if (Stuples[i].key == Rtuples[hit - 1].key) {
        join_tids->first->push_back(Rtuples[hit - 1].payload);
        join_tids->second->push_back(Stuples[i].payload);
      }
    }
  }
  /* PROBE-LOOP END  */

  /* clean up temp */
  free(bucket);
  free(next);
}

/**
* RJ: Radix Join.
*
* The "Radix Join" implementation denoted as RJ implements
* the single-threaded original multipass radix cluster join idea
* by Manegold et al.
*
* @param relR  input relation R - inner relation
* @param relS  input relation S - inner relation
* @param join_tids result join TIDs
* @param NUMBER_OF_PASSESs
 *@param NUMBER_OF_RADIX_BITS
*/
template <class T>
void radix_hash_joins<T>::RJ(relation_t__<T>* relR, relation_t__<T>* relS,
                             PositionListPairPtr join_tids,
                             const unsigned int NUMBER_OF_PASSES,
                             const unsigned int NUMBER_OF_RADIX_BITS) {
  typedef tuple_t__<T> tuple_t__;
  typedef relation_t__<T> relation_t__;

  unsigned int i;

  relation_t__ *outRelR, *outRelS;

  outRelR = (relation_t__*)malloc(sizeof(relation_t__));
  outRelS = (relation_t__*)malloc(sizeof(relation_t__));

  /* calculate RELATION_PADDING */
  /**some padding space is allocated for relations in order to
  *  avoid L1 conflict misses and PADDING_TUPLES is placed between
  *  partitions in pass-1 of partitioning and SMALL_PADDING_TUPLES is placed
  *  between partitions in pass-2 of partitioning. 3 is a magic number.
  */
  const int FANOUT_PASS1__ = (1 << (NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES));
  const int FANOUT_PASS2__ =
      (1 << (NUMBER_OF_RADIX_BITS - (NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES)));
  /**
   * Put an odd number of cache lines between partitions in pass-2:
   * Here we put 3 cache lines.
   */
  const int SMALL_PADDING_TUPLES__ =
      (3 * CACHE_LINE_SIZE__ / sizeof(tuple_t__));
  const int PADDING_TUPLES__ = (SMALL_PADDING_TUPLES__ * (FANOUT_PASS2__ + 1));
  /** @warning This padding must be allocated at the end of relation */
  const int RELATION_PADDING__ =
      (PADDING_TUPLES__ * FANOUT_PASS1__ * sizeof(tuple_t__));

  /* allocate temporary space for partitioning */
  /* TODO: padding problem */
  size_t sz = relR->num_tuples * sizeof(tuple_t__) + RELATION_PADDING__;
  outRelR->tuples = (tuple_t__*)malloc(sz);
  outRelR->num_tuples = relR->num_tuples;

  sz = relS->num_tuples * sizeof(tuple_t__) + RELATION_PADDING__;
  outRelS->tuples = (tuple_t__*)malloc(sz);
  outRelS->num_tuples = relS->num_tuples;

  /***** do the multi-pass partitioning *****/
  if (NUMBER_OF_PASSES == 1) {
    /* apply radix-clustering on relation R for pass-1 */
    radix_cluster_nopadding(outRelR, relR, 0, NUMBER_OF_RADIX_BITS);
    relR = outRelR;

    /* apply radix-clustering on relation S for pass-1 */
    radix_cluster_nopadding(outRelS, relS, 0, NUMBER_OF_RADIX_BITS);
    relS = outRelS;
  } else if (NUMBER_OF_PASSES == 2) {
    /* apply radix-clustering on relation R for pass-1 */
    radix_cluster_nopadding(outRelR, relR, 0,
                            NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES);

    /* apply radix-clustering on relation S for pass-1 */
    radix_cluster_nopadding(outRelS, relS, 0,
                            NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES);

    /* apply radix-clustering on relation R for pass-2 */
    radix_cluster_nopadding(
        relR, outRelR, NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES,
        NUMBER_OF_RADIX_BITS - (NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES));

    /* apply radix-clustering on relation S for pass-2 */
    radix_cluster_nopadding(
        relS, outRelS, NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES,
        NUMBER_OF_RADIX_BITS - (NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES));

    /* clean up temporary relations */
    free(outRelR->tuples);
    free(outRelS->tuples);
    free(outRelR);
    free(outRelS);
  } else if (NUMBER_OF_PASSES >= 3 && NUMBER_OF_PASSES % 2 == 1) {
    unsigned int i;
    const unsigned int MASK_SIZE = NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES;

    for (i = 0; i < NUMBER_OF_PASSES - 1; i += 2) {
      /* apply radix-clustering on relation R for pass-1 */
      radix_cluster_nopadding(outRelR, relR, (i)*MASK_SIZE, MASK_SIZE);

      /* apply radix-clustering on relation S for pass-1 */
      radix_cluster_nopadding(outRelS, relS, (i)*MASK_SIZE, MASK_SIZE);

      /* apply radix-clustering on relation R for pass-2 */
      radix_cluster_nopadding(relR, outRelR, (i + 1) * MASK_SIZE, MASK_SIZE);

      /* apply radix-clustering on relation S for pass-2 */
      radix_cluster_nopadding(relS, outRelS, (i + 1) * MASK_SIZE, MASK_SIZE);
    }

    /* apply radix-clustering on relation R for pass-3 */
    radix_cluster_nopadding(outRelR, relR, i * MASK_SIZE,
                            NUMBER_OF_RADIX_BITS - (i * MASK_SIZE));

    /* apply radix-clustering on relation S for pass-3 */
    radix_cluster_nopadding(outRelS, relS, i * MASK_SIZE,
                            NUMBER_OF_RADIX_BITS - (i * MASK_SIZE));

    relR = outRelR;
    relS = outRelS;
  } else if (NUMBER_OF_PASSES >= 4 && NUMBER_OF_PASSES % 2 == 0) {
    unsigned int i;
    const unsigned int MASK_SIZE = NUMBER_OF_RADIX_BITS / NUMBER_OF_PASSES;

    for (i = 0; i < NUMBER_OF_PASSES - 3; i += 2) {
      /* apply radix-clustering on relation R for pass-1 */
      radix_cluster_nopadding(outRelR, relR, (i)*MASK_SIZE, MASK_SIZE);

      /* apply radix-clustering on relation S for pass-1 */
      radix_cluster_nopadding(outRelS, relS, (i)*MASK_SIZE, MASK_SIZE);

      /* apply radix-clustering on relation R for pass-2 */
      radix_cluster_nopadding(relR, outRelR, (i + 1) * MASK_SIZE, MASK_SIZE);

      /* apply radix-clustering on relation S for pass-2 */
      radix_cluster_nopadding(relS, outRelS, (i + 1) * MASK_SIZE, MASK_SIZE);
    }

    /* apply radix-clustering on relation R for pass-1 */
    radix_cluster_nopadding(outRelR, relR, (i)*MASK_SIZE, MASK_SIZE);

    /* apply radix-clustering on relation S for pass-1 */
    radix_cluster_nopadding(outRelS, relS, (i)*MASK_SIZE, MASK_SIZE);

    /* apply radix-clustering on relation R for pass-3 */
    radix_cluster_nopadding(relR, outRelR, (i + 1) * MASK_SIZE,
                            NUMBER_OF_RADIX_BITS - ((i + 1) * MASK_SIZE));

    /* apply radix-clustering on relation S for pass-3 */
    radix_cluster_nopadding(relS, outRelS, (i + 1) * MASK_SIZE,
                            NUMBER_OF_RADIX_BITS - ((i + 1) * MASK_SIZE));

    /* clean up temporary relations */
    free(outRelR->tuples);
    free(outRelS->tuples);
    free(outRelR);
    free(outRelS);
  } else {
    COGADB_ERROR("The control flow shouldn't go in here...",
                 "The control flow shouldn't go in here...")
  }

  int* R_count_per_cluster =
      (int*)calloc((1 << NUMBER_OF_RADIX_BITS), sizeof(int));
  int* S_count_per_cluster =
      (int*)calloc((1 << NUMBER_OF_RADIX_BITS), sizeof(int));

  /* compute number of tuples per cluster */
  for (i = 0; i < relR->num_tuples; i++) {
    uint32_t idx = (relR->tuples[i].key) & ((1 << NUMBER_OF_RADIX_BITS) - 1);
    R_count_per_cluster[idx]++;
  }
  for (i = 0; i < relS->num_tuples; i++) {
    uint32_t idx = (relS->tuples[i].key) & ((1 << NUMBER_OF_RADIX_BITS) - 1);
    S_count_per_cluster[idx]++;
  }

  /* build hashtable on inner */
  int r, s; /* start index of next clusters */
  r = s = 0;
  for (i = 0; i < (1 << NUMBER_OF_RADIX_BITS); i++) {
    relation_t__ tmpR, tmpS;

    if (R_count_per_cluster[i] > 0 && S_count_per_cluster[i] > 0) {
      tmpR.num_tuples = R_count_per_cluster[i];
      tmpR.tuples = relR->tuples + r;
      r += R_count_per_cluster[i];

      tmpS.num_tuples = S_count_per_cluster[i];
      tmpS.tuples = relS->tuples + s;
      s += S_count_per_cluster[i];

      bucket_chaining_join(&tmpR, &tmpS, join_tids, NUMBER_OF_RADIX_BITS);
    } else {
      r += R_count_per_cluster[i];
      s += S_count_per_cluster[i];
    }
  }

  /* clean-up temporary buffers */
  free(S_count_per_cluster);
  free(R_count_per_cluster);

  if (NUMBER_OF_PASSES % 2 == 1) {
    /* clean up temporary relations */
    free(outRelR->tuples);
    free(outRelS->tuples);
    free(outRelR);
    free(outRelS);
  }
}

template <class T>
const PositionListPairPtr radix_hash_joins<T>::serial_radix_hash_join(
    T* build_relation, size_t br_size, T* probe_relation, size_t pr_size) {
  typedef tuple_t__<T> tuple_t__;
  typedef relation_t__<T> relation_t__;

  /* compute number of passes (STANDARD: 2 passes, 14 radix bits) */
  /** number of passes in multipass partitioning, currently fixed at 2. */
  const unsigned int NUMBER_OF_PASSES = 2;
  /** number of total radix bits used for partitioning. */
  const unsigned int NUMBER_OF_RADIX_BITS = 14;

  PositionListPairPtr join_tids(new PositionListPair());
  join_tids->first = createPositionList(0, hype::PD_Memory_0);
  join_tids->second = createPositionList(0, hype::PD_Memory_0);

  relation_t__ *build_rel, *probe_rel;

  build_rel = (relation_t__*)malloc(sizeof(relation_t__));
  probe_rel = (relation_t__*)malloc(sizeof(relation_t__));

  size_t sz = br_size * sizeof(tuple_t__);
  build_rel->tuples = (tuple_t__*)malloc(sz);
  build_rel->num_tuples = br_size;

  sz = pr_size * sizeof(tuple_t__);
  probe_rel->tuples = (tuple_t__*)malloc(sz);
  probe_rel->num_tuples = pr_size;

  for (size_t i = 0; i < br_size; i++) {
    (build_rel->tuples[i]).key = build_relation[i];
    (build_rel->tuples[i]).payload = i;
  }

  for (size_t i = 0; i < pr_size; i++) {
    (probe_rel->tuples[i]).key = probe_relation[i];
    (probe_rel->tuples[i]).payload = i;
  }

  RJ(build_rel, probe_rel, join_tids, NUMBER_OF_PASSES, NUMBER_OF_RADIX_BITS);

  free(build_rel->tuples);
  free(probe_rel->tuples);
  free(build_rel);
  free(probe_rel);

  return join_tids;
}

const PositionListPairPtr parallel_radix_hash_join(int* build_relation,
                                                   size_t br_size,
                                                   int* probe_relation,
                                                   size_t pr_size) {
  const unsigned int NUMBER_OF_THREADS = boost::thread::hardware_concurrency();

  relation_t_pro *build_rel, *probe_rel;

  build_rel = (relation_t_pro*)malloc(sizeof(relation_t_pro));
  probe_rel = (relation_t_pro*)malloc(sizeof(relation_t_pro));

  size_t sz = br_size * sizeof(tuple_t_pro);
  build_rel->tuples = (tuple_t_pro*)MALLOC(sz);
  build_rel->num_tuples = br_size;

  sz = pr_size * sizeof(tuple_t_pro);
  probe_rel->tuples = (tuple_t_pro*)MALLOC(sz);
  probe_rel->num_tuples = pr_size;

  for (size_t i = 0; i < br_size; i++) {
    (build_rel->tuples[i]).key = build_relation[i];
    (build_rel->tuples[i]).payload = i;
  }

  for (size_t i = 0; i < pr_size; i++) {
    (probe_rel->tuples[i]).key = probe_relation[i];
    (probe_rel->tuples[i]).payload = i;
  }

  std::vector<CoGaDB::PositionListPairPtr>* join_tid_lists =
      new std::vector<CoGaDB::PositionListPairPtr>();

  int64_t matches =
      PRO(build_rel, probe_rel, NUMBER_OF_THREADS, join_tid_lists);

  // merge tid lists
  // create result join tids
  PositionListPairPtr join_tids(new PositionListPair());
  join_tids->first = createPositionList(
      0, hype::PD_Memory_0);  // PositionListPtr(new PositionList());
  join_tids->second = createPositionList(
      0, hype::PD_Memory_0);  // PositionListPtr(new PositionList());

  // copy partial tids into final array
  for (size_t i = 0; i < join_tid_lists->size(); i++) {
    PositionList::iterator join_tid_fit1 =
        join_tid_lists->at(i)->first->begin();
    PositionList::iterator join_tid_fit2 = join_tid_lists->at(i)->first->end();
    join_tids->first->insert(join_tid_fit1, join_tid_fit2);
  }

  for (size_t i = 0; i < join_tid_lists->size(); i++) {
    PositionList::iterator join_tid_sit1 =
        join_tid_lists->at(i)->second->begin();
    PositionList::iterator join_tid_sit2 = join_tid_lists->at(i)->second->end();
    join_tids->second->insert(join_tid_sit1, join_tid_sit2);
  }

  delete_relation(build_rel);
  delete_relation(probe_rel);
  free(build_rel);
  free(probe_rel);
  delete join_tid_lists;

  //                return matches;
  return join_tids;
}

/** \endinternal */
template class radix_hash_joins<int32_t>;
template class radix_hash_joins<uint32_t>;
template class radix_hash_joins<uint64_t>;

}  // end namespace main_memory_joins
}  // end namespace CDK
}  // end namespace CoGaDB
