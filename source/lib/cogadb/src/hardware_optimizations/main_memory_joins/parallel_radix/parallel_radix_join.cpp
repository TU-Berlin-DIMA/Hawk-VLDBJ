/**
 * @file    parallel_radix_join.c
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Sun Feb 20:19:51 2012
 * @version $Id: parallel_radix_join.c 3017 2012-12-07 10:56:20Z bcagri $
 *
 * @brief  Provides implementations for several variants of Radix Hash Join.
 *
 * (c) 2012, ETH Zurich, Systems Group
 *
 * The following code is entirely based on the source code package
 * 'multicore-hashjoins-0.1.tar.gz' which is available online from
 * the website http://www.systems.ethz.ch/projects/paralleljoins.
 * The original author is Cagri Balkesen from ETH Zurich, Systems Group.
 *
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>   /* pthread_* */
#include <sched.h>     /* CPU_ZERO, CPU_SET */
#include <smmintrin.h> /* simd only for 32-bit keys – SSE4.1 */
#include <stdio.h>     /* printf */
#include <stdlib.h>    /* malloc, posix_memalign */
#include <sys/time.h>  /* gettimeofday */

#include "hardware_optimizations/main_memory_joins/parallel_radix/cpu_mapping.h" /* get_cpu_id */
#include "hardware_optimizations/main_memory_joins/parallel_radix/parallel_radix_join.h"
#include "hardware_optimizations/main_memory_joins/parallel_radix/prj_params.h" /* constant parameters */
#include "hardware_optimizations/main_memory_joins/parallel_radix/task_queue.h" /* task_queue_* */
//#include "hardware_optimizations/parallel_radix/generator.h"        /*
// numa_localize() */

#include <backends/cpu/cpu_backend.hpp>
#include <core/base_column.hpp>
#include <core/column.hpp>
#include <hardware_optimizations/malloc.hpp>

/** \internal */

#ifndef BARRIER_ARRIVE
/** barrier wait macro */
#define BARRIER_ARRIVE(B, RV)                           \
  RV = pthread_barrier_wait(B);                         \
  if (RV != 0 && RV != PTHREAD_BARRIER_SERIAL_THREAD) { \
    printf("Couldn't wait on barrier\n");               \
    exit(EXIT_FAILURE);                                 \
  }
#endif

/** checks malloc() result */
#ifndef MALLOC_CHECK
#define MALLOC_CHECK(M)                                            \
  if (!M) {                                                        \
    printf("[ERROR] MALLOC_CHECK: %s : %d\n", __FILE__, __LINE__); \
    perror(": malloc() failed!\n");                                \
    exit(EXIT_FAILURE);                                            \
  }
#endif

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

#ifndef MAX
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#endif

/** Debug msg logging method */
#ifdef DEBUG
#define DEBUGMSG(COND, MSG, ...)                    \
  if (COND) {                                       \
    fprintf(stdout, "[DEBUG] " MSG, ##__VA_ARGS__); \
  }
#else
#define DEBUGMSG(COND, MSG, ...)
#endif

/* just to enable compilation with g++ */
#if defined(__cplusplus)
#define restrict __restrict__
#endif

/** An experimental feature to allocate input relations numa-local */
extern int numalocalize; /* defined in generator.c */

typedef struct arg_t arg_t;
typedef struct part_t part_t;
typedef struct synctimer_t synctimer_t;
typedef int64_t (*JoinFunction)(const relation_t_pro *const,
                                const relation_t_pro *const,
                                relation_t_pro *const, arg_t *args);

/** holds the arguments passed to each thread */
struct arg_t {
  int32_t **histR;
  tuple_t_pro *relR;
  tuple_t_pro *tmpR;
  int32_t **histS;
  tuple_t_pro *relS;
  tuple_t_pro *tmpS;

  int32_t numR;
  int32_t numS;
  int32_t totalR;
  int32_t totalS;

  task_queue_t *join_queue;
  task_queue_t *part_queue;
#ifdef SKEW_HANDLING_PRO
  task_queue_t *skew_queue;
  task_t **skewtask;
#endif
  pthread_barrier_t *barrier;
  JoinFunction join_function;
  int64_t result;
  uint32_t my_tid;
  uint32_t nthreads;

  // MODIFIED
  CoGaDB::PositionListPairPtr join_tids;

  /* stats about the thread */
  //    int32_t        parts_processed;
  //    uint64_t       timer1, timer2, timer3;
  //    struct timeval start, end; //WARNING: is this attribute used?
} __attribute__((aligned(CACHE_LINE_SIZE_PRO)));

/** holds arguments passed for partitioning */
struct part_t {
  tuple_t_pro *rel;
  tuple_t_pro *tmp;
  int32_t **hist;
  int32_t *output;
  arg_t *thrargs;
  uint32_t num_tuples;
  uint32_t total_tuples;
  int32_t R;
  uint32_t D;
  int relidx; /* 0: R, 1: S */
  uint32_t padding;
} __attribute__((aligned(CACHE_LINE_SIZE_PRO)));

static void *alloc_aligned(size_t size) {
  void *ret;
  int rv;
  rv = posix_memalign((void **)&ret, CACHE_LINE_SIZE_PRO, size);

  if (rv) {
    perror("alloc_aligned() failed: out of memory");
    return 0;
  }

  return ret;
}

/** \endinternal */

/**
 * @defgroup Radix Radix Join Implementation Variants
 * @{
 */

/**
 *  This algorithm builds the hashtable using the bucket chaining idea and used
 *  in PRO implementation. Join between given two relations is evaluated using
 *  the "bucket chaining" algorithm proposed by Manegold et al. It is used after
 *  the partitioning phase, which is common for all algorithms. Moreover, R and
 *  S typically fit into L2 or at least R and |R|*sizeof(int) fits into L2
 *cache.
 *
 * @param R input relation R
 * @param S input relation S
 *
 * @return number of result tuples
 */
int64_t bucket_chaining_join(const relation_t_pro *const R,
                             const relation_t_pro *const S,
                             relation_t_pro *const tmpR, arg_t *args) {
  int *next, *bucket;
  const uint32_t numR = R->num_tuples;
  uint32_t N = numR;
  int64_t matches = 0;

  NEXT_POW_2(N);
  /* N <<= 1; */
  const uint32_t MASK = (N - 1) << (NUM_RADIX_BITS_PRO);

  next = (int *)malloc(sizeof(int) * numR);
  /* posix_memalign((void**)&next, CACHE_LINE_SIZE, numR * sizeof(int)); */
  bucket = (int *)calloc(N, sizeof(int));

  const tuple_t_pro *const Rtuples = R->tuples;
  for (uint32_t i = 0; i < numR;) {
    uint32_t idx = HASH_BIT_MODULO(R->tuples[i].key, MASK, NUM_RADIX_BITS_PRO);
    next[i] = bucket[idx];
    bucket[idx] = ++i; /* we start pos's from 1 instead of 0 */
  }

  const tuple_t_pro *const Stuples = S->tuples;
  const uint32_t numS = S->num_tuples;

  /* PROBE- LOOP */
  for (uint32_t i = 0; i < numS; i++) {
    uint32_t idx = HASH_BIT_MODULO(Stuples[i].key, MASK, NUM_RADIX_BITS_PRO);

    for (int hit = bucket[idx]; hit > 0; hit = next[hit - 1]) {
      if (Stuples[i].key == Rtuples[hit - 1].key) {
        /* TODO: copy to the result buffer, we skip it */
        matches++;
        // MODIFIED
        args->join_tids->first->push_back(Rtuples[hit - 1].payload);
        args->join_tids->second->push_back(Stuples[i].payload);
      }
    }
  }
  /* PROBE-LOOP END  */

  /* clean up temp */
  free(bucket);
  free(next);

  return matches;
}

/**
 * Radix clustering algorithm (originally described by Manegold et al)
 * The algorithm mimics the 2-pass radix clustering algorithm from
 * Kim et al. The difference is that it does not compute
 * prefix-sum, instead the sum (offset in the code) is computed iteratively.
 *
 * @warning This method puts padding between clusters, see
 * radix_cluster_nopadding for the one without padding.
 *
 * @param outRel [out] result of the partitioning
 * @param inRel [in] input relation
 * @param hist [out] number of tuples in each partition
 * @param R cluster bits
 * @param D radix bits per pass
 * @returns tuples per partition.
 */
void radix_cluster(relation_t_pro *restrict outRel,
                   relation_t_pro *restrict inRel, int32_t *restrict hist,
                   int R, int D) {
  uint32_t i;
  uint32_t M = ((1 << D) - 1) << R;
  uint32_t offset;
  uint32_t fanOut = 1 << D;

  /* the following are fixed size when D is same for all the passes,
     and can be re-used from call to call. Allocating in this function
     just in case D differs from call to call. */
  uint32_t dst[fanOut];

  /* count tuples per cluster */
  for (i = 0; i < inRel->num_tuples; i++) {
    uint32_t idx = HASH_BIT_MODULO(inRel->tuples[i].key, M, R);
    hist[idx]++;
  }
  offset = 0;
  /* determine the start and end of each cluster depending on the counts. */
  for (i = 0; i < fanOut; i++) {
    /* dst[i]      = outRel->tuples + offset; */
    /* determine the beginning of each partitioning by adding some
       padding to avoid L1 conflict misses during scatter. */
    dst[i] = offset + i * SMALL_PADDING_TUPLES_PRO;
    offset += hist[i];
  }

  /* copy tuples to their corresponding clusters at appropriate offsets */
  for (i = 0; i < inRel->num_tuples; i++) {
    uint32_t idx = HASH_BIT_MODULO(inRel->tuples[i].key, M, R);
    outRel->tuples[dst[idx]] = inRel->tuples[i];
    ++dst[idx];
  }
}

/**
 * This function implements the radix clustering of a given input
 * relations. The relations to be clustered are defined in task_t and after
 * clustering, each partition pair is added to the join_queue to be joined.
 *
 * @param task description of the relation to be partitioned
 * @param join_queue task queue to add join tasks after clustering
 */
void serial_radix_partition(task_t *const task, task_queue_t *join_queue,
                            const int R, const int D) {
  int i;
  uint32_t offsetR = 0, offsetS = 0;
  const int fanOut = 1 << D; /*(NUM_RADIX_BITS / NUM_PASSES);*/
  int32_t *outputR, *outputS;

  outputR = (int32_t *)calloc(fanOut + 1, sizeof(int32_t));
  outputS = (int32_t *)calloc(fanOut + 1, sizeof(int32_t));
  /* TODO: measure the effect of memset() */
  /* memset(outputR, 0, fanOut * sizeof(int32_t)); */
  radix_cluster(&task->tmpR, &task->relR, outputR, R, D);

  /* memset(outputS, 0, fanOut * sizeof(int32_t)); */
  radix_cluster(&task->tmpS, &task->relS, outputS, R, D);

  /* task_t t; */
  for (i = 0; i < fanOut; i++) {
    if (outputR[i] > 0 && outputS[i] > 0) {
      task_t *t = task_queue_get_slot_atomic(join_queue);
      t->relR.num_tuples = outputR[i];
      t->relR.tuples =
          task->tmpR.tuples + offsetR + i * SMALL_PADDING_TUPLES_PRO;
      t->tmpR.tuples =
          task->relR.tuples + offsetR + i * SMALL_PADDING_TUPLES_PRO;
      offsetR += outputR[i];

      t->relS.num_tuples = outputS[i];
      t->relS.tuples =
          task->tmpS.tuples + offsetS + i * SMALL_PADDING_TUPLES_PRO;
      t->tmpS.tuples =
          task->relS.tuples + offsetS + i * SMALL_PADDING_TUPLES_PRO;
      offsetS += outputS[i];

      /* task_queue_copy_atomic(join_queue, &t); */
      task_queue_add_atomic(join_queue, t);
    } else {
      offsetR += outputR[i];
      offsetS += outputS[i];
    }
  }
  free(outputR);
  free(outputS);
}

/**
 * This function implements the parallel radix partitioning of a given input
 * relation. Parallel partitioning is done by histogram-based relation
 * re-ordering as described by Kim et al. Parallel partitioning method is
 * commonly used by all parallel radix join algorithms.
 *
 * @param part description of the relation to be partitioned
 */
void parallel_radix_partition(part_t *const part) {
  const tuple_t_pro *restrict rel = part->rel;
  int32_t **hist = part->hist;
  int32_t *restrict output = part->output;

  const uint32_t my_tid = part->thrargs->my_tid;
  const uint32_t nthreads = part->thrargs->nthreads;
  const uint32_t num_tuples = part->num_tuples;

  const int32_t R = part->R;
  const int32_t D = part->D;
  const uint32_t fanOut = 1 << D;
  const uint32_t MASK = (fanOut - 1) << R;
  const uint32_t padding = part->padding;

  int32_t sum = 0;
  uint32_t i, j;
  int rv;

  int32_t dst[fanOut + 1];

  /* compute local histogram for the assigned region of rel */
  /* compute histogram */
  int32_t *my_hist = hist[my_tid];

  for (i = 0; i < num_tuples; i++) {
    uint32_t idx = HASH_BIT_MODULO(rel[i].key, MASK, R);
    my_hist[idx]++;
  }

  /* compute local prefix sum on hist */
  for (i = 0; i < fanOut; i++) {
    sum += my_hist[i];
    my_hist[i] = sum;
  }

  /* wait at a barrier until each thread complete histograms */
  BARRIER_ARRIVE(part->thrargs->barrier, rv);
  /* barrier global sync point-1 */

  /* determine the start and end of each cluster */
  for (i = 0; i < my_tid; i++) {
    for (j = 0; j < fanOut; j++) output[j] += hist[i][j];
  }
  for (i = my_tid; i < nthreads; i++) {
    for (j = 1; j < fanOut; j++) output[j] += hist[i][j - 1];
  }

  for (i = 0; i < fanOut; i++) {
    output[i] += i * padding;  // PADDING_TUPLES;
    dst[i] = output[i];
  }
  output[fanOut] = part->total_tuples + fanOut * padding;  // PADDING_TUPLES;

  tuple_t_pro *restrict tmp = part->tmp;

  /* Copy tuples to their corresponding clusters */
  for (i = 0; i < num_tuples; i++) {
    uint32_t idx = HASH_BIT_MODULO(rel[i].key, MASK, R);
    tmp[dst[idx]] = rel[i];
    ++dst[idx];
  }
}

/**
 * @defgroup SoftwareManagedBuffer Optimized Partitioning Using SW-buffers
 * @{
 */
typedef union {
  struct {
    tuple_t_pro tuples[CACHE_LINE_SIZE_PRO / sizeof(tuple_t_pro)];
  } tuples;
  struct {
    tuple_t_pro tuples[CACHE_LINE_SIZE_PRO / sizeof(tuple_t_pro) - 1];
    int32_t slot;
  } data;
} cacheline_t;

#define TUPLESPERCACHELINE (CACHE_LINE_SIZE_PRO / sizeof(tuple_t_pro))

/**
 * Makes a non-temporal write of 64 bytes from src to dst.
 * Uses vectorized non-temporal stores if available, falls
 * back to assignment copy.
 *
 * @param dst
 * @param src
 *
 * @return
 */
static inline void store_nontemp_64B(void *dst, void *src) {
#ifdef __AVX__
  register __m256i *d1 = (__m256i *)dst;
  register __m256i s1 = *((__m256i *)src);
  register __m256i *d2 = d1 + 1;
  register __m256i s2 = *(((__m256i *)src) + 1);

  _mm256_stream_si256(d1, s1);
  _mm256_stream_si256(d2, s2);

#elif defined(__SSE2__)

  register __m128i *d1 = (__m128i *)dst;
  register __m128i *d2 = d1 + 1;
  register __m128i *d3 = d1 + 2;
  register __m128i *d4 = d1 + 3;
  register __m128i s1 = *(__m128i *)src;
  register __m128i s2 = *((__m128i *)src + 1);
  register __m128i s3 = *((__m128i *)src + 2);
  register __m128i s4 = *((__m128i *)src + 3);

  _mm_stream_si128(d1, s1);
  _mm_stream_si128(d2, s2);
  _mm_stream_si128(d3, s3);
  _mm_stream_si128(d4, s4);

#else
  /* just copy with assignment */
  *(cacheline_t *)dst = *(cacheline_t *)src;

#endif
}

/**
 * This function implements the parallel radix partitioning of a given input
 * relation. Parallel partitioning is done by histogram-based relation
 * re-ordering as described by Kim et al. Parallel partitioning method is
 * commonly used by all parallel radix join algorithms. However this
 * implementation is further optimized to benefit from write-combining and
 * non-temporal writes.
 *
 * @param part description of the relation to be partitioned
 */
void parallel_radix_partition_optimized(part_t *const part) {
  const tuple_t_pro *restrict rel = part->rel;
  int32_t **hist = part->hist;
  int32_t *restrict output = part->output;

  const uint32_t my_tid = part->thrargs->my_tid;
  const uint32_t nthreads = part->thrargs->nthreads;
  const uint32_t num_tuples = part->num_tuples;

  const int32_t R = part->R;
  const int32_t D = part->D;
  const uint32_t fanOut = 1 << D;
  const uint32_t MASK = (fanOut - 1) << R;
  const uint32_t padding = part->padding;

  int32_t sum = 0;
  uint32_t i, j;
  int rv;

  /* compute local histogram for the assigned region of rel */
  /* compute histogram */
  int32_t *my_hist = hist[my_tid];

  for (i = 0; i < num_tuples; i++) {
    uint32_t idx = HASH_BIT_MODULO(rel[i].key, MASK, R);
    my_hist[idx]++;
  }

  /* compute local prefix sum on hist */
  for (i = 0; i < fanOut; i++) {
    sum += my_hist[i];
    my_hist[i] = sum;
  }

  /* wait at a barrier until each thread complete histograms */
  BARRIER_ARRIVE(part->thrargs->barrier, rv);

  /* determine the start and end of each cluster */
  for (i = 0; i < my_tid; i++) {
    for (j = 0; j < fanOut; j++) output[j] += hist[i][j];
  }
  for (i = my_tid; i < nthreads; i++) {
    for (j = 1; j < fanOut; j++) output[j] += hist[i][j - 1];
  }

  /* uint32_t pre; /\* nr of tuples to cache-alignment *\/ */
  tuple_t_pro *restrict tmp = part->tmp;
  /* software write-combining buffer */
  cacheline_t buffer[fanOut] __attribute__((aligned(CACHE_LINE_SIZE_PRO)));

  for (i = 0; i < fanOut; i++) {
    uint32_t off = output[i] + i * padding;
    /* pre        = (off + TUPLESPERCACHELINE) & ~(TUPLESPERCACHELINE-1); */
    /* pre       -= off; */
    output[i] = off;
    buffer[i].data.slot = off;
  }
  output[fanOut] = part->total_tuples + fanOut * padding;

  /* Copy tuples to their corresponding clusters */
  for (i = 0; i < num_tuples; i++) {
    uint32_t idx = HASH_BIT_MODULO(rel[i].key, MASK, R);
    uint32_t slot = buffer[idx].data.slot;
    tuple_t_pro *tup = (tuple_t_pro *)(buffer + idx);
    uint32_t slotMod = (slot) & (TUPLESPERCACHELINE - 1);
    tup[slotMod] = rel[i];

    if (slotMod == (TUPLESPERCACHELINE - 1)) {
      /* write out 64-Bytes with non-temporal store */
      store_nontemp_64B((tmp + slot - (TUPLESPERCACHELINE - 1)),
                        (buffer + idx));
      /* writes += TUPLESPERCACHELINE; */
    }

    buffer[idx].data.slot = slot + 1;
  }
  /* _mm_sfence (); */

  /* write out the remainders in the buffer */
  for (i = 0; i < fanOut; i++) {
    uint32_t slot = buffer[i].data.slot;
    uint32_t sz = (slot) & (TUPLESPERCACHELINE - 1);
    slot -= sz;
    for (uint32_t j = 0; j < sz; j++) {
      tmp[slot] = buffer[i].data.tuples[j];
      slot++;
    }
  }
}

/** @} */

/**
 * The main thread of parallel radix join. It does partitioning in parallel with
 * other threads and during the join phase, picks up join tasks from the task
 * queue and calls appropriate JoinFunction to compute the join task.
 *
 * @param param
 *
 * @return
 */
void *prj_thread(void *param) {
  arg_t *args = (arg_t *)param;
  int32_t my_tid = args->my_tid;

  const int fanOut = 1 << (NUM_RADIX_BITS_PRO / NUM_PASSES_PRO);
  const int R = (NUM_RADIX_BITS_PRO / NUM_PASSES_PRO);
  const int D = (NUM_RADIX_BITS_PRO - (NUM_RADIX_BITS_PRO / NUM_PASSES_PRO));
  const int thresh1 = MAX((1 << D), (1 << R)) * THRESHOLD1_PRO(args->nthreads);

  uint64_t results = 0;
  int i;
  int rv;

  part_t part;
  task_t *task;
  task_queue_t *part_queue;
  task_queue_t *join_queue;
#ifdef SKEW_HANDLING_PRO
  task_queue_t *skew_queue;
#endif

  int32_t *outputR = (int32_t *)calloc((fanOut + 1), sizeof(int32_t));
  int32_t *outputS = (int32_t *)calloc((fanOut + 1), sizeof(int32_t));
  MALLOC_CHECK((outputR && outputS));

  part_queue = args->part_queue;
  join_queue = args->join_queue;
#ifdef SKEW_HANDLING_PRO
  skew_queue = args->skew_queue;
#endif

  args->histR[my_tid] = (int32_t *)calloc(fanOut, sizeof(int32_t));
  args->histS[my_tid] = (int32_t *)calloc(fanOut, sizeof(int32_t));

  /* in the first pass, partitioning is done together by all threads */

  //    args->parts_processed = 0;

  /* wait at a barrier until each thread starts and then start the timer */
  BARRIER_ARRIVE(args->barrier, rv);

  /********** 1st pass of multi-pass partitioning ************/
  part.R = 0;
  part.D = NUM_RADIX_BITS_PRO / NUM_PASSES_PRO;
  part.thrargs = args;
  part.padding = PADDING_TUPLES_PRO;

  /* 1. partitioning for relation R */
  part.rel = args->relR;
  part.tmp = args->tmpR;
  part.hist = args->histR;
  part.output = outputR;
  part.num_tuples = args->numR;
  part.total_tuples = args->totalR;
  part.relidx = 0;

#ifdef USE_SWWC_OPTIMIZED_PART
  parallel_radix_partition_optimized(&part);
#else
  parallel_radix_partition(&part);
#endif

  /* 2. partitioning for relation S */
  part.rel = args->relS;
  part.tmp = args->tmpS;
  part.hist = args->histS;
  part.output = outputS;
  part.num_tuples = args->numS;
  part.total_tuples = args->totalS;
  part.relidx = 1;

#ifdef USE_SWWC_OPTIMIZED_PART
  parallel_radix_partition_optimized(&part);
#else
  parallel_radix_partition(&part);
#endif

  /* wait at a barrier until each thread copies out */
  BARRIER_ARRIVE(args->barrier, rv);

  /********** end of 1st partitioning phase ******************/

  /* 3. first thread creates partitioning tasks for 2nd pass */
  if (my_tid == 0) {
    for (i = 0; i < fanOut; i++) {
      int32_t ntupR = outputR[i + 1] - outputR[i] - PADDING_TUPLES_PRO;
      int32_t ntupS = outputS[i + 1] - outputS[i] - PADDING_TUPLES_PRO;

#ifdef SKEW_HANDLING_PRO
      if (ntupR > thresh1 || ntupS > thresh1) {
        DEBUGMSG(1, "Adding to skew_queue= R:%d, S:%d\n", ntupR, ntupS);

        task_t *t = task_queue_get_slot(skew_queue);

        t->relR.num_tuples = t->tmpR.num_tuples = ntupR;
        t->relR.tuples = args->tmpR + outputR[i];
        t->tmpR.tuples = args->relR + outputR[i];

        t->relS.num_tuples = t->tmpS.num_tuples = ntupS;
        t->relS.tuples = args->tmpS + outputS[i];
        t->tmpS.tuples = args->relS + outputS[i];

        task_queue_add(skew_queue, t);
      } else
#endif
          if (ntupR > 0 && ntupS > 0) {
        task_t *t = task_queue_get_slot(part_queue);

        t->relR.num_tuples = t->tmpR.num_tuples = ntupR;
        t->relR.tuples = args->tmpR + outputR[i];
        t->tmpR.tuples = args->relR + outputR[i];

        t->relS.num_tuples = t->tmpS.num_tuples = ntupS;
        t->relS.tuples = args->tmpS + outputS[i];
        t->tmpS.tuples = args->relS + outputS[i];

        task_queue_add(part_queue, t);
      }
    }

    /* debug partitioning task queue */
    DEBUGMSG(1, "Pass-2: # partitioning tasks = %d\n", part_queue->count);
  }

  /* wait at a barrier until first thread adds all partitioning tasks */
  BARRIER_ARRIVE(args->barrier, rv);

/************ 2nd pass of multi-pass partitioning ********************/
/* 4. now each thread further partitions and add to join task queue **/

#if NUM_PASSES_PRO == 1
  /* If the partitioning is single pass we directly add tasks from pass-1 */
  task_queue_t *swap = join_queue;
  join_queue = part_queue;
  /* part_queue is used as a temporary queue for handling skewed parts */
  part_queue = swap;

#elif NUM_PASSES_PRO == 2

  while ((task = task_queue_get_atomic(part_queue))) {
    serial_radix_partition(task, join_queue, R, D);
  }

#else
#warning Only 2-pass partitioning is implemented, set NUM_PASSES to 2!
#endif

#ifdef SKEW_HANDLING_PRO
  /* Partitioning pass-2 for skewed relations */
  part.R = R;
  part.D = D;
  part.thrargs = args;
  part.padding = SMALL_PADDING_TUPLES_PRO;

  while (1) {
    if (my_tid == 0) {
      *args->skewtask = task_queue_get_atomic(skew_queue);
    }
    BARRIER_ARRIVE(args->barrier, rv);
    if (*args->skewtask == NULL) break;

    DEBUGMSG((my_tid == 0), "Got skew task = R: %d, S: %d\n",
             (*args->skewtask)->relR.num_tuples,
             (*args->skewtask)->relS.num_tuples);

    int32_t numperthr = (*args->skewtask)->relR.num_tuples / args->nthreads;
    const int fanOut2 = (1 << D);

    free(outputR);
    free(outputS);

    outputR = (int32_t *)calloc(fanOut2 + 1, sizeof(int32_t));
    outputS = (int32_t *)calloc(fanOut2 + 1, sizeof(int32_t));

    free(args->histR[my_tid]);
    free(args->histS[my_tid]);

    args->histR[my_tid] = (int32_t *)calloc(fanOut2, sizeof(int32_t));
    args->histS[my_tid] = (int32_t *)calloc(fanOut2, sizeof(int32_t));

    /* wait until each thread allocates memory */
    BARRIER_ARRIVE(args->barrier, rv);

    /* 1. partitioning for relation R */
    part.rel = (*args->skewtask)->relR.tuples + my_tid * numperthr;
    part.tmp = (*args->skewtask)->tmpR.tuples;
    part.hist = args->histR;
    part.output = outputR;
    part.num_tuples =
        (my_tid == (args->nthreads - 1))
            ? ((*args->skewtask)->relR.num_tuples - my_tid * numperthr)
            : numperthr;
    part.total_tuples = (*args->skewtask)->relR.num_tuples;
    part.relidx = 2; /* meaning this is pass-2, no syncstats */
    parallel_radix_partition(&part);

    numperthr = (*args->skewtask)->relS.num_tuples / args->nthreads;
    /* 2. partitioning for relation S */
    part.rel = (*args->skewtask)->relS.tuples + my_tid * numperthr;
    part.tmp = (*args->skewtask)->tmpS.tuples;
    part.hist = args->histS;
    part.output = outputS;
    part.num_tuples =
        (my_tid == (args->nthreads - 1))
            ? ((*args->skewtask)->relS.num_tuples - my_tid * numperthr)
            : numperthr;
    part.total_tuples = (*args->skewtask)->relS.num_tuples;
    part.relidx = 2; /* meaning this is pass-2, no syncstats */
    parallel_radix_partition(&part);

    /* wait at a barrier until each thread copies out */
    BARRIER_ARRIVE(args->barrier, rv);

    /* first thread adds join tasks */
    if (my_tid == 0) {
      const int THR1 = THRESHOLD1_PRO(args->nthreads);

      for (i = 0; i < fanOut2; i++) {
        int32_t ntupR = outputR[i + 1] - outputR[i] - SMALL_PADDING_TUPLES_PRO;
        int32_t ntupS = outputS[i + 1] - outputS[i] - SMALL_PADDING_TUPLES_PRO;
        if (ntupR > THR1 || ntupS > THR1) {
          DEBUGMSG(1, "Large join task = R: %d, S: %d\n", ntupR, ntupS);

          /* use part_queue temporarily */
          for (int k = 0; k < args->nthreads; k++) {
            int ns = (k == args->nthreads - 1)
                         ? (ntupS - k * (ntupS / args->nthreads))
                         : (ntupS / args->nthreads);
            task_t *t = task_queue_get_slot(part_queue);

            t->relR.num_tuples = t->tmpR.num_tuples = ntupR;
            t->relR.tuples = (*args->skewtask)->tmpR.tuples + outputR[i];
            t->tmpR.tuples = (*args->skewtask)->relR.tuples + outputR[i];

            t->relS.num_tuples = t->tmpS.num_tuples = ns;  // ntupS;
            t->relS.tuples = (*args->skewtask)->tmpS.tuples + outputS[i]  //;
                             + k * (ntupS / args->nthreads);
            t->tmpS.tuples = (*args->skewtask)->relS.tuples + outputS[i]  //;
                             + k * (ntupS / args->nthreads);

            task_queue_add(part_queue, t);
          }
        } else if (ntupR > 0 && ntupS > 0) {
          task_t *t = task_queue_get_slot(join_queue);

          t->relR.num_tuples = t->tmpR.num_tuples = ntupR;
          t->relR.tuples = (*args->skewtask)->tmpR.tuples + outputR[i];
          t->tmpR.tuples = (*args->skewtask)->relR.tuples + outputR[i];

          t->relS.num_tuples = t->tmpS.num_tuples = ntupS;
          t->relS.tuples = (*args->skewtask)->tmpS.tuples + outputS[i];
          t->tmpS.tuples = (*args->skewtask)->relS.tuples + outputS[i];

          task_queue_add(join_queue, t);

          DEBUGMSG(1, "Join added = R: %d, S: %d\n", t->relR.num_tuples,
                   t->relS.num_tuples);
        }
      }
    }
  }

  /* add large join tasks in part_queue to the front of the join queue */
  if (my_tid == 0) {
    while ((task = task_queue_get_atomic(part_queue)))
      task_queue_add(join_queue, task);
  }

#endif

  free(outputR);
  free(outputS);

  /* wait at a barrier until all threads add all join tasks */
  BARRIER_ARRIVE(args->barrier, rv)

  DEBUGMSG((my_tid == 0), "Number of join tasks = %d\n", join_queue->count);

  while ((task = task_queue_get_atomic(join_queue))) {
    /* do the actual join. join method differs for different algorithms,
       i.e. bucket chaining, histogram-based, histogram-based with simd &
       prefetching  */
    results += args->join_function(&task->relR, &task->relS, &task->tmpR, args);

    //        args->parts_processed ++;
  }

  args->result = results;

  return 0;
}

/**
 * The template function for different joins: Basically each parallel radix join
 * has a initialization step, partitioning step and build-probe steps. All our
 * parallel radix implementations have exactly the same initialization and
 * partitioning steps. Difference is only in the build-probe step. Here are all
 * the parallel radix join implemetations and their Join (build-probe)
 *functions:
 *
 * - PRO,  Parallel Radix Join Optimized --> bucket_chaining_join()
 */
int64_t join_init_run(
    relation_t_pro *relR, relation_t_pro *relS, JoinFunction jf,
    unsigned int nthreads,
    std::vector<CoGaDB::PositionListPairPtr> *join_tid_lists) {
  int rv;
  std::vector<pthread_t> tid(nthreads);
  pthread_attr_t attr;
  pthread_barrier_t barrier;
  cpu_set_t set;
  std::vector<arg_t> args(nthreads);

  int32_t **histR, **histS;
  tuple_t_pro *tmpRelR, *tmpRelS;
  uint32_t numperthr[2];
  int64_t result = 0;

  task_queue_t *part_queue, *join_queue;
#ifdef SKEW_HANDLING_PRO
  task_queue_t *skew_queue;
  task_t *skewtask = NULL;
  skew_queue = task_queue_init(FANOUT_PASS1_PRO);
#endif
  part_queue = task_queue_init(FANOUT_PASS1_PRO);
  join_queue = task_queue_init((1 << NUM_RADIX_BITS_PRO));

  /* allocate temporary space for partitioning */
  tmpRelR = static_cast<tuple_t_pro *>(alloc_aligned(
      relR->num_tuples * sizeof(tuple_t_pro) + RELATION_PADDING_PRO));
  tmpRelS = static_cast<tuple_t_pro *>(alloc_aligned(
      relS->num_tuples * sizeof(tuple_t_pro) + RELATION_PADDING_PRO));
  MALLOC_CHECK((tmpRelR && tmpRelS));
  /** Not an elegant way of passing whether we will numa-localize, but this
      feature is experimental anyway. */
  //    if(numalocalize) { //WARNING: uncommented
  //        numa_localize(tmpRelR, relR->num_tuples, nthreads);
  //        numa_localize(tmpRelS, relS->num_tuples, nthreads);
  //    }

  /* allocate histograms arrays, actual allocation is local to threads */
  histR = static_cast<int32_t **>(alloc_aligned(nthreads * sizeof(int32_t *)));
  histS = static_cast<int32_t **>(alloc_aligned(nthreads * sizeof(int32_t *)));
  MALLOC_CHECK((histR && histS));

  rv = pthread_barrier_init(&barrier, NULL, nthreads);
  if (rv != 0) {
    printf("[ERROR] Couldn't create the barrier\n");
    exit(EXIT_FAILURE);
  }

  pthread_attr_init(&attr);

  /* first assign chunks of relR & relS for each thread */
  numperthr[0] = relR->num_tuples / nthreads;
  numperthr[1] = relS->num_tuples / nthreads;
  for (auto i = 0u; i < nthreads; i++) {
    auto cpu_idx = get_cpu_id(i);

    DEBUGMSG(1, "Assigning thread-%d to CPU-%d\n", i, cpu_idx);

    CPU_ZERO(&set);
    CPU_SET(cpu_idx, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

    args[i].relR = relR->tuples + i * numperthr[0];
    args[i].tmpR = tmpRelR;
    args[i].histR = histR;

    args[i].relS = relS->tuples + i * numperthr[1];
    args[i].tmpS = tmpRelS;
    args[i].histS = histS;

    args[i].numR = static_cast<int32_t>(
        (i == (nthreads - 1)) ? (relR->num_tuples - i * numperthr[0])
                              : numperthr[0]);
    args[i].numS = static_cast<int32_t>(
        (i == (nthreads - 1)) ? (relS->num_tuples - i * numperthr[1])
                              : numperthr[1]);
    args[i].totalR = static_cast<int32_t>(relR->num_tuples);
    args[i].totalS = static_cast<int32_t>(relS->num_tuples);

    args[i].my_tid = i;
    args[i].part_queue = part_queue;
    args[i].join_queue = join_queue;
#ifdef SKEW_HANDLING_PRO
    args[i].skew_queue = skew_queue;
    args[i].skewtask = &skewtask;
#endif
    args[i].barrier = &barrier;
    args[i].join_function = jf;
    args[i].nthreads = nthreads;

    // MODIFIED
    args[i].join_tids =
        CoGaDB::PositionListPairPtr(new CoGaDB::PositionListPair());
    args[i].join_tids->first = CoGaDB::createPositionList(0, hype::PD_Memory_0);
    args[i].join_tids->second =
        CoGaDB::createPositionList(0, hype::PD_Memory_0);

    rv = pthread_create(&tid[i], &attr, prj_thread, (void *)&args[i]);
    if (rv) {
      printf("[ERROR] return code from pthread_create() is %d\n", rv);
      exit(-1);
    }
  }

  /* wait for threads to finish */
  for (auto i = 0u; i < nthreads; ++i) {
    pthread_join(tid[i], NULL);
    result += args[i].result;

    // MODIFIED
    join_tid_lists->push_back(args[i].join_tids);
  }

  /* clean up */
  for (auto i = 0u; i < nthreads; ++i) {
    free(histR[i]);
    free(histS[i]);
  }
  free(histR);
  free(histS);
  task_queue_free(part_queue);
  task_queue_free(join_queue);
#ifdef SKEW_HANDLING_PRO
  task_queue_free(skew_queue);
#endif
  free(tmpRelR);
  free(tmpRelS);

  return result;
}

/** \copydoc PRO */
int64_t PRO(relation_t_pro *relR, relation_t_pro *relS, int nthreads,
            std::vector<CoGaDB::PositionListPairPtr> *join_tid_lists) {
  return join_init_run(relR, relS, bucket_chaining_join, nthreads,
                       join_tid_lists);
}

/** @} */
