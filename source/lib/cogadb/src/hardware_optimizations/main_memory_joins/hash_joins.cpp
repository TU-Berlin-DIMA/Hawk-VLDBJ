

#include <boost/any.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>

#include <hardware_optimizations/main_memory_joins/hash_joins.hpp>
#include <util/time_measurement.hpp>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <getopt.h>

#include <ctype.h>

#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "datagen/datagen.h"
#include "hashtable/hashtable.h"
#include "join/hashjoin.h"
#include "join/partition.h"
#include "schema.h"
#include "time/time.h"

#ifdef __cplusplus
}
#endif

#include <backends/cpu/hashtable.hpp>
#include <core/column.hpp>

//#define VALIDATE_JOIN_RESULTS

namespace CoGaDB {

namespace CDK {

namespace main_memory_joins {

const PositionListPairPtr serial_hash_join(int* build_column, size_t br_size,
                                           int* probe_column, size_t pr_size) {
  PositionListPairPtr join_tids(new PositionListPair());
  join_tids->first = createPositionList(
      0, hype::PD_Memory_0);  // PositionListPtr(new PositionList());
  join_tids->second = createPositionList(
      0, hype::PD_Memory_0);  // PositionListPtr(new PositionList());

  hashtable_t* hashtable;

  /* Build phase */
  hashtable = hash_new(br_size);

  for (unsigned long r = 0; r < br_size; r++) {
    tuple_t t = {static_cast<ht_key_t>(build_column[r]), r};
    hash_put(hashtable, t);  // R.tuples[r]);
  }

  /* Join phase */
  for (unsigned long s = 0; s < pr_size; s++) {
    unsigned long hash = HASH(probe_column[s]) & hashtable->mask;
    hash_bucket_t* bucket = &hashtable->buckets[hash];

    while (bucket) {
      for (unsigned int i = 0; i < bucket->count; i++) {
        if (bucket->tuples[i].key == (uint64_t)probe_column[s]) {
          join_tids->first->push_back(bucket->tuples[i].value);
          join_tids->second->push_back(s);
        }
      }
      bucket = bucket->next;
    }
  }

  hash_release(hashtable);
  return join_tids;
}

//            void hash_join_pruning_thread(hashtable_t *hashtable, int*
//            probe_column, TID begin_index, TID end_index, PositionListPairPtr
//            join_tids) {
//
//                /* Join phase */
//                for (unsigned long s = begin_index; s < end_index; ++s) {
//                    unsigned long hash = HASH(probe_column[s]) &
//                    hashtable->mask;
//                    hash_bucket_t *bucket = &hashtable->buckets[hash];
//
//                    while (bucket) {
//                        for (unsigned int i = 0; i < bucket->count; i++) {
//                            if (bucket->tuples[i].key == probe_column[s]) {
//                                join_tids->first->push_back(bucket->tuples[i].value);
//                                join_tids->second->push_back(s);
//                            }
//                        }
//                        bucket = bucket->next;
//                    }
//                }
//
//            }

typedef TypedHashTable<TID, TID> TypedHashTable;
typedef boost::shared_ptr<TypedHashTable> HashTablePtr;

//[old_hash_table] void hash_join_pruning_thread(hashtable_t * hashtable, TID*
// probe_column, TID begin_index, TID end_index, PositionListPairPtr join_tids)
// {
void hash_join_pruning_thread(HashTablePtr hashtable, TID* probe_column,
                              TID begin_index, TID end_index,
                              PositionListPairPtr join_tids) {
  /* Join phase */
  for (unsigned long s = begin_index; s < end_index; ++s) {
    //[old_hash_table] unsigned long hash = HASH(probe_column[s]) &
    // hashtable->mask;
    TypedHashTable::hash_bucket_t* bucket =
        hashtable->getBucket(probe_column[s]);
    //[old_hash_table] hash_bucket_t *bucket = &(hashtable->buckets[hash]);

    while (bucket) {
      for (unsigned int i = 0; i < bucket->count; i++) {
        if (bucket->tuples[i].key == probe_column[s]) {
          //[old_hash_table]
          // join_tids->first->push_back(bucket->tuples[i].value);
          join_tids->first->push_back(bucket->tuples[i].payload);
          join_tids->second->push_back(s);
        }
      }
      bucket = bucket->next;
    }
  }
}

void write_partial_result_tids_to_output_thread(
    TID* join_tids_table1, TID* join_tids_table2, TID* join_tids_result_table1,
    TID* join_tids_result_table2, size_t thread_id, size_t number_of_threads,
    TID begin_index_result, TID end_index_result) {
  size_t result_size = end_index_result - begin_index_result;
  // copy memory chunk to result array
  std::memcpy(&join_tids_result_table1[begin_index_result], join_tids_table1,
              result_size * sizeof(TID));
  std::memcpy(&join_tids_result_table2[begin_index_result], join_tids_table2,
              result_size * sizeof(TID));
}

const PositionListPairPtr parallel_hash_join(TID* build_relation,
                                             size_t br_size,
                                             TID* probe_relation,
                                             size_t pr_size) {
  Timestamp build_hashtable_begin = getTimestamp();

  //[old_hash_table] hashtable_t *hashtable;
  HashTablePtr hashtable = boost::make_shared<TypedHashTable>(br_size);

  /* Build phase */
  //[old_hash_table] hashtable = hash_new(br_size);

  Timestamp build_hashtable_end = getTimestamp();

  for (TID r = 0; r < br_size; r++) {
    TypedHashTable::tuple_t t = {build_relation[r], r};
    //[old_hash_table]hash_put(hashtable, t); //R.tuples[r]);
    hashtable->put(t);
  }

  // probe larger relation
  Timestamp prune_hashtable_begin = getTimestamp();

  std::vector<PositionListPairPtr> partial_results;
  const size_t number_of_threads = omp_get_max_threads();
  for (size_t i = 0; i < number_of_threads; ++i) {
    PositionListPairPtr join_tids(new PositionListPair());
    join_tids->first = createPositionList(0, hype::PD_Memory_0);
    join_tids->second = createPositionList(0, hype::PD_Memory_0);
    partial_results.push_back(join_tids);
  }

  size_t thread_id, num_threads, partition_size;
  TID begin_index, end_index;
#pragma omp parallel default(shared) private( \
    thread_id, num_threads, partition_size, begin_index, end_index)
  {
    thread_id = omp_get_thread_num();
    num_threads = omp_get_num_threads();
    /* size of partition */
    partition_size = pr_size / num_threads;

    begin_index = thread_id * partition_size; /* starting array index */
    if (thread_id == num_threads - 1)         /* last thread may do more */
      partition_size = pr_size - begin_index;

    end_index = begin_index + partition_size;

    // call pruning thread
    hash_join_pruning_thread(hashtable, probe_relation, begin_index, end_index,
                             partial_results[thread_id]);
  }

  // compute offset position of partial results
  std::vector<size_t> thread_result_sizes(number_of_threads);
  std::vector<size_t> write_indexes(number_of_threads + 1);
  size_t total_result_size = 0;
  for (size_t i = 0; i < number_of_threads; i++) {
    thread_result_sizes[i] = partial_results[i]->first->size();
    // compute exclusive prefix sum!
    write_indexes[i] = total_result_size;
    total_result_size += thread_result_sizes[i];
  }
  write_indexes[number_of_threads] = total_result_size;

  // create result join tids
  PositionListPairPtr join_tids(new PositionListPair());
  join_tids->first = createPositionList(
      total_result_size,
      hype::PD_Memory_0);  // PositionListPtr(new PositionList());
  join_tids->second = createPositionList(
      total_result_size,
      hype::PD_Memory_0);  // PositionListPtr(new PositionList());

// copy partial tids into final array
#pragma omp parallel for
  for (size_t i = 0; i < number_of_threads; i++) {
    if (thread_result_sizes[i] > 0) {
      write_partial_result_tids_to_output_thread(
          partial_results[i]->first->data(), partial_results[i]->second->data(),
          join_tids->first->data(), join_tids->second->data(), i,
          number_of_threads, write_indexes[i], write_indexes[i + 1]);
    }
  }
  //[old_hash_table] hash_release(hashtable);
  return join_tids;
}

//            const PositionListPairPtr parallel_hash_join(int* build_relation,
//            size_t br_size, int* probe_relation, size_t pr_size) {
//
//                Timestamp build_hashtable_begin = getTimestamp();
//
//                hashtable_t *hashtable;
//
//                /* Build phase */
//                hashtable = hash_new(br_size);
//
//                Timestamp build_hashtable_end = getTimestamp();
//
//                for (TID r = 0; r < br_size; r++) {
//                    tuple_t t = {build_relation[r], r};
//                    hash_put(hashtable, t); //R.tuples[r]);
//                }
//
//                //probe larger relation
//                Timestamp prune_hashtable_begin = getTimestamp();
//
//                std::vector<PositionListPairPtr> partial_results;
//                const size_t number_of_threads = omp_get_max_threads();
//                for (size_t i = 0; i < number_of_threads; ++i) {
//                    PositionListPairPtr join_tids(new PositionListPair());
//                    join_tids->first = createPositionList(0,
//                    hype::PD_Memory_0);
//                    join_tids->second = createPositionList(0,
//                    hype::PD_Memory_0);
//                    partial_results.push_back(join_tids);
//                }
//
//                size_t thread_id, num_threads, partition_size;
//                TID begin_index, end_index;
//                #pragma omp parallel default(shared) private(thread_id,
//                num_threads, partition_size, begin_index, end_index)
//                {
//                    thread_id = omp_get_thread_num();
//                    num_threads = omp_get_num_threads();
//                    /* size of partition */
//                    partition_size = pr_size / num_threads;
//
//                    begin_index = thread_id * partition_size; /* starting
//                    array index */
//                    if (thread_id == num_threads - 1)
//                        /* last thread may do more */
//                        partition_size = pr_size - begin_index;
//
//                    end_index = begin_index + partition_size;
//
//                    //call pruning thread
//                    hash_join_pruning_thread(hashtable, probe_relation,
//                    begin_index, end_index, partial_results[thread_id]);
//                }
//
//                //compute offset position of partial results
//                std::vector<size_t> thread_result_sizes(number_of_threads);
//                std::vector<size_t> write_indexes(number_of_threads + 1);
//                size_t total_result_size = 0;
//                for (size_t i = 0; i < number_of_threads; i++) {
//                    thread_result_sizes[i] =
//                    partial_results[i]->first->size();
//                    //compute exclusive prefix sum!
//                    write_indexes[i] = total_result_size;
//                    total_result_size += thread_result_sizes[i];
//                }
//                write_indexes[number_of_threads] = total_result_size;
//
//                //create result join tids
//                PositionListPairPtr join_tids(new PositionListPair());
//                join_tids->first = createPositionList(total_result_size,
//                hype::PD_Memory_0); //PositionListPtr(new PositionList());
//                join_tids->second = createPositionList(total_result_size,
//                hype::PD_Memory_0); //PositionListPtr(new PositionList());
//
//                //copy partial tids into final array
//                #pragma omp parallel for
//                for (size_t i = 0; i < number_of_threads; i++) {
//                    if (thread_result_sizes[i] > 0) {
//                        write_partial_result_tids_to_output_thread(
//                                partial_results[i]->first->data(),
//                                partial_results[i]->second->data(),
//                                join_tids->first->data(),
//                                join_tids->second->data(),
//                                i, number_of_threads,
//                                write_indexes[i],
//                                write_indexes[i + 1]);
//                    }
//                }
//                hash_release(hashtable);
//                return join_tids;
//            }

}  // end namespace main_memory_joins
}  // end namespace CDK
}  // end namespace CoGaDB
