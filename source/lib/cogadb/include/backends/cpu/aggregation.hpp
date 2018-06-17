/*
 * File:   aggregation.hpp
 * Author: sebastian
 *
 * Created on 10. Mai 2015, 11:06
 */

#ifndef AGGREGATION_HPP
#define AGGREGATION_HPP

#include <omp.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <typeinfo>
#include <utility>

#include <boost/make_shared.hpp>
#include <boost/unordered_map.hpp>

#include <compression/dictionary_compressed_column.hpp>
#include <core/column.hpp>
#include <util/column_grouping_keys.hpp>
#include <util/getname.hpp>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <backends/cpu/hashtable.hpp>

namespace CoGaDB {

  struct AggregationPayload {
    TID tid;
    double value;
  };

  inline std::pair<ColumnPtr, ColumnPtr> hash_aggregation_sum(
      ColumnGroupingKeysPtr grouping_keys, const double* values,
      size_t num_elements) {
    assert(grouping_keys->keys->size() == num_elements);

    typedef GroupingKeys::value_type GroupingKeysType;

    GroupingKeysType* keys = grouping_keys->keys->data();

    typedef TypedHashTable<GroupingKeysType, AggregationPayload> TypedHashTable;
    typedef typename TypedHashTable::hash_bucket_t hash_bucket_t;
    typedef TypedHashTable::tuple_t tuple_t;

    TypedHashTable hashtable(num_elements);

    /* Build phase */
    Timestamp begin = getTimestamp();

    size_t number_of_rows = num_elements;
    for (size_t i = 0; i < number_of_rows; ++i) {
      hash_bucket_t* bucket = hashtable.getBucket(keys[i]);

      bool found = false;
      while (bucket) {
        for (size_t j = 0; j < bucket->count; j++) {
          if (bucket->tuples[j].key == keys[i]) {
            bucket->tuples[j].payload.value += values[i];
            found = true;
          }
        }
        bucket = bucket->next;
      }
      // if the group key does not exist in the hash table, insert the key
      // and initialize with the current value
      if (!found) {
        AggregationPayload p = {i, values[i]};
        tuple_t t = {keys[i], p};
        hashtable.put(t);
      }
    }

    Timestamp end = getTimestamp();

    std::cout << "Pure Aggregation Time for Hash based aggregation for "
              << num_elements
              << " elements: " << double(end - begin) / (1000 * 1000) << "ms"
              << std::endl;
    std::cout << "Bandwidth: "
              << (double(num_elements * sizeof(double)) /
                  (1024 * 1024 * 1024)) /
                     (double(end - begin) / (1000 * 1000 * 1000))
              << "GB/s" << std::endl;

    Column<TID>* new_keys =
        new Column<TID>(grouping_keys->keys->getName(), OID);

    boost::shared_ptr<Column<double> > aggregated_values(
        new Column<double>("", DOUBLE));

    // if there is nothing to do, return valid but empty colums
    if (grouping_keys->keys->size() == 0) {
      return std::pair<ColumnPtr, ColumnPtr>(
          ColumnPtr(new_keys),
          aggregated_values);  // ColumnPtr(aggregated_values));
    }

    // fetch aggregation result from hash table
    for (size_t bucketid = 0; bucketid < hashtable.hashtable->num_buckets;
         ++bucketid) {
      hash_bucket_t* bucket = &hashtable.hashtable->buckets[bucketid];
      while (bucket) {
        for (size_t j = 0; j < bucket->count; j++) {
          new_keys->insert(bucket->tuples[j].payload.tid);
          aggregated_values->insert(bucket->tuples[j].payload.value);
        }
        bucket = bucket->next;
      }
    }

    return std::pair<ColumnPtr, ColumnPtr>(
        ColumnPtr(new_keys),
        aggregated_values);  // ColumnPtr(aggregated_values));
  }

  inline void hash_aggregation_sum_thread(
      const GroupingKeys::value_type* keys, const double* values,
      //            size_t num_elements,
      size_t begin_index, size_t end_index,
      TypedHashTable<GroupingKeys::value_type, AggregationPayload>* hashtable) {
    assert(hashtable != NULL);

    typedef TypedHashTable<GroupingKeys::value_type, AggregationPayload>
        TypedHashTable;
    typedef TypedHashTable::hash_bucket_t hash_bucket_t;
    typedef TypedHashTable::tuple_t tuple_t;

    for (size_t i = begin_index; i < end_index; ++i) {
      hash_bucket_t* bucket = hashtable->getBucket(keys[i]);
      bool found = false;
      while (bucket) {
        for (size_t j = 0; j < bucket->count; j++) {
          if (bucket->tuples[j].key == keys[i]) {
            bucket->tuples[j].payload.value += values[i];
            found = true;
            break;
          }
        }
        bucket = bucket->next;
      }
      // if the group key does not exist in the hash table, insert the key
      // and initialize with the current value
      if (!found) {
        AggregationPayload p = {i, values[i]};
        tuple_t t = {keys[i], p};
        hashtable->put(t);
      }
    }
  }

  inline std::pair<ColumnPtr, ColumnPtr> parallel_hash_aggregation_sum(
      ColumnGroupingKeysPtr grouping_keys, const double* values,
      size_t num_elements) {
    typedef GroupingKeys::value_type GroupingKeysType;
    typedef TypedHashTable<GroupingKeysType, AggregationPayload> TypedHashTable;
    typedef boost::shared_ptr<TypedHashTable> HashTablePtr;
    typedef typename TypedHashTable::hash_bucket_t hash_bucket_t;

    typedef std::vector<HashTablePtr> HashTables;

    assert(grouping_keys->keys->size() == num_elements);

    GroupingKeysType* keys = grouping_keys->keys->data();

    size_t num_threads = boost::thread::hardware_concurrency();

    HashTables hashtables(num_threads);

    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(num_threads));

    TID thread_id, partition_size;
    TID begin_index, end_index;
#pragma omp parallel default(shared) private( \
    thread_id, num_threads, partition_size, begin_index, end_index)
    {
      thread_id = static_cast<TID>(omp_get_thread_num());
      num_threads = static_cast<size_t>(omp_get_num_threads());
      hashtables[thread_id] =
          boost::make_shared<TypedHashTable>(num_elements / num_threads);

      /* size of partition */
      partition_size = num_elements / num_threads;

      begin_index = thread_id * partition_size; /* starting array index */
      if (thread_id == num_threads - 1) {
        /* last thread may do more */
        partition_size = num_elements - begin_index;
      }

      end_index = begin_index + partition_size;

      hash_aggregation_sum_thread(keys, values, begin_index, end_index,
                                  hashtables[thread_id].get());
    }

    Timestamp begin = getTimestamp();

    TypedHashTable hashtable(num_elements / num_threads);

    // combine hash tables
    for (size_t i = 0; i < num_threads; ++i) {
      // iterate over aggregation result from hash table
      for (size_t bucketid = 0;
           bucketid < hashtables[i]->hashtable->num_buckets; ++bucketid) {
        hash_bucket_t* bucket = &hashtables[i]->hashtable->buckets[bucketid];
        // iterate over bucket of partial hash table
        while (bucket) {
          for (size_t j = 0; j < bucket->count; j++) {
            // look up whether current key is in result hash table,
            // if yes, aggregate values and create the entry if not
            hash_bucket_t* ht_result_bucket =
                hashtable.getBucket(bucket->tuples[j].key);
            bool found = false;
            while (ht_result_bucket) {
              for (size_t k = 0; k < ht_result_bucket->count; k++) {
                if (ht_result_bucket->tuples[k].key == bucket->tuples[j].key) {
                  ht_result_bucket->tuples[k].payload.value +=
                      bucket->tuples[j].payload.value;
                  found = true;
                  break;
                }
              }
              ht_result_bucket = ht_result_bucket->next;
            }
            // if the group key does not exist in the hash table, insert the key
            // and initialize with the current value
            if (!found) {
              hashtable.put(bucket->tuples[j]);
            }
          }
          bucket = bucket->next;
        }
      }
    }

    Timestamp end = getTimestamp();

    double size_gb =
        (double(num_elements * sizeof(double)) / (1024 * 1024 * 1024));
    double time_in_seconds = (double(end - begin) / (1000 * 1000 * 1000));

    std::cout << "Pure Aggregation Time for Hash based aggregation for "
              << num_elements
              << " elements: " << double(end - begin) / (1000 * 1000) << "ms"
              << std::endl;
    std::cout << "Size of Input Data in GB: " << size_gb << std::endl;
    std::cout << "Time in seconds: " << time_in_seconds << std::endl;
    std::cout << "Bandwidth: " << size_gb / time_in_seconds << "GB/s"
              << std::endl;

    Column<TID>* new_keys =
        new Column<TID>(grouping_keys->keys->getName(), OID);

    boost::shared_ptr<Column<double> > aggregated_values(
        new Column<double>("", DOUBLE));

    // if there is nothing to do, return valid but empty colums
    if (grouping_keys->keys->size() == 0) {
      return std::pair<ColumnPtr, ColumnPtr>(
          ColumnPtr(new_keys),
          aggregated_values);  // ColumnPtr(aggregated_values));
    }

    // fetch aggregation result from hash table
    for (size_t bucketid = 0; bucketid < hashtable.hashtable->num_buckets;
         ++bucketid) {
      hash_bucket_t* bucket = &hashtable.hashtable->buckets[bucketid];
      while (bucket) {
        for (size_t j = 0; j < bucket->count; j++) {
          new_keys->insert(bucket->tuples[j].payload.tid);
          aggregated_values->insert(bucket->tuples[j].payload.value);
        }
        bucket = bucket->next;
      }
    }

    return std::pair<ColumnPtr, ColumnPtr>(
        ColumnPtr(new_keys),
        aggregated_values);  // ColumnPtr(aggregated_values));
  }

  //    inline std::pair<ColumnPtr, ColumnPtr> hash_aggregation_sum(
  //            ColumnGroupingKeysPtr grouping_keys,
  //            const double* values, size_t num_elements
  //            ) {
  //
  //        assert(grouping_keys->keys->size() == num_elements);
  //        //take a dictionary compressed column in case key column is a
  //        VARCHAR column, and a plain column otherwise
  //        //typedef typename
  //        boost::mpl::if_<boost::mpl::equal_to<U,std::string>,DictionaryCompressedColumn<U>,Column<U>
  //        >::type KeyColumnType;
  //        //        typedef typename
  //        TypedColumnImplementationGenerator<U>::ColumnType KeyColumnType;
  //
  //        typedef GroupingKeys::value_type GroupingKeysType;
  //
  //        GroupingKeysType* keys = grouping_keys->keys->data();
  //
  //        typedef GroupingKeysType U;
  //
  //        typedef std::pair<TID, double> Payload;
  //
  //        typedef boost::unordered_map<GroupingKeysType, Payload,
  //        boost::hash<GroupingKeysType>, std::equal_to<GroupingKeysType> >
  //        HashTable;
  //
  //        //create hash table
  //        HashTable hash_table;
  //
  //        std::pair<typename HashTable::iterator, typename
  //        HashTable::iterator> range;
  //        typename HashTable::iterator it;
  //        size_t number_of_rows = num_elements;
  //        for (size_t i = 0; i < number_of_rows; ++i) {
  //            it = hash_table.find(keys[i]);
  //            if (it != hash_table.end()) {
  //                it->second.second+=values[i];
  //            } else {
  //                std::pair<typename HashTable::iterator, bool> ret =
  //                hash_table.insert(std::make_pair(keys[i], Payload(i,
  //                double(0))));
  //                if (ret.second) {
  //                    ret.first->second.second=values[i];
  //                }
  //            }
  //        }
  //
  //
  //        Column<U>* new_keys = new Column<U>(grouping_keys->keys->getName(),
  //        grouping_keys->keys->getType());
  //        //aggregated_values->setName(values->getName());
  //        //Column<float>* aggregated_values = new
  //        Column<float>(values->getName(),FLOAT);
  //
  //        boost::shared_ptr<Column<double> > aggregated_values(new
  //        Column<double>("", DOUBLE));
  //
  //        //if there is nothing to do, return valid but empty colums
  //        if (grouping_keys->keys->size() == 0) {
  //            return std::pair<ColumnPtr, ColumnPtr>(ColumnPtr(new_keys),
  //            aggregated_values); //ColumnPtr(aggregated_values));
  //        }
  //
  //        for (it = hash_table.begin(); it != hash_table.end(); ++it) {
  //            new_keys->insert(it->second.first);
  //            aggregated_values->insert(it->second.second);
  //        }
  //
  //        return std::pair<ColumnPtr, ColumnPtr>(ColumnPtr(new_keys),
  //        aggregated_values); //ColumnPtr(aggregated_values));
  //    }

}  // end namespace CoGaDB

#endif /* AGGREGATION_HPP */
