
#include <backends/cpu/hashtable.hpp>
#include <backends/cpu/join.hpp>
#include <core/column.hpp>
#include <core/global_definitions.hpp>
#include <util/getname.hpp>

namespace CoGaDB {

template <typename T>
typename CPU_Join<T>::JoinFunctionPtr CPU_Join<T>::get(
    const JoinType& join_type) {
  typedef CPU_Join<T>::JoinFunctionPtr JoinFunctionPtr;
  typedef std::map<JoinType, JoinFunctionPtr> JoinTypeMap;
  static JoinTypeMap map;
  static bool initialized = false;
  if (!initialized) {
    map.insert(std::make_pair(INNER_JOIN, &CPU_Join<T>::inner_join));
    map.insert(std::make_pair(LEFT_OUTER_JOIN, &CPU_Join<T>::left_outer_join));
    map.insert(
        std::make_pair(RIGHT_OUTER_JOIN, &CPU_Join<T>::right_outer_join));
    map.insert(std::make_pair(FULL_OUTER_JOIN, &CPU_Join<T>::full_outer_join));
    initialized = true;
  }
  if (map.find(join_type) != map.end()) {
    return map.find(join_type)->second;
  } else {
    COGADB_FATAL_ERROR("Detected use of unsupported Join Type!", "");
    return JoinFunctionPtr();
  }
}

template <typename T>
const PositionListPairPtr CPU_Join<T>::inner_join(T* join_column1,
                                                  size_t left_num_elements,
                                                  T* join_column2,
                                                  size_t right_num_elements,
                                                  const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr CPU_Join<T>::left_outer_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr CPU_Join<T>::right_outer_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr CPU_Join<T>::full_outer_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return PositionListPairPtr();
}

template <typename T>
typename CPU_Semi_Join<T>::TIDSemiJoinFunctionPtr
CPU_Semi_Join<T>::getTIDSemiJoin(const JoinType& join_type) {
  typedef CPU_Semi_Join<T>::TIDSemiJoinFunctionPtr TIDSemiJoinFunctionPtr;

  typedef std::map<JoinType, TIDSemiJoinFunctionPtr> JoinTypeMap;
  static JoinTypeMap map;
  static bool initialized = false;
  if (!initialized) {
    map.insert(
        std::make_pair(LEFT_SEMI_JOIN, &CPU_Semi_Join<T>::tid_left_semi_join));
    map.insert(std::make_pair(RIGHT_SEMI_JOIN,
                              &CPU_Semi_Join<T>::tid_right_semi_join));
    map.insert(std::make_pair(LEFT_ANTI_SEMI_JOIN,
                              &CPU_Semi_Join<T>::tid_left_anti_semi_join));
    map.insert(std::make_pair(RIGHT_ANTI_SEMI_JOIN,
                              &CPU_Semi_Join<T>::tid_right_anti_semi_join));
    initialized = true;
  }
  if (map.find(join_type) != map.end()) {
    return map.find(join_type)->second;
  } else {
    COGADB_FATAL_ERROR(
        "Detected use of unsupported Join Type: " << util::getName(join_type),
        "");
    return TIDSemiJoinFunctionPtr();
  }
}

template <typename T>
typename CPU_Semi_Join<T>::BitmapSemiJoinFunctionPtr
CPU_Semi_Join<T>::getBitmapSemiJoin(const JoinType& join_type) {
  typedef CPU_Semi_Join<T>::BitmapSemiJoinFunctionPtr BitmapSemiJoinFunctionPtr;
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return BitmapSemiJoinFunctionPtr();
}

template <typename T>
const PositionListPtr serial_tid_left_semi_join(T* join_column1,
                                                size_t left_num_elements,
                                                T* join_column2,
                                                size_t right_num_elements,
                                                const JoinParam& param) {
  typedef boost::unordered_multimap<T, std::pair<TID, bool>, boost::hash<T>,
                                    std::equal_to<T> >
      HashTable;

  PositionListPtr join_tids = createPositionList();

  // create hash table on left relation
  HashTable hashtable;
  size_t hash_table_size = left_num_elements;
  size_t join_column_size = right_num_elements;

  for (TID i = 0; i < hash_table_size; i++)
    hashtable.insert(std::make_pair(join_column1[i], std::make_pair(i, false)));

  // probe right relation
  std::pair<typename HashTable::iterator, typename HashTable::iterator> range;
  typename HashTable::iterator it;
  for (TID i = 0; i < join_column_size; i++) {
    range = hashtable.equal_range(join_column2[i]);
    for (it = range.first; it != range.second; ++it) {
      if (it->first == join_column2[i]) {
        if (!it->second.second) {
          join_tids->push_back(it->second.first);
          it->second.second = true;
        }
      }
    }
  }
  return join_tids;
}

struct SemiJoinPayload {
  TID tid;
  bool flag;
};

struct EmptyPayload {};

/*
 * \brief Computes left semi join and assumes that the left relation is
 * smaller than the right relation.
 * \details This function builds a hash table on the left relation, and
 * probes the right relation with the hash table. For matching keys, it
 * checks whether the tuple in the hash table is already part of the result
 * to avoid duplicates. For this, it uses a flag in the payload of the
 * hash entry, and writes the result only in case the flag is false. After
 * writing the tuple identifier (also in the payload of the hash entry), it
 * sets the flag to true and continues to search for more matching keys in
 * the bucket. This is neccessary, because the left relation may contain
 * duplicate tuples, and all these tuples have to be part of the result.
 */
template <typename T>
const PositionListPtr optimized_left_smaller_right_tid_left_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  PositionListPtr join_tids = createPositionList();

  typedef TypedHashTable<T, SemiJoinPayload> TypedHashTable;
  typedef typename TypedHashTable::hash_bucket_t hash_bucket_t;
  typedef typename TypedHashTable::tuple_t tuple_t;

  // create hash table on left relation
  // to reduce the number of buckets to explore, we tell
  // the hash table to allocate memory for more tuples
  TypedHashTable hashtable(left_num_elements * 10);

  size_t hash_table_size = left_num_elements;

  for (TID i = 0; i < hash_table_size; i++) {
    tuple_t t = {join_column1[i], {i, false}};
    hashtable.put(t);
  }

// probe right relation
#pragma omp parallel for
  for (size_t i = 0; i < right_num_elements; ++i) {
    hash_bucket_t* bucket = hashtable.getBucket(join_column2[i]);
    while (bucket) {
      for (size_t j = 0; j < bucket->count; j++) {
        if (bucket->tuples[j].key == join_column2[i]) {
#pragma omp critical  // WriteResult
          {
            if (!bucket->tuples[j].payload.flag) {
              join_tids->insert(bucket->tuples[j].payload.tid);
              bucket->tuples[j].payload.flag = true;
            }
          }
        }
      }
      bucket = bucket->next;
    }
  }
  return join_tids;
}

/*
 * \brief Computes left semi join and assumes that the right relation is
 * smaller than the left relation.
 * \details This function builds a hash table on the right relation, and
 * probes the left relation with the hash table. For matching keys, the
 * positions are written into the result PositionList.
 */
template <typename T>
const PositionListPtr optimized_left_greater_right_tid_left_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  PositionListPtr join_tids = createPositionList();

  typedef TypedHashTable<T, EmptyPayload> TypedHashTable;
  typedef typename TypedHashTable::hash_bucket_t hash_bucket_t;
  typedef typename TypedHashTable::tuple_t tuple_t;

  // create hash table on right relation
  TypedHashTable hashtable(right_num_elements);

  for (TID i = 0; i < right_num_elements; i++) {
    tuple_t t = {join_column2[i]};
    hashtable.put(t);
  }

  // probe left relation, and write result id if key matches
  for (size_t i = 0; i < left_num_elements; ++i) {
    hash_bucket_t* bucket = hashtable.getBucket(join_column1[i]);
    while (bucket) {
      for (size_t j = 0; j < bucket->count; j++) {
        if (bucket->tuples[j].key == join_column1[i]) {
          join_tids->push_back(i);
          // this is neccessary to leave also the while loop
          // in case there are duplicate tuples in the hash table,
          // we may only write the result once!
          bucket = NULL;
          break;
        }
      }
      if (bucket) bucket = bucket->next;
    }
  }

  return join_tids;
}

template <typename T>
const PositionListPtr CPU_Semi_Join<T>::tid_left_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
//#define COGADB_TEST_CPU_SEMI_JOIN
#ifdef COGADB_TEST_CPU_SEMI_JOIN
  std::cout << "Left Num Elements:  " << left_num_elements << std::endl;
  std::cout << "Right Num Elements: " << right_num_elements << std::endl;

  Timestamp begin = getTimestamp();
  PositionListPtr tids1 = optimized_left_greater_right_tid_left_semi_join(
      join_column1, left_num_elements, join_column2, right_num_elements, param);
  Timestamp end = getTimestamp();

  std::cout
      << "Time Semi Join optimized_left_greater_right_tid_left_semi_join: "
      << double(end - begin) / (1000 * 1000) << "ms" << std::endl;

  begin = getTimestamp();
  PositionListPtr tids2 = optimized_left_smaller_right_tid_left_semi_join(
      join_column1, left_num_elements, join_column2, right_num_elements, param);
  end = getTimestamp();

  std::cout
      << "Time Semi Join optimized_left_smaller_right_tid_left_semi_join: "
      << double(end - begin) / (1000 * 1000) << "ms" << std::endl;

  size_t n = std::min(tids1->size(), tids2->size());

  TID* array1 = tids1->data();
  TID* array2 = tids2->data();

  std::sort(array1, array1 + tids1->size());
  std::sort(array2, array2 + tids2->size());

  //        std::cout << "Results of Semi Join: " << std::endl;
  //        for(size_t i=0;i<n;++i){
  //            std::cout << array1[i] << ", " << array2[i] << std::endl;
  //        }
  assert(tids1->is_equal(tids2));
#endif
  if (left_num_elements <= right_num_elements) {
    return optimized_left_smaller_right_tid_left_semi_join(
        join_column1, left_num_elements, join_column2, right_num_elements,
        param);
  } else {
    return optimized_left_greater_right_tid_left_semi_join(
        join_column1, left_num_elements, join_column2, right_num_elements,
        param);
  }
}

template <>
const PositionListPtr CPU_Semi_Join<std::string>::tid_left_semi_join(
    std::string* join_column1, size_t left_num_elements,
    std::string* join_column2, size_t right_num_elements,
    const JoinParam& param) {
  return PositionListPtr();
}

template <>
const PositionListPtr CPU_Semi_Join<char*>::tid_left_semi_join(
    char** join_column1, size_t left_num_elements, char** join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return PositionListPtr();
}

template <typename T>
const PositionListPtr CPU_Semi_Join<T>::tid_right_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return CPU_Semi_Join<T>::tid_left_semi_join(
      join_column2, right_num_elements, join_column1, left_num_elements, param);
}

template <typename T>
const PositionListPtr CPU_Semi_Join<T>::tid_left_anti_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  typedef boost::unordered_multimap<T, std::pair<TID, bool>, boost::hash<T>,
                                    std::equal_to<T> >
      HashTable;

  PositionListPtr join_tids = createPositionList();

  // create hash table
  HashTable hashtable;
  size_t hash_table_size = left_num_elements;
  size_t join_column_size = right_num_elements;

  for (TID i = 0; i < hash_table_size; i++)
    hashtable.insert(std::make_pair(join_column1[i], std::make_pair(i, false)));

  // probe larger relation
  std::pair<typename HashTable::iterator, typename HashTable::iterator> range;
  typename HashTable::iterator it;
  for (TID i = 0; i < join_column_size; i++) {
    range = hashtable.equal_range(join_column2[i]);
    bool matched = false;
    for (it = range.first; it != range.second; ++it) {
      // mark entries in hash table that have a hit
      if (it->first == join_column2[i]) {
        it->second.second = true;
      }
    }
  }
  // return ids of tuples that did not match any key (bit flag is false)
  for (it = hashtable.begin(); it != hashtable.end(); ++it) {
    if (!it->second.second) join_tids->push_back(it->second.first);
  }
  return join_tids;
}

template <typename T>
const PositionListPtr CPU_Semi_Join<T>::tid_right_anti_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return CPU_Semi_Join<T>::tid_left_anti_semi_join(
      join_column2, right_num_elements, join_column1, left_num_elements, param);
  //        COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  //        return PositionListPtr();
}

template <typename T>
const BitmapPtr CPU_Semi_Join<T>::bitmap_left_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return BitmapPtr();
}

template <typename T>
const BitmapPtr CPU_Semi_Join<T>::bitmap_right_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return BitmapPtr();
}

template <typename T>
const BitmapPtr CPU_Semi_Join<T>::bitmap_left_anti_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return BitmapPtr();
}

template <typename T>
const BitmapPtr CPU_Semi_Join<T>::bitmap_right_anti_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return BitmapPtr();
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(CPU_Join)
COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(CPU_Semi_Join)
}
