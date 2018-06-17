

#include <hardware_optimizations/primitives.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define PRIME 1442968193
#include <assert.h>
#include <math.h>
#include <stdint.h>

#include <iostream>
//#include <cstdint>
#include <bitset>
#include <boost/shared_ptr.hpp>
#include <cstring>
#include <vector>

#include <util/time_measurement.hpp>

#include <core/lookup_array.hpp>
#include <lookup_table/join_index.hpp>
#include <lookup_table/lookup_table.hpp>
#include <util/begin_ptr.hpp>
#include <util/column_grouping_keys.hpp>
#include "backends/processor_backend.hpp"
#include "core/selection_expression.hpp"
#include "persistence/storage_manager.hpp"
#include "util/utility_functions.hpp"

#include <tbb/parallel_sort.h>
#include <compression/dictionary_compressed_column.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wvla"

//#include <cuda_runtime.h>
//#define THRUST_DEVICE_SYSTEM_CUDA    1
//#define THRUST_DEVICE_SYSTEM_OMP     2
//#define THRUST_DEVICE_SYSTEM_TBB     3
//#define THRUST_DEVICE_SYSTEM_CPP     4
#define THRUST_DEVICE_SYSTEM 3
#include <thrust/sort.h>

#include <util/getname.hpp>

namespace CoGaDB {
// CoGaDB Database Kernel
namespace CDK {

using namespace std;

typedef struct {
  TID begin_index;
  TID end_index;
} RadixBucket;

// typedef struct{
//        int key;
//        TID value;
//} RadixPair;

struct RadixPair {
  RadixPair(int key, TID value);
  int key;
  TID value;
};

RadixPair::RadixPair(int key_, TID value_) : key(key_), value(value_) {}

class RadixHashTable {
 public:
  RadixHashTable(RadixPair* value_array, uint32_t* bucket_borders,
                 size_t number_of_buckets, uint8_t total_number_of_radix_bits);
  ~RadixHashTable();
  void print();
  unsigned int getNumberOfBuckets() const;
  class Bucket {
   public:
    Bucket(TID begin_index, TID end_index, RadixPair* value_array,
           unsigned int bucket_id, uint8_t total_number_of_radix_bits);

    RadixPair* begin();

    RadixPair* end();

   private:
    TID begin_index_;
    TID end_index_;
    RadixPair* value_array_;
  };
  typedef boost::shared_ptr<Bucket> BucketPtr;
  BucketPtr getBucket(unsigned int id);

 private:
  std::vector<uint32_t> bucket_borders_;
  RadixPair* value_array_;
  size_t number_of_buckets_;
  uint8_t total_number_of_radix_bits_;
};

typedef boost::shared_ptr<RadixHashTable> RadixHashTablePtr;

RadixHashTable::RadixHashTable(RadixPair* value_array, uint32_t* bucket_borders,
                               size_t number_of_buckets,
                               uint8_t total_number_of_radix_bits)
    : bucket_borders_(bucket_borders, bucket_borders + number_of_buckets),
      value_array_(value_array),
      number_of_buckets_(number_of_buckets - 1),
      total_number_of_radix_bits_(total_number_of_radix_bits) {
  assert(value_array_ != NULL);
}

RadixHashTable::~RadixHashTable() {
  if (value_array_) free(value_array_);
}

void RadixHashTable::print() {
  unsigned int bitmask = pow(2, total_number_of_radix_bits_) - 1;
  cout << (bitset<32>)bitmask << endl;
  cout << "Number of Buckets: " << number_of_buckets_ << endl;
  for (unsigned int i = 0; i < number_of_buckets_; ++i) {
    cout << "Cluster " << i
         << " size: " << bucket_borders_[i + 1] - bucket_borders_[i] << " "
         << bucket_borders_[i] << endl;
    for (unsigned int j = 0; j < bucket_borders_[i + 1] - bucket_borders_[i];
         ++j) {
      cout << value_array_[j + bucket_borders_[i]].key << " Radix: "
           << (bitset<32>)(value_array_[j + bucket_borders_[i]].key & bitmask)
           << "  " << (void*)&value_array_[j] << endl;
    }
  }
}

unsigned int RadixHashTable::getNumberOfBuckets() const {
  return number_of_buckets_;
}

RadixHashTable::BucketPtr RadixHashTable::getBucket(unsigned int id) {
  if (id < number_of_buckets_) {
    // Note: bucket_borders_ has a size of number_of_buckets_+1, hence, it is
    // always save to dereference bucket_borders_[id+1]!
    return BucketPtr(new Bucket(bucket_borders_[id], bucket_borders_[id + 1],
                                value_array_, id, total_number_of_radix_bits_));
  } else {
    return BucketPtr();  // return NULL Pointer
  }
}

RadixHashTable::Bucket::Bucket(TID begin_index, TID end_index,
                               RadixPair* value_array, unsigned int bucket_id,
                               uint8_t total_number_of_radix_bits)
    : begin_index_(begin_index),
      end_index_(end_index),
      value_array_(value_array) {}

RadixPair* RadixHashTable::Bucket::begin() {
  return &value_array_[begin_index_];
}

RadixPair* RadixHashTable::Bucket::end() { return &value_array_[end_index_]; }

typedef unsigned char BIT;

void radix_cluster(RadixPair* array, size_t array_size,
                   uint8_t total_number_of_bits, uint8_t* bits_per_pass,
                   size_t number_of_passess, int* buckets,
                   size_t total_number_of_buckets, size_t pass_id,
                   unsigned int global_cluster_id, RadixPair* result_0,
                   RadixPair* result_1) {
  // if(pass_id>0) cout << "Partitioning Bucket of size " << array_size << endl;
  // compute bitmask of bits to consider in this pass
  int bitmask = 0;
  for (unsigned int i = 0; i < bits_per_pass[pass_id]; ++i) {
    bitmask = bitmask << 1;
    bitmask = bitmask | 1;
    // cout << "Bitmask: " << (bitset<32>) bitmask << endl;
  }
  // compute position in this pass
  int number_of_processed_bits = 0;
  for (unsigned int i = 0; i < pass_id; i++) {
    number_of_processed_bits += bits_per_pass[i];
  }
  // cout << "Number of Bits Processed: " << number_of_processed_bits << endl;
  // shift current bits to correct position for this pass
  bitmask <<= number_of_processed_bits;
  // number of clusters
  // cout << "#Clusters: 2^(" << number_of_processed_bits+bits_per_pass[pass_id]
  // << ")=" << pow(2,number_of_processed_bits+bits_per_pass[pass_id]) << endl;
  uint32_t number_of_clusters = pow(
      2,
      bits_per_pass
          [pass_id]);  // pow(2,number_of_processed_bits+bits_per_pass[pass_id]);
                       // //2^(number_of_processed_bits+bits_per_pass[pass_id])
  std::vector<uint32_t> bucket_sizes(number_of_clusters, 0);
  //	for(unsigned int i=0;i<number_of_clusters;++i){
  //		bucket_sizes[i]=0;
  //	}
  for (unsigned int i = 0; i < array_size; i++) {
    int cluster_number = (array[i].key & bitmask) >> number_of_processed_bits;

    // cout << "put " << array[i].key << " in cluster number: " <<
    // cluster_number << endl;
    bucket_sizes[cluster_number]++;
    // cout << "New bucket Size" << bucket_sizes[cluster_number] << endl;
  }
  // compute prefix sum to get begin and end index for each bucket
  std::vector<uint32_t> bucket_borders(number_of_clusters + 1);
  bucket_borders[0] = 0;
  for (uint32_t i = 1; i <= number_of_clusters; ++i) {
    bucket_borders[i] = bucket_sizes[i - 1] + bucket_borders[i - 1];
  }

  auto insert_positions = bucket_borders;

  // int* result_array = (int*) malloc(sizeof(int)*array_size);
  // insert data in buckets
  for (uint32_t i = 0; i < array_size; i++) {
    int cluster_number = (array[i].key & bitmask) >> number_of_processed_bits;
    if (!quiet && verbose && debug)
      cout << "write pair at position " << i << "(" << array[i].key << ","
           << array[i].value << ")"
           << " to result_array at postion " << insert_positions[cluster_number]
           << endl;
    result_0[insert_positions[cluster_number]++] = array[i];
  }
  if (!quiet && verbose && debug) {
    for (uint32_t i = 0; i < array_size; i++) {
      cout << "Result: " << result_0[i].key
           << " vs. Input Array: " << array[i].key << endl;
    }
  }

  if (!quiet && verbose) {
    cout << "Pass: " << pass_id
         << " (Total Number of Passes: " << number_of_passess << ")" << endl;
    cout << "Bitmask: " << (bitset<32>)bitmask << endl;
    cout << "Bits in this pass: " << (int)bits_per_pass[pass_id] << endl;
    cout << "#Radix Cluster: " << number_of_clusters << endl;
    cout << "Content of Radix Hash Table: " << endl;
  }

  if (!quiet && verbose && debug) {
    for (unsigned int i = 0; i < number_of_clusters; ++i) {
      cout << "Cluster " << i << " size: " << bucket_sizes[i] << " "
           << bucket_borders[i] << endl;
      for (unsigned int j = bucket_borders[i]; j < bucket_borders[i + 1]; ++j) {
        cout << result_0[j].key << " Radix: " << (bitset<32>)result_0[j].key
             << "  " << (void*)&result_0[j]
             << endl;  //(bitset<32>)((result_0[j].key & bitmask) ) << endl;
                       ////>> number_of_processed_bits) << endl;
      }
    }

    cout << "Current Partition:" << endl;
    for (unsigned int i = 0; i < array_size; i++) {
      cout << result_0[i].key << endl;
    }
  }
  if (pass_id + 1 < number_of_passess) {
    for (unsigned int i = 0; i < number_of_clusters; ++i) {
      if (!quiet && verbose && debug)
        cout << "==================================================" << endl;
      // unsigned int new_number_of_buckets =
      // number_of_buckets*bits_per_pass[i+1];
      if (!quiet && verbose && debug)
        cout << "Radix Bucket ID in Pass " << pass_id << " is: "
             << i * pow(2, total_number_of_bits - number_of_processed_bits)
             << endl;
      unsigned int new_global_cluster_id =
          global_cluster_id +
          i * pow(2,
                  total_number_of_bits -
                      (number_of_processed_bits +
                       bits_per_pass
                           [pass_id]));  // i*pow(2,total_number_of_bits-number_of_processed_bits);
      //			radix_cluster(result_0+bucket_borders[i],
      // bucket_sizes[i], total_number_of_bits,
      //							  bits_per_pass,
      // number_of_passess, buckets, total_number_of_buckets, pass_id+1,
      //							  new_global_cluster_id,
      //							  result_1+bucket_borders[i],
      // result_0+bucket_borders[i]);

      radix_cluster(&result_0[bucket_borders[i]], bucket_sizes[i],
                    total_number_of_bits, bits_per_pass, number_of_passess,
                    buckets, total_number_of_buckets, pass_id + 1,
                    new_global_cluster_id, &result_1[bucket_borders[i]],
                    &result_0[bucket_borders[i]]);

      //			unsigned int global_bucket_id = ;
      //			buckets[pass_id*number_of_clusters]
    }
  } else {
    // write bucket size in Hash Table
    for (unsigned int i = 0; i < number_of_clusters; ++i) {
      if (!quiet && verbose && debug)
        cout << "Writing Result Bucket Size (" << bucket_sizes[i]
             << ") for Global RadixBucket " << global_cluster_id + i
             << " (Buckets in Total: " << total_number_of_buckets << ")"
             << endl;
      buckets[global_cluster_id + i] = bucket_sizes[i];
    }
  }

  // if(result_array) free(result_array);
}

RadixHashTablePtr createRadixHashTable(int* array, size_t array_size,
                                       int total_number_of_bits,
                                       int total_number_of_passes) {
  RadixPair* result_0 = (RadixPair*)malloc(array_size * sizeof(RadixPair));
  RadixPair* result_1 = (RadixPair*)malloc(array_size * sizeof(RadixPair));

  for (unsigned int i = 0; i < array_size; i++) {
    result_1[i] = RadixPair(array[i], i);
  }
  //        unsigned int i=0;
  //        unsigned int unrolled_loop_size=(array_size/8)*8;
  //        //the for loop header caused a high CPI rate thats why we unrolled
  //        it by hand
  //	for(i=0;i<unrolled_loop_size;i+=8){
  //	    result_1[i]=RadixPair(array[i],i);
  //            result_1[i+1]=RadixPair(array[i+1],i+1);
  //            result_1[i+2]=RadixPair(array[i+2],i+2);
  //            result_1[i+3]=RadixPair(array[i+3],i+3);
  //            result_1[i+4]=RadixPair(array[i+4],i+4);
  //            result_1[i+5]=RadixPair(array[i+5],i+5);
  //            result_1[i+6]=RadixPair(array[i+6],i+6);
  //            result_1[i+7]=RadixPair(array[i+7],i+7);
  //	}
  //	for(;i<array_size;i++){
  //		result_1[i]=RadixPair(array[i],i);
  //	}

  //        unsigned int i=0;
  //        //cout << "array size " << array_size << " " <<
  //        array_size/pow(2,i)<< endl;
  //        while(true){
  //            if(array_size/pow(2,i)<100) break;
  //            ++i;
  //        }
  //        unsigned int total_number_of_bits=i;
  //        //cout << "use " << i << " Bits to cluster in " <<
  //        (total_number_of_bits+2)/3 << "passes" << endl;
  //	unsigned int number_of_passes=(i+2)/3; //3; //2;
  //	uint8_t bits_per_pass[number_of_passes];
  //        for(i=0;i<number_of_passes;++i){
  //            bits_per_pass[i]=3;
  //        }

  //        unsigned int number_of_passes=4; //2;
  //        uint8_t bits_per_pass[number_of_passes];
  //	bits_per_pass[0]=6;
  //	bits_per_pass[1]=3;
  //	bits_per_pass[2]=3;
  //        bits_per_pass[3]=3;
  //	unsigned int total_number_of_bits=15;

  unsigned int number_of_passes = total_number_of_passes;  // 2;
  std::vector<uint8_t> bits_per_pass(number_of_passes);
  bits_per_pass[0] = (total_number_of_bits / number_of_passes) +
                     (total_number_of_bits % number_of_passes);
  for (unsigned int i = 1; i < number_of_passes; ++i) {
    bits_per_pass[i] = total_number_of_bits / number_of_passes;
  }
  cout << "RadixHashTable: total_bits: " << total_number_of_bits
       << " #passes: " << number_of_passes << " Bits per pass: (";
  for (unsigned int i = 0; i < number_of_passes; ++i) {
    cout << (int)bits_per_pass[i] << ",";  //<< endl;
  }
  cout << ")" << endl;

  unsigned int total_number_of_buckets = pow(2, total_number_of_bits);
  std::vector<int> buckets(total_number_of_buckets);
  // buckets[0]=v.size(); //data in one bucket at beginning

  radix_cluster(result_1, array_size, total_number_of_bits,
                bits_per_pass.data(), number_of_passes, buckets.data(),
                total_number_of_buckets, 0, 0, result_0, result_1);

  // compute prefix sum to get begin and end index for each bucket
  std::vector<uint32_t> bucket_borders(total_number_of_buckets + 1);
  bucket_borders[0] = 0;
  for (unsigned int i = 1; i < total_number_of_buckets + 1; ++i) {
    bucket_borders[i] = buckets[i - 1] + bucket_borders[i - 1];
  }

  RadixPair* result;
  if (number_of_passes % 2 == 0) {
    result = result_1;
    free(result_0);
  } else {
    result = result_0;
    free(result_1);
  }

  return RadixHashTablePtr(new RadixHashTable(result, bucket_borders.data(),
                                              total_number_of_buckets + 1,
                                              total_number_of_bits));
}

// template<typename T>
void RadixPair_nested_loop_join(RadixPair* __restrict__ column1,
                                const size_t& col1_array_size,
                                RadixPair* __restrict__ column2,
                                const size_t& col2_array_size,
                                PositionListPairPtr join_tids) {
  assert(column1 != NULL);
  assert(column2 != NULL);

  unsigned int join_column1_size = col1_array_size;
  unsigned int join_column2_size = col2_array_size;

  for (unsigned int i = 0; i < join_column1_size; i++) {
    for (unsigned int j = 0; j < join_column2_size; j++) {
      if (column1[i].key == column2[j].key) {
        // std::cout << "MATCH: (" << column1[i].key << "," << column1[i].value
        // << "; " << column2[j].key << "," << column2[j].value << ")" <<
        // std::endl;
        join_tids->first->push_back(column1[i].value);
        join_tids->second->push_back(column2[j].value);
      }
    }
  }
  // return join_tids;
}

void RadixPair_serial_hash_join(RadixPair* __restrict__ column1,
                                const size_t& col1_array_size,
                                RadixPair* __restrict__ column2,
                                const size_t& col2_array_size,
                                PositionListPairPtr join_tids) {
  if (col2_array_size < col1_array_size) {
    return RadixPair_serial_hash_join(column2, col2_array_size, column1,
                                      col1_array_size, join_tids);
  }

  typedef boost::unordered_multimap<int, TID, boost::hash<int>,
                                    std::equal_to<int> >
      HashTable;

  //				PositionListPairPtr join_tids( new
  // PositionListPair());
  //				join_tids->first =
  // PositionListPtr=createPositionList();
  //				join_tids->second =
  // PositionListPtr=createPositionList();

  Timestamp build_hashtable_begin = getTimestamp();
  // create hash table
  HashTable hashtable;
  unsigned int hash_table_size = col1_array_size;
  unsigned int join_column_size = col2_array_size;

  assert(col2_array_size >= col1_array_size);
  unsigned int pos = 0;

  for (unsigned int i = 0; i < hash_table_size; i++)
    hashtable.insert(std::pair<int, TID>(column1[i].key, column1[i].value));
  Timestamp build_hashtable_end = getTimestamp();
  //        std::cout << "Number of Buckets: " << hashtable.bucket_count() <<
  //        std::endl;
  //        for(unsigned int i=0;i< hashtable.bucket_count();i++){
  //            std::cout << "Size of Bucket '" << i << "': " <<
  //            hashtable.bucket_size(i) << std::endl;
  //        }

  // probe larger relation
  Timestamp prune_hashtable_begin = getTimestamp();

  std::pair<typename HashTable::iterator, typename HashTable::iterator> range;
  typename HashTable::iterator it;
  for (unsigned int i = 0; i < join_column_size; i++) {
    range = hashtable.equal_range(column2[i].key);
    for (it = range.first; it != range.second; ++it) {
      if (it->first == column2[i].key) {  //(*join_column)[i]){
        //                                  join_tids_table1[pos]=it->second;
        //                                  join_tids_table2[pos]=i;
        //                                  pos++;
        // pos2=++pos1;
        //                                join_tids_table1[pos1++]=it->second;
        //                                join_tids_table2[pos2++]=i;
        join_tids->first->push_back(it->second);
        join_tids->second->push_back(column2[i].value);

        // cout << "match! " << it->second << ", " << i << "	"  << it->first
        // << endl;
      }
    }
  }

  Timestamp prune_hashtable_end = getTimestamp();

  if (!quiet && verbose)
    std::cout << "Hash Join: Build Phase: "
              << double(build_hashtable_end - build_hashtable_begin) /
                     (1000 * 1000)
              << "ms"
              << "Pruning Phase: "
              << double(prune_hashtable_end - prune_hashtable_begin) /
                     (1000 * 1000)
              << "ms" << std::endl;
}

namespace selection {

void bitwise_and_thread(char* left_input_bitmap, char* right_input_bitmap,
                        TID begin_index, TID end_index, char* result_bitmap) {
  assert(begin_index % 8 == 0);
  // process (#Bits+7/8) Byte [Rest Bits of the Byte as well]
  TID new_end_index = (end_index + 7) / 8;
  for (TID i = begin_index; i < new_end_index; i++) {
    result_bitmap[i] = left_input_bitmap[i] & right_input_bitmap[i];
  }
}

void bitwise_or_thread(char* left_input_bitmap, char* right_input_bitmap,
                       TID begin_index, TID end_index, char* result_bitmap) {
  assert(begin_index % 8 == 0);
  // process (#Bits+7/8) Byte [Rest Bits of the Byte as well]
  TID new_end_index = (end_index + 7) / 8;
  for (TID i = begin_index; i < new_end_index; i++) {
    result_bitmap[i] = left_input_bitmap[i] | right_input_bitmap[i];
  }
}

BitmapPtr bitwise_and(BitmapPtr left_input_bitmap,
                      BitmapPtr right_input_bitmap) {
  assert(left_input_bitmap != NULL);
  assert(right_input_bitmap != NULL);
  // check whether number of elements is equal for both input bitmaps
  assert(left_input_bitmap->size() == right_input_bitmap->size());
  // set non init flag, we will overwrite the values anyway
  BitmapPtr result(new Bitmap(left_input_bitmap->size(), false, false));
  // do the work
  bitwise_and_thread(left_input_bitmap->data(), right_input_bitmap->data(), 0,
                     left_input_bitmap->size(), result->data());
  return result;
}

BitmapPtr bitwise_or(BitmapPtr left_input_bitmap,
                     BitmapPtr right_input_bitmap) {
  assert(left_input_bitmap != NULL);
  assert(right_input_bitmap != NULL);
  // check whether number of elements is equal for both input bitmaps
  assert(left_input_bitmap->size() == right_input_bitmap->size());
  // set non init flag, we will overwrite the values anyway
  BitmapPtr result(new Bitmap(left_input_bitmap->size(), false, false));
  // do the work
  bitwise_or_thread(left_input_bitmap->data(), right_input_bitmap->data(), 0,
                    left_input_bitmap->size(), result->data());
  return result;
}

namespace variants {

#ifdef ENABLE_SIMD_ACCELERATION

template <class T>
void SIMD_selection_thread(T* array, size_t column_size, unsigned int thread_id,
                           unsigned int number_of_threads, const T value,
                           const ValueComparator comp, TID* result_tids,
                           size_t* result_size) {
  return CoGaDB::CDK::selection::variants::selection_thread(
      array, column_size, thread_id, number_of_threads, value, comp,
      result_tids, result_size);
}

template <class T>
void bf_SIMD_selection_thread(T* array, size_t column_size,
                              unsigned int thread_id,
                              unsigned int number_of_threads, const T value,
                              const ValueComparator comp, TID* result_tids,
                              size_t* result_size) {
  return CoGaDB::CDK::selection::variants::bf_selection_thread(
      array, column_size, thread_id, number_of_threads, value, comp,
      result_tids, result_size);
}

template <>
void SIMD_selection_thread(float* array, size_t column_size,
                           unsigned int thread_id,
                           unsigned int number_of_threads, const float value,
                           const ValueComparator comp, TID* result_tids,
                           size_t* result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>& >(*col);
  // number of elements per thread
  unsigned int chunk_size = column_size / number_of_threads;
  unsigned int begin_index = chunk_size * thread_id;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    chunk_size = column_size - (number_of_threads - 1) * chunk_size;
  }
  // cout << "Thread " << thread_id << " begin index: " <<  begin_index << " end
  // index: " << end_index << endl;
  TID pos = begin_index;
  /* unsigned int* result_array = new unsigned int[chunk_size];
   assert(result_array!=NULL);
   */

  CoGaDB::simd_selection_float_thread(array + begin_index, chunk_size, value,
                                      comp, result_tids, &pos, begin_index);
  //}
  // std::cout << "ID: " << thread_id << " SIZE: " << pos << std::endl;
  // write result size to array
  *result_size = pos;
}

template <>
void SIMD_selection_thread(int* array, size_t column_size,
                           unsigned int thread_id,
                           unsigned int number_of_threads,
                           const int value_for_comparison,
                           const ValueComparator comp, TID* result_tids,
                           size_t* result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>& >(*col);
  // number of elements per thread
  unsigned int chunk_size = column_size / number_of_threads;
  unsigned int begin_index = chunk_size * thread_id;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    chunk_size = column_size - (number_of_threads - 1) * chunk_size;
  }
  // cout << "Thread " << thread_id << " begin index: " <<  begin_index << " end
  // index: " << end_index << endl;
  TID pos = begin_index;
  /* unsigned int* result_array = new unsigned int[chunk_size];
   assert(result_array!=NULL);
   */

  CoGaDB::simd_selection_int_thread(array + begin_index, chunk_size,
                                    value_for_comparison, comp, result_tids,
                                    &pos, begin_index);
  //}
  // std::cout << "ID: " << thread_id << " SIZE: " << pos << std::endl;
  // write result size to array
  *result_size = pos;
}

template <>
void bf_SIMD_selection_thread(float* array, size_t column_size,
                              unsigned int thread_id,
                              unsigned int number_of_threads, const float value,
                              const ValueComparator comp, TID* result_tids,
                              size_t* result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>& >(*col);
  // number of elements per thread
  unsigned int chunk_size = column_size / number_of_threads;
  unsigned int begin_index = chunk_size * thread_id;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    chunk_size = column_size - (number_of_threads - 1) * chunk_size;
  }
  // cout << "Thread " << thread_id << " begin index: " <<  begin_index << " end
  // index: " << end_index << endl;
  TID pos = begin_index;
  /* unsigned int* result_array = new unsigned int[chunk_size];
   assert(result_array!=NULL);
   */

  CoGaDB::bf_simd_selection_float_thread(array + begin_index, chunk_size, value,
                                         comp, result_tids, &pos, begin_index);
  //}
  // std::cout << "ID: " << thread_id << " SIZE: " << pos << std::endl;
  // write result size to array
  *result_size = pos;
}

template <>
void bf_SIMD_selection_thread(int* array, size_t column_size,
                              unsigned int thread_id,
                              unsigned int number_of_threads,
                              const int value_for_comparison,
                              const ValueComparator comp, TID* result_tids,
                              size_t* result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>& >(*col);
  // number of elements per thread
  unsigned int chunk_size = column_size / number_of_threads;
  unsigned int begin_index = chunk_size * thread_id;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    chunk_size = column_size - (number_of_threads - 1) * chunk_size;
  }
  // cout << "Thread " << thread_id << " begin index: " <<  begin_index << " end
  // index: " << end_index << endl;
  TID pos = begin_index;
  /* unsigned int* result_array = new unsigned int[chunk_size];
   assert(result_array!=NULL);
   */

  CoGaDB::bf_simd_selection_int_thread(array + begin_index, chunk_size,
                                       value_for_comparison, comp, result_tids,
                                       &pos, begin_index);
  //}
  // std::cout << "ID: " << thread_id << " SIZE: " << pos << std::endl;
  // write result size to array
  *result_size = pos;
}

template <class T>
void unrolled_SIMD_selection_thread(T* array, size_t column_size,
                                    unsigned int thread_id,
                                    unsigned int number_of_threads,
                                    const T value, const ValueComparator comp,
                                    TID* result_tids, size_t* result_size) {
  return CoGaDB::CDK::selection::variants::unrolled_selection_thread(
      array, column_size, thread_id, number_of_threads, value, comp,
      result_tids, result_size);
}

template <>
void unrolled_SIMD_selection_thread(float* array, size_t column_size,
                                    unsigned int thread_id,
                                    unsigned int number_of_threads,
                                    const float value_for_comparison,
                                    const ValueComparator comp,
                                    TID* result_tids, size_t* result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>& >(*col);
  // number of elements per thread
  unsigned int chunk_size = column_size / number_of_threads;
  unsigned int begin_index = chunk_size * thread_id;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    chunk_size = column_size - (number_of_threads - 1) * chunk_size;
  }
  // cout << "Thread " << thread_id << " begin index: " <<  begin_index << " end
  // index: " << end_index << endl;
  TID pos = begin_index;
  /*    unsigned int* result_array = new unsigned int[chunk_size];
   assert(result_array!=NULL);
   */
  CoGaDB::CDK::selection::variants::unrolled_simd_selection_float_thread(
      array + begin_index, chunk_size, value_for_comparison, comp, result_tids,
      &pos, begin_index);
  //}
  // std::cout << "ID: " << thread_id << " SIZE: " << pos-begin_index <<
  // std::endl;
  // write result size to array
  *result_size = pos;
}

template <>
void unrolled_SIMD_selection_thread(int* array, size_t column_size,
                                    unsigned int thread_id,
                                    unsigned int number_of_threads,
                                    const int value_for_comparison,
                                    const ValueComparator comp,
                                    TID* result_tids, size_t* result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>& >(*col);
  // number of elements per thread
  unsigned int chunk_size = column_size / number_of_threads;
  unsigned int begin_index = chunk_size * thread_id;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    chunk_size = column_size - (number_of_threads - 1) * chunk_size;
  }
  // cout << "Thread " << thread_id << " begin index: " <<  begin_index << " end
  // index: " << end_index << endl;
  TID pos = begin_index;
  /*    unsigned int* result_array = new unsigned int[chunk_size];
      assert(result_array!=NULL);
    */
  CoGaDB::CDK::selection::variants::unrolled_simd_selection_int_thread(
      array + begin_index, chunk_size, value_for_comparison, comp, result_tids,
      &pos, begin_index);
  //}
  // std::cout << "ID: " << thread_id << " SIZE: " << pos-begin_index <<
  // std::endl;
  // write result size to array
  *result_size = pos;
}

template <class T>
void bf_unrolled_SIMD_selection_thread(T* array, size_t column_size,
                                       unsigned int thread_id,
                                       unsigned int number_of_threads,
                                       const T value,
                                       const ValueComparator comp,
                                       TID* result_tids, size_t* result_size) {
  return CoGaDB::CDK::selection::variants::bf_unrolled_selection_thread(
      array, column_size, thread_id, number_of_threads, value, comp,
      result_tids, result_size);
}

template <>
void bf_unrolled_SIMD_selection_thread(float* array, size_t column_size,
                                       unsigned int thread_id,
                                       unsigned int number_of_threads,
                                       const float value_for_comparison,
                                       const ValueComparator comp,
                                       TID* result_tids, size_t* result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>& >(*col);
  // number of elements per thread
  unsigned int chunk_size = column_size / number_of_threads;
  unsigned int begin_index = chunk_size * thread_id;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    chunk_size = column_size - (number_of_threads - 1) * chunk_size;
  }
  // cout << "Thread " << thread_id << " begin index: " <<  begin_index << " end
  // index: " << end_index << endl;
  TID pos = begin_index;
  /*    unsigned int* result_array = new unsigned int[chunk_size];
   assert(result_array!=NULL);
   */
  CoGaDB::CDK::selection::variants::bf_unrolled_simd_selection_float_thread(
      array + begin_index, chunk_size, value_for_comparison, comp, result_tids,
      &pos, begin_index);
  //}
  // std::cout << "ID: " << thread_id << " SIZE: " << pos-begin_index <<
  // std::endl;
  // write result size to array
  *result_size = pos;
}

template <>
void bf_unrolled_SIMD_selection_thread(int* array, size_t column_size,
                                       unsigned int thread_id,
                                       unsigned int number_of_threads,
                                       const int value_for_comparison,
                                       const ValueComparator comp,
                                       TID* result_tids, size_t* result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>& >(*col);
  // number of elements per thread
  unsigned int chunk_size = column_size / number_of_threads;
  unsigned int begin_index = chunk_size * thread_id;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    chunk_size = column_size - (number_of_threads - 1) * chunk_size;
  }
  // cout << "Thread " << thread_id << " begin index: " <<  begin_index << " end
  // index: " << end_index << endl;
  TID pos = begin_index;
  /*    unsigned int* result_array = new unsigned int[chunk_size];
   assert(result_array!=NULL);
   */
  CoGaDB::CDK::selection::variants::bf_unrolled_simd_selection_int_thread(
      array + begin_index, chunk_size, value_for_comparison, comp, result_tids,
      &pos, begin_index);
  //}
  // std::cout << "ID: " << thread_id << " SIZE: " << pos-begin_index <<
  // std::endl;
  // write result size to array
  *result_size = pos;
}

// serial SIMD selection
template <class T>
const PositionListPtr serial_SIMD_selection(
    T* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  return CoGaDB::CDK::selection::variants::serial_selection(
      array, array_size, value_for_comparison, comp);
}

template <>
const PositionListPtr serial_SIMD_selection(
    float* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  float value;

  if (value_for_comparison.type() != typeid(float)) {
    // allow comparison with itnegers as well
    if (value_for_comparison.type() == typeid(int)) {
      value = boost::any_cast<int>(value_for_comparison);
    } else {
      COGADB_FATAL_ERROR(std::string("Typemismatch for column") +
                             std::string(" Column Type: ") +
                             typeid(float).name() +
                             std::string(" filter value type: ") +
                             value_for_comparison.type().name(),
                         "");
    }
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<float>(value_for_comparison);
  }

  return CoGaDB::simd_selection_float(array, array_size, value, comp);
}

template <>
const PositionListPtr serial_SIMD_selection(
    int* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  int value;

  if (value_for_comparison.type() != typeid(int)) {
    COGADB_FATAL_ERROR(std::string("Typemismatch for column") +
                           std::string(" Column Type: ") + typeid(int).name() +
                           std::string(" filter value type: ") +
                           value_for_comparison.type().name(),
                       "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<int>(value_for_comparison);
  }
  return CoGaDB::simd_selection_int(array, array_size, value, comp);
}

template <class T>
const PositionListPtr serial_bf_SIMD_selection(
    T* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  return CoGaDB::CDK::selection::variants::serial_bf_selection(
      array, array_size, value_for_comparison, comp);
}

template <>
const PositionListPtr serial_bf_SIMD_selection(
    float* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  float value;

  if (value_for_comparison.type() != typeid(float)) {
    // allow comparison with itnegers as well
    if (value_for_comparison.type() == typeid(int)) {
      value = boost::any_cast<int>(value_for_comparison);
    } else {
      COGADB_FATAL_ERROR(std::string("Typemismatch for column") +
                             std::string(" Column Type: ") +
                             typeid(float).name() +
                             std::string(" filter value type: ") +
                             value_for_comparison.type().name(),
                         "");
    }
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<float>(value_for_comparison);
  }

  return CoGaDB::bf_simd_selection_float(array, array_size, value, comp);
}

template <>
const PositionListPtr serial_bf_SIMD_selection(
    int* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  int value;

  if (value_for_comparison.type() != typeid(int)) {
    COGADB_FATAL_ERROR(std::string("Typemismatch for column") +
                           std::string(" Column Type: ") + typeid(int).name() +
                           std::string(" filter value type: ") +
                           value_for_comparison.type().name(),
                       "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<int>(value_for_comparison);
  }
  return CoGaDB::bf_simd_selection_int(array, array_size, value, comp);
}

template <class T>
const PositionListPtr serial_unrolled_SIMD_selection(
    T* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  return CoGaDB::CDK::selection::variants::serial_unrolled_selection(
      array, array_size, value_for_comparison, comp);
};

template <>
const PositionListPtr serial_unrolled_SIMD_selection(
    int* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  int value;

  if (value_for_comparison.type() != typeid(int)) {
    COGADB_FATAL_ERROR(std::string("Typemismatch for column") +
                           std::string(" Column Type: ") + typeid(int).name() +
                           std::string(" filter value type: ") +
                           value_for_comparison.type().name(),
                       "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<int>(value_for_comparison);
  }
  if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;
  PositionListPtr result_tids = createPositionList();

  unsigned int result_size = 0;

  // size_t chunk_size = std::floor( array_size/8);

  CoGaDB::CDK::selection::variants::unrolled_simd_selection_int_thread(
      array, array_size, value, comp, result_tids->data(), &result_size, 0);

  return result_tids;
};

template <>
const PositionListPtr serial_unrolled_SIMD_selection(
    float* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  float value;

  if (value_for_comparison.type() != typeid(float)) {
    COGADB_FATAL_ERROR(
        std::string("Typemismatch for column") + std::string(" Column Type: ") +
            typeid(float).name() + std::string(" filter value type: ") +
            value_for_comparison.type().name(),
        "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<float>(value_for_comparison);
  }
  if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;
  PositionListPtr result_tids = createPositionList();
  unsigned int result_size = 0;

  CoGaDB::CDK::selection::variants::unrolled_simd_selection_float_thread(
      array, array_size, value, comp, result_tids->data(), &result_size, 0);

  return result_tids;
};

template <class T>
const PositionListPtr serial_bf_unrolled_SIMD_selection(
    T* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  return CoGaDB::CDK::selection::variants::serial_bf_unrolled_selection(
      array, array_size, value_for_comparison, comp);
};

template <>
const PositionListPtr serial_bf_unrolled_SIMD_selection(
    int* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  int value;

  if (value_for_comparison.type() != typeid(int)) {
    COGADB_FATAL_ERROR(std::string("Typemismatch for column") +
                           std::string(" Column Type: ") + typeid(int).name() +
                           std::string(" filter value type: ") +
                           value_for_comparison.type().name(),
                       "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<int>(value_for_comparison);
  }
  if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;
  PositionListPtr result_tids = createPositionList();
  unsigned int result_size = 0;

  CoGaDB::CDK::selection::variants::bf_unrolled_simd_selection_int_thread(
      array, array_size, value, comp, result_tids->data(), &result_size, 0);

  return result_tids;
};

template <>
const PositionListPtr serial_bf_unrolled_SIMD_selection(
    float* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  float value;

  if (value_for_comparison.type() != typeid(float)) {
    COGADB_FATAL_ERROR(
        std::string("Typemismatch for column") + std::string(" Column Type: ") +
            typeid(float).name() + std::string(" filter value type: ") +
            value_for_comparison.type().name(),
        "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<float>(value_for_comparison);
  }
  if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;
  PositionListPtr result_tids = createPositionList();

  unsigned int result_size = 0;

  // size_t chunk_size = std::floor( array_size/8);

  CoGaDB::CDK::selection::variants::bf_unrolled_simd_selection_float_thread(
      array, array_size, value, comp, result_tids->data(), &result_size, 0);

  return result_tids;
};

template <class T>
const PositionListPtr parallel_unrolled_SIMD_selection(
    T* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  return CoGaDB::CDK::selection::variants::parallel_unrolled_selection(
      array, array_size, value_for_comparison, comp, number_of_threads);
};

template <>
const PositionListPtr parallel_unrolled_SIMD_selection(
    float* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  PositionListPtr result_tids = createPositionList();
  // unsigned int number_of_threads=4;
  float value;

  if (value_for_comparison.type() != typeid(float)) {
    COGADB_FATAL_ERROR(
        std::string("Typemismatch for column") + std::string(" Column Type: ") +
            typeid(float).name() + std::string(" filter value type: ") +
            value_for_comparison.type().name(),
        "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<float>(value_for_comparison);
  }

  TID* result_tids_array = (unsigned int*)malloc(array_size * sizeof(int));

  std::vector<size_t> result_sizes(number_of_threads);
  boost::thread_group threads;
  // create a PositionListPtr of the maximal result size, so
  // that we can write the result tids in parallel to th vector
  // without the default latency
  threads.add_thread(new boost::thread(
      boost::bind(&CoGaDB::CDK::selection::resize_PositionListPtr_thread,
                  result_tids, array_size)));
  for (unsigned int i = 0; i < number_of_threads; i++) {
    // create a selection thread
    threads.add_thread(new boost::thread(boost::bind(
        &CoGaDB::CDK::selection::variants::unrolled_SIMD_selection_thread<
            float>,
        array, array_size, i, number_of_threads, value, comp, result_tids_array,
        &result_sizes[i])));
  }
  threads.join_all();

  std::vector<size_t> prefix_sum(number_of_threads + 1);
  prefix_sum[0] = 0;
  for (unsigned int i = 1; i < number_of_threads + 1; i++) {
    prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
  }

  // copy result chunks in vector
  unsigned int chunk_size = array_size / number_of_threads;
  for (unsigned int i = 0; i < number_of_threads; i++) {
    unsigned int begin_index = chunk_size * i;
    threads.add_thread(new boost::thread(
        boost::bind(&write_selection_result_thread, i, number_of_threads,
                    array_size, result_tids_array, result_tids, result_sizes[i],
                    prefix_sum[i], prefix_sum[i + 1])));
  }
  threads.join_all();
  // fit positionlist to actual result length
  result_tids->resize(prefix_sum[number_of_threads]);

  free(result_tids_array);

  return result_tids;
};

template <>
const PositionListPtr parallel_unrolled_SIMD_selection(
    int* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  PositionListPtr result_tids = createPositionList();
  // unsigned int number_of_threads=4;
  int value;

  if (value_for_comparison.type() != typeid(int)) {
    COGADB_FATAL_ERROR(std::string("Typemismatch for column") +
                           std::string(" Column Type: ") + typeid(int).name() +
                           std::string(" filter value type: ") +
                           value_for_comparison.type().name(),
                       "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<int>(value_for_comparison);
  }

  TID* result_tids_array = (unsigned int*)malloc(array_size * sizeof(int));

  std::vector<size_t> result_sizes(number_of_threads);
  boost::thread_group threads;
  // create a PositionListPtr of the maximal result size, so
  // that we can write the result tids in parallel to th vector
  // without the default latency
  threads.add_thread(new boost::thread(
      boost::bind(&CoGaDB::CDK::selection::resize_PositionListPtr_thread,
                  result_tids, array_size)));
  for (unsigned int i = 0; i < number_of_threads; i++) {
    // create a selection thread
    threads.add_thread(new boost::thread(boost::bind(
        &CoGaDB::CDK::selection::variants::unrolled_SIMD_selection_thread<int>,
        array, array_size, i, number_of_threads, value, comp, result_tids_array,
        &result_sizes[i])));
  }
  threads.join_all();

  std::vector<size_t> prefix_sum(number_of_threads + 1);
  prefix_sum[0] = 0;
  for (unsigned int i = 1; i < number_of_threads + 1; i++) {
    prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
  }

  // copy result chunks in vector
  unsigned int chunk_size = array_size / number_of_threads;
  for (unsigned int i = 0; i < number_of_threads; i++) {
    unsigned int begin_index = chunk_size * i;
    threads.add_thread(new boost::thread(
        boost::bind(&write_selection_result_thread, i, number_of_threads,
                    array_size, result_tids_array, result_tids, result_sizes[i],
                    prefix_sum[i], prefix_sum[i + 1])));
  }
  threads.join_all();
  // fit positionlist to actual result length
  result_tids->resize(prefix_sum[number_of_threads]);

  free(result_tids_array);

  return result_tids;
};

template <class T>
const PositionListPtr parallel_bf_unrolled_SIMD_selection(
    T* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  return CoGaDB::CDK::selection::variants::parallel_unrolled_selection(
      array, array_size, value_for_comparison, comp, number_of_threads);
};

template <>
const PositionListPtr parallel_bf_unrolled_SIMD_selection(
    float* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  PositionListPtr result_tids = createPositionList();
  // unsigned int number_of_threads=4;
  float value;

  if (value_for_comparison.type() != typeid(float)) {
    COGADB_FATAL_ERROR(
        std::string("Typemismatch for column") + std::string(" Column Type: ") +
            typeid(float).name() + std::string(" filter value type: ") +
            value_for_comparison.type().name(),
        "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<float>(value_for_comparison);
  }

  TID* result_tids_array = (unsigned int*)malloc(array_size * sizeof(int));

  std::vector<size_t> result_sizes(number_of_threads);
  boost::thread_group threads;
  // create a PositionListPtr of the maximal result size, so
  // that we can write the result tids in parallel to th vector
  // without the default latency
  threads.add_thread(new boost::thread(
      boost::bind(&CoGaDB::CDK::selection::resize_PositionListPtr_thread,
                  result_tids, array_size)));
  for (unsigned int i = 0; i < number_of_threads; i++) {
    // create a selection thread
    threads.add_thread(new boost::thread(boost::bind(
        &CoGaDB::CDK::selection::variants::bf_unrolled_SIMD_selection_thread<
            float>,
        array, array_size, i, number_of_threads, value, comp, result_tids_array,
        &result_sizes[i])));
  }
  threads.join_all();

  std::vector<size_t> prefix_sum(number_of_threads + 1);
  prefix_sum[0] = 0;
  for (unsigned int i = 1; i < number_of_threads + 1; i++) {
    prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
  }

  // copy result chunks in vector
  unsigned int chunk_size = array_size / number_of_threads;
  for (unsigned int i = 0; i < number_of_threads; i++) {
    unsigned int begin_index = chunk_size * i;
    threads.add_thread(new boost::thread(
        boost::bind(&write_selection_result_thread, i, number_of_threads,
                    array_size, result_tids_array, result_tids, result_sizes[i],
                    prefix_sum[i], prefix_sum[i + 1])));
  }
  threads.join_all();
  // fit positionlist to actual result length
  result_tids->resize(prefix_sum[number_of_threads]);

  free(result_tids_array);

  return result_tids;
};

template <>
const PositionListPtr parallel_bf_unrolled_SIMD_selection(
    int* __restrict__ array, size_t array_size,
    const boost::any& value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  PositionListPtr result_tids = createPositionList();
  // unsigned int number_of_threads=4;
  int value;

  if (value_for_comparison.type() != typeid(int)) {
    COGADB_FATAL_ERROR(std::string("Typemismatch for column") +
                           std::string(" Column Type: ") + typeid(int).name() +
                           std::string(" filter value type: ") +
                           value_for_comparison.type().name(),
                       "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<int>(value_for_comparison);
  }

  TID* result_tids_array = (unsigned int*)malloc(array_size * sizeof(int));

  std::vector<size_t> result_sizes(number_of_threads);
  boost::thread_group threads;
  // create a PositionListPtr of the maximal result size, so
  // that we can write the result tids in parallel to th vector
  // without the default latency
  threads.add_thread(new boost::thread(
      boost::bind(&CoGaDB::CDK::selection::resize_PositionListPtr_thread,
                  result_tids, array_size)));
  for (unsigned int i = 0; i < number_of_threads; i++) {
    // create a selection thread
    threads.add_thread(new boost::thread(boost::bind(
        &CoGaDB::CDK::selection::variants::bf_unrolled_SIMD_selection_thread<
            int>,
        array, array_size, i, number_of_threads, value, comp, result_tids_array,
        &result_sizes[i])));
  }
  threads.join_all();

  std::vector<size_t> prefix_sum(number_of_threads + 1);
  prefix_sum[0] = 0;
  for (unsigned int i = 1; i < number_of_threads + 1; i++) {
    prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
  }

  // copy result chunks in vector
  unsigned int chunk_size = array_size / number_of_threads;
  for (unsigned int i = 0; i < number_of_threads; i++) {
    unsigned int begin_index = chunk_size * i;
    threads.add_thread(new boost::thread(
        boost::bind(&write_selection_result_thread, i, number_of_threads,
                    array_size, result_tids_array, result_tids, result_sizes[i],
                    prefix_sum[i], prefix_sum[i + 1])));
  }
  threads.join_all();
  // fit positionlist to actual result length
  result_tids->resize(prefix_sum[number_of_threads]);

  free(result_tids_array);

  return result_tids;
};

#endif

}  // End namespace variants

}  // End namespace selection

namespace join {

const PositionListPairPtr radix_join(int* __restrict__ column1,
                                     const size_t& col1_array_size,
                                     int* __restrict__ column2,
                                     const size_t& col2_array_size) {
  // TODO: FIXME: implement heuristic to choose number of passes and number of
  // bits
  return radix_join(column1, col1_array_size, column2, col2_array_size, 13, 2);
}

const PositionListPairPtr radix_join(int* __restrict__ column1,
                                     const size_t& col1_array_size,
                                     int* __restrict__ column2,
                                     const size_t& col2_array_size,
                                     int total_number_of_bits,
                                     int number_of_passes) {
  PositionListPairPtr join_tids(new PositionListPair());
  join_tids->first = createPositionList();
  join_tids->second = createPositionList();

  if (col1_array_size == 0 || col2_array_size == 0) {
    return join_tids;
  }
  // reserve memory (assume a normal PK FK Join, Implementation can handle
  // arbitrary joins!)
  join_tids->first->reserve(std::max(col1_array_size, col2_array_size));
  join_tids->second->reserve(std::max(col1_array_size, col2_array_size));

  CoGaDB::Timestamp begin_radix_cluster_step = CoGaDB::getTimestamp();
  RadixHashTablePtr hash_table_col1 = createRadixHashTable(
      column1, col1_array_size, total_number_of_bits, number_of_passes);
  RadixHashTablePtr hash_table_col2 = createRadixHashTable(
      column2, col2_array_size, total_number_of_bits, number_of_passes);
  CoGaDB::Timestamp end_radix_cluster_step = CoGaDB::getTimestamp();

  assert(hash_table_col1->getNumberOfBuckets() ==
         hash_table_col1->getNumberOfBuckets());

  if (!quiet && verbose && debug)
    cout << "Number of Buckets: " << hash_table_col1->getNumberOfBuckets()
         << endl;
  CoGaDB::Timestamp begin_join_step = CoGaDB::getTimestamp();
  for (unsigned int i = 0; i < hash_table_col1->getNumberOfBuckets(); ++i) {
    RadixHashTable::BucketPtr bucket_col1 = hash_table_col1->getBucket(i);
    RadixHashTable::BucketPtr bucket_col2 = hash_table_col2->getBucket(i);

    RadixPair* begin_col1 = bucket_col1->begin();
    size_t array_size_col1 = bucket_col1->end() - bucket_col1->begin();

    RadixPair* begin_col2 = bucket_col2->begin();
    size_t array_size_col2 = bucket_col2->end() - bucket_col2->begin();

    if (!quiet && verbose && debug) {
      cout << "Column1: Bucket  " << i << " size: " << array_size_col1 << endl;
      cout << "Column2: Bucket  " << i << " size: " << array_size_col2 << endl;
      cout << "Perform Nested Loop Join: " << endl;
    }

    // RadixPair_nested_loop_join(begin_col1, array_size_col1, begin_col2,
    // array_size_col2, join_tids);
    RadixPair_serial_hash_join(begin_col1, array_size_col1, begin_col2,
                               array_size_col2, join_tids);
  }
  CoGaDB::Timestamp end_join_step = CoGaDB::getTimestamp();

  cout << "Time Radix Clustering: "
       << double(end_radix_cluster_step - begin_radix_cluster_step) /
              (1000 * 1000)
       << "ms" << endl;
  cout << "Time Joining Clusters: "
       << double(end_join_step - begin_join_step) / (1000 * 1000) << "ms"
       << endl;
  return join_tids;
}

// Idea: invisible Join creates Lookup Table for Fact Table, then the system
// just joins the prefiltered fact table with the dimension tables

// typedef std::pair<Table, ValueValuePredicate, KNF> InvisibleJoinSelection;
// struct InvisibleJoinSelection{
//    std::string table_name;
//    Predicate join_pred;
//    KNF_Selection_Expression knf_sel_expr;
//};
//
// typedef std::list<InvisibleJoinSelection> InvisibleJoinSelectionList;

// int bitmap_join(T* pk_column, size_t pk_column_size , T* fk_column, size_t
// fk_column_size, char* bitmap_for_matching_fks){
//
//
//}

void buildBitmapForFactTable(TablePtr fact_table,
                             InvisibleJoinSelection inv_join_sel,
                             char** bitmaps, unsigned int thread_id) {
  // PositionListPtr dim_tids =
  // complex_selection(inv_join_sel.first,inv_join_sel.second);
  TablePtr dim_tab_sel_result =
      Table::selection(CoGaDB::getTablebyName(inv_join_sel.table_name),
                       inv_join_sel.knf_sel_expr, LOOKUP, SERIAL);

  std::cout << "Result size of filtered Dimension Table '"
            << inv_join_sel.table_name << "': '"
            << dim_tab_sel_result->getNumberofRows() << "' rows" << std::endl;
  ColumnPtr col = dim_tab_sel_result->getColumnbyName(
      inv_join_sel.join_pred.getColumn1Name());
  boost::shared_ptr<LookupArray<int> > lookup_array =
      boost::dynamic_pointer_cast<LookupArray<int> >(col);
  PositionListPtr matching_tids = lookup_array->getPositionList();

  JoinIndexPtr join_index = JoinIndexes::instance().getJoinIndex(
      CoGaDB::getTablebyName(inv_join_sel.table_name),
      inv_join_sel.join_pred.getColumn1Name(), fact_table,
      inv_join_sel.join_pred.getColumn2Name());
  PositionListPtr result_rows_fact_table =
      fetchMatchingTIDsFromJoinIndex(join_index, matching_tids);
  char* matching_rows_fact_table_bitmap =
      (char*)calloc((fact_table->getNumberofRows() + 7) / 8, sizeof(char));
  CDK::selection::convertPositionListToBitmap(result_rows_fact_table,
                                              matching_rows_fact_table_bitmap,
                                              fact_table->getNumberofRows());

  // setBitmapMatchingTIDsFromJoinIndex(join_index, matching_tids,
  // matching_rows_fact_table_bitmap, fact_table->getNumberofRows());
  // write result bitmap to thread specific location
  bitmaps[thread_id] = matching_rows_fact_table_bitmap;
  // CDK::selection::print_bitmap(matching_rows_fact_table_bitmap,
  // fk_column->size());
  unsigned int num_result_rows = CDK::selection::countSetBitsInBitmap(
      matching_rows_fact_table_bitmap, fact_table->getNumberofRows());
  std::cout << "Result size of Join from Fact Table and "
            << inv_join_sel.table_name << ": '" << num_result_rows << "' rows"
            << std::endl;
}

void pk_fk_semi_join_thread(TablePtr filtered_fact_tab,
                            InvisibleJoinSelection inv_join_sel,
                            LookupTablePtr* result, unsigned int thread_id) {
  TablePtr dim_table = getTablebyName(inv_join_sel.table_name);
  TablePtr join_result = BaseTable::pk_fk_join(
      dim_table, inv_join_sel.join_pred.getColumn1Name(), filtered_fact_tab,
      inv_join_sel.join_pred.getColumn2Name(), HASH_JOIN, LOOKUP);

  std::cout << "Dim " << inv_join_sel.table_name
            << ": #Rows: " << join_result->getNumberofRows() << std::endl;
  std::list<std::string> columns_dim_table;
  TableSchema schema = dim_table->getSchema();
  TableSchema::iterator sit;
  for (sit = schema.begin(); sit != schema.end(); ++sit) {
    columns_dim_table.push_back(sit->second);
  }

  TablePtr dimension_table_semi_join =
      BaseTable::projection(join_result, columns_dim_table, MATERIALIZE, CPU);
  LookupTablePtr lookup_table =
      boost::dynamic_pointer_cast<LookupTable>(dimension_table_semi_join);
  // dimension_semi_joins.push_back(lookup_table);
  *result = lookup_table;
}

TablePtr invisibleJoin(TablePtr fact_table,
                       InvisibleJoinSelectionList dimensions) {
  InvisibleJoinSelectionList::iterator it;
  const uint32_t NUM_DIM_TABLES = dimensions.size();
  std::vector<char*> bitmaps(NUM_DIM_TABLES);

  Timestamp begin_create_matching_rows_bitmap = getTimestamp();
  unsigned int counter = 0;

  // for(it=dimensions.begin();it!=dimensions.end();++it){
  //    //PositionListPtr dim_tids = complex_selection(it->first,it->second);
  //    TablePtr dim_tab_sel_result =
  //    Table::selection(CoGaDB::getTablebyName(it->table_name),
  //    it->knf_sel_expr, LOOKUP, SERIAL);
  //
  //    std::cout << "Result size of filtered Dimension Table " <<
  //    it->table_name << ": " << dim_tab_sel_result->getNumberofRows() << "
  //    rows" << std::endl;
  //    ColumnPtr col =
  //    dim_tab_sel_result->getColumnbyName(it->join_pred.getColumn1Name());
  //    boost::shared_ptr<LookupArray<int> > lookup_array =
  //    boost::dynamic_pointer_cast<LookupArray<int> >(col);
  //    PositionListPtr matching_tids = lookup_array->getPositionList();
  //
  //    JoinIndexPtr join_index =
  //    JoinIndexes::instance().getJoinIndex(CoGaDB::getTablebyName(it->table_name),
  //    it->join_pred.getColumn1Name(), fact_table,
  //    it->join_pred.getColumn2Name());
  //    PositionListPtr result_rows_fact_table =
  //    fetchMatchingTIDsFromJoinIndex(join_index, matching_tids);
  //    char* matching_rows_fact_table_bitmap = (char*)
  //    calloc((fact_table->getNumberofRows()+7)/8,sizeof(char));
  //    convertPositionListToBitmap(result_rows_fact_table,
  //    matching_rows_fact_table_bitmap, fact_table->getNumberofRows());
  //
  //    //setBitmapMatchingTIDsFromJoinIndex(join_index, matching_tids,
  //    matching_rows_fact_table_bitmap, fact_table->getNumberofRows());
  //
  //    bitmaps[counter++]=matching_rows_fact_table_bitmap;
  //    //CDK::selection::print_bitmap(matching_rows_fact_table_bitmap,
  //    fk_column->size());
  //    unsigned int num_result_rows =
  //    CDK::selection::countSetBitsInBitmap(matching_rows_fact_table_bitmap,
  //    fact_table->getNumberofRows());
  //    std::cout << "Result size of Join from Fact Table and " <<
  //    it->table_name << ": " << num_result_rows << " rows" << std::endl;
  //}

  // for(it=dimensions.begin();it!=dimensions.end();++it){
  //    buildBitmapForFactTable(fact_table, *it, bitmaps, counter++);
  //}

  hype::ProcessingDeviceID id = hype::PD0;
  ProcessorSpecification proc_spec(id);

  boost::thread_group threads;
  for (it = dimensions.begin(); it != dimensions.end(); ++it) {
    threads.add_thread(new boost::thread(boost::bind(
        &buildBitmapForFactTable, fact_table, *it, bitmaps.data(), counter++)));
  }
  threads.join_all();

  Timestamp end_create_matching_rows_bitmap = getTimestamp();

  Timestamp begin_combine_bitmaps = getTimestamp();
  char* matching_rows_bitmap =
      (char*)malloc((fact_table->getNumberofRows() + 7) / 8);
  // char* matching_rows_bitmap2 = (char*)
  // malloc((fact_table->getNumberofRows()+7)/8);
  memset(matching_rows_bitmap, 255, ((fact_table->getNumberofRows() + 7) / 8));
  // memset(matching_rows_bitmap2, 255, ((fact_table->getNumberofRows()+7)/8));
  for (unsigned int i = 0; i < NUM_DIM_TABLES; ++i) {
    CDK::selection::bitwise_and_thread(matching_rows_bitmap, bitmaps[i], (TID)0,
                                       (TID)fact_table->getNumberofRows(),
                                       matching_rows_bitmap);
    // std::swap(matching_rows_bitmap,matching_rows_bitmap2);
    // char* tmp=matching_rows_bitmap2;
    // matching_rows_bitmap2=matching_rows_bitmap;
    // matching_rows_bitmap=tmp;
    // bitmap_and_operation(bitmaps[0],bitmaps[i]);
  }
  for (unsigned int i = 0; i < NUM_DIM_TABLES; ++i) {
    free(bitmaps[i]);
  }
  PositionListPtr fact_table_tids =
      CDK::selection::createPositionListfromBitmap(
          matching_rows_bitmap, fact_table->getNumberofRows());

  free(matching_rows_bitmap);
  LookupTablePtr filtered_fact_tab = createLookupTableforUnaryOperation(
      "InvisibleJoin(Fact_Table)", fact_table, fact_table_tids, proc_spec);
  Timestamp end_combine_bitmaps = getTimestamp();

  cout << "Size of Filtered Fact Table (invisible Join): "
       << filtered_fact_tab->getNumberofRows() << "rows" << endl;

  Timestamp begin_construct_result = getTimestamp();
  TablePtr result = filtered_fact_tab;
  for (it = dimensions.begin(); it != dimensions.end(); ++it) {
    //	PositionList dim_tab_tids = fetchValuesFromDimTable(it->first,
    // fact_table, fact_table_tids);
    //	LookupColumn dim_table_lc = createLookupColumn(it->first,dim_tab_tids);
    //	result.concat(dim_table_lc);
    hype::ProcessingDeviceID id = hype::PD0;
    ProcessorSpecification proc_spec(id);
    JoinParam param(proc_spec, HASH_JOIN);
    result = Table::join(
        result, it->join_pred.getColumn2Name(), getTablebyName(it->table_name),
        it->join_pred.getColumn1Name(), param);  // HASH_JOIN, LOOKUP);
  }

  //        std::vector<LookupTablePtr> dimension_semi_joins(dimensions.size());
  //        unsigned int thread_counter=0;
  //        for(it=dimensions.begin();it!=dimensions.end();++it){
  //            threads.add_thread(new
  //            boost::thread(boost::bind(&CDK::join::pk_fk_semi_join_thread,filtered_fact_tab,
  //            *it, &dimension_semi_joins[thread_counter], thread_counter)));
  //            thread_counter++;
  //        }
  //
  //        threads.join_all();
  //
  //        LookupTablePtr
  //        result=boost::dynamic_pointer_cast<LookupTable>(filtered_fact_tab);
  //        for(unsigned int i=0;i<dimension_semi_joins.size();++i){
  //            result=LookupTable::concatenate("",*result,*dimension_semi_joins[i]);
  //        }
  Timestamp end_construct_result = getTimestamp();
  // result.concat(fact_tab);

  cout << "Stable Invisible Join: Time for Creating Matching Rows Bitmaps: "
       << double(end_create_matching_rows_bitmap -
                 begin_create_matching_rows_bitmap) /
              (1000 * 1000)
       << "ms" << std::endl;
  cout << "Stable Invisible Join: Time for Combining Bitmaps: "
       << double(end_combine_bitmaps - begin_combine_bitmaps) / (1000 * 1000)
       << "ms" << std::endl;
  cout << "Stable: Time for Creating Matching Rows TID List: "
       << double(end_create_matching_rows_bitmap -
                 begin_create_matching_rows_bitmap) /
                  (1000 * 1000) +
              double(end_combine_bitmaps - begin_combine_bitmaps) /
                  (1000 * 1000)
       << "ms" << std::endl;

  cout << "Stable Invisible Join: Time for Result Construction: "
       << double(end_construct_result - begin_construct_result) / (1000 * 1000)
       << "ms" << std::endl;

  //    char* dim_table_bitmap = bitmap_scan(it->first,it->second);
  //    dim_join_column = fetch_values(it->first,
  //    ValueValuePredicate->getColumnName(), dim_table_bitmap);

  return result;
}

}  // end namespace join

namespace aggregation {

typedef GroupingKeys::value_type GroupingKeyType;

//            PositionListPtr groupby(const
//            std::vector<DictionaryCompressedCol*>& dict_compressed_columns);
//
//            unsigned int getGreaterPowerOfTwo(unsigned int val);
//            uint32_t createBitmask(unsigned int num_set_bits);

uint32_t createBitmask(unsigned int num_set_bits) {
  uint32_t result = 0;
  for (unsigned int i = 0; i < num_set_bits; ++i) {
    result = result << 1;
    result = result | 1;
  }
  return result;
}

struct BitPack {
  BitPack(uint32_t* packed_values, size_t num_values, size_t num_bits);
  uint32_t* packed_values;
  size_t num_values;
  size_t num_bits;
};
typedef boost::shared_ptr<BitPack> BitPackPtr;

inline BitPack::BitPack(uint32_t* packed_values_, size_t num_values_,
                        size_t num_bits_)
    : packed_values(packed_values_),
      num_values(num_values_),
      num_bits(num_bits_) {}

BitPackPtr BitPackColumn(ColumnPtr col);
BitPackPtr BitPackArray(uint32_t* values, size_t num_rows);
size_t getNumberOfBitsForArrayElements(uint32_t* values, size_t num_rows);

//            void ArrayLeftShift(uint32_t* values, size_t num_rows, size_t
//            num_bits_to_shift);
//            void ArrayBitwiseOr(uint32_t* left_input, uint32_t* right_input,
//            size_t size, uint32_t* result_bitmap);
//            void AddTidsToGroupidArray(uint32_t* bitpacked_groupids, uint64_t
//            bitpacked_result, size_t size);

void ArrayLeftShift(ColumnGroupingKeysPtr col_group_keys, size_t num_rows,
                    size_t num_bits_to_shift) {
  if (!col_group_keys) return;
  GroupingKeyType* values = col_group_keys->keys->data();
  assert(values != NULL);
  if (num_bits_to_shift == 0) return;
  // shift values
  for (unsigned int i = 0; i < num_rows; ++i) {
    values[i] = values[i] << num_bits_to_shift;
  }
}

void ArrayBitwiseOr(GroupingKeyType* left_input, GroupingKeyType* right_input,
                    size_t size, GroupingKeyType* result_bitmap) {
  for (TID i = 0; i < size; i++) {
    result_bitmap[i] = left_input[i] | right_input[i];
  }
}

void AddTidsToGroupidArray(uint32_t* bitpacked_groupids,
                           uint64_t* bitpacked_result, size_t size) {
  for (uint32_t i = 0; i < size; i++) {
    uint64_t val = bitpacked_groupids[i];
    // std::cout << "groupid "<< bitpacked_groupids[i] << "(" << val << ")" <<
    // std::endl;
    val <<= 32;
    // std::cout << "shift(groupid) "<< std::bitset<32>(val >> 32) <<
    // std::bitset<32>(val) << std::endl;
    val = val | i;
    bitpacked_result[i] = val;
    // std::cout << "Pack groupid "<< bitpacked_groupids[i] << " (" <<
    // std::bitset<32>(bitpacked_groupids[i])  << ") and Position " << i << " to
    // " << bitpacked_result[i] << " (" << std::bitset<32>(bitpacked_result[i]
    // >> 32) << std::bitset<32>(bitpacked_result[i]) << ")" << std::endl;
  }
}

PositionListPtr fetchPositionListFromBitPackedArray(uint64_t* bitpacked_array,
                                                    size_t size) {
  PositionListPtr tids = createPositionList();
  tids->resize(size);
  TID* tid_array = tids->data();

  for (TID i = 0; i < size; i++) {
    uint64_t val = bitpacked_array[i];
    val = val & 0xffffffff;
    tid_array[i] = val;
  }
  return tids;
}

// we handle signed int columns as uint columns, because in case an
// int value is negative, bit 31 is setm hence, we need the complete
// 31 bit to store a negative number
// which is automatically handled by the bitpacking (in case the
// column contains negative elements, the maximum element has bit 31 set)
uint32_t* getRawColumnData(ColumnPtr col) {
  if (col->getType() == INT) {
    // is LookupArray?
    if (!col->isMaterialized() && !col->isCompressed()) {
      boost::shared_ptr<LookupArray<int> > lookup_array =
          boost::dynamic_pointer_cast<LookupArray<int> >(col);
      int* values = lookup_array->materializeToArray();
      assert(values != NULL);
      return (uint32_t*)values;
    } else if (col->isMaterialized()) {
      boost::shared_ptr<Column<int> > column =
          boost::dynamic_pointer_cast<Column<int> >(col);
      int* values = column->data();
      assert(values != NULL);
      // size_t num_bits = getNumberOfBitsForArrayElements(uint32_t* values,
      // size_t num_rows)
      uint32_t* result = (uint32_t*)malloc(sizeof(uint32_t) * col->size());
      assert(result != NULL);
      std::memcpy(result, values, sizeof(uint32_t) * col->size());
      return result;
    } else {
      COGADB_ERROR(
          "BitPacking of Arbitrarily Compressed Columns (INT) not yet "
          "supported!",
          "");
      return NULL;
    }
  } else if (col->getType() == FLOAT) {
    COGADB_ERROR("BitPacking of Columns (FLOAT) not yet supported!", "");
    return NULL;
  } else if (col->getType() == VARCHAR) {
    if (col->isCompressed()) {
      DictionaryCompressedCol* dict_compressed_col =
          dynamic_cast<DictionaryCompressedCol*>(col.get());
      assert(dict_compressed_col != NULL);
      uint32_t* values = dict_compressed_col->getIdData();
      // result=BitPackArray((uint32_t*)values,col->size());
      uint32_t* result = (uint32_t*)malloc(sizeof(uint32_t) * col->size());
      assert(result != NULL);
      std::memcpy(result, values, sizeof(uint32_t) * col->size());
      return result;
      // is Lookup Array?
    } else if (!col->isMaterialized() && !col->isCompressed()) {
      boost::shared_ptr<LookupArray<std::string> > lookup_array =
          boost::dynamic_pointer_cast<LookupArray<std::string> >(col);
      PositionListPtr tids = lookup_array->getPositionList();
      boost::shared_ptr<ColumnBaseTyped<std::string> > typed_column =
          lookup_array->getIndexedColumn();
      // assert(typed_column->isCompressed());
      DictionaryCompressedCol* dict_compressed_col =
          dynamic_cast<DictionaryCompressedCol*>(typed_column.get());
      if (!dict_compressed_col) return NULL;
      assert(dict_compressed_col != NULL);
      uint32_t* values = dict_compressed_col->getIdData();

      //                        ColumnBaseTyped<std::string>* col_values =
      //                        dynamic_cast<ColumnBaseTyped<std::string>*>(typed_column.get());
      //                        assert(col_values!=NULL);
      //                        for(unsigned int i=0;i<col->size();++i){
      //                            std::cout << "id " << values[i] << " Value:
      //                            "<< (*col_values)[i] << std::endl;
      //                        }

      // result=BitPackArray((uint32_t*)values,col->size());
      uint32_t* result = (uint32_t*)malloc(sizeof(uint32_t) * col->size());
      assert(result != NULL);
      // fetch compressed values from base column
      CDK::util::parallel_gather(values, tids->data(), col->size(), result);
      // std::memcpy(result,values,sizeof(uint32_t)*col->size());

      //                        ColumnBaseTyped<std::string>* col_values =
      //                        dynamic_cast<ColumnBaseTyped<std::string>*>(typed_column.get());
      //                        assert(col_values!=NULL);
      //                        for(unsigned int i=0;i<tids->size();++i){
      //                            std::cout << "id " << values[(*tids)[i]] <<
      //                            " fetched id: " << result[i] << " Value: "<<
      //                            (*col_values)[(*tids)[i]] << std::endl;
      //                        }

      return result;
    }
  } else {
    COGADB_FATAL_ERROR("Unknown Type!", "");
    return NULL;
  }
  return NULL;
}

size_t getNumberOfBitsForArrayElements(uint32_t* values, size_t num_rows) {
  assert(values != NULL);
  // scan column and identify greatest value
  uint32_t max_value = std::numeric_limits<uint32_t>::min();
  for (unsigned int i = 0; i < num_rows; ++i) {
    if (max_value < values[i]) {
      max_value = values[i];
    }
  }
  // these are the number of bits we need to represent the actual values without
  // information loss
  unsigned int num_of_bits = getGreaterPowerOfTwo(max_value);
  return num_of_bits;
}

ColumnGroupingKeysPtr computeColumnGroupingKeysGeneric(
    const std::vector<ColumnPtr>& columns,
    const ProcessorSpecification& proc_spec) {
  assert(hype::util::isCPU(proc_spec.proc_id));

  hype::ProcessingDeviceMemoryID mem_id =
      hype::util::getMemoryID(proc_spec.proc_id);

  typedef boost::shared_ptr<std::string> StringColumnPtr;

  if (columns.empty()) return ColumnGroupingKeysPtr();

  std::vector<ColumnPtr> placed_columns;
  for (unsigned int i = 0; i < columns.size(); ++i) {
    ColumnPtr placed_col = copy_if_required(columns[i], proc_spec);
    if (!placed_col) {
      COGADB_ERROR(
          "Placement of Column " << columns[i]->getName() << " Failed!", "");
      return ColumnGroupingKeysPtr();
    }
    placed_columns.push_back(placed_col);
  }

  std::vector<StringDenseValueColumnPtr> column_pointers;
  std::vector<std::string*> data_pointers;
  for (unsigned int i = 0; i < placed_columns.size(); ++i) {
    StringDenseValueColumnPtr ptr =
        placed_columns[i]
            ->convertToDenseValueStringColumn();  // getRawColumnData(columns[i]);
    if (!ptr) {
      COGADB_ERROR(
          "computeColumnGroupingKeysGeneric failed for column "
              << placed_columns[i]->getName() << "("
              << CoGaDB::util::getName(placed_columns[i]->getColumnType())
              << ", " << CoGaDB::util::getName(placed_columns[i]->getType())
              << ")",
          "");
      return ColumnGroupingKeysPtr();
    }
    column_pointers.push_back(ptr);
    data_pointers.push_back(ptr->data());
  }

  typedef std::map<std::string, ColumnGroupingKeys::GroupingKeysType>
      Dictionary;
  //                typedef
  //                boost::unordered_map<std::string,ColumnGroupingKeys::GroupingKeysType>
  //                Dictionary;

  const size_t number_of_rows = placed_columns.front()->size();
  const size_t number_of_columns = placed_columns.size();
  ColumnGroupingKeys::GroupingKeysType max_id = 0;

  ColumnPtr group_ids_untyped = createColumn(OID, "", PLAIN_MATERIALIZED);
  GroupingKeysPtr grouping_keys =
      boost::dynamic_pointer_cast<GroupingKeys>(group_ids_untyped);
  if (!grouping_keys) return ColumnGroupingKeysPtr();
  Dictionary dict;
  Dictionary::const_iterator dict_cit;
  ColumnGroupingKeysPtr result_group_ids(new ColumnGroupingKeys(mem_id));

  for (size_t i = 0; i < number_of_rows; ++i) {
    std::string group;
    for (size_t col_id = 0; col_id < number_of_columns; ++col_id) {
      group.append(data_pointers[col_id][i]);
    }
    dict_cit = dict.find(group);
    if (dict_cit == dict.end()) {
      ++max_id;
      std::pair<Dictionary::const_iterator, bool> ret =
          dict.insert(std::make_pair(group, max_id));
      dict_cit = ret.first;
    }
    result_group_ids->keys->insert(dict_cit->second);
  }
  result_group_ids->required_number_of_bits =
      ColumnGroupingKeys::getGreaterPowerOfTwo(max_id);
  return result_group_ids;
}

ColumnGroupingKeysPtr computeColumnGroupingKeys(
    const std::vector<ColumnPtr>& columns,
    const ProcessorSpecification& proc_spec) {
  //                return computeColumnGroupingKeysGeneric(columns, proc_spec);

  hype::ProcessingDeviceMemoryID mem_id =
      hype::util::getMemoryID(proc_spec.proc_id);

  if (columns.empty()) return ColumnGroupingKeysPtr();

  std::vector<ColumnPtr> placed_columns;
  for (unsigned int i = 0; i < columns.size(); ++i) {
    ColumnPtr placed_col = copy_if_required(columns[i], proc_spec);
    if (!placed_col) {
      COGADB_ERROR(
          "Placement of Column " << columns[i]->getName() << " Failed!", "");
      return ColumnGroupingKeysPtr();
    }
    placed_columns.push_back(placed_col);
  }

  // place columns

  // uint32_t* column_raw_pointers[columns.size()];
  std::vector<ColumnGroupingKeysPtr> column_raw_pointers;  //[columns.size()];
  size_t number_of_rows = placed_columns.front()->size();
  for (unsigned int i = 0; i < placed_columns.size(); ++i) {
    ColumnGroupingKeysPtr ptr = placed_columns[i]->createColumnGroupingKeys(
        proc_spec);  // getRawColumnData(columns[i]);
    if (!ptr) {
      return computeColumnGroupingKeysGeneric(columns, proc_spec);
    }
    column_raw_pointers.push_back(ptr);
  }

  std::vector<uint64_t> bits_per_column(placed_columns.size());
  for (unsigned int i = 0; i < placed_columns.size(); ++i) {
    //                    bits_per_column[i]=getNumberOfBitsForArrayElements(column_raw_pointers[i],
    //                    number_of_rows);
    bits_per_column[i] = column_raw_pointers[i]->required_number_of_bits;
  }

  uint64_t total_number_of_bits =
      std::accumulate(bits_per_column.begin(), bits_per_column.end(), 0ul);

  // can we apply our bit packing?
  if (total_number_of_bits >= sizeof(GroupingKeyType) * 8) {
    COGADB_ERROR(
        "Maximum Number of Bits for optimized groupby exceeded: max value: "
            << sizeof(GroupingKeyType) * 8 << " Got: " << total_number_of_bits,
        "");
    /* if we run on a CPU, we will execute the fallback algorithm
     * otherwise, we just return a NULL pointer
     */
    if (hype::util::isCPU(proc_spec.proc_id)) {
      std::cout << "Fallback to generic groupby..." << std::endl;
      return computeColumnGroupingKeysGeneric(columns, proc_spec);
    } else {
      return ColumnGroupingKeysPtr();
    }
  }

  // we get the number of bits to shift for each column by computing
  // the prefix sum of each columns number of bits
  std::vector<uint64_t> bits_to_shift(placed_columns.size() + 1);
  serial_prefixsum(bits_per_column.data(), placed_columns.size(),
                   bits_to_shift.data());

  for (unsigned int i = 0; i < placed_columns.size(); ++i) {
    // ArrayLeftShift(column_raw_pointers[i], number_of_rows, bits_to_shift[i]);

    ProcessorBackend<TID>* backend =
        ProcessorBackend<TID>::get(proc_spec.proc_id);
    BitShiftParam shift_param(proc_spec, SHIFT_BITS_LEFT, bits_to_shift[i]);
    if (!backend->bit_shift(column_raw_pointers[i], shift_param)) {
      COGADB_ERROR("Bitshift operation failed!", "");
      return ColumnGroupingKeysPtr();
    }
  }

  for (unsigned int i = 1; i < columns.size(); ++i) {
    // ArrayBitwiseOr(bitpacked_groupids, column_raw_pointers[i]->keys->data(),
    // number_of_rows, bitpacked_groupids);
    ProcessorBackend<TID>* backend =
        ProcessorBackend<TID>::get(proc_spec.proc_id);
    BitwiseCombinationParam bit_comb_param(proc_spec, BITWISE_OR);
    if (!backend->bitwise_combination(column_raw_pointers[0],
                                      column_raw_pointers[i], bit_comb_param)) {
      COGADB_ERROR("BitwiseCombination operation failed!", "");
      return ColumnGroupingKeysPtr();
    }
  }
  column_raw_pointers[0]->required_number_of_bits = total_number_of_bits;
  return column_raw_pointers[0];
}

//            PositionListPtr groupby(const
//            std::vector<DictionaryCompressedCol*>& dict_compressed_columns){
//                if(dict_compressed_columns.empty()) return PositionListPtr();
//                std::vector<unsigned int> bit_sizes;
//                for(unsigned int i=0;i<dict_compressed_columns.size();++i){
//                    uint32_t max_id =
//                    dict_compressed_columns[i]->getLargestID();
//                    unsigned int power_of_two = getGreaterPowerOfTwo(max_id);
//                    bit_sizes.push_back(power_of_two);
//                }
//                unsigned int total_num_bits_groupkey =
//                std::accumulate(bit_sizes.begin(),bit_sizes.end(),0);
//
//                if(total_num_bits_groupkey>sizeof(uint32_t)*8){
//                    return PositionListPtr();
//                }
//                unsigned int
//                column_size=dict_compressed_columns.front()->getNumberOfRows();
//                uint64_t* sort_array=(uint64_t*)
//                malloc(sizeof(uint64_t)*column_size);
//
//                for(unsigned int j=0;j<column_size;++j){
//                    unsigned int current_start_bit=0;
//                    uint64_t current_value=0;
//                    for(unsigned int
//                    i=0;i<dict_compressed_columns.size();++i){
//                        //pack bits to one compresed value
//                        //low 32 bits: TID of origin
//                        //high 32 bits: packed group key
//                        //group key <unused rest bits><bits col n>...<bits col
//                        1><bits col 0>
//                        uint32_t bitmask = createBitmask(bit_sizes[i]);
//                        bitmask = bitmask << current_start_bit;
//                        //(dict_compressed_columns[i] & bitmask)
//                        unsigned int val =
//                        (dict_compressed_columns[i]->getCompressedValues().at(j));
//                        current_value |= (val & bitmask);
//                        //move to upper 32-63 bits!
//                        current_value = current_value << 32;
//                        //make as lower part of the key the current TID
//                        current_value |= j;
//
//                        current_start_bit+=bit_sizes[i];
//                    }
//                    sort_array[j]=current_value;
//                }
//                //std::sort(sort_array,sort_array+column_size);
//                tbb::parallel_sort(sort_array,sort_array+column_size);
//
//                PositionListPtr result_tids=createPositionList();
//                result_tids->resize(column_size);
//
//                //extract TIDs from array of sorted <packed_groupkeys,tids>
//                uint32_t bitmask=pow(2,32)-1;
//                for(unsigned int j=0;j<column_size;++j){
//                    (*result_tids)[j]=(TID)(sort_array[j] & bitmask);
//                }
//
//                free(sort_array);
//                return result_tids;
//
//            }

unsigned int getGreaterPowerOfTwo(unsigned int val) {
  int current_power_of_two = 0;
  while (pow(2, current_power_of_two) <= val) {
    current_power_of_two++;
  }
  return current_power_of_two;
}
}

}  // end namespace CDK
}  // end namespace CoGaDB

#pragma GCC diagnostic pop
