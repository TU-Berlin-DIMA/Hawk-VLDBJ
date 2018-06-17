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
using namespace std;

typedef unsigned int TID;

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
  RadixHashTable(RadixPair* value_array, int* bucket_borders,
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
    unsigned int bucket_id_;
    uint8_t total_number_of_radix_bits_;
  };
  typedef boost::shared_ptr<Bucket> BucketPtr;
  BucketPtr getBucket(unsigned int id);

 private:
  std::vector<int> bucket_borders_;
  RadixPair* value_array_;
  size_t number_of_buckets_;
  uint8_t total_number_of_radix_bits_;
};

typedef boost::shared_ptr<RadixHashTable> RadixHashTablePtr;

RadixHashTable::RadixHashTable(RadixPair* value_array, int* bucket_borders,
                               size_t number_of_buckets,
                               uint8_t total_number_of_radix_bits)
    : value_array_(value_array),
      bucket_borders_(bucket_borders, bucket_borders + number_of_buckets),
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
      value_array_(value_array),
      bucket_id_(bucket_id),
      total_number_of_radix_bits_(total_number_of_radix_bits) {}

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
  if (pass_id > 0) cout << "Partitioning Bucket of size " << array_size << endl;
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
  cout << "Number of Bits Processed: " << number_of_processed_bits << endl;
  // shift current bits to correct position for this pass
  bitmask <<= number_of_processed_bits;
  // number of clusters
  // cout << "#Clusters: 2^(" << number_of_processed_bits+bits_per_pass[pass_id]
  // << ")=" << pow(2,number_of_processed_bits+bits_per_pass[pass_id]) << endl;
  int number_of_clusters = pow(
      2,
      bits_per_pass
          [pass_id]);  // pow(2,number_of_processed_bits+bits_per_pass[pass_id]);
                       // //2^(number_of_processed_bits+bits_per_pass[pass_id])
  int bucket_sizes[number_of_clusters];
  //	for(unsigned int i=0;i<number_of_clusters;++i){
  //		bucket_sizes[i]=0;
  //	}
  std::memset(bucket_sizes, 0, number_of_clusters * sizeof(int));
  for (unsigned int i = 0; i < array_size; i++) {
    int cluster_number = (array[i].key & bitmask) >> number_of_processed_bits;

    cout << "put " << array[i].key << " in cluster number: " << cluster_number
         << endl;
    bucket_sizes[cluster_number]++;
    cout << "New bucket Size" << bucket_sizes[cluster_number] << endl;
  }
  // compute prefix sum to get begin and end index for each bucket
  int bucket_borders[number_of_clusters + 1];
  bucket_borders[0] = 0;
  for (unsigned int i = 1; i <= number_of_clusters; ++i) {
    bucket_borders[i] = bucket_sizes[i - 1] + bucket_borders[i - 1];
  }
  int insert_positions[number_of_clusters + 1];
  std::memcpy(insert_positions, bucket_borders,
              (number_of_clusters + 1) * sizeof(int));

  // int* result_array = (int*) malloc(sizeof(int)*array_size);
  // insert data in buckets
  for (unsigned int i = 0; i < array_size; i++) {
    int cluster_number = (array[i].key & bitmask) >> number_of_processed_bits;
    cout << "write pair at position " << i << "(" << array[i].key << ","
         << array[i].value << ")"
         << " to result_array at postion " << insert_positions[cluster_number]
         << endl;
    result_0[insert_positions[cluster_number]++] = array[i];
  }

  for (unsigned int i = 0; i < array_size; i++) {
    cout << "Result: " << result_0[i].key
         << " vs. Input Array: " << array[i].key << endl;
  }

  cout << "Pass: " << pass_id
       << " (Total Number of Passes: " << number_of_passess << ")" << endl;
  cout << "Bitmask: " << (bitset<32>)bitmask << endl;
  cout << "Bits in this pass: " << (int)bits_per_pass[pass_id] << endl;
  cout << "#Radix Cluster: " << number_of_clusters << endl;
  cout << "Content of Radix Hash Table: " << endl;

  for (unsigned int i = 0; i < number_of_clusters; ++i) {
    cout << "Cluster " << i << " size: " << bucket_sizes[i] << " "
         << bucket_borders[i] << endl;
    for (unsigned int j = bucket_borders[i]; j < bucket_borders[i + 1]; ++j) {
      cout << result_0[j].key << " Radix: " << (bitset<32>)result_0[j].key
           << "  " << (void*)&result_0[j]
           << endl;  //(bitset<32>)((result_0[j].key & bitmask) ) << endl; //>>
      // number_of_processed_bits) << endl;
    }
  }

  cout << "Current Partition:" << endl;
  for (unsigned int i = 0; i < array_size; i++) {
    cout << result_0[i].key << endl;
  }

  if (pass_id + 1 < number_of_passess) {
    for (unsigned int i = 0; i < number_of_clusters; ++i) {
      cout << "==================================================" << endl;
      // unsigned int new_number_of_buckets =
      // number_of_buckets*bits_per_pass[i+1];
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
      cout << "Writing Result Bucket Size (" << bucket_sizes[i]
           << ") for Global RadixBucket " << global_cluster_id + i
           << " (Buckets in Total: " << total_number_of_buckets << ")" << endl;
      buckets[global_cluster_id + i] = bucket_sizes[i];
    }
  }

  // if(result_array) free(result_array);
}

RadixHashTablePtr createRadixHashTable(int* array, size_t array_size) {
  RadixPair* result_0 = (RadixPair*)malloc(array_size * sizeof(RadixPair));
  RadixPair* result_1 = (RadixPair*)malloc(array_size * sizeof(RadixPair));

  for (unsigned int i = 0; i < array_size; i++) {
    result_1[i] = RadixPair(array[i], i);
  }
  unsigned int number_of_passes = 2;  // 2;
  uint8_t bits_per_pass[number_of_passes];
  bits_per_pass[0] = 3;
  bits_per_pass[1] = 2;
  unsigned int total_number_of_bits = 5;

  unsigned int total_number_of_buckets = pow(2, total_number_of_bits);
  int buckets[total_number_of_buckets];
  // buckets[0]=v.size(); //data in one bucket at beginning

  radix_cluster(result_1, array_size, total_number_of_bits, bits_per_pass,
                number_of_passes, buckets, total_number_of_buckets, 0, 0,
                result_0, result_1);

  // compute prefix sum to get begin and end index for each bucket
  int bucket_borders[total_number_of_buckets + 1];
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

  return RadixHashTablePtr(new RadixHashTable(result, bucket_borders,
                                              total_number_of_buckets + 1,
                                              total_number_of_bits));
}

// template<typename T>
void nested_loop_join(RadixPair* __restrict__ column1,
                      const size_t& col1_array_size,
                      RadixPair* __restrict__ column2,
                      const size_t& col2_array_size) {
  assert(column1 != NULL);
  assert(column2 != NULL);

  //                PositionListPairPtr join_tids( new PositionListPair());
  //                join_tids->first = PositionListPtr=createPositionList();
  //                join_tids->second = PositionListPtr=createPositionList();

  unsigned int join_column1_size = col1_array_size;
  unsigned int join_column2_size = col2_array_size;

  for (unsigned int i = 0; i < join_column1_size; i++) {
    for (unsigned int j = 0; j < join_column2_size; j++) {
      if (column1[i].key == column2[j].key) {
        std::cout << "MATCH: (" << column1[i].key << "," << column1[i].value
                  << "; " << column2[j].key << "," << column2[j].value << ")"
                  << std::endl;
        //                                        join_tids->first->push_back(i);
        //                                        join_tids->second->push_back(j);
      }
    }
  }
  // return join_tids;
}

void simple_nested_loop_join(int* __restrict__ column1,
                             const size_t& col1_array_size,
                             int* __restrict__ column2,
                             const size_t& col2_array_size) {
  assert(column1 != NULL);
  assert(column2 != NULL);

  //                PositionListPairPtr join_tids( new PositionListPair());
  //                join_tids->first = PositionListPtr=createPositionList();
  //                join_tids->second = PositionListPtr=createPositionList();

  unsigned int join_column1_size = col1_array_size;
  unsigned int join_column2_size = col2_array_size;

  for (unsigned int i = 0; i < join_column1_size; i++) {
    for (unsigned int j = 0; j < join_column2_size; j++) {
      if (column1[i] == column2[j]) {
        std::cout << "MATCH: (" << column1[i] << "," << i << "; " << column2[j]
                  << "," << j << ")" << std::endl;
        //                                        join_tids->first->push_back(i);
        //                                        join_tids->second->push_back(j);
      }
    }
  }
  // return join_tids;
}

void radix_join(int* __restrict__ column1, const size_t& col1_array_size,
                int* __restrict__ column2, const size_t& col2_array_size) {
  RadixHashTablePtr hash_table_col1 =
      createRadixHashTable(column1, col1_array_size);
  RadixHashTablePtr hash_table_col2 =
      createRadixHashTable(column2, col2_array_size);

  assert(hash_table_col1->getNumberOfBuckets() ==
         hash_table_col1->getNumberOfBuckets());

  cout << "Number of Buckets: " << hash_table_col1->getNumberOfBuckets()
       << endl;
  for (unsigned int i = 0; i < hash_table_col1->getNumberOfBuckets(); ++i) {
    RadixHashTable::BucketPtr bucket_col1 = hash_table_col1->getBucket(i);
    RadixHashTable::BucketPtr bucket_col2 = hash_table_col2->getBucket(i);

    RadixPair* begin_col1 = bucket_col1->begin();
    size_t array_size_col1 = bucket_col1->end() - bucket_col1->begin();

    RadixPair* begin_col2 = bucket_col2->begin();
    size_t array_size_col2 = bucket_col2->end() - bucket_col2->begin();

    cout << "Column1: Bucket  " << i << " size: " << array_size_col1 << endl;
    cout << "Column2: Bucket  " << i << " size: " << array_size_col2 << endl;
    cout << "Perform Nested Loop Join: " << endl;
    nested_loop_join(begin_col1, array_size_col1, begin_col2, array_size_col2);
  }
}

int main(int argc, char* argv[]) {
  {
    cout << "Keys: " << endl;
    std::vector<int> v;
    for (unsigned int i = 0; i < 100; i++) {
      v.push_back(rand() % 100);
    }
    // std::vector<RadixPair> v;
    // for(unsigned int i=0;i<100;i++){
    //	RadixPair p(rand()%100,i); //{rand()%100,i};
    //	v.push_back(p);
    //	//cout << p.key << "," << p.value << endl;
    // }
    // for(unsigned int i=0;i<v.size();i++){
    // 	cout << v[i].key << "," << v[i].value << endl;
    // }

    RadixHashTablePtr hash_table = createRadixHashTable(&v[0], v.size());

    hash_table->print();

    cout << "Number of Buckets: " << hash_table->getNumberOfBuckets() << endl;
    for (unsigned int i = 0; i < hash_table->getNumberOfBuckets(); ++i) {
      RadixHashTable::BucketPtr bucket = hash_table->getBucket(i);
      RadixPair* begin = bucket->begin();
      size_t array_size = bucket->end() - bucket->begin();
      cout << "Bucket  " << i << " size: " << array_size << endl;
      for (unsigned int j = 0; j < array_size; ++j) {
        cout << "Key: " << begin[j].key << "\tValue:" << begin[j].value << endl;
      }
    }

    //	for(unsigned int i=0;i<number_of_buckets_;++i){
    //		cout << "Cluster " << i << " size: " <<
    // bucket_borders_[i+1]-bucket_borders_[i] << " " << bucket_borders_[i] <<
    // endl;
    //		for(unsigned int
    // j=0;j<bucket_borders_[i+1]-bucket_borders_[i];++j){
    //			cout << value_array_[j+bucket_borders_[i]].key << "
    // Radix:
    //"
    //<<
    //(bitset<32>)(value_array_[j+bucket_borders_[i]].key & bitmask) << "  " <<
    //(void*) &value_array_[j]<< endl;
    //		}
    //	}

    // exit(0);
  }

  {
    cout << "Primary Keys: " << endl;
    std::vector<int> prim_keys;
    for (unsigned int i = 0; i < 100; i++) {
      prim_keys.push_back(i);
    }

    cout << "Foreign Keys: " << endl;
    std::vector<int> foreign_keys;
    for (unsigned int i = 0; i < 100; i++) {
      foreign_keys.push_back(rand() % prim_keys.size());
    }
    cout << "==================================================" << endl;
    cout << "Perform Radix Join: " << endl;
    // Perform Radix Join:
    radix_join(&prim_keys[0], prim_keys.size(), &foreign_keys[0],
               foreign_keys.size());

    cout << "==================================================" << endl;
    cout << "Perform Nested Loop Join: " << endl;
    // Perform Nested Loop Join:
    simple_nested_loop_join(&prim_keys[0], prim_keys.size(), &foreign_keys[0],
                            foreign_keys.size());
    exit(0);
  }

  std::vector<RadixPair> v;
  for (unsigned int i = 0; i < 100; i++) {
    RadixPair p(rand() % 100, i);  //{rand()%100,i};
    v.push_back(p);
    // cout << p.key << "," << p.value << endl;
  }
  for (unsigned int i = 0; i < v.size(); i++) {
    cout << v[i].key << "," << v[i].value << endl;
  }

  RadixPair* result_0 = (RadixPair*)malloc(v.size() * sizeof(RadixPair));
  RadixPair* result_1 = (RadixPair*)malloc(v.size() * sizeof(RadixPair));

  // {
  // uint8_t bits_per_pass[1];
  // bits_per_pass[0]=6;
  // unsigned int total_number_of_bits=6;
  // unsigned int total_number_of_buckets=pow(2,total_number_of_bits);
  // int buckets[total_number_of_buckets];
  // //buckets[0]=v.size(); //data in one bucket at beginning

  // radix_cluster(&v[0], v.size(), total_number_of_bits, bits_per_pass, 1,
  // buckets, 1, 0, result_0, result_1);
  // }

  {
    // unsigned int number_of_passes=1; //2;
    // uint8_t bits_per_pass[number_of_passes];
    // bits_per_pass[0]=6;
    // unsigned int total_number_of_bits=6;

    unsigned int number_of_passes = 2;  // 2;
    uint8_t bits_per_pass[number_of_passes];
    bits_per_pass[0] = 3;
    bits_per_pass[1] = 2;
    unsigned int total_number_of_bits = 5;

    // unsigned int number_of_passes=3; //2;
    // uint8_t bits_per_pass[number_of_passes];
    // bits_per_pass[0]=3;
    // bits_per_pass[1]=2;
    // bits_per_pass[2]=1;
    // unsigned int total_number_of_bits=6;

    unsigned int total_number_of_buckets = pow(2, total_number_of_bits);
    int buckets[total_number_of_buckets];
    // buckets[0]=v.size(); //data in one bucket at beginning

    radix_cluster(&v[0], v.size(), total_number_of_bits, bits_per_pass,
                  number_of_passes, buckets, total_number_of_buckets, 0, 0,
                  result_0, result_1);

    RadixPair* result;
    if (number_of_passes % 2 == 0) {
      result = result_1;
    } else {
      result = result_0;
    }

    // compute prefix sum to get begin and end index for each bucket
    int bucket_borders[total_number_of_buckets + 1];
    bucket_borders[0] = 0;
    for (unsigned int i = 1; i < total_number_of_buckets + 1; ++i) {
      bucket_borders[i] = buckets[i - 1] + bucket_borders[i - 1];
    }

    cout << "==================================================" << endl;

    unsigned int bitmask = pow(2, total_number_of_bits) - 1;
    cout << (bitset<32>)bitmask << endl;

    for (unsigned int i = 0; i < total_number_of_buckets; ++i) {
      cout << "Cluster " << i << " size: " << buckets[i] << " "
           << bucket_borders[i] << endl;
      for (unsigned int j = 0; j < buckets[i]; ++j) {
        cout << result[j + bucket_borders[i]].key << " Radix: "
             << (bitset<32>)(result[j + bucket_borders[i]].key & bitmask)
             << "  " << (void*)&result[j] << endl;
      }
    }

    //	for(unsigned int i=0;i<total_number_of_buckets;++i){
    //		for(unsigned int j=0;j<buckets[i];++j){
    //			cout << result_0[j+buckets[i]].key << endl;
    //		}
    //	}

    //	cout << "==================================================" << endl;

    //	for(unsigned int i=0;i<total_number_of_buckets;++i){
    //		for(unsigned int j=0;j<buckets[i];++j){
    //			cout << result_1[j+buckets[i]].key << endl;
    //		}
    //	}
  }

  if (result_0) free(result_0);
  if (result_1) free(result_1);

  return 0;
}
