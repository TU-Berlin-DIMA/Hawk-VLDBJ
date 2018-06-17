#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define PRIME 1442968193
#include <math.h>
#include <stdint.h>

#include <iostream>
//#include <cstdint>
#include <bitset>
#include <cstring>
#include <vector>
using namespace std;

typedef unsigned int TID;

typedef struct {
  TID begin_index;
  TID end_index;
} RadixBucket;

typedef struct {
  RadixBucket* buckets;
  size_t number_of_buckets;
} RadixHashTable;

typedef struct {
  TID key;
  int value;
} RadixPair;

typedef unsigned char BIT;

void radix_cluster(int* array, size_t array_size, uint8_t total_number_of_bits,
                   uint8_t* bits_per_pass, size_t number_of_passess,
                   int* buckets, size_t total_number_of_buckets, size_t pass_id,
                   unsigned int global_cluster_id, int* result_0,
                   int* result_1) {
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
    int cluster_number = (array[i] & bitmask) >> number_of_processed_bits;
    // cout << "cluster number: " << cluster_number << endl;
    bucket_sizes[cluster_number]++;
    // cout << "New bucket Size" << bucket_sizes[cluster_number] << endl;
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
    int cluster_number = (array[i] & bitmask) >> number_of_processed_bits;
    result_0[insert_positions[cluster_number]++] = array[i];
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
    for (unsigned int j = 0; j < bucket_sizes[i]; ++j) {
      cout << result_0[j + bucket_borders[i]] << " Radix: "
           << (bitset<32>)((result_0[j + bucket_borders[i]] & bitmask) >>
                           number_of_processed_bits)
           << endl;
    }
  }

  if (pass_id + 1 < number_of_passess) {
    for (unsigned int i = 0; i < number_of_clusters; ++i) {
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
      radix_cluster(result_0 + bucket_borders[i], bucket_sizes[i],
                    total_number_of_bits, bits_per_pass, number_of_passess,
                    buckets, total_number_of_buckets, pass_id + 1,
                    new_global_cluster_id, result_1 + bucket_borders[i],
                    result_0 + bucket_borders[i]);

      //			unsigned int global_bucket_id = ;
      //			buckets[pass_id*number_of_clusters]
    }
  } else {
    // write bucket size in Hash Table
    for (unsigned int i = 0; i < number_of_clusters; ++i) {
      cout << "Writing Result Bucket Size for Global RadixBucket "
           << global_cluster_id + i
           << " (Buckets in Total: " << total_number_of_buckets << ")" << endl;
      buckets[global_cluster_id + i] = bucket_sizes[i];
    }
  }

  // if(result_array) free(result_array);
}

int main(int argc, char* argv[]) {
  std::vector<int> v;
  for (unsigned int i = 0; i < 100; i++) {
    v.push_back(rand() % 100);
  }

  int* result_0 = (int*)malloc(v.size() * sizeof(int));
  int* result_1 = (int*)malloc(v.size() * sizeof(int));

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

    // unsigned int number_of_passes=2; //2;
    // uint8_t bits_per_pass[number_of_passes];
    // bits_per_pass[0]=3;
    // bits_per_pass[1]=2;
    // unsigned int total_number_of_bits=5;

    unsigned int number_of_passes = 2;  // 2;
    uint8_t bits_per_pass[number_of_passes];
    bits_per_pass[0] = 3;
    bits_per_pass[1] = 2;
    unsigned int total_number_of_bits = 5;

    unsigned int total_number_of_buckets = pow(2, total_number_of_bits);
    int buckets[total_number_of_buckets];
    // buckets[0]=v.size(); //data in one bucket at beginning

    radix_cluster(&v[0], v.size(), total_number_of_bits, bits_per_pass,
                  number_of_passes, buckets, total_number_of_buckets, 0, 0,
                  result_0, result_1);

    int* result;
    if (number_of_passes % 2 == 0) {
      result = result_1;
    } else {
      result = result_0;
    }

    // compute prefix sum to get begin and end index for each bucket
    int bucket_borders[total_number_of_buckets + 1];
    bucket_borders[0] = 0;
    for (unsigned int i = 1; i <= total_number_of_buckets; ++i) {
      bucket_borders[i] = buckets[i - 1] + bucket_borders[i - 1];
    }

    unsigned int bitmask = pow(2, total_number_of_bits) - 1;
    cout << (bitset<32>)bitmask << endl;

    for (unsigned int i = 0; i < total_number_of_buckets; ++i) {
      cout << "Cluster " << i << " size: " << buckets[i] << " "
           << bucket_borders[i] << endl;
      for (unsigned int j = 0; j < buckets[i]; ++j) {
        cout << result[j + bucket_borders[i]] << " Radix: "
             << (bitset<32>)(result[j + bucket_borders[i]] & bitmask) << endl;
      }
    }
  }

  if (result_0) free(result_0);
  if (result_1) free(result_1);

  return 0;
}
