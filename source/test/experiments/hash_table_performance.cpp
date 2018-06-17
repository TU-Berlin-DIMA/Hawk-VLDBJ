
#include <fstream>
#include <iostream>

#include <stdint.h>

#include <boost/random.hpp>

#include <algorithm>
#include <queue>
#include <string>
#include <utility>

#include <backends/cpu/hashtable.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_sort.h>
#include <boost/unordered_map.hpp>
#include <google/dense_hash_map>
#include <unordered_map>

#include <boost/chrono.hpp>
#include <cstdlib>

// needs -lrt (real-time lib)
// 1970-01-01 epoch UTC time, 1 mcs resolution (divide by 1M to get time_t)

typedef uint64_t Timestamp;
Timestamp getTimestamp();

using namespace boost::chrono;

Timestamp getTimestamp() {
  high_resolution_clock::time_point tp = high_resolution_clock::now();
  nanoseconds dur = tp.time_since_epoch();

  return (Timestamp)dur.count();
}

typedef uint64_t TID;

bool testHashTablePerformance() {
  //        size_t num_build_elements=1000000;
  //        size_t num_probe_elements=10000000;

  const size_t num_build_elements = static_cast<std::size_t>(1000) * 1000 * 50;
  const size_t num_probe_elements = static_cast<std::size_t>(1000) * 1000 * 10;
  const size_t num_distinct_keys = num_build_elements;

  //        std::cout << "Insert number of build elements: ";
  //        std::cin >> num_build_elements;
  //        std::cout << "Insert number of probe elements: ";
  //        std::cin >> num_probe_elements;

  TID* build_keys = (TID*)malloc(num_build_elements * sizeof(TID));
  TID* probe_keys = (TID*)malloc(num_probe_elements * sizeof(TID));

  assert(build_keys != NULL);
  assert(probe_keys != NULL);

  std::cout << "Generating data: " << std::endl;

  for (size_t i = 0; i < num_build_elements; ++i) {
    build_keys[i] = rand() % num_distinct_keys;  // 1000000;
  }

  for (size_t i = 0; i < num_probe_elements; ++i) {
    probe_keys[i] = rand() % num_distinct_keys;  // 1000000;
  }

  std::cout << "Start Benchmark: " << std::endl;

#define USE_SIMPLE_HASH_TABLE

//        google::dense_hash_map<TID,TID> google_ht;

#ifdef USE_SIMPLE_HASH_TABLE
  typedef CoGaDB::TypedHashTable<TID, TID> HashTable;
  HashTable ht(num_build_elements);
#else
  //        using namespace std::tr1;
  //        typedef std::tr1::unordered_map<TID,TID,std::tr1::hash<TID>,
  //        std::tr1::equal_to<TID> > HashTable;
  //        typedef std::tr1::unordered_multimap<TID,TID> HashTable;
  //        typedef boost::unordered_multimap<TID,TID> HashTable;

  typedef google::dense_hash_map<TID, TID> HashTable;
  //        typedef tbb::concurrent_unordered_multimap<TID,TID> HashTable;
  HashTable ht;
  ht.set_empty_key(std::numeric_limits<TID>::max());
  //        ht.resize(num_build_elements);
  typedef HashTable::iterator HashTableIterator;

#endif
  size_t counter = 0;
  Timestamp begin_build = getTimestamp();
  {
    //        COGADB_PCM_START_PROFILING("hash_build",std::cout);

    //#pragma omp parallel for
    for (size_t i = 0; i < num_build_elements; ++i) {
#ifdef USE_SIMPLE_HASH_TABLE
      HashTable::tuple_t t = {build_keys[i], i};
      ht.put(t);
#else
      //           std::pair<HashTableIterator,bool> ret =
      //           ht.insert(std::make_pair(build_keys[i], i));
      //           assert(ret.second==true);
      //           assert(ret.first!=ht.end());

      ht.insert(std::make_pair(build_keys[i], i));

//           HashTableIterator ret = ht.insert(std::make_pair(build_keys[i],
//           i));
//           assert(ret!=ht.end());
#endif
    }
    //        COGADB_PCM_STOP_PROFILING("hash_build", std::cout,
    //        num_build_elements,
    //                sizeof(TID), false, false, true);
  }
  Timestamp end_build = getTimestamp();

  Timestamp begin_probe = getTimestamp();
  {
    //        COGADB_PCM_START_PROFILING("hash_probe",std::cout);

    //#pragma omp parallel for
    for (size_t i = 0; i < num_probe_elements; ++i) {
#ifdef USE_SIMPLE_HASH_TABLE
      HashTable::hash_bucket_t* bucket = ht.getBucket(probe_keys[i]);
      if (bucket) do {
          for (size_t bucket_slot = 0; bucket_slot < bucket->count;
               bucket_slot++) {
            if (bucket->tuples[bucket_slot].key == probe_keys[i]) {
              counter++;
            }
          }
          bucket = bucket->next;
        } while (bucket);
#else
      std::pair<HashTableIterator, HashTableIterator> range =
          ht.equal_range(probe_keys[i]);
      HashTableIterator it;
      for (it = range.first; it != range.second; ++it) {
        counter++;
      }
#endif
    }

    //        COGADB_PCM_STOP_PROFILING("hash_probe", std::cout,
    //        num_probe_elements,
    //                sizeof(TID), false, false, true);
  }
  Timestamp end_probe = getTimestamp();

  std::cout << "#Matches: " << counter << std::endl;

#ifndef USE_SIMPLE_HASH_TABLE
  std::cout << "Hash Table contains " << ht.size() << " elements" << std::endl;
#endif

  //	Timestamp begin_sort = getTimestamp();
  //        tbb::parallel_sort(build_keys,build_keys+num_build_elements);
  //	Timestamp end_sort = getTimestamp();
  //
  //        Timestamp begin_binary_search_probe = getTimestamp();
  //        size_t counter=0;
  //        for(size_t i=0;i<num_probe_elements;++i){
  //            std::pair<TID*,TID*> bounds;
  //            bounds=std::equal_range
  //            (build_keys,build_keys+num_build_elements, probe_keys[i]);
  //            TID* cit;
  //            for(cit=bounds.first; cit!=bounds.second;++cit){
  //                counter++;
  //            }
  //        }
  //        Timestamp end_binary_search_probe = getTimestamp();
  //

  double time_hash_build_in_sec =
      double(end_build - begin_build) / (1000 * 1000 * 1000);
  double time_hash_probe_in_sec =
      double(end_probe - begin_probe) / (1000 * 1000 * 1000);
  //        double time_sort_in_sec =
  //        double(end_sort-begin_sort)/(1000*1000*1000);
  //        double time_binary_search_probe_in_sec =
  //        double(end_binary_search_probe-begin_binary_search_probe)/(1000*1000*1000);

  std::cout << "Build Time: " << time_hash_build_in_sec << "s\t"
            << (double(sizeof(TID) * num_build_elements) /
                (1024 * 1024 * 1024)) /
                   time_hash_build_in_sec
            << " GB/s\t"
            << "AVG Access Time: "
            << (time_hash_build_in_sec / num_build_elements) *
                   (1000 * 1000 * 1000)
            << "ns" << std::endl;
  std::cout << "Probe Time: " << time_hash_probe_in_sec << "s\t"
            << (double(sizeof(TID) * num_probe_elements) /
                (1024 * 1024 * 1024)) /
                   time_hash_probe_in_sec
            << " GB/s\t"
            << "AVG Access Time: "
            << (time_hash_probe_in_sec / num_probe_elements) *
                   (1000 * 1000 * 1000)
            << "ns" << std::endl;

//        std::cout << "Sort Time: "
//                << time_sort_in_sec << "s\t"
//                <<
//                (double(sizeof(TID)*num_build_elements)/(1024*1024*1024))/time_sort_in_sec
//                << " GB/s\t"
//                << "AVG Access Time: " <<
//                (time_sort_in_sec/num_build_elements)*(1000*1000*1000) << "ns"
//                << std::endl;
//
//        std::cout << "Binary Search Probe Time: " <<
//        time_binary_search_probe_in_sec << "s\t"
//                <<
//                (double(sizeof(TID)*num_probe_elements)/(1024*1024*1024))/time_binary_search_probe_in_sec
//                << " GB/s\t"
//                << "AVG Access Time: " <<
//                (time_binary_search_probe_in_sec/num_probe_elements)*(1000*1000*1000)
//                << "ns"
//                << std::endl;

#ifdef USE_SIMPLE_HASH_TABLE
  ht.printStatistics();
#endif

  free(build_keys);
  free(probe_keys);

  return true;
}

int main() {
  srand(0);

  google::dense_hash_map<TID, TID> google_ht;

  google_ht.set_empty_key(std::numeric_limits<TID>::max());

  testHashTablePerformance();

  return 0;
}
