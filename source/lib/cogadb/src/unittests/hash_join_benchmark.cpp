#include <hardware_optimizations/main_memory_joins/parallel_radix/prj_params.h>
#include <core/column.hpp>
#include <hardware_optimizations/main_memory_joins/hash_joins.hpp>
#include <hardware_optimizations/main_memory_joins/radix_hash_joins.hpp>

#include <boost/thread/thread.hpp>

#include <iostream>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

namespace CoGaDB {

namespace unit_tests {

typedef boost::shared_ptr<Column<int> > IntColumnPtr;

inline double getMilliseconds(Timestamp end, Timestamp begin) {
  return (double(end - begin) / (1000 * 1000));
}

void generate_build_column(IntColumnPtr build_column, size_t BUILD_SIZE) {
  size_t i;
  build_column->clearContent();
  for (i = 0; i < BUILD_SIZE; ++i) {
    build_column->insert(int(random() % BUILD_SIZE));
  }
  std::random_shuffle(build_column->begin(), build_column->end());
}

void generate_probe_column(IntColumnPtr probe_column, size_t PROBE_SIZE,
                           IntColumnPtr build_column, size_t BUILD_SIZE,
                           double match_rate_modifed) {
  size_t i;
  probe_column->clearContent();
  for (i = 0; i < PROBE_SIZE * match_rate_modifed; ++i) {
    probe_column->insert(int(
        build_column->data()[random() % BUILD_SIZE]));  // random() % NUM_TUPLES
  }
  for (i = PROBE_SIZE * match_rate_modifed; i < PROBE_SIZE; ++i) {
    probe_column->insert(
        int(BUILD_SIZE + (random() % BUILD_SIZE)));  // random() % NUM_TUPLES
  }
  std::random_shuffle(probe_column->begin(), probe_column->end());
}

bool hash_join_benchmark() {
  unsigned int num_runs = 1;
  const unsigned int DELTA = 30000;
  const unsigned int MIN_SIZE = 30000;
  const unsigned int MAX_SIZE = 6000000;

  Timestamp begin, end;
  double ms;
  long double ms_avg[5];
  size_t i, k, l;
  double scale_factor = 1;
  double match_rate_modifier = 1;
  IntColumnPtr r_column(new Column<int>("R", INT));
  IntColumnPtr s_column(new Column<int>("S", INT));
  size_t TUPLES_IN_R, TUPLES_IN_S;

  std::cout << "starting hash_join_benchmark..." << std::endl;
#ifdef HAVE_CONFIG_H
  std::cout << "COGADB_L1_CACHE_SIZE = " << COGADB_L1_CACHE_SIZE << std::endl;
  std::cout << "COGADB_L1_CACHELINE_SIZE = " << COGADB_L1_CACHELINE_SIZE
            << std::endl;
  std::cout << "COGADB_L1_CACHE_ASSOCIATIVITY = "
            << COGADB_L1_CACHE_ASSOCIATIVITY << std::endl;
#endif
  std::cout << "L1_CACHE_SIZE_CB = " << L1_CACHE_SIZE_PRO << std::endl;
  std::cout << "CACHE_LINE_SIZE_CB = " << CACHE_LINE_SIZE_PRO << std::endl;
  std::cout << "L1_ASSOCIATIVITY_CB = " << L1_ASSOCIATIVITY_PRO << std::endl;

  std::cout << "boost::thread::hardware_concurrency(): "
            << boost::thread::hardware_concurrency() << std::endl;

  std::cout << "Enter Number Of Runs [>1]: " << std::endl;
  std::cin >> num_runs;
  std::cout << "Enter Scale Factor [>0.0001]: " << std::endl;
  std::cin >> scale_factor;
  std::cout << "Enter Match Rate Modifier[0.0-1.0]: " << std::endl;
  std::cin >> match_rate_modifier;

  for (i = MIN_SIZE; i <= MAX_SIZE; i += DELTA) {
    TUPLES_IN_R = scale_factor * i;
    TUPLES_IN_S = scale_factor * MAX_SIZE;
    std::cout << "TUPLES_IN_R: " << TUPLES_IN_R << std::endl;
    generate_build_column(r_column, TUPLES_IN_R);
    std::cout << "r_column->size(): " << r_column->size() << std::endl;

    std::cout << "TUPLES_IN_S: " << TUPLES_IN_S << std::endl;
    generate_probe_column(s_column, TUPLES_IN_S, r_column, TUPLES_IN_R,
                          match_rate_modifier);
    std::cout << "s_column->size(): " << s_column->size() << std::endl;

    //            r_column->print();
    //            s_column->print();

    for (k = 0; k < 5; k++) {
      ms_avg[k] = 0;
    }

    size_t matches = r_column->hash_join(s_column)->first->size();
    std::cout << "Hash Join Result Size:  " << matches << std::endl;

    std::cout << "RUNS: ";
    for (l = 0; l < num_runs; l++) {
      std::cout << l << " " << std::flush;

      //        std::cout << "Performing Hash Join: " << std::endl;
      begin = getTimestamp();
      PositionListPairPtr hash_tid_pair = r_column->hash_join(s_column);
      end = getTimestamp();
      ms = getMilliseconds(end, begin);
      //            std::cout << "Hash Join:  " << ms << "ms" << std::endl;
      ms_avg[0] += ms;

      begin = getTimestamp();
      //        std::cout << "Performing Hash Join (C): " << std::endl;
      PositionListPairPtr hash2_tid_pair =
          CDK::main_memory_joins::hash_joins<int>::serial_hash_join(
              r_column->data(), r_column->size(), s_column->data(),
              s_column->size());
      end = getTimestamp();
      ms = getMilliseconds(end, begin);
      //            std::cout << "Hash Join (C):  " << ms << "ms" << std::endl;
      ms_avg[1] += ms;

      begin = getTimestamp();
      //        std::cout << "Performing Parallel Hash Join (C + OpenMP): " <<
      //        std::endl;
      PositionListPairPtr hash3_tid_pair =
          CDK::main_memory_joins::hash_joins<int>::parallel_hash_join(
              r_column->data(), r_column->size(), s_column->data(),
              s_column->size());
      end = getTimestamp();
      ms = getMilliseconds(end, begin);
      //            std::cout << "Parallel Hash Join (C + OpemMP):  " << ms <<
      //            "ms" << std::endl;
      ms_avg[2] += ms;

      begin = getTimestamp();
      //        std::cout << "Performing Radix Hash Join: " << std::endl;
      PositionListPairPtr hash4_tid_pair =
          CDK::main_memory_joins::radix_hash_joins<int>::serial_radix_hash_join(
              r_column->data(), r_column->size(), s_column->data(),
              s_column->size());
      end = getTimestamp();
      ms = getMilliseconds(end, begin);
      //            std::cout << "Radix Hash Join:  " << ms << "ms" <<
      //            std::endl;
      ms_avg[3] += ms;

      begin = getTimestamp();
      //        std::cout << "Performing Parallel Radix Hash Join: " <<
      //        std::endl;
      PositionListPairPtr hash5_tid_pair =
          CDK::main_memory_joins::parallel_radix_hash_join(
              r_column->data(), r_column->size(), s_column->data(),
              s_column->size());
      end = getTimestamp();
      ms = getMilliseconds(end, begin);
      //            std::cout << "Parallel Radix Hash Join:  " << ms << "ms" <<
      //            std::endl;
      ms_avg[4] += ms;
    }
    std::cout << std::endl;
    std::cout << "Hash Join:  " << ms_avg[0] / num_runs << "ms" << std::endl;
    std::cout << "Hash Join (C):  " << ms_avg[1] / num_runs << "ms"
              << std::endl;
    std::cout << "Parallel Hash Join (C + OpemMP):  " << ms_avg[2] / num_runs
              << "ms" << std::endl;
    std::cout << "Radix Hash Join:  " << ms_avg[3] / num_runs << "ms"
              << std::endl;
    std::cout << "Parallel Radix Hash Join:  " << ms_avg[4] / num_runs << "ms"
              << std::endl;
  }

  std::cout << "finished hash_join_benchmark" << std::endl;
  return true;
}
}
}
