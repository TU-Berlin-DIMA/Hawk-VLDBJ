
#include <cassert>
#include <core/column.hpp>
#include <hardware_optimizations/main_memory_joins/hash_joins.hpp>

#include <boost/unordered_map.hpp>
#include <iostream>

//#include "hardware_optimizations/main_memory_joins/radix_hash_joins.hpp"
namespace CoGaDB {

// using namespace query_processing;

using namespace std;

namespace unit_tests {

typedef std::pair<TID, TID> JoinPair;
typedef std::vector<JoinPair> JoinPairs;

JoinPairs convertToSortedJoinPairs(PositionListPairPtr tid_pairs) {
  JoinPairs jp;

  assert(tid_pairs->first->size() == tid_pairs->second->size());
  TID* first = tid_pairs->first->data();
  TID* second = tid_pairs->second->data();

  for (size_t i = 0; i < tid_pairs->first->size(); i++) {
    jp.push_back(JoinPair(first[i], second[i]));
  }

  std::sort(jp.begin(), jp.end());

  return jp;
}

void error_report(std::string error_message, JoinPairs jp_left,
                  JoinPairs jp_right) {
  // we assume the left side join pairs are correct

  size_t number_of_jps = jp_left.size();
  std::cout << "#Correct Matches: " << jp_left.size() << std::endl;
  if (jp_left.size() != jp_right.size()) {
    std::cout << "#Counted Matches: " << jp_right.size() << std::endl;
    number_of_jps = min(jp_left.size(), jp_right.size());
  }

  for (size_t i = 0; i < number_of_jps; i++) {
    //                cout << "Correct: " << jp_left[i].first << ", " <<
    //                jp_left[i].second << endl;
    //                   if(jp_left[i]!=jp_right[i])
    //                     cout << "\tIncorrect: " <<  jp_right[i].first << ", "
    //                     <<  jp_right[i].second << endl;
    //                //break;

    //            if(jp_left[i]!=jp_right[i]){
    //                cout << "Variant:" << endl
    //                     << "\tCorrect: " << jp_left[i].first << ", " <<
    //                     jp_left[i].second << endl
    //                     << "\tIncorrect: " <<  jp_right[i].first << ", " <<
    //                     jp_right[i].second << endl;
    //                //break;
    //            }
  }

  COGADB_FATAL_ERROR(error_message, "");
}

bool main_memory_join_tests() {
  double scale_factor = 10;
  std::cout << "Enter Scale Factor: " << endl;
  std::cin >> scale_factor;

  const size_t TUPLES_IN_R = scale_factor * 30000;
  const size_t TUPLES_IN_S = scale_factor * 6000000;

  typedef boost::shared_ptr<Column<int> > IntColumnPtr;
  IntColumnPtr r_column(new Column<int>("R", INT));
  IntColumnPtr s_column(new Column<int>("S", INT));

  std::cout << "Tuples in R: " << TUPLES_IN_R << std::endl;
  for (size_t i = 0; i < TUPLES_IN_R; ++i) {
    r_column->insert(int(random() % TUPLES_IN_R));
  }

  std::cout << "Tuples in S: " << TUPLES_IN_S << std::endl;
  for (size_t i = 0; i < TUPLES_IN_S; ++i) {
    s_column->insert(int(random() % TUPLES_IN_R));
  }
  //        r_column->print();
  //        s_column->print();

  std::cout << "Performing Nested Loop Join: " << std::endl;
  PositionListPairPtr nlj_tid_pair = r_column->nested_loop_join(s_column);

  std::cout << "Performing Hash Join: " << std::endl;
  PositionListPairPtr hash_tid_pair = r_column->hash_join(s_column);

  std::cout << "Performing Hash Join (C): " << std::endl;
  PositionListPairPtr hash2_tid_pair = CDK::main_memory_joins::serial_hash_join(
      r_column->data(), r_column->size(), s_column->data(), s_column->size());

  // std::cout << "Performing Parallel Hash Join (C + OpenMP): " << std::endl;
  // PositionListPairPtr hash3_tid_pair =
  // CDK::main_memory_joins::parallel_hash_join(r_column->data(),r_column->size(),
  // s_column->data(), s_column->size());

  //#ifndef __APPLE__
  //        std::cout << "Performing Serial Radix Hash Join: " << std::endl;
  //        PositionListPairPtr hash4_tid_pair =
  //        CDK::main_memory_joins::radix_hash_joins<int>::serial_radix_hash_join(r_column->data(),r_column->size(),
  //        s_column->data(), s_column->size());

  //        std::cout << "Performing Parallel Radix Hash Join: " << std::endl;
  //        PositionListPairPtr hash5_tid_pair =
  //        CDK::main_memory_joins::parallel_radix_hash_join(r_column->data(),r_column->size(),
  //        s_column->data(), s_column->size());
  //#endif

  JoinPairs nlj = convertToSortedJoinPairs(nlj_tid_pair);
  JoinPairs hash = convertToSortedJoinPairs(hash_tid_pair);
  JoinPairs hash2 = convertToSortedJoinPairs(hash2_tid_pair);
  // JoinPairs hash3 = convertToSortedJoinPairs(hash3_tid_pair);

  //#ifndef __APPLE__
  //        JoinPairs hash4 = convertToSortedJoinPairs(hash4_tid_pair);
  //        JoinPairs hash5 = convertToSortedJoinPairs(hash5_tid_pair);
  //#endif

  std::cout << "NLJ Matches: " << nlj.size() << endl;
  std::cout << "HJ Matches: " << hash.size() << endl;
  std::cout << "HJ (C) Matches: " << hash2.size() << endl;
  // std::cout << "Parallel HJ (C + OpenMP) Matches: " << hash3.size() << endl;

  //#ifndef __APPLE__
  //        std::cout << "Serial RHJ Matches: " << hash4.size() << endl;
  //        std::cout << "Parallel RHJ Matches: " << hash5.size() << endl;
  //#endif

  if (nlj != hash) {
    error_report("Wrong Result for Hash Join!", nlj, hash);
  }
  if (nlj != hash2) {
    error_report("Wrong Result for Hash Join (C)!", nlj, hash2);
  }
  // if(nlj!=hash3){
  //    error_report("Wrong Result for Parallel Hash Join (C + OpenMP)!",nlj,
  //    hash3);
  //}
  //#ifndef __APPLE__
  //       if(nlj!=hash4){
  //           error_report("Wrong Result for Serial Radix Hash Join!",nlj,
  //           hash4);
  //       }
  //       if(nlj!=hash5){
  //           error_report("Wrong Result for Serial Radix Hash Join!",nlj,
  //           hash5);
  //       }
  //#endif

  return true;
}
}
}  // end namespace CogaDB
