

#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>

#include <boost/generator_iterator.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

#include <hardware_optimizations/primitives.hpp>

#include <unittests/unittests.hpp>
#include "util/begin_ptr.hpp"

#include <core/selection_expression.hpp>

#include <hardware_optimizations/main_memory_joins/hash_joins.hpp>

namespace CoGaDB {

namespace CDK {
namespace join {
template <typename T>
const PositionListPairPtr block_nested_loop_join(T* __restrict__ column1,
                                                 const size_t& col1_array_size,
                                                 T* __restrict__ column2,
                                                 const size_t& col2_array_size,
                                                 unsigned int block_size) {
  if (col1_array_size < col2_array_size) {
    PositionListPairPtr ret = block_nested_loop_join(
        column2, col2_array_size, column1, col1_array_size, block_size);
    std::swap(ret->first, ret->second);
    return ret;
  }
  std::cout << "Block Size: " << block_size << std::endl;

  assert(column1 != NULL);
  assert(column2 != NULL);

  PositionListPairPtr join_tids(new PositionListPair());
  join_tids->first = createPositionList();
  join_tids->second = createPositionList();

  size_t join_column1_size = col1_array_size;
  size_t join_column2_size = col2_array_size;

  unsigned int i = 0;
  unsigned int j = 0;

  for (; i + block_size < join_column1_size; i += block_size) {
    j = 0;
    for (; j + block_size < join_column2_size; j += block_size) {
      for (unsigned o = 0; o < block_size; ++o) {
        for (unsigned u = 0; u < block_size; ++u) {
          if (column1[i + o] == column2[j + u]) {
            if (debug)
              std::cout << "MATCH: (" << i + o << "," << j + u << ")"
                        << std::endl;
            join_tids->first->push_back(i + o);
            join_tids->second->push_back(j + u);
          }
        }
      }
    }
    for (; j < join_column2_size; j++) {
      for (unsigned o = 0; o < block_size; ++o) {
        if (column1[i + o] == column2[j]) {
          if (debug)
            std::cout << "MATCH: (" << i + o << "," << j << ")" << std::endl;
          join_tids->first->push_back(i + o);
          join_tids->second->push_back(j);
        }
      }
    }
  }

  for (; i < join_column1_size; i++) {
    for (j = 0; j < join_column2_size; j++) {
      if (column1[i] == column2[j]) {
        if (debug) std::cout << "MATCH: (" << i << "," << j << ")" << std::endl;
        join_tids->first->push_back(i);
        join_tids->second->push_back(j);
      }
    }
  }
  return join_tids;
}

}  // end namespace join

}  // end namespace CDK

namespace unit_tests {

using namespace std;

//    bool cdk_gather_test();
//    bool cdk_selection_tid_test();
//    bool cdk_selection_bitmap_test();
//    bool cdk_join_test();
//    bool cdk_join_performance_test();

bool primitives_unittests() {
  if (!cdk_gather_test()) {
    std::cerr << "Error Unittests: Gather Test Failed!" << std::endl;
    return false;
  }

  if (!cdk_selection_tid_test()) {
    std::cerr << "Error Unittests: Selection TID Test Failed!" << std::endl;
    return false;
  }

  if (!cdk_selection_bitmap_test()) {
    std::cerr << "Error Unittests: Selection Bitmap Test Failed!" << std::endl;
    return false;
  }

  if (!cdk_join_test()) {
    std::cerr << "Error Unittests: Join Test Failed!" << std::endl;
    return false;
  }

  //        if(!cdk_join_performance_test()){
  //            std::cerr << "Error Unittests: Join Test Failed!" << std::endl;
  //            return false;
  //        }

  return true;
}

bool cdk_gather_test() {
  const long long NUMBER_OF_ELEMENTS = 6 * 1000 * 1000;
  bool return_value = true;
  for (unsigned int u = 0; u < 10; ++u) {
    std::vector<int> array(NUMBER_OF_ELEMENTS);
    std::vector<TID> tid_array(NUMBER_OF_ELEMENTS);
    unsigned int number_of_threads = 8;

    for (unsigned int i = 0; i < NUMBER_OF_ELEMENTS; ++i) {
      array[i] = rand() % 100;
      tid_array[i] =
          rand() % NUMBER_OF_ELEMENTS;  // generate valid tids w.r.t. array
    }

    // std::sort(tid_array,tid_array+NUMBER_OF_ELEMENTS);
    for (unsigned int i = 0; i < 1; ++i) {
      uint64_t begin, end;
      double rel_time;
      uint64_t second = 1000 * 1000 * 1000;
      std::vector<int> serial_gather_result_array(NUMBER_OF_ELEMENTS);
      std::vector<int> result_array(NUMBER_OF_ELEMENTS);
      // serial gather
      begin = getTimestamp();
      CoGaDB::CDK::util::serial_gather(array.data(), tid_array.data(),
                                       NUMBER_OF_ELEMENTS,
                                       serial_gather_result_array.data());
      end = getTimestamp();
      assert(end > begin);
      rel_time = second / (end - begin);  /// 1000*1000*1000
      if (!quiet)
        std::cout << "Time Serial Gather Operation: "
                  << double(end - begin) / (1000 * 1000) << "ms ("
                  << (double(sizeof(int) * NUMBER_OF_ELEMENTS) * rel_time) /
                         (1024 * 1024)
                  << "MB/s)" << std::endl;
      // parallel gather
      begin = getTimestamp();
      CoGaDB::CDK::util::parallel_gather(
          array.data(), tid_array.data(), NUMBER_OF_ELEMENTS,
          result_array.data(), number_of_threads);
      end = getTimestamp();
      assert(end > begin);
      rel_time = second / (end - begin);  /// 1000*1000*1000
      if (!quiet)
        std::cout << "Time Parallel Gather Operation: "
                  << double(end - begin) / (1000 * 1000) << "ms ("
                  << (double(sizeof(int) * NUMBER_OF_ELEMENTS) * rel_time) /
                         (1024 * 1024)
                  << "MB/s)" << std::endl;
      // check for correctness
      bool ret =
          std::equal(serial_gather_result_array.begin(),
                     serial_gather_result_array.end(), result_array.begin());
      if (ret) {
        return_value = true;
      } else {
        return_value = false;
        COGADB_FATAL_ERROR(
            "Gather Unittests failed! At least one algorithm works incorrect!",
            "");
      }
    }
  }
  return return_value;
}

bool cdk_gather_performance_test() {
  const int NUMBER_OF_ELEMENTS = 6 * 1000 * 1000;
  std::vector<int> array(NUMBER_OF_ELEMENTS);
  std::vector<TID> tid_array(NUMBER_OF_ELEMENTS);
  unsigned int number_of_threads = 8;
  std::cout << "Enter Number of Threads: " << std::endl;
  //        std::cin >> number_of_threads;

  for (unsigned int i = 0; i < NUMBER_OF_ELEMENTS; ++i) {
    array[i] = rand() % 100;
    tid_array[i] =
        rand() % NUMBER_OF_ELEMENTS;  // generate valid tids w.r.t. array
  }

  std::sort(tid_array.begin(), tid_array.end());

  for (unsigned int i = 0; i < 20; ++i) {
    uint64_t begin = getTimestamp();
    std::vector<int> result_array(NUMBER_OF_ELEMENTS);
    CoGaDB::CDK::util::parallel_gather(array.data(), tid_array.data(),
                                       NUMBER_OF_ELEMENTS, result_array.data(),
                                       number_of_threads);
    uint64_t end = getTimestamp();
    assert(end > begin);
    uint64_t second = 1000 * 1000 * 1000;
    double rel_time = second / (end - begin);  /// 1000*1000*1000
    std::cout << "Time Gather Operation: "
              << double(end - begin) / (1000 * 1000) << "ms ("
              << (double(sizeof(int) * NUMBER_OF_ELEMENTS) * rel_time) /
                     (1024 * 1024)
              << "MB/s)" << std::endl;
  }

  return true;
}

bool cdk_selection_tid_test() {
  int selection_value;
  ValueComparator selection_comparison_value;  // 0 EQUAL, 1 LESSER, 2 LARGER

  boost::mt19937 rng;
  boost::uniform_int<> selection_values(0, 1000);
  boost::uniform_int<> filter_condition(0, 2);

  std::vector<int> column;
  // fill column
  for (unsigned int i = 0; i < 10 * 1000 * 1000; i++) {
    column.push_back(int(rand() % 1000));
  }

  for (unsigned int i = 0; i < 100; i++) {
    selection_value = selection_values(rng);
    selection_comparison_value =
        static_cast<ValueComparator>(filter_condition(rng));  // rand()%3;
    int* column_array = hype::util::begin_ptr(column);

    Timestamp begin;
    Timestamp end;

    PositionListPtr serial_selection_tids;
    PositionListPtr parallel_selection_tids;
    // PositionListPtr lock_free_parallel_selection_tids;

    {
      begin = getTimestamp();
      serial_selection_tids = CoGaDB::CDK::selection::serial_selection(
          column_array, column.size(), boost::any(selection_value),
          selection_comparison_value);  // column->selection(selection_value,selection_comparison_value);
      end = getTimestamp();
      cout << "Serial Selection: " << double(end - begin) / (1000 * 1000)
           << "ms" << endl;
    }
    {
      begin = getTimestamp();
      parallel_selection_tids = CoGaDB::CDK::selection::parallel_selection(
          column_array, column.size(), boost::any(selection_value),
          selection_comparison_value,
          4);  // column->parallel_selection(selection_value,selection_comparison_value,number_of_threads);
      end = getTimestamp();
      cout << "Parallel Selection: " << double(end - begin) / (1000 * 1000)
           << "ms" << endl;
    }
    //                            {
    //                            begin=getTimestamp();
    //                            lock_free_parallel_selection_tids =
    //                            column->lock_free_parallel_selection(selection_value,selection_comparison_value,number_of_threads);
    //                            end=getTimestamp();
    //                            cout << "Lock Free Parallel Selection: " <<
    //                            double(end-begin)/(1000*1000) << "ms" << endl;
    //                            }

    if ((*serial_selection_tids) != (*parallel_selection_tids)) {
      cout << "TID lists are not equal!" << endl;
      cout << "Serial Selection result size: " << serial_selection_tids->size()
           << endl;
      cout << "Parallel Selection result size: "
           << parallel_selection_tids->size() << endl;

      ////                                for(unsigned int i=0;i<size;i++){
      ////                                    cout << "Serial id: " <<
      ///(*serial_selection_tids)[i] << " \tParallel id:"<<
      ///(*parallel_selection_tids)[i] << endl;
      ////                                }
      //                                for(unsigned int i=0;i<size;i++){
      //                                    if((*serial_selection_tids)[i]!=(*parallel_selection_tids)[i])
      //                                         cout << "Serial id: " <<
      //                                         (*serial_selection_tids)[i] <<
      //                                         " \tParallel id:"<<
      //                                         (*parallel_selection_tids)[i]
      //                                         << endl;
      //                                }
      //                                if(size<serial_selection_tids->size()){
      //                                    cout << "Detected additional values
      //                                    for serial selection " << endl;
      //                                    for(unsigned int
      //                                    i=size;i<serial_selection_tids->size();i++){
      //                                           cout << "id: " << i << " val:
      //                                           " <<
      //                                           (*serial_selection_tids)[i]
      //                                           << endl;
      //                                    }
      //                                }
      COGADB_FATAL_ERROR(
          "Selection Unittests failed! At least one algorithm works incorrect!",
          "");
    }
    //                            assert((*serial_selection_tids)==(*lock_free_parallel_selection_tids));
  }
  return true;
}

bool cdk_selection_bitmap_test() {
  int selection_value;
  ValueComparator selection_comparison_value;  // 0 EQUAL, 1 LESSER, 2 LARGER

  cout << "Enter number of threads:" << endl;

  std::vector<int> column;
  // fill column
  for (unsigned int i = 0; i < 10 * 1000 * 1000; i++) {
    column.push_back(int(rand() % 1000));
  }

  selection_value = 3;  // selection_values(rng);
  selection_comparison_value =
      EQUAL;  //(ValueComparator)filter_condition(rng); //rand()%3;
  int* column_array = hype::util::begin_ptr(column);

  PositionListPtr serial_selection_tids;

  serial_selection_tids = CoGaDB::CDK::selection::serial_selection(
      column_array, column.size(), boost::any(selection_value),
      selection_comparison_value);
  char* result_bitmap =
      static_cast<char*>(calloc((column.size() + 7) / 8, sizeof(char)));
  CDK::selection::scan_column_equal_bitmap_thread(
      column_array, 0, column.size(), selection_value, result_bitmap);
  PositionListPtr bitmap_selection_tids =
      CDK::selection::createPositionListfromBitmap(result_bitmap,
                                                   column.size());

  free(result_bitmap);
  assert((*bitmap_selection_tids) == (*serial_selection_tids));

  return true;
}

bool cdk_unrolling_performance_test() {
  int selection_value;
  ValueComparator selection_comparison_value;  // 0 EQUAL, 1 LESSER, 2 LARGER

  std::vector<int> column;
  // fill column
  for (unsigned int i = 0; i < 60 * 1000 * 1000; i++) {
    column.push_back(int(rand() % 1000));
  }

  auto sel_values = boost::make_shared<vector<int>>();
  for (int i = 0; i < 11; i++) {
    sel_values->push_back(i * 100);
  }

  selection_comparison_value = static_cast<ValueComparator>(0);
  int* column_array = hype::util::begin_ptr(column);

  Timestamp begin;
  Timestamp end;

  PositionListPtr serial_selection_tids;
  PositionListPtr unrolled2_selection_tids;
  PositionListPtr unrolled3_selection_tids;
  PositionListPtr unrolled4_selection_tids;
  PositionListPtr unrolled5_selection_tids;
  PositionListPtr unrolled6_selection_tids;
  PositionListPtr unrolled7_selection_tids;
  PositionListPtr unrolled8_selection_tids;

  PositionListPtr serial_bf_selection_tids;
  PositionListPtr bf_unrolled2_selection_tids;
  PositionListPtr bf_unrolled3_selection_tids;
  PositionListPtr bf_unrolled4_selection_tids;
  PositionListPtr bf_unrolled5_selection_tids;
  PositionListPtr bf_unrolled6_selection_tids;
  PositionListPtr bf_unrolled7_selection_tids;
  PositionListPtr bf_unrolled8_selection_tids;

  cout << "Serial_Selection\t\t"
       << "Serial_BF_Selection\t\t"
       << "Serial_Unrolled2_Selection\t\t"
       << "Serial_BF_Unrolled2_Selection\t\t"
       << "Serial_Unrolled3_Selection\t\t"
       << "Serial_BF_Unrolled3_Selection\t\t"
       << "Serial_Unrolled4_Selection\t\t"
       << "Serial_BF_Unrolled4_Selection\t\t"
       << "Serial_Unrolled5_Selection\t\t"
       << "Serial_BF_Unrolled5_Selection\t\t"
       << "Serial_Unrolled6_Selection\t\t"
       << "Serial_BF_Unrolled6_Selection\t\t"
       << "Serial_Unrolled7_Selection\t\t"
       << "Serial_BF_Unrolled7_Selection\t\t"
       << "Serial_Unrolled8_Selection\t\t"
       << "Serial_BF_Unrolled8_Selection\t\t" << endl;

  for (unsigned int i = 0; i < 10; i++) {
    size_t col_size =
        static_cast<size_t>((((i % 10) + 1) / 10.0) * column.size());
    cout << "Col_Size: " << col_size << endl;
    for (unsigned int j = 0; j < 11; j++) {
      selection_value = sel_values->at(j % 11);
      for (unsigned int anz_iter = 0; anz_iter < 100; anz_iter++) {
        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_selection_tids =
              CoGaDB::CDK::selection::variants::serial_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(serial_selection_tids->size() * 100 / col_size) << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_bf_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(serial_bf_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled2_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled2_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled2_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled2_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled2_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled2_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled3_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled3_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled3_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled3_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled3_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled3_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled4_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled4_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled4_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled4_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled4_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled4_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled5_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled5_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled5_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled5_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled5_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled5_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled6_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled6_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled6_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled6_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled6_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled6_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled7_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled7_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled7_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled7_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled7_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled7_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled8_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled8_selection_tids->size() * 100 / col_size)
               << "\n";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled8_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled8_selection_tids->size() * 100 / col_size)
               << "\n";
        }
      }
    }
  }

  return true;
}

bool cdk_unrolling_performance_test_float() {
  int selection_value;
  ValueComparator selection_comparison_value;  // 0 EQUAL, 1 LESSER, 2 LARGER

  std::vector<int> column;
  // fill column
  for (unsigned int i = 0; i < 60 * 1000 * 1000; i++) {
    column.push_back(int(rand() % 1000));
  }

  boost::shared_ptr<vector<int>> sel_values(new vector<int>());
  for (unsigned int i = 0; i < 11; i++) {
    sel_values->push_back(int(i * 100));
  }

  selection_comparison_value = static_cast<ValueComparator>(0);
  int* column_array = hype::util::begin_ptr(column);

  Timestamp begin;
  Timestamp end;

  PositionListPtr serial_selection_tids;
  PositionListPtr unrolled2_selection_tids;
  PositionListPtr unrolled3_selection_tids;
  PositionListPtr unrolled4_selection_tids;
  PositionListPtr unrolled5_selection_tids;
  PositionListPtr unrolled6_selection_tids;
  PositionListPtr unrolled7_selection_tids;
  PositionListPtr unrolled8_selection_tids;

  PositionListPtr serial_bf_selection_tids;
  PositionListPtr bf_unrolled2_selection_tids;
  PositionListPtr bf_unrolled3_selection_tids;
  PositionListPtr bf_unrolled4_selection_tids;
  PositionListPtr bf_unrolled5_selection_tids;
  PositionListPtr bf_unrolled6_selection_tids;
  PositionListPtr bf_unrolled7_selection_tids;
  PositionListPtr bf_unrolled8_selection_tids;

  cout << "Serial_Selection\t\t"
       << "Serial_BF_Selection\t\t"
       << "Serial_Unrolled2_Selection\t\t"
       << "Serial_BF_Unrolled2_Selection\t\t"
       << "Serial_Unrolled3_Selection\t\t"
       << "Serial_BF_Unrolled3_Selection\t\t"
       << "Serial_Unrolled4_Selection\t\t"
       << "Serial_BF_Unrolled4_Selection\t\t"
       << "Serial_Unrolled5_Selection\t\t"
       << "Serial_BF_Unrolled5_Selection\t\t"
       << "Serial_Unrolled6_Selection\t\t"
       << "Serial_BF_Unrolled6_Selection\t\t"
       << "Serial_Unrolled7_Selection\t\t"
       << "Serial_BF_Unrolled7_Selection\t\t"
       << "Serial_Unrolled8_Selection\t\t"
       << "Serial_BF_Unrolled8_Selection\t\t" << endl;

  for (unsigned int i = 0; i < 10; i++) {
    size_t col_size =
        static_cast<size_t>((((i % 10) + 1) / 10.0) * column.size());
    cout << "Col_Size: " << col_size << endl;
    for (unsigned int j = 0; j < 11; j++) {
      selection_value = sel_values->at(j % 11);
      for (unsigned int anz_iter = 0; anz_iter < 100; anz_iter++) {
        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_selection_tids =
              CoGaDB::CDK::selection::variants::serial_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(serial_selection_tids->size() * 100 / col_size) << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_bf_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(serial_bf_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled2_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled2_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled2_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled2_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled2_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled2_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled3_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled3_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled3_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled3_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled3_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled3_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled4_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled4_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled4_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled4_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled4_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled4_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled5_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled5_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled5_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled5_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled5_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled5_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled6_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled6_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled6_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled6_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled6_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled6_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled7_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled7_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled7_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled7_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled7_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled7_selection_tids->size() * 100 / col_size)
               << "\t";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          unrolled8_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(unrolled8_selection_tids->size() * 100 / col_size)
               << "\n";
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          bf_unrolled8_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();

          cout << double(end - begin) / (1000 * 1000) << "\t"
               << floor(bf_unrolled8_selection_tids->size() * 100 / col_size)
               << "\n";
        }
      }
    }
  }

  return true;
}

bool cdk_selection_performance_test_float() {
#define BENCHMARK true

  float selection_value;
  ValueComparator selection_comparison_value;  // 0 EQUAL, 1 LESSER, 2 LARGER

  unsigned int number_of_threads = boost::thread::hardware_concurrency();
  cout << "Used Threads:" << boost::thread::hardware_concurrency() << endl;

  std::vector<float> column;
  // fill column
  for (unsigned int i = 0; i < 60 * 1000 * 1000; i++) {
    column.push_back(float(rand() % 1000));
  }

  auto sel_values = boost::make_shared<vector<float>>();
  for (unsigned int i = 0; i < 11; i++) {
    sel_values->push_back(float(i * 100.f));
  }

  if (BENCHMARK) {
    cout << "Serial_Selection\t\t"
         << "Serial_BF_Selection\t\t"
         << "Serial_Unrolled_Selection\t\t"
         << "Serial_BF_Unrolled_Selection\t\t";
#ifdef ENABLE_SIMD_ACCELERATION
    cout << "Serial_SIMD_Selection\t\t"
         << "Serial_BF_SIMD_Selection\t\t"
         << "Serial_Unrolled_SIMD_Selection\t\t"
         << "Serial_BF_Unrolled_SIMD_Selection\t\t";
#endif
    cout << "Parallel_Selection\t\t"
         << "Parallel_BF_Selection\t\t"
         << "Parallel_Unrolled_Selection\t\t"
         << "Parallel_BF_Unrolled_Selection\t\t";
#ifdef ENABLE_SIMD_ACCELERATION
    cout << "Parallel_SIMD_Selection\t\t"
         << "Parallel_BF_SIMD_Selection\t\t"
         << "Parallel_Unrolled_SIMD_Selection\t\t"
         << "Parallel_BF_Unrolled_SIMD_Selection\t\t";
#endif
    cout << endl;
  }

  for (unsigned int i = 0; i < 10; i++) {
    size_t col_size =
        static_cast<size_t>((((i % 10) + 1) / 10.0) * column.size());
    cout << "Col_Size: " << col_size << endl;
    for (unsigned int j = 0; j < 11; j++) {
      selection_value = sel_values->at(j % 11);
      for (unsigned int anz_iter = 0; anz_iter < 100; anz_iter++) {
        selection_comparison_value = static_cast<ValueComparator>(0);
        float* column_array = hype::util::begin_ptr(column);

        Timestamp begin;
        Timestamp end;

        PositionListPtr serial_selection_tids;
        PositionListPtr serial_unrolled_selection_tids;
        PositionListPtr parallel_selection_tids;
        PositionListPtr parallel_unrolled_selection_tids;

        PositionListPtr serial_bf_selection_tids;
        PositionListPtr serial_bf_unrolled_selection_tids;
        PositionListPtr parallel_bf_selection_tids;
        PositionListPtr parallel_bf_unrolled_selection_tids;

        {
          begin = getTimestamp();
          serial_selection_tids =
              CoGaDB::CDK::selection::variants::serial_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_selection_tids->size() * 100 / col_size)
                 << "\t";
          } else {
            cout << "Serial Selection: " << double(end - begin) / (1000 * 1000)
                 << "ms"
                 << " Selectivity: "
                 << floor(serial_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          serial_bf_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_bf_selection_tids->size() * 100 / col_size)
                 << "\t";
          } else {
            cout << "Serial Selection: " << double(end - begin) / (1000 * 1000)
                 << "ms"
                 << " Selectivity: "
                 << floor(serial_bf_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          serial_unrolled_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << "\t";
          } else {
            cout << "Serial Unrolled Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          serial_bf_unrolled_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_bf_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << "\t";
          } else {
            cout << "Serial Unrolled Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_bf_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

#ifdef ENABLE_SIMD_ACCELERATION
        PositionListPtr serial_simd_selection_tids;
        PositionListPtr serial_unrolled_simd_selection_tids;
        PositionListPtr parallel_simd_selection_tids;
        PositionListPtr parallel_unrolled_simd_selection_tids;
        PositionListPtr serial_bf_simd_selection_tids;
        PositionListPtr serial_bf_unrolled_simd_selection_tids;
        PositionListPtr parallel_bf_simd_selection_tids;
        PositionListPtr parallel_bf_unrolled_simd_selection_tids;

        {
          begin = getTimestamp();
          //                    serial_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_SIMD_selection(column_array,
          //                    (int) (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_simd_selection_tids =
              CoGaDB::CDK::selection::variants::serial_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Serial_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(serial_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_simd_selection_tids->size() * 100 / col_size)
                 << "\t";

          } else {
            cout << "Serial SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_simd_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }
        {
          begin = getTimestamp();
          //                    serial_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_SIMD_selection(column_array,
          //                    (int) (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_bf_simd_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Serial_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(serial_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_bf_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Serial SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_bf_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          serial_unrolled_simd_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Parallel_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Serial Unrolled SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          serial_bf_unrolled_simd_selection_tids = CoGaDB::CDK::selection::
              variants::serial_bf_unrolled_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Parallel_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_bf_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Serial Unrolled SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_bf_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }
#endif
        {
          begin = getTimestamp();
          parallel_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_selection_tids->size() * 100 / col_size)
                 << "\t";
          } else {
            cout << "Parallel Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          parallel_bf_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_bf_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_bf_selection_tids->size() * 100 / col_size)
                 << "\t";
          } else {
            cout << "Parallel Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_bf_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          parallel_unrolled_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << "\t";
          } else {
            cout << "Parallel Unrolled Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          parallel_bf_unrolled_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_bf_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_bf_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << "\t";
          } else {
            cout << "Parallel Unrolled Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_bf_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

#ifdef ENABLE_SIMD_ACCELERATION

        {
          begin = getTimestamp();
          //                    parallel_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_simd_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();

          if (BENCHMARK) {
            //                        cout << "Parallel_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_simd_selection_tids->size() * 100 / col_size)
                 << "\t";

          } else {
            cout << "Parallel SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_simd_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    parallel_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_bf_simd_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_bf_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();

          if (BENCHMARK) {
            //                        cout << "Parallel_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_bf_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Parallel SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_bf_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    parallel_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_unrolled_simd_selection_tids = CoGaDB::CDK::selection::
              variants::parallel_unrolled_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();

          if (BENCHMARK) {
            //                        cout << "Parallel_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Parallel Unrolled SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    parallel_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_bf_unrolled_simd_selection_tids = CoGaDB::CDK::selection::
              variants::parallel_bf_unrolled_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();

          if (BENCHMARK) {
            //                        cout << "Parallel_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_bf_unrolled_simd_selection_tids->size() *
                          100 / col_size)
                 << "\t";

          } else {
            cout << "Parallel Unrolled SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_bf_unrolled_simd_selection_tids->size() *
                          100 / col_size)
                 << endl;
          }
        }

#endif
        if (BENCHMARK) {
          cout << endl;
        }

#ifdef ENABLE_SIMD_ACCELERATION
        if ((*serial_selection_tids) != (*serial_unrolled_selection_tids) ||
            (*serial_unrolled_selection_tids) !=
                (*serial_simd_selection_tids) ||
            (*serial_simd_selection_tids) != (*parallel_selection_tids) ||
            (*parallel_selection_tids) != (*parallel_unrolled_selection_tids) ||
            (*parallel_unrolled_selection_tids) !=
                (*parallel_simd_selection_tids) ||
            (*parallel_simd_selection_tids) !=
                (*serial_unrolled_simd_selection_tids) ||
            (*serial_unrolled_simd_selection_tids) !=
                (*parallel_unrolled_simd_selection_tids) ||
            (*serial_bf_selection_tids) !=
                (*serial_bf_unrolled_selection_tids) ||
            (*serial_bf_unrolled_selection_tids) !=
                (*serial_bf_simd_selection_tids) ||
            (*serial_bf_simd_selection_tids) != (*parallel_bf_selection_tids) ||
            (*parallel_bf_selection_tids) !=
                (*parallel_bf_unrolled_selection_tids) ||
            (*parallel_bf_unrolled_selection_tids) !=
                (*parallel_bf_simd_selection_tids) ||
            (*parallel_bf_simd_selection_tids) !=
                (*serial_bf_unrolled_simd_selection_tids) ||
            (*serial_bf_unrolled_simd_selection_tids) !=
                (*parallel_bf_unrolled_simd_selection_tids) ||
            (*serial_bf_selection_tids) != (*serial_selection_tids)) {
#else
        if ((*serial_selection_tids) != (*serial_unrolled_selection_tids) ||
            (*serial_unrolled_selection_tids) != (*parallel_selection_tids) ||
            (*parallel_selection_tids) != (*parallel_unrolled_selection_tids) ||
            (*serial_selection_tids) != (*serial_bf_selection_tids) ||
            (*serial_bf_selection_tids) !=
                (*serial_bf_unrolled_selection_tids) ||
            (*serial_bf_unrolled_selection_tids) !=
                (*parallel_bf_selection_tids) ||
            (*parallel_bf_selection_tids) !=
                (*parallel_bf_unrolled_selection_tids)

                ) {
#endif
          cout << "TID lists are not equal!" << endl;
          cout << "Serial Selection result size: "
               << serial_selection_tids->size() << endl;
          std::copy(serial_selection_tids->begin(),
                    serial_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Branch-free Selection result size: "
               << serial_bf_selection_tids->size() << endl;
          std::copy(serial_bf_selection_tids->begin(),
                    serial_bf_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Unrolled Selection result size: "
               << serial_unrolled_selection_tids->size() << endl;
          std::copy(serial_unrolled_selection_tids->begin(),
                    serial_unrolled_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Branch-free Unrolled Selection result size: "
               << serial_bf_unrolled_selection_tids->size() << endl;
          std::copy(serial_bf_unrolled_selection_tids->begin(),
                    serial_bf_unrolled_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));

#ifdef ENABLE_SIMD_ACCELERATION
          cout << "Serial SIMD Selection result size: "
               << serial_simd_selection_tids->size() << endl;
          std::copy(serial_simd_selection_tids->begin(),
                    serial_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Branch-free SIMD Selection result size: "
               << serial_bf_simd_selection_tids->size() << endl;
          std::copy(serial_bf_simd_selection_tids->begin(),
                    serial_bf_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Unrolled SIMD Selection result size: "
               << serial_unrolled_simd_selection_tids->size() << endl;
          std::copy(serial_unrolled_simd_selection_tids->begin(),
                    serial_unrolled_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Branch-free Unrolled SIMD Selection result size: "
               << serial_bf_unrolled_simd_selection_tids->size() << endl;
          std::copy(serial_bf_unrolled_simd_selection_tids->begin(),
                    serial_bf_unrolled_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
#endif
          cout << "Parallel Selection result size: "
               << parallel_selection_tids->size() << endl;
          std::copy(parallel_selection_tids->begin(),
                    parallel_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Branch-free Selection result size: "
               << parallel_bf_selection_tids->size() << endl;
          std::copy(parallel_bf_selection_tids->begin(),
                    parallel_bf_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Unrolled Selection result size: "
               << parallel_unrolled_selection_tids->size() << endl;
          std::copy(parallel_unrolled_selection_tids->begin(),
                    parallel_unrolled_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Branch-free Unrolled Selection result size: "
               << parallel_bf_unrolled_selection_tids->size() << endl;
          std::copy(parallel_bf_unrolled_selection_tids->begin(),
                    parallel_bf_unrolled_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));

#ifdef ENABLE_SIMD_ACCELERATION
          cout << "Parallel SIMD Selection result size: "
               << parallel_simd_selection_tids->size() << endl;
          std::copy(parallel_simd_selection_tids->begin(),
                    parallel_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Branch-free SIMD Selection result size: "
               << parallel_bf_simd_selection_tids->size() << endl;
          std::copy(parallel_bf_simd_selection_tids->begin(),
                    parallel_bf_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          //                            std::copy(parallel_simd_selection_tids->begin(),
          //                            parallel_simd_selection_tids->end(),
          //                            std::ostream_iterator<int>(std::cout,
          //                            "\n"));
          cout << "Parallel Unrolled SIMD Selection result size: "
               << parallel_unrolled_simd_selection_tids->size() << endl;
          std::copy(parallel_unrolled_simd_selection_tids->begin(),
                    parallel_unrolled_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Branch-free Unrolled SIMD Selection result size: "
               << parallel_bf_unrolled_simd_selection_tids->size() << endl;
          std::copy(parallel_bf_unrolled_simd_selection_tids->begin(),
                    parallel_bf_unrolled_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
#endif
          //                            std::copy(parallel_unrolled_simd_selection_tids->begin(),
          //                            parallel_unrolled_simd_selection_tids->end(),
          //                            std::ostream_iterator<int>(std::cout,
          //                            "\n"));

          // unsigned int
          // size=std::min(serial_selection_tids->size(),parallel_selection_tids->size());
          ////                                for(unsigned int i=0;i<size;i++){
          ////                                    cout << "Serial id: " <<
          ///(*serial_selection_tids)[i] << " \tParallel id:"<<
          ///(*parallel_selection_tids)[i] << endl;
          ////                                }
          //                                for(unsigned int i=0;i<size;i++){
          //                                    if((*serial_selection_tids)[i]!=(*parallel_selection_tids)[i])
          //                                         cout << "Serial id: " <<
          //                                         (*serial_selection_tids)[i]
          //                                         << " \tParallel id:"<<
          //                                         (*parallel_selection_tids)[i]
          //                                         << endl;
          //                                }
          //                                if(size<serial_selection_tids->size()){
          //                                    cout << "Detected additional
          //                                    values for serial selection " <<
          //                                    endl;
          //                                    for(unsigned int
          //                                    i=size;i<serial_selection_tids->size();i++){
          //                                           cout << "id: " << i << "
          //                                           val: " <<
          //                                           (*serial_selection_tids)[i]
          //                                           << endl;
          //                                    }
          //                                }
          COGADB_FATAL_ERROR(
              "Selection Unittests failed! At least one algorithm works "
              "incorrect!",
              "");
        }
        //                            assert((*serial_selection_tids)==(*lock_free_parallel_selection_tids));
      }
    }
  }
  return true;
}

bool cdk_selection_bitmap_performance_test() {
#define BENCHMARK true

  int selection_value;
  ValueComparator selection_comparison_value;  // 0 EQUAL, 1 LESSER, 2 LARGER

  std::vector<int> column;
  // fill column
  for (unsigned int i = 0; i < 60 * 1000 * 1000; i++) {
    column.push_back(int(rand() % 1000));
  }

  auto sel_values = boost::make_shared<std::vector<int>>();
  for (int i = 0; i < 11; i++) {
    sel_values->push_back(i * 100);
  }

  if (BENCHMARK) {
    cout << "Serial_Bitmap_Selection\t\t"
         << "Serial_BF_Bitmap_Selection\t\t"
         << "Serial_Unrolled_Selection\t\t"
         << "Serial_BF_Unrolled_Selection\t\t";
#ifdef ENABLE_SIMD_ACCELERATION
    cout << "Serial_SIMD_Selection\t\t"
         << "Serial_BF_SIMD_Selection\t\t"
         << "Serial_Unrolled_SIMD_Selection\t\t"
         << "Serial_BF_Unrolled_SIMD_Selection\t\t";
#endif
    cout << "Parallel_Selection\t\t"
         << "Parallel_BF_Selection\t\t"
         << "Parallel_Unrolled_Selection\t\t"
         << "Parallel_BF_Unrolled_Selection\t\t";
#ifdef ENABLE_SIMD_ACCELERATION
    cout << "Parallel_SIMD_Selection\t\t"
         << "Parallel_BF_SIMD_Selection\t\t"
         << "Parallel_Unrolled_SIMD_Selection\t\t"
         << "Parallel_BF_Unrolled_SIMD_Selection\t\t";
#endif
    cout << endl;
  }

  for (unsigned int i = 0; i < 10; i++) {
    size_t col_size =
        static_cast<size_t>((((i % 10) + 1) / 10.0) * column.size());
    cout << "Col_Size: " << col_size << endl;
    for (unsigned int j = 0; j < 11; j++) {
      selection_value = sel_values->at(j % 11);
      for (unsigned int anz_iter = 0; anz_iter < 100; anz_iter++) {
        selection_comparison_value = static_cast<ValueComparator>(0);
        int* column_array = hype::util::begin_ptr(column);

        Timestamp begin;
        Timestamp end;

        char* serial_selection_tids;
        char* serial_bf_selection_tids;

        {
          begin = getTimestamp();
          serial_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bitmap_selection<int>(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t" << j * 10
                 << "\t";
          } else {
            cout << "Serial Selection: " << double(end - begin) / (1000 * 1000)
                 << "ms"
                 << " Selectivity: " << double(end - begin) / (1000 * 1000)
                 << "\t" << j * 10 << "\t" << endl;
          }
        }

        {
          begin = getTimestamp();
          serial_bf_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_bitmap_selection<int>(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            cout << double(end - begin) / (1000 * 1000) << "\t" << j * 10
                 << "\t";

          } else {
            cout << "Serial Selection: " << double(end - begin) / (1000 * 1000)
                 << "ms"
                 << " Selectivity: " << j * 10 << endl;
          }
        }

        //                                {
        //
        //                                    begin=getTimestamp();
        //                                    //
        //                                    serial_unrolled_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::serial_unrolled_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    serial_unrolled_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::serial_unrolled_selection(column_array,col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //
        //                                        //                        cout
        //                                        <<
        //                                        "Serial_Unrolled_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(serial_unrolled_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<<  j*10  << "\t";
        //
        //                                    }else{
        //                                        cout << "Serial Unrolled
        //                                        Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10    << endl;
        //                                    }
        //                                }
        //
        //
        //
        //                                {
        //
        //                                    begin=getTimestamp();
        //                                    //
        //                                    serial_unrolled_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::serial_unrolled_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    serial_bf_unrolled_selection_tids
        //                                    =
        //                                    CoGaDB::CDK::selection::variants::serial_bf_unrolled_selection(column_array,col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //
        //                                        //                        cout
        //                                        <<
        //                                        "Serial_Unrolled_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(serial_unrolled_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10 << "\t";
        //
        //                                    }else{
        //                                        cout << "Serial Unrolled
        //                                        Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //
        //
        //
        //#ifdef ENABLE_SIMD_ACCELERATION
        //                                {
        //
        //                                    begin=getTimestamp();
        //                                    //
        //                                    serial_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::serial_SIMD_selection(column_array,
        //                                    (int) (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    serial_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::serial_SIMD_selection(column_array,
        //                                    col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Serial_SIMD_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(serial_simd_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10 << "\t";
        //
        //                                    }else{
        //                                        cout << "Serial SIMD
        //                                        Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //                                {
        //
        //                                    begin=getTimestamp();
        //                                    //
        //                                    serial_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::serial_SIMD_selection(column_array,
        //                                    (int) (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    serial_bf_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::serial_bf_SIMD_selection(column_array,
        //                                    col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Serial_SIMD_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(serial_simd_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10 << "\t";
        //
        //                                    }else{
        //                                        cout << "Serial SIMD
        //                                        Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //                                {
        //                                    begin=getTimestamp();
        //                                    serial_unrolled_simd_selection_tids
        //                                    =
        //                                    CoGaDB::CDK::selection::variants::serial_unrolled_SIMD_selection(column_array,
        //                                    col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Parallel_Selection\t" <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10 << "\t";
        //
        //                                    }else{
        //                                        cout << "Serial Unrolled SIMD
        //                                        Selection: "  <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //                                {
        //                                    begin=getTimestamp();
        //                                    serial_bf_unrolled_simd_selection_tids
        //                                    =
        //                                    CoGaDB::CDK::selection::variants::serial_bf_unrolled_SIMD_selection(column_array,
        //                                    col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Parallel_Selection\t" <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10 << "\t";
        //
        //                                    }else{
        //                                        cout << "Serial Unrolled SIMD
        //                                        Selection: "  <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //#endif
        //                                {
        //                                    begin=getTimestamp();
        //                                    //
        //                                    parallel_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    parallel_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_selection(column_array,col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Parallel_Selection\t" <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10  << "\t";
        //
        //                                    }else{
        //                                        cout << "Parallel Selection: "
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //
        //                                {
        //                                    begin=getTimestamp();
        //                                    //
        //                                    parallel_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    parallel_bf_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_bf_selection(column_array,col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Parallel_Selection\t" <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10  << "\t";
        //
        //                                    }else{
        //                                        cout << "Parallel Selection: "
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //
        //                                {
        //                                    begin=getTimestamp();
        //                                    //
        //                                    parallel_unrolled_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_unrolled_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    parallel_unrolled_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_unrolled_selection(column_array,col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        <<
        //                                        "Parallel_Unrolled_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_unrolled_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout<<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10  << "\t";
        //
        //                                    }else{
        //                                        cout << "Parallel Unrolled
        //                                        Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //
        //
        //                                {
        //                                    begin=getTimestamp();
        //                                    //
        //                                    parallel_unrolled_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_unrolled_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    parallel_bf_unrolled_selection_tids
        //                                    =
        //                                    CoGaDB::CDK::selection::variants::parallel_bf_unrolled_selection(column_array,col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    end=getTimestamp();
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        <<
        //                                        "Parallel_Unrolled_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_unrolled_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout<<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10  << "\t";
        //
        //                                    }else{
        //                                        cout << "Parallel Unrolled
        //                                        Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //
        //#ifdef ENABLE_SIMD_ACCELERATION
        //
        //
        //                                {
        //                                    begin=getTimestamp();
        //                                    //
        //                                    parallel_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    parallel_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,
        //                                    col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    end=getTimestamp();
        //
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Parallel_SIMD_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_simd_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10  << "\t" ;
        //
        //                                    }else{
        //                                        cout << "Parallel SIMD
        //                                        Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //
        //
        //                                {
        //                                    begin=getTimestamp();
        //                                    //
        //                                    parallel_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    parallel_bf_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_bf_SIMD_selection(column_array,
        //                                    col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    end=getTimestamp();
        //
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Parallel_SIMD_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_simd_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10  << "\t" ;
        //
        //                                    }else{
        //                                        cout << "Parallel SIMD
        //                                        Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //
        //
        //                                {
        //                                    begin=getTimestamp();
        //                                    //
        //                                    parallel_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    parallel_unrolled_simd_selection_tids
        //                                    =
        //                                    CoGaDB::CDK::selection::variants::parallel_unrolled_SIMD_selection(column_array,
        //                                    col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    end=getTimestamp();
        //
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Parallel_SIMD_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_simd_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10  << "\t";
        //
        //                                    }else{
        //                                        cout << "Parallel Unrolled
        //                                        SIMD Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //
        //                                {
        //                                    begin=getTimestamp();
        //                                    //
        //                                    parallel_simd_selection_tids =
        //                                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
        //                                    (column.size()/(i+1)),
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    parallel_bf_unrolled_simd_selection_tids
        //                                    =
        //                                    CoGaDB::CDK::selection::variants::parallel_bf_unrolled_SIMD_selection(column_array,
        //                                    col_size,
        //                                    boost::any(selection_value),
        //                                    selection_comparison_value,
        //                                    number_of_threads);
        //                                    end=getTimestamp();
        //
        //                                    if(BENCHMARK){
        //                                        //                        cout
        //                                        << "Parallel_SIMD_Selection\t"
        //                                        <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms\t"<<
        //                                        floor(parallel_simd_selection_tids->size()*100
        //                                        /(column.size()/(i+1)))  <<
        //                                        endl;
        //                                        cout <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "\t"<< j*10  << "\t";
        //
        //                                    }else{
        //                                        cout << "Parallel Unrolled
        //                                        SIMD Selection: " <<
        //                                        double(end-begin)/(1000*1000)
        //                                        << "ms" << " Selectivity: " <<
        //                                        j*10 << endl;
        //                                    }
        //                                }
        //
        //

        //#endif
        if (BENCHMARK) {
          cout << endl;
        }

        //                            {
        //                            begin=getTimestamp();
        //                            lock_free_parallel_selection_tids =
        //                            column->lock_free_parallel_selection(selection_value,selection_comparison_value,number_of_threads);
        //                            end=getTimestamp();
        //                            cout << "Lock Free Parallel Selection: "
        //                            << double(end-begin)/(1000*1000) << "ms"
        //                            << endl;
        //                            }

        //
        //
        //#ifdef ENABLE_SIMD_ACCELERATION
        //                                if((*serial_selection_tids)!=(*serial_unrolled_selection_tids)
        //                                ||  (*serial_unrolled_selection_tids)
        //                                != (*serial_simd_selection_tids) ||
        //                                (*serial_simd_selection_tids) !=
        //                                (*parallel_selection_tids)||
        //                                (*parallel_selection_tids) !=
        //                                (*parallel_unrolled_selection_tids)||
        //                                (*parallel_unrolled_selection_tids) !=
        //                                (*parallel_simd_selection_tids) ||
        //                                (*parallel_simd_selection_tids)!=(*serial_unrolled_simd_selection_tids)
        //                                ||
        //                                (*serial_unrolled_simd_selection_tids)!=(*parallel_unrolled_simd_selection_tids)
        //                                ||
        //                                (*serial_bf_selection_tids)!=(*serial_bf_unrolled_selection_tids)
        //                                ||
        //                                (*serial_bf_unrolled_selection_tids)
        //                                != (*serial_bf_simd_selection_tids) ||
        //                                (*serial_bf_simd_selection_tids) !=
        //                                (*parallel_bf_selection_tids)||
        //                                (*parallel_bf_selection_tids) !=
        //                                (*parallel_bf_unrolled_selection_tids)||
        //                                (*parallel_bf_unrolled_selection_tids)
        //                                != (*parallel_bf_simd_selection_tids)
        //                                ||
        //                                (*parallel_bf_simd_selection_tids)!=(*serial_bf_unrolled_simd_selection_tids)
        //                                ||
        //                                (*serial_bf_unrolled_simd_selection_tids)!=(*parallel_bf_unrolled_simd_selection_tids)
        //                                ||
        //                                (*serial_bf_selection_tids)!=(*serial_selection_tids)
        //                                ){
        //#else
        //                                    if((*serial_selection_tids)!=(*serial_unrolled_selection_tids)
        //                                    ||
        //                                    (*serial_unrolled_selection_tids)
        //                                    != (*parallel_selection_tids)||
        //                                    (*parallel_selection_tids) !=
        //                                    (*parallel_unrolled_selection_tids)||
        //                                    (*serial_selection_tids)!=(*serial_bf_selection_tids)
        //                                    ||
        //                                    (*serial_bf_selection_tids)!=(*serial_bf_unrolled_selection_tids)
        //                                    ||
        //                                    (*serial_bf_unrolled_selection_tids)
        //                                    != (*parallel_bf_selection_tids)||
        //                                    (*parallel_bf_selection_tids) !=
        //                                    (*parallel_bf_unrolled_selection_tids)
        //                                       ){
        //#endif
        if (strcmp(serial_bf_selection_tids, serial_selection_tids) != 0) {
          cout << "Bitmaps are not equal!" << endl;
          for (int i = 0; serial_selection_tids[i] != '\0'; i++) {
            printf("serial: %x, bf: %x \n", serial_selection_tids[i],
                   serial_bf_selection_tids[i]);
          }
          // cout << "Serial Selection result size: " <<
          // serial_selection_tids->size() << endl;
          // std::copy(serial_selection_tids->begin(),
          // serial_selection_tids->end(), std::ostream_iterator<int>(std::cout,
          // "\t"));
          // cout << "Serial Branch-free Selection result size: " <<
          // serial_bf_selection_tids->size() << endl;
          // std::copy(serial_bf_selection_tids->begin(),
          // serial_bf_selection_tids->end(),
          // std::ostream_iterator<int>(std::cout, "\t"));
          //                                        cout << "Serial Unrolled
          //                                        Selection result size: " <<
          //                                        serial_unrolled_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(serial_unrolled_selection_tids->begin(),
          //                                        serial_unrolled_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        cout << "Serial Branch-free
          //                                        Unrolled Selection result
          //                                        size: " <<
          //                                        serial_bf_unrolled_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(serial_bf_unrolled_selection_tids->begin(),
          //                                        serial_bf_unrolled_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //
          //#ifdef ENABLE_SIMD_ACCELERATION
          //                                        cout << "Serial SIMD
          //                                        Selection result size: " <<
          //                                        serial_simd_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(serial_simd_selection_tids->begin(),
          //                                        serial_simd_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        cout << "Serial Branch-free
          //                                        SIMD Selection result size:
          //                                        " <<
          //                                        serial_bf_simd_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(serial_bf_simd_selection_tids->begin(),
          //                                        serial_bf_simd_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        cout << "Serial Unrolled
          //                                        SIMD Selection result size:
          //                                        " <<
          //                                        serial_unrolled_simd_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(serial_unrolled_simd_selection_tids->begin(),
          //                                        serial_unrolled_simd_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        cout << "Serial Branch-free
          //                                        Unrolled SIMD Selection
          //                                        result size: " <<
          //                                        serial_bf_unrolled_simd_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(serial_bf_unrolled_simd_selection_tids->begin(),
          //                                        serial_bf_unrolled_simd_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //#endif
          //                                        cout << "Parallel Selection
          //                                        result size: " <<
          //                                        parallel_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(parallel_selection_tids->begin(),
          //                                        parallel_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        cout << "Parallel
          //                                        Branch-free Selection result
          //                                        size: " <<
          //                                        parallel_bf_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(parallel_bf_selection_tids->begin(),
          //                                        parallel_bf_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        cout << "Parallel Unrolled
          //                                        Selection result size: " <<
          //                                        parallel_unrolled_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(parallel_unrolled_selection_tids->begin(),
          //                                        parallel_unrolled_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        cout << "Parallel
          //                                        Branch-free Unrolled
          //                                        Selection result size: " <<
          //                                        parallel_bf_unrolled_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(parallel_bf_unrolled_selection_tids->begin(),
          //                                        parallel_bf_unrolled_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //
          //#ifdef ENABLE_SIMD_ACCELERATION
          //                                        cout << "Parallel SIMD
          //                                        Selection result size: " <<
          //                                        parallel_simd_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(parallel_simd_selection_tids->begin(),
          //                                        parallel_simd_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        cout << "Parallel
          //                                        Branch-free SIMD Selection
          //                                        result size: " <<
          //                                        parallel_bf_simd_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(parallel_bf_simd_selection_tids->begin(),
          //                                        parallel_bf_simd_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        //
          //                                        std::copy(parallel_simd_selection_tids->begin(),
          //                                        parallel_simd_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\n"));
          //                                        cout << "Parallel Unrolled
          //                                        SIMD Selection result size:
          //                                        " <<
          //                                        parallel_unrolled_simd_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(parallel_unrolled_simd_selection_tids->begin(),
          //                                        parallel_unrolled_simd_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //                                        cout << "Parallel
          //                                        Branch-free Unrolled SIMD
          //                                        Selection result size: " <<
          //                                        parallel_bf_unrolled_simd_selection_tids->size()
          //                                        << endl;
          //                                        std::copy(parallel_bf_unrolled_simd_selection_tids->begin(),
          //                                        parallel_bf_unrolled_simd_selection_tids->end(),
          //                                        std::ostream_iterator<int>(std::cout,
          //                                        "\t"));
          //#endif

          //                            std::copy(parallel_unrolled_simd_selection_tids->begin(),
          //                            parallel_unrolled_simd_selection_tids->end(),
          //                            std::ostream_iterator<int>(std::cout,
          //                            "\n"));

          // unsigned int
          // size=std::min(serial_selection_tids->size(),parallel_selection_tids->size());
          ////                                for(unsigned int i=0;i<size;i++){
          ////                                    cout << "Serial id: " <<
          ///(*serial_selection_tids)[i] << " \tParallel id:"<<
          ///(*parallel_selection_tids)[i] << endl;
          ////                                }
          //                                for(unsigned int i=0;i<size;i++){
          //                                    if((*serial_selection_tids)[i]!=(*parallel_selection_tids)[i])
          //                                         cout << "Serial id: " <<
          //                                         (*serial_selection_tids)[i]
          //                                         << " \tParallel id:"<<
          //                                         (*parallel_selection_tids)[i]
          //                                         << endl;
          //                                }
          //                                if(size<serial_selection_tids->size()){
          //                                    cout << "Detected additional
          //                                    values for serial selection " <<
          //                                    endl;
          //                                    for(unsigned int
          //                                    i=size;i<serial_selection_tids->size();i++){
          //                                           cout << "id: " << i << "
          //                                           val: " <<
          //                                           (*serial_selection_tids)[i]
          //                                           << endl;
          //                                    }
          //                                }
          COGADB_FATAL_ERROR(
              "Selection Unittests failed! At least one algorithm works "
              "incorrect!",
              "");
        }
        free(serial_selection_tids);
        free(serial_bf_selection_tids);
        //                            assert((*serial_selection_tids)==(*lock_free_parallel_selection_tids));
      }
    }
  }

  return true;
}

bool cdk_selection_performance_test() {
#define BENCHMARK true

  int selection_value;
  ValueComparator selection_comparison_value;  // 0 EQUAL, 1 LESSER, 2 LARGER

  // cout << "Enter number of threads:" << endl;
  unsigned int number_of_threads = boost::thread::hardware_concurrency();

  cout << "Used Threads:" << boost::thread::hardware_concurrency() << endl;

  // cin >> number_of_threads;
  // number_of_threads=8;

  boost::mt19937 rng;
  boost::uniform_int<> selection_values(0, 1000);
  boost::uniform_int<> filter_condition(0, 2);

  std::vector<int> column;
  // fill column
  for (unsigned int i = 0; i < 60 * 1000 * 1000; i++) {
    //     for(unsigned int i=0;i<10000;i++){
    column.push_back(int(rand() % 1000));
  }

  boost::shared_ptr<vector<int>> sel_values(new vector<int>());
  for (unsigned int i = 0; i < 11; i++) {
    sel_values->push_back(i * 100);
    //             sel_values->push_back(selection_values(rng));
  }
  //            sort(sel_values->begin(), sel_values->end());

  if (BENCHMARK) {
    cout << "Serial_Selection\t\t"
         << "Serial_BF_Selection\t\t"
         << "Serial_Unrolled_Selection\t\t"
         << "Serial_BF_Unrolled_Selection\t\t";
#ifdef ENABLE_SIMD_ACCELERATION
    cout << "Serial_SIMD_Selection\t\t"
         << "Serial_BF_SIMD_Selection\t\t"
         << "Serial_Unrolled_SIMD_Selection\t\t"
         << "Serial_BF_Unrolled_SIMD_Selection\t\t";
#endif
    cout << "Parallel_Selection\t\t"
         << "Parallel_BF_Selection\t\t"
         << "Parallel_Unrolled_Selection\t\t"
         << "Parallel_BF_Unrolled_Selection\t\t";
#ifdef ENABLE_SIMD_ACCELERATION
    cout << "Parallel_SIMD_Selection\t\t"
         << "Parallel_BF_SIMD_Selection\t\t"
         << "Parallel_Unrolled_SIMD_Selection\t\t"
         << "Parallel_BF_Unrolled_SIMD_Selection\t\t";
#endif
    cout << endl;
  }

  for (unsigned int i = 0; i < 10; i++) {
    size_t col_size = (((float)((i % 10) + 1) / 10)) * column.size();
    cout << "Col_Size: " << col_size << endl;
    for (unsigned int j = 0; j < 11; j++) {
      selection_value = sel_values->at(j % 11);
      for (unsigned int anz_iter = 0; anz_iter < 100; anz_iter++) {
        //   for(unsigned int i=0;i< 11*100 ;i++){
        // selection_value = sel_values->at(i % 11 );

        // selection_comparison_value = (ValueComparator)filter_condition(rng);
        // //rand()%3;
        selection_comparison_value = (ValueComparator)0;
        int* column_array = hype::util::begin_ptr(column);

        Timestamp begin;
        Timestamp end;

        PositionListPtr serial_selection_tids;
        PositionListPtr serial_unrolled_selection_tids;
        PositionListPtr serial_simd_selection_tids;
        PositionListPtr serial_unrolled_simd_selection_tids;
        PositionListPtr parallel_selection_tids;
        PositionListPtr parallel_unrolled_selection_tids;
        PositionListPtr parallel_simd_selection_tids;
        PositionListPtr parallel_unrolled_simd_selection_tids;

        PositionListPtr serial_bf_selection_tids;
        PositionListPtr serial_bf_unrolled_selection_tids;
        PositionListPtr serial_bf_simd_selection_tids;
        PositionListPtr serial_bf_unrolled_simd_selection_tids;
        PositionListPtr parallel_bf_selection_tids;
        PositionListPtr parallel_bf_unrolled_selection_tids;
        PositionListPtr parallel_bf_simd_selection_tids;
        PositionListPtr parallel_bf_unrolled_simd_selection_tids;
        // PositionListPtr lock_free_parallel_selection_tids;

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_selection_tids =
              CoGaDB::CDK::selection::variants::serial_selection<int>(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Serial_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(serial_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_selection_tids->size() * 100 / col_size)
                 << "\t";

          } else {
            cout << "Serial Selection: " << double(end - begin) / (1000 * 1000)
                 << "ms"
                 << " Selectivity: "
                 << floor(serial_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    serial_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_selection<int>(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_bf_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_selection<int>(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Serial_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(serial_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_bf_selection_tids->size() * 100 / col_size)
                 << "\t";

          } else {
            cout << "Serial Selection: " << double(end - begin) / (1000 * 1000)
                 << "ms"
                 << " Selectivity: "
                 << floor(serial_bf_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    serial_unrolled_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_unrolled_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_unrolled_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Serial_Unrolled_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(serial_unrolled_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Serial Unrolled Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    serial_unrolled_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_unrolled_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_bf_unrolled_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Serial_Unrolled_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(serial_unrolled_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_bf_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Serial Unrolled Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_bf_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

#ifdef ENABLE_SIMD_ACCELERATION
        {
          begin = getTimestamp();
          //                    serial_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_SIMD_selection(column_array,
          //                    (int) (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_simd_selection_tids =
              CoGaDB::CDK::selection::variants::serial_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Serial_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(serial_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_simd_selection_tids->size() * 100 / col_size)
                 << "\t";

          } else {
            cout << "Serial SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_simd_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }
        {
          begin = getTimestamp();
          //                    serial_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::serial_SIMD_selection(column_array,
          //                    (int) (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value);
          serial_bf_simd_selection_tids =
              CoGaDB::CDK::selection::variants::serial_bf_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Serial_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(serial_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_bf_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Serial SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_bf_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          serial_unrolled_simd_selection_tids =
              CoGaDB::CDK::selection::variants::serial_unrolled_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Parallel_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Serial Unrolled SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          serial_bf_unrolled_simd_selection_tids = CoGaDB::CDK::selection::
              variants::serial_bf_unrolled_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Parallel_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(serial_bf_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Serial Unrolled SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(serial_bf_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }
#endif
        {
          begin = getTimestamp();
          //                    parallel_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Parallel_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_selection_tids->size() * 100 / col_size)
                 << "\t";

          } else {
            cout << "Parallel Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    parallel_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_bf_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_bf_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Parallel_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_bf_selection_tids->size() * 100 / col_size)
                 << "\t";

          } else {
            cout << "Parallel Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_bf_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    parallel_unrolled_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_unrolled_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_unrolled_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Parallel_Unrolled_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_unrolled_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Parallel Unrolled Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    parallel_unrolled_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_unrolled_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_bf_unrolled_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_bf_unrolled_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();
          if (BENCHMARK) {
            //                        cout << "Parallel_Unrolled_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_unrolled_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_bf_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Parallel Unrolled Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_bf_unrolled_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

#ifdef ENABLE_SIMD_ACCELERATION

        {
          begin = getTimestamp();
          //                    parallel_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_simd_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();

          if (BENCHMARK) {
            //                        cout << "Parallel_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_simd_selection_tids->size() * 100 / col_size)
                 << "\t";

          } else {
            cout << "Parallel SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_simd_selection_tids->size() * 100 / col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    parallel_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_bf_simd_selection_tids =
              CoGaDB::CDK::selection::variants::parallel_bf_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();

          if (BENCHMARK) {
            //                        cout << "Parallel_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_bf_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Parallel SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_bf_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    parallel_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_unrolled_simd_selection_tids = CoGaDB::CDK::selection::
              variants::parallel_unrolled_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();

          if (BENCHMARK) {
            //                        cout << "Parallel_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << "\t";

          } else {
            cout << "Parallel Unrolled SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_unrolled_simd_selection_tids->size() * 100 /
                          col_size)
                 << endl;
          }
        }

        {
          begin = getTimestamp();
          //                    parallel_simd_selection_tids =
          //                    CoGaDB::CDK::selection::variants::parallel_SIMD_selection(column_array,(int)
          //                    (column.size()/(i+1)),
          //                    boost::any(selection_value),
          //                    selection_comparison_value, number_of_threads);
          parallel_bf_unrolled_simd_selection_tids = CoGaDB::CDK::selection::
              variants::parallel_bf_unrolled_SIMD_selection(
                  column_array, col_size, boost::any(selection_value),
                  selection_comparison_value, number_of_threads);
          end = getTimestamp();

          if (BENCHMARK) {
            //                        cout << "Parallel_SIMD_Selection\t" <<
            //                        double(end-begin)/(1000*1000) << "ms\t"<<
            //                        floor(parallel_simd_selection_tids->size()*100
            //                        /(column.size()/(i+1)))  << endl;
            cout << double(end - begin) / (1000 * 1000) << "\t"
                 << floor(parallel_bf_unrolled_simd_selection_tids->size() *
                          100 / col_size)
                 << "\t";

          } else {
            cout << "Parallel Unrolled SIMD Selection: "
                 << double(end - begin) / (1000 * 1000) << "ms"
                 << " Selectivity: "
                 << floor(parallel_bf_unrolled_simd_selection_tids->size() *
                          100 / col_size)
                 << endl;
          }
        }

#endif
        if (BENCHMARK) {
          cout << endl;
        }

//                            {
//                            begin=getTimestamp();
//                            lock_free_parallel_selection_tids =
//                            column->lock_free_parallel_selection(selection_value,selection_comparison_value,number_of_threads);
//                            end=getTimestamp();
//                            cout << "Lock Free Parallel Selection: " <<
//                            double(end-begin)/(1000*1000) << "ms" << endl;
//                            }

#ifdef ENABLE_SIMD_ACCELERATION
        if ((*serial_selection_tids) != (*serial_unrolled_selection_tids) ||
            (*serial_unrolled_selection_tids) !=
                (*serial_simd_selection_tids) ||
            (*serial_simd_selection_tids) != (*parallel_selection_tids) ||
            (*parallel_selection_tids) != (*parallel_unrolled_selection_tids) ||
            (*parallel_unrolled_selection_tids) !=
                (*parallel_simd_selection_tids) ||
            (*parallel_simd_selection_tids) !=
                (*serial_unrolled_simd_selection_tids) ||
            (*serial_unrolled_simd_selection_tids) !=
                (*parallel_unrolled_simd_selection_tids) ||
            (*serial_bf_selection_tids) !=
                (*serial_bf_unrolled_selection_tids) ||
            (*serial_bf_unrolled_selection_tids) !=
                (*serial_bf_simd_selection_tids) ||
            (*serial_bf_simd_selection_tids) != (*parallel_bf_selection_tids) ||
            (*parallel_bf_selection_tids) !=
                (*parallel_bf_unrolled_selection_tids) ||
            (*parallel_bf_unrolled_selection_tids) !=
                (*parallel_bf_simd_selection_tids) ||
            (*parallel_bf_simd_selection_tids) !=
                (*serial_bf_unrolled_simd_selection_tids) ||
            (*serial_bf_unrolled_simd_selection_tids) !=
                (*parallel_bf_unrolled_simd_selection_tids) ||
            (*serial_bf_selection_tids) != (*serial_selection_tids)) {
#else
        if ((*serial_selection_tids) != (*serial_unrolled_selection_tids) ||
            (*serial_unrolled_selection_tids) != (*parallel_selection_tids) ||
            (*parallel_selection_tids) != (*parallel_unrolled_selection_tids) ||
            (*serial_selection_tids) != (*serial_bf_selection_tids) ||
            (*serial_bf_selection_tids) !=
                (*serial_bf_unrolled_selection_tids) ||
            (*serial_bf_unrolled_selection_tids) !=
                (*parallel_bf_selection_tids) ||
            (*parallel_bf_selection_tids) !=
                (*parallel_bf_unrolled_selection_tids)) {
#endif
          cout << "TID lists are not equal!" << endl;
          cout << "Serial Selection result size: "
               << serial_selection_tids->size() << endl;
          std::copy(serial_selection_tids->begin(),
                    serial_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Branch-free Selection result size: "
               << serial_bf_selection_tids->size() << endl;
          std::copy(serial_bf_selection_tids->begin(),
                    serial_bf_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Unrolled Selection result size: "
               << serial_unrolled_selection_tids->size() << endl;
          std::copy(serial_unrolled_selection_tids->begin(),
                    serial_unrolled_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Branch-free Unrolled Selection result size: "
               << serial_bf_unrolled_selection_tids->size() << endl;
          std::copy(serial_bf_unrolled_selection_tids->begin(),
                    serial_bf_unrolled_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));

#ifdef ENABLE_SIMD_ACCELERATION
          cout << "Serial SIMD Selection result size: "
               << serial_simd_selection_tids->size() << endl;
          std::copy(serial_simd_selection_tids->begin(),
                    serial_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Branch-free SIMD Selection result size: "
               << serial_bf_simd_selection_tids->size() << endl;
          std::copy(serial_bf_simd_selection_tids->begin(),
                    serial_bf_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Unrolled SIMD Selection result size: "
               << serial_unrolled_simd_selection_tids->size() << endl;
          std::copy(serial_unrolled_simd_selection_tids->begin(),
                    serial_unrolled_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Serial Branch-free Unrolled SIMD Selection result size: "
               << serial_bf_unrolled_simd_selection_tids->size() << endl;
          std::copy(serial_bf_unrolled_simd_selection_tids->begin(),
                    serial_bf_unrolled_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
#endif
          cout << "Parallel Selection result size: "
               << parallel_selection_tids->size() << endl;
          std::copy(parallel_selection_tids->begin(),
                    parallel_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Branch-free Selection result size: "
               << parallel_bf_selection_tids->size() << endl;
          std::copy(parallel_bf_selection_tids->begin(),
                    parallel_bf_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Unrolled Selection result size: "
               << parallel_unrolled_selection_tids->size() << endl;
          std::copy(parallel_unrolled_selection_tids->begin(),
                    parallel_unrolled_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Branch-free Unrolled Selection result size: "
               << parallel_bf_unrolled_selection_tids->size() << endl;
          std::copy(parallel_bf_unrolled_selection_tids->begin(),
                    parallel_bf_unrolled_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));

#ifdef ENABLE_SIMD_ACCELERATION
          cout << "Parallel SIMD Selection result size: "
               << parallel_simd_selection_tids->size() << endl;
          std::copy(parallel_simd_selection_tids->begin(),
                    parallel_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Branch-free SIMD Selection result size: "
               << parallel_bf_simd_selection_tids->size() << endl;
          std::copy(parallel_bf_simd_selection_tids->begin(),
                    parallel_bf_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          //                            std::copy(parallel_simd_selection_tids->begin(),
          //                            parallel_simd_selection_tids->end(),
          //                            std::ostream_iterator<int>(std::cout,
          //                            "\n"));
          cout << "Parallel Unrolled SIMD Selection result size: "
               << parallel_unrolled_simd_selection_tids->size() << endl;
          std::copy(parallel_unrolled_simd_selection_tids->begin(),
                    parallel_unrolled_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
          cout << "Parallel Branch-free Unrolled SIMD Selection result size: "
               << parallel_bf_unrolled_simd_selection_tids->size() << endl;
          std::copy(parallel_bf_unrolled_simd_selection_tids->begin(),
                    parallel_bf_unrolled_simd_selection_tids->end(),
                    std::ostream_iterator<int>(std::cout, "\t"));
#endif

          //                            std::copy(parallel_unrolled_simd_selection_tids->begin(),
          //                            parallel_unrolled_simd_selection_tids->end(),
          //                            std::ostream_iterator<int>(std::cout,
          //                            "\n"));

          // unsigned int
          // size=std::min(serial_selection_tids->size(),parallel_selection_tids->size());
          ////                                for(unsigned int i=0;i<size;i++){
          ////                                    cout << "Serial id: " <<
          ///(*serial_selection_tids)[i] << " \tParallel id:"<<
          ///(*parallel_selection_tids)[i] << endl;
          ////                                }
          //                                for(unsigned int i=0;i<size;i++){
          //                                    if((*serial_selection_tids)[i]!=(*parallel_selection_tids)[i])
          //                                         cout << "Serial id: " <<
          //                                         (*serial_selection_tids)[i]
          //                                         << " \tParallel id:"<<
          //                                         (*parallel_selection_tids)[i]
          //                                         << endl;
          //                                }
          //                                if(size<serial_selection_tids->size()){
          //                                    cout << "Detected additional
          //                                    values for serial selection " <<
          //                                    endl;
          //                                    for(unsigned int
          //                                    i=size;i<serial_selection_tids->size();i++){
          //                                           cout << "id: " << i << "
          //                                           val: " <<
          //                                           (*serial_selection_tids)[i]
          //                                           << endl;
          //                                    }
          //                                }
          COGADB_FATAL_ERROR(
              "Selection Unittests failed! At least one algorithm works "
              "incorrect!",
              "");
        }
        //                            assert((*serial_selection_tids)==(*lock_free_parallel_selection_tids));
      }
    }
  }
  return true;
}

bool cdk_join_test() {
  for (unsigned int u = 0; u < 10; u++) {
    std::vector<int> pk_column;
    std::vector<int> fk_column;

    for (unsigned int i = 0; i < 300 * 10; ++i) {
      pk_column.push_back(int(i));
    }

    for (unsigned int i = 0; i < 10000 * 100; ++i) {
      fk_column.push_back(rand() % 1000);
    }
    // for(unsigned int j=0;j<100;j++){
    Timestamp begin;
    Timestamp end;

    begin = getTimestamp();
    PositionListPairPtr result_nested_loop_join =
        CoGaDB::CDK::join::nested_loop_join(
            hype::util::begin_ptr(pk_column), pk_column.size(),
            hype::util::begin_ptr(fk_column), fk_column.size());
    end = getTimestamp();
    cout << "NLJ: " << double(end - begin) / (1000 * 1000) << "ms" << endl;

    begin = getTimestamp();
    PositionListPairPtr result_blocked_nested_loop_join =
        CoGaDB::CDK::join::block_nested_loop_join(
            hype::util::begin_ptr(pk_column), pk_column.size(),
            hype::util::begin_ptr(fk_column), fk_column.size(), 1000);
    end = getTimestamp();
    cout << "Block-wise NLJ: " << double(end - begin) / (1000 * 1000) << "ms"
         << endl;

    begin = getTimestamp();
    PositionListPairPtr result_hash_join = CoGaDB::CDK::join::serial_hash_join(
        hype::util::begin_ptr(pk_column), pk_column.size(),
        hype::util::begin_ptr(fk_column),
        fk_column
            .size());  // BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab,
                       // "FK_ID",HASH_JOIN, LOOKUP,CPU);
    end = getTimestamp();
    cout << "Serial HJ: " << double(end - begin) / (1000 * 1000) << "ms"
         << endl;

    begin = getTimestamp();
    PositionListPairPtr result_radix_join = CoGaDB::CDK::join::radix_join(
        hype::util::begin_ptr(pk_column), pk_column.size(),
        hype::util::begin_ptr(fk_column),
        fk_column
            .size());  // BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab,
                       // "FK_ID",HASH_JOIN, LOOKUP,CPU);
    end = getTimestamp();
    cout << "Radix Join: " << double(end - begin) / (1000 * 1000) << "ms"
         << endl;

    std::set<std::pair<TID, TID>> nlj_set;
    assert(result_nested_loop_join->first->size() ==
           result_nested_loop_join->second->size());
    for (unsigned int i = 0; i < result_nested_loop_join->first->size(); ++i) {
      // std::cout << result_nested_loop_join->first->operator [](i) << "," <<
      // result_nested_loop_join->second->operator [](i) << std::endl;
      nlj_set.insert(
          std::make_pair(result_nested_loop_join->first->operator[](i),
                         result_nested_loop_join->second->operator[](i)));
    }

    std::set<std::pair<TID, TID>> block_nlj_set;
    assert(result_blocked_nested_loop_join->first->size() ==
           result_blocked_nested_loop_join->second->size());
    for (unsigned int i = 0; i < result_blocked_nested_loop_join->first->size();
         ++i) {
      // std::cout << result_blocked_nested_loop_join->first->operator [](i) <<
      // "," << result_blocked_nested_loop_join->second->operator [](i) <<
      // std::endl;
      block_nlj_set.insert(std::make_pair(
          result_blocked_nested_loop_join->first->operator[](i),
          result_blocked_nested_loop_join->second->operator[](i)));
    }

    std::set<std::pair<TID, TID>> shj_set;
    assert(result_hash_join->first->size() == result_hash_join->second->size());
    for (unsigned int i = 0; i < result_hash_join->first->size(); ++i) {
      shj_set.insert(std::make_pair(result_hash_join->first->operator[](i),
                                    result_hash_join->second->operator[](i)));
    }

    std::set<std::pair<TID, TID>> rdx_set;
    assert(result_radix_join->first->size() ==
           result_radix_join->second->size());
    for (unsigned int i = 0; i < result_radix_join->first->size(); ++i) {
      // std::cout << result_radix_join->first->operator [](i) << "," <<
      // result_radix_join->second->operator [](i) << std::endl;
      rdx_set.insert(std::make_pair(result_radix_join->first->operator[](i),
                                    result_radix_join->second->operator[](i)));
    }

    // check equality
    cout << "NLJ Result size: " << nlj_set.size()
         << " Block-wise NLJ: " << block_nlj_set.size() << endl;
    assert(nlj_set == block_nlj_set);
    assert(nlj_set == shj_set);
    assert(nlj_set == rdx_set);

    //#ifdef ENABLE_GPU_ACCELERATION
    //	TablePtr result_gpu_merge_join =
    // BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab,
    //"FK_ID",SORT_MERGE_JOIN_2, LOOKUP,GPU);
    //#endif

    //        std::list<std::string> column_names;
    //        column_names.push_back("PK_ID");
    //        column_names.push_back("FK_ID");
    //        column_names.push_back("VAL_Table1");
    //        column_names.push_back("VAL_Table2");
    //
    //
    ////        result can be the sam,e but differ in sort rder, so first sort
    /// both reatliosn and the compare tables
    ////        Hint: assumes that complex sort works!
    //        result_nested_loop_join =
    //        BaseTable::sort(result_nested_loop_join,column_names);
    //        result_hash_join =
    //        BaseTable::sort(result_hash_join,column_names,ASCENDING,LOOKUP);
    //        result_parallel_hash_join =
    //        BaseTable::sort(result_parallel_hash_join,column_names,ASCENDING,LOOKUP);
    //#ifdef ENABLE_GPU_ACCELERATION
    //	result_gpu_merge_join =
    // BaseTable::sort(result_gpu_merge_join,column_names);
    //#endif
    //        /*
    //        if(result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(result_sort_merge_join->getColumnbyName("PK_ID"))){
    //            cerr << "Error in SortMergeJoin! PK_ID Column not correct!" <<
    //            endl;
    //            return false;
    //        }
    //        if(result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(result_sort_merge_join->getColumnbyName("FK_ID"))){
    //            cerr << "Error in SortMergeJoin! FK_ID Column not correct!" <<
    //            endl;
    //            return false;
    //        }*/
    //        if(!result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(result_hash_join->getColumnbyName("PK_ID"))){
    //            cerr << "Error in HashJoin! PK_ID Column not correct!" <<
    //            endl;
    //            cout << "PK_ID Column Sizes: NLJ: " <<
    //            result_nested_loop_join->getColumnbyName("PK_ID")->size() << "
    //            HJ: " << result_hash_join->getColumnbyName("PK_ID")->size() <<
    //            endl;
    //            return false;
    //        }
    //        if(!result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(result_hash_join->getColumnbyName("FK_ID"))){
    //            cerr << "Error in HashJoin! FK_ID Column not correct!" <<
    //            endl;
    //            return false;
    //        }
    //
    //        if(!result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(result_parallel_hash_join->getColumnbyName("PK_ID"))){
    //            cerr << "Error in parallel HashJoin! PK_ID Column not
    //            correct!" << endl;
    //            cout << "PK_ID Column Sizes: NLJ: " <<
    //            result_nested_loop_join->getColumnbyName("PK_ID")->size() << "
    //            HJ: " <<
    //            result_parallel_hash_join->getColumnbyName("PK_ID")->size() <<
    //            endl;
    //            return false;
    //        }
    //        if(!result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(result_parallel_hash_join->getColumnbyName("FK_ID"))){
    //            cerr << "Error in parallel HashJoin! FK_ID Column not
    //            correct!" << endl;
    //            return false;
    //        }
    //
    //#ifdef ENABLE_GPU_ACCELERATION
    //        if(!result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(result_gpu_merge_join->getColumnbyName("PK_ID"))){
    //            cerr << "Error in GPU SortMergeJoin! PK_ID Column not
    //            correct!" << endl;
    //            cout << "Nested Loop Join:" << endl;
    //            result_nested_loop_join->getColumnbyName("PK_ID")->print();
    //            cout << "GPU Merge Join:" << endl;
    //            result_gpu_merge_join->getColumnbyName("PK_ID")->print();
    //            cout << "Nested Loop Join:" << endl;
    //            result_nested_loop_join->print();
    //            cout << "GPU Merge Join:" << endl;
    //            result_gpu_merge_join->print();
    //            return false;
    //        }
    //        if(!result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(result_gpu_merge_join->getColumnbyName("FK_ID"))){
    //            cerr << "Error in GPU SortMergeJoin! FK_ID Column not
    //            correct!" << endl;
    //            return false;
    //        }
    //#endif

    //}
  }

  return true;
}

bool cdk_invisiblejoin_test() {
  //            unsigned int dimension_table_sizes[4];
  //            dimension_table_sizes[0]=300;
  //            dimension_table_sizes[1]=2000;
  //            dimension_table_sizes[2]=4000;
  //            dimension_table_sizes[3]=1000;
  //            unsigned int fact_table_size=6*1000*1000; //10000;
  //            //6*1000*1000;
  //
  //            TablePtr dim_tab1;
  //            TablePtr dim_tab2;
  //            TablePtr dim_tab3;
  //            TablePtr dim_tab4;
  //            TablePtr fact_table;
  //
  //            {
  //                TableSchema schema;
  //                schema.push_back(Attribut(INT,"DIM1_PK_ID"));
  //                schema.push_back(Attribut(INT,"DIM1_ATTRIBUTE"));
  //
  //                dim_tab1=TablePtr(new Table("DimensionTable1",schema));
  //
  //                for(unsigned int i=0;i<dimension_table_sizes[0];++i){
  //                    {Tuple t; t.push_back(int(i)); t.push_back(rand()%1000);
  //                    dim_tab1->insert(t);}
  //                }
  //                if(!dim_tab1->setPrimaryKeyConstraint("DIM1_PK_ID")){
  //                    COGADB_FATAL_ERROR("Failed to set Primary Key
  //                    Constraint!","");
  //                }
  //                //getGlobalTableList().push_back(new_table);
  //                addToGlobalTableList(dim_tab1);
  //            }
  //
  //            {
  //                TableSchema schema;
  //                schema.push_back(Attribut(INT,"DIM2_PK_ID"));
  //                schema.push_back(Attribut(INT,"DIM2_ATTRIBUTE"));
  //
  //                dim_tab2=TablePtr(new Table("DimensionTable2",schema));
  //                for(unsigned int i=0;i<dimension_table_sizes[1];++i){
  //                    {Tuple t; t.push_back(int(i)); t.push_back(rand()%1000);
  //                    dim_tab2->insert(t);}
  //                }
  //                if(!dim_tab2->setPrimaryKeyConstraint("DIM2_PK_ID")){
  //                    COGADB_FATAL_ERROR("Failed to set Primary Key
  //                    Constraint!","");
  //                }
  //                addToGlobalTableList(dim_tab2);
  //            }
  //
  //            {
  //                TableSchema schema;
  //                schema.push_back(Attribut(INT,"DIM3_PK_ID"));
  //                schema.push_back(Attribut(INT,"DIM3_ATTRIBUTE"));
  //
  //                dim_tab3=TablePtr(new Table("DimensionTable3",schema));
  //                for(unsigned int i=0;i<dimension_table_sizes[2];++i){
  //                    {Tuple t; t.push_back(int(i)); t.push_back(rand()%1000);
  //                    dim_tab3->insert(t);}
  //                }
  //                if(!dim_tab3->setPrimaryKeyConstraint("DIM3_PK_ID")){
  //                    COGADB_FATAL_ERROR("Failed to set Primary Key
  //                    Constraint!","");
  //                }
  //                addToGlobalTableList(dim_tab3);
  //            }
  //
  //            {
  //                TableSchema schema;
  //                schema.push_back(Attribut(INT,"DIM4_PK_ID"));
  //                schema.push_back(Attribut(INT,"DIM4_ATTRIBUTE"));
  //
  //                dim_tab4=TablePtr(new Table("DimensionTable4",schema));
  //                for(unsigned int i=0;i<dimension_table_sizes[3];++i){
  //                    {Tuple t; t.push_back(int(i)); t.push_back(rand()%1000);
  //                    dim_tab4->insert(t);}
  //                }
  //                if(!dim_tab4->setPrimaryKeyConstraint("DIM4_PK_ID")){
  //                    COGADB_FATAL_ERROR("Failed to set Primary Key
  //                    Constraint!","");
  //                }
  //                addToGlobalTableList(dim_tab4);
  //            }
  //
  //            TableSchema schema2;
  //            schema2.push_back(Attribut(INT,"DIM1_FK_ID"));
  //            schema2.push_back(Attribut(INT,"DIM2_FK_ID"));
  //            schema2.push_back(Attribut(INT,"DIM3_FK_ID"));
  //            schema2.push_back(Attribut(INT,"DIM4_FK_ID"));
  //            //schema2.push_back(Attribut(INT,"VAL_Table2"));
  //
  //            fact_table=TablePtr(new Table("FactTable",schema2));
  //
  //            for(unsigned int i=0;i<fact_table_size;++i){
  //                //generate also foreign keys with no matching primary key
  //                (in case a primary key table is prefiltered)
  //                {Tuple t;
  //                    t.push_back(int(rand()%dimension_table_sizes[0]));
  //                    t.push_back(int(rand()%dimension_table_sizes[1]));
  //                    t.push_back(int(rand()%dimension_table_sizes[2]));
  //                    t.push_back(int(rand()%dimension_table_sizes[3]));
  //                    fact_table->insert(t);
  //                }
  //            }
  //            if(!fact_table->setForeignKeyConstraint("DIM1_FK_ID","DIM1_PK_ID",
  //            "DimensionTable1")){ COGADB_ERROR("Failed to set Foreign Key
  //            Constraint!",""); return false;}
  //            if(!fact_table->setForeignKeyConstraint("DIM2_FK_ID","DIM2_PK_ID",
  //            "DimensionTable2")){ COGADB_ERROR("Failed to set Foreign Key
  //            Constraint!",""); return false;}
  //            if(!fact_table->setForeignKeyConstraint("DIM3_FK_ID","DIM3_PK_ID",
  //            "DimensionTable3")){ COGADB_ERROR("Failed to set Foreign Key
  //            Constraint!",""); return false;}
  //            if(!fact_table->setForeignKeyConstraint("DIM4_FK_ID","DIM4_PK_ID",
  //            "DimensionTable4")){ COGADB_ERROR("Failed to set Foreign Key
  //            Constraint!",""); return false;}
  //
  //            addToGlobalTableList(fact_table);
  //
  //            KNF_Selection_Expression knf_expr_dim1; //LO_QUANTITY<25 AND
  //            LO_DISCOUNT>0.99 AND LO_DISCOUNT<3.01
  //            {
  //                Disjunction d;
  //                d.push_back(Predicate("DIM1_ATTRIBUTE", boost::any(500),
  //                ValueConstantPredicate, LESSER)); //LO_DISCOUNT>0.99
  //                knf_expr_dim1.disjunctions.push_back(d);
  //            }
  //            KNF_Selection_Expression knf_expr_dim2; //LO_QUANTITY<25 AND
  //            LO_DISCOUNT>0.99 AND LO_DISCOUNT<3.01
  //            {
  //                Disjunction d;
  //                d.push_back(Predicate("DIM2_ATTRIBUTE", boost::any(500),
  //                ValueConstantPredicate, LESSER)); //LO_DISCOUNT>0.99
  //                knf_expr_dim2.disjunctions.push_back(d);
  //            }
  //            KNF_Selection_Expression knf_expr_dim3; //LO_QUANTITY<25 AND
  //            LO_DISCOUNT>0.99 AND LO_DISCOUNT<3.01
  //            {
  //                Disjunction d;
  //                d.push_back(Predicate("DIM3_ATTRIBUTE", boost::any(500),
  //                ValueConstantPredicate, LESSER)); //LO_DISCOUNT>0.99
  //                knf_expr_dim3.disjunctions.push_back(d);
  //            }
  //            KNF_Selection_Expression knf_expr_dim4; //LO_QUANTITY<25 AND
  //            LO_DISCOUNT>0.99 AND LO_DISCOUNT<3.01
  //            {
  //                Disjunction d;
  //                d.push_back(Predicate("DIM4_ATTRIBUTE", boost::any(500),
  //                ValueConstantPredicate, LESSER)); //LO_DISCOUNT>0.99
  //                knf_expr_dim4.disjunctions.push_back(d);
  //            }
  //
  //            std::string s;
  //            std::cout << "Press Enter to start benchmark" << std::endl;
  //            std::cin >> s;
  //
  //            for(unsigned int i=0;i<10;++i){
  //                Timestamp begin_normal_star_join = getTimestamp();
  //                TablePtr filtered_dim_tab1 = BaseTable::selection(dim_tab1,
  //                knf_expr_dim1, LOOKUP);
  //                TablePtr filtered_dim_tab2 = BaseTable::selection(dim_tab2,
  //                knf_expr_dim2, LOOKUP);
  //                TablePtr filtered_dim_tab3 = BaseTable::selection(dim_tab3,
  //                knf_expr_dim3, LOOKUP);
  //                TablePtr filtered_dim_tab4 = BaseTable::selection(dim_tab4,
  //                knf_expr_dim4, LOOKUP);
  //
  //                std::cout << "Output Size selection for Table " <<
  //                filtered_dim_tab1->getName() << ": " <<
  //                filtered_dim_tab1->getNumberofRows() << std::endl;
  //                std::cout << "Output Size selection for Table " <<
  //                filtered_dim_tab2->getName() << ": " <<
  //                filtered_dim_tab2->getNumberofRows() << std::endl;
  //                std::cout << "Output Size selection for Table " <<
  //                filtered_dim_tab3->getName() << ": " <<
  //                filtered_dim_tab3->getNumberofRows() << std::endl;
  //                std::cout << "Output Size selection for Table " <<
  //                filtered_dim_tab4->getName() << ": " <<
  //                filtered_dim_tab4->getNumberofRows() << std::endl;
  //
  //                //perform normal joins
  //                TablePtr join_result1 =
  //                BaseTable::join(fact_table,"DIM1_FK_ID",filtered_dim_tab1,
  //                "DIM1_PK_ID",HASH_JOIN, LOOKUP,CPU);
  //                TablePtr join_result2 =
  //                BaseTable::join(join_result1,"DIM2_FK_ID",filtered_dim_tab2,
  //                "DIM2_PK_ID",HASH_JOIN, LOOKUP,CPU);
  //                TablePtr join_result3 =
  //                BaseTable::join(join_result2,"DIM3_FK_ID",filtered_dim_tab3,
  //                "DIM3_PK_ID",HASH_JOIN, LOOKUP,CPU);
  //                TablePtr join_result4 =
  //                BaseTable::join(join_result3,"DIM4_FK_ID",filtered_dim_tab4,
  //                "DIM4_PK_ID",HASH_JOIN, LOOKUP,CPU);
  //
  //                Timestamp end_normal_star_join = getTimestamp();
  //
  //                std::cout << "Output Size JOIN(FactTable,DIM1): " <<
  //                BaseTable::join(fact_table,"DIM1_FK_ID",filtered_dim_tab1,
  //                "DIM1_PK_ID",HASH_JOIN, LOOKUP,CPU)->getNumberofRows() << "
  //                rows" << std::endl;
  //                std::cout << "Output Size JOIN(FactTable,DIM2): " <<
  //                BaseTable::join(fact_table,"DIM2_FK_ID",filtered_dim_tab2,
  //                "DIM2_PK_ID",HASH_JOIN, LOOKUP,CPU)->getNumberofRows() << "
  //                rows" << std::endl;
  //                std::cout << "Output Size JOIN(FactTable,DIM3): " <<
  //                BaseTable::join(fact_table,"DIM3_FK_ID",filtered_dim_tab3,
  //                "DIM3_PK_ID",HASH_JOIN, LOOKUP,CPU)->getNumberofRows() << "
  //                rows" << std::endl;
  //                std::cout << "Output Size JOIN(FactTable,DIM4): " <<
  //                BaseTable::join(fact_table,"DIM4_FK_ID",filtered_dim_tab4,
  //                "DIM4_PK_ID",HASH_JOIN, LOOKUP,CPU)->getNumberofRows() << "
  //                rows" << std::endl;
  //
  //
  //
  //                cout << "Size of Filtered Fact Table (Normal Joins): " <<
  //                join_result4->getNumberofRows() << "rows" << endl;
  //
  //                Timestamp begin_normal_invisible_join = getTimestamp();
  //                //Perform Invisible Join
  //                InvisibleJoinSelectionList dimensions;
  //                dimensions.push_back(InvisibleJoinSelection("DimensionTable1",
  //                Predicate("DIM1_PK_ID",
  //                std::string("DIM1_FK_ID"),ValueValuePredicate, EQUAL),
  //                knf_expr_dim1));
  //                dimensions.push_back(InvisibleJoinSelection("DimensionTable2",
  //                Predicate("DIM2_PK_ID",
  //                std::string("DIM2_FK_ID"),ValueValuePredicate, EQUAL),
  //                knf_expr_dim2));
  //                dimensions.push_back(InvisibleJoinSelection("DimensionTable3",
  //                Predicate("DIM3_PK_ID",
  //                std::string("DIM3_FK_ID"),ValueValuePredicate, EQUAL),
  //                knf_expr_dim3));
  //                dimensions.push_back(InvisibleJoinSelection("DimensionTable4",
  //                Predicate("DIM4_PK_ID",
  //                std::string("DIM4_FK_ID"),ValueValuePredicate, EQUAL),
  //                knf_expr_dim4));
  //                TablePtr inv_join_result =
  //                CDK::join::invisibleJoin(fact_table, dimensions);
  //                Timestamp end_normal_invisible_join = getTimestamp();
  //
  //                assert(inv_join_result->getNumberofRows()==join_result4->getNumberofRows());
  //
  //
  //
  //                cout << "Time for Normal Star Join: " <<
  //                double(end_normal_star_join-begin_normal_star_join)/(1000*1000)
  //                << "ms" << std::endl;
  //                cout << "Time for Invisible Join: " <<
  //                double(end_normal_invisible_join-begin_normal_invisible_join)/(1000*1000)
  //                << "ms" << std::endl;
  //            }
  return true;
}

bool cdk_join_performance_test() {
  for (unsigned int u = 1; u < 101; u++) {
    std::vector<int> pk_column;
    std::vector<int> fk_column;

    for (unsigned int i = 0; i < 300 * 1000 * (double(u) / 10); ++i) {
      pk_column.push_back(int(i));
    }

    for (unsigned int i = 0; i < 1 * 1000 * 1000 * (double(u) / 10); ++i) {
      fk_column.push_back(rand() % 1000);
    }
    for (unsigned int j = 0; j < 1; j++) {
      Timestamp begin;
      Timestamp end;

      //        begin=getTimestamp();
      //        PositionListPairPtr result_nested_loop_join =
      //        CoGaDB::CDK::join::nested_loop_join(hype::util::begin_ptr(pk_column),pk_column.size(),
      //                                                                                          hype::util::begin_ptr(fk_column),fk_column.size());
      //        end=getTimestamp();
      //        cout << "NLJ: " << double(end-begin)/(1000*1000) << "ms" <<
      //        endl;

      //        begin=getTimestamp();
      //        PositionListPairPtr result_blocked_nested_loop_join =
      //        CoGaDB::CDK::join::block_nested_loop_join(hype::util::begin_ptr(pk_column),pk_column.size(),
      //                                                                                                        hype::util::begin_ptr(fk_column),fk_column.size());
      //        end=getTimestamp();
      //        cout << "Block-wise NLJ: " << double(end-begin)/(1000*1000) <<
      //        "ms" << endl;

      PositionListPairPtr result_blocked_nested_loop_join;
      //        for(unsigned int i=1;i<11;++i){
      //            begin=getTimestamp();
      //            unsigned int block_size = 9;
      //            result_blocked_nested_loop_join =
      //            CoGaDB::CDK::join::block_nested_loop_join(hype::util::begin_ptr(pk_column),pk_column.size(),
      //            hype::util::begin_ptr(fk_column),fk_column.size(),
      //            block_size);
      //            end=getTimestamp();
      //            cout << "Block-wise NLJ: " << double(end-begin)/(1000*1000)
      //            << "ms (Blocksize: " << block_size*sizeof(int) << " bytes )"
      //            << endl;
      //        }
      //        for(unsigned int i=1;i<11;++i){
      //            begin=getTimestamp();
      //            unsigned int block_size = i;
      //            result_blocked_nested_loop_join =
      //            CoGaDB::CDK::join::block_nested_loop_join(hype::util::begin_ptr(pk_column),pk_column.size(),
      //            hype::util::begin_ptr(fk_column),fk_column.size(),
      //            block_size);
      //            end=getTimestamp();
      //            cout << "Block-wise NLJ: " << double(end-begin)/(1000*1000)
      //            << "ms (Blocksize: " << block_size*sizeof(int) << " bytes )"
      //            << endl;
      //        }
      //        for(unsigned int i=1;i<11;++i){
      //            begin=getTimestamp();
      //            unsigned int block_size = 10*i;
      //            result_blocked_nested_loop_join =
      //            CoGaDB::CDK::join::block_nested_loop_join(hype::util::begin_ptr(pk_column),pk_column.size(),
      //            hype::util::begin_ptr(fk_column),fk_column.size(),
      //            block_size);
      //            end=getTimestamp();
      //            cout << "Block-wise NLJ: " << double(end-begin)/(1000*1000)
      //            << "ms (Blocksize: " << block_size*sizeof(int) << " bytes )"
      //            << endl;
      //        }
      //        for(unsigned int i=1;i<11;++i){
      //            begin=getTimestamp();
      //            unsigned int block_size = 1000000*i;
      //            result_blocked_nested_loop_join =
      //            CoGaDB::CDK::join::block_nested_loop_join(hype::util::begin_ptr(pk_column),pk_column.size(),
      //            hype::util::begin_ptr(fk_column),fk_column.size(),
      //            block_size);
      //            end=getTimestamp();
      //            cout << "Block-wise NLJ: " << double(end-begin)/(1000*1000)
      //            << "ms (Blocksize: " << block_size*sizeof(int) << " bytes )"
      //            << endl;
      //        }

      std::cout << "Size PK Table: " << 300 * 1000 * (double(u) / 10)
                << "rows, Size FK Table: " << 1 * 1000 * 1000 * (double(u) / 10)
                << std::endl;

      begin = getTimestamp();
      PositionListPairPtr result_hash_join = CoGaDB::CDK::join::serial_hash_join(
          hype::util::begin_ptr(pk_column), pk_column.size(),
          hype::util::begin_ptr(fk_column),
          fk_column
              .size());  // BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab,
                         // "FK_ID",HASH_JOIN, LOOKUP,CPU);
      end = getTimestamp();
      cout << "Serial HJ: " << double(end - begin) / (1000 * 1000) << "ms"
           << endl;

      begin = getTimestamp();
      PositionListPairPtr result_hash_join2 =
          CoGaDB::CDK::main_memory_joins::serial_hash_join(
              hype::util::begin_ptr(pk_column), pk_column.size(),
              hype::util::begin_ptr(fk_column),
              fk_column
                  .size());  // BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab,
                             // "FK_ID",HASH_JOIN, LOOKUP,CPU);
      end = getTimestamp();
      cout << "Serial HJ (C Implementation): "
           << double(end - begin) / (1000 * 1000) << "ms" << endl;

      begin = getTimestamp();
      // PositionListPairPtr result_hash_join3 =
      // CoGaDB::CDK::main_memory_joins::parallel_hash_join(hype::util::begin_ptr(pk_column),pk_column.size(),
      //                                                                                         hype::util::begin_ptr(fk_column),fk_column.size()); //BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab, "FK_ID",HASH_JOIN, LOOKUP,CPU);
      end = getTimestamp();
      cout << "Parallel HJ (C + OpenMP Implementation): "
           << double(end - begin) / (1000 * 1000) << "ms" << endl;

      //        begin=getTimestamp();
      // 	PositionListPairPtr result_radix_join =
      // CoGaDB::CDK::join::radix_join(hype::util::begin_ptr(pk_column),pk_column.size(),
      //                                                                              hype::util::begin_ptr(fk_column),fk_column.size(), 15, 4); //BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab, "FK_ID",HASH_JOIN, LOOKUP,CPU);
      //        end=getTimestamp();
      //        cout << "Radix Join: " << double(end-begin)/(1000*1000) << "ms"
      //        << endl;
      PositionListPairPtr result_radix_join;
      begin = getTimestamp();
      result_radix_join = CoGaDB::CDK::join::radix_join(
          hype::util::begin_ptr(pk_column), pk_column.size(),
          hype::util::begin_ptr(fk_column), fk_column.size(), 13,
          2);  // BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab,
               // "FK_ID",HASH_JOIN, LOOKUP,CPU);
      end = getTimestamp();
      // cout << "Radix Join: " << "(passes: " << j << ", total_num_bits: " << i
      // << " ):"  << double(end-begin)/(1000*1000) << "ms" << endl;
      cout << "Radix Join: " << double(end - begin) / (1000 * 1000) << "ms"
           << endl;

      //        for(unsigned int i=3;i<27;++i){
      //            for(unsigned int j=1;j<10;++j){
      //
      //                begin=getTimestamp();
      //                result_radix_join =
      //                CoGaDB::CDK::join::radix_join(hype::util::begin_ptr(pk_column),pk_column.size(),
      //                                                                                      hype::util::begin_ptr(fk_column),fk_column.size(),i,j); //BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab, "FK_ID",HASH_JOIN, LOOKUP,CPU);
      //                end=getTimestamp();
      //                cout << "Radix Join: " << "(passes: " << j << ",
      //                total_num_bits: " << i << " ):"  <<
      //                double(end-begin)/(1000*1000) << "ms" << endl;
      //            }
      //            if(i>10) i++;
      //            //if(i>20) i++;
      //        }
    }
  }
  return true;
}

}  // end namespace unit_tests

}  // end namespace CoGaDB
