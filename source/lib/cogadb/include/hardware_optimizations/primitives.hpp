#pragma once

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <boost/any.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>

#include <core/base_table.hpp>
#include <core/bitmap.hpp>
#include <core/global_definitions.hpp>
#include <core/positionlist.hpp>
#include <core/selection_expression.hpp>

#include <core/column.hpp>
#include <hardware_optimizations/main_memory_joins/hash_joins.hpp>
#include <hardware_optimizations/simd_acceleration.hpp>
#include <util/time_measurement.hpp>

#include <util/functions.hpp>

#define COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i, array_tids, \
                                               pos, COMPARATOR)             \
  if (array[i] COMPARATOR value) {                                          \
    array_tids[pos++] = i;                                                  \
  }
#define COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i, array_tids, \
                                                 pos, COMPARATOR)             \
  array_tids[pos] = i;                                                        \
  pos += (array[i] COMPARATOR value);
#define COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i, pos, \
                                               result_bitmap)        \
  result_bitmap[pos++] = (array[i] == value);

#define COGADB_SELECTION_BODY_WRITE_BMP_BRANCH(array, value, i, bmp, pos, \
                                               COMPARATOR)                \
  if (array[i] COMPARATOR value) {                                        \
    bmp[i / 8] |= 1 << i % 8;                                             \
  }
#define COGADB_SELECTION_BODY_WRITE_BMP_NOBRANCH(array, value, i, bmp, pos, \
                                                 COMPARATOR)                \
  bmp[i / 8] |= (array[i] COMPARATOR value) << i % 8;

/*
#ifdef ENABLE_BRANCHING_SCAN
#define COGADB_SELECTION_BODY_WRITE_TID(array,value,i,array_tids,pos,COMPARATOR)
COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array,value,i,array_tids,pos,COMPARATOR)
#else
#define COGADB_SELECTION_BODY_WRITE_TID(array,value,i,array_tids,pos,COMPARATOR)
COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i,array_tids,pos,COMPARATOR)
#endif
*/

namespace CoGaDB {
  class LookupTable;
  typedef boost::shared_ptr<LookupTable> LookupTablePtr;

  class DictionaryCompressedCol;

  struct InvisibleJoinSelection {
    InvisibleJoinSelection(std::string table_name, Predicate join_pred,
                           KNF_Selection_Expression knf_sel_expr);
    std::string table_name;
    Predicate join_pred;
    KNF_Selection_Expression knf_sel_expr;
  };

  inline InvisibleJoinSelection::InvisibleJoinSelection(
      std::string table_name_a, Predicate join_pred_a,
      KNF_Selection_Expression knf_sel_expr_a)
      : table_name(table_name_a),
        join_pred(join_pred_a),
        knf_sel_expr(knf_sel_expr_a) {}

  typedef std::list<InvisibleJoinSelection> InvisibleJoinSelectionList;

  // CoGaDB Database Kernel
  namespace CDK {

    namespace util {

      template <typename T>
      void gather_thread(T* __restrict__ array, TID* __restrict__ tid_array,
                         const TID& begin_index, const TID& end_index,
                         T* __restrict__ result_array) {
        TID pos = begin_index;
        const size_t chunk_size = end_index - begin_index;
        for (size_t i = 0; i < chunk_size; ++i) {
          result_array[pos] = array[tid_array[pos]];
          pos++;
        }
      }

      template <typename T>
      void parallel_gather(T* __restrict__ array, TID* __restrict__ tid_array,
                           const size_t& array_size,
                           T* __restrict__ result_array,
                           unsigned int number_of_threads =
                               boost::thread::hardware_concurrency()) {
        std::vector<size_t> result_sizes(number_of_threads);
        boost::thread_group threads;

        for (unsigned int thread_id = 0; thread_id < number_of_threads;
             ++thread_id) {
          // number of elements per thread
          size_t chunk_size = array_size / number_of_threads;
          TID begin_index = chunk_size * thread_id;
          TID end_index;
          if (thread_id + 1 == number_of_threads) {
            // process until end of input array
            end_index = array_size;
          } else {
            end_index = (chunk_size) * (thread_id + 1);
          }

          // gather_thread(array, tid_array, begin_index, end_index,
          // result_array);
          // create a gather thread
          threads.add_thread(new boost::thread(
              boost::bind(&gather_thread<T>, array, tid_array, begin_index,
                          end_index, result_array)));
        }
        threads.join_all();
      }

      template <class T>
      void serial_gather(T* __restrict__ array, TID* __restrict__ tid_array,
                         const size_t& tid_array_size,
                         T* __restrict__ result_array) {
        for (TID i = 0; i < tid_array_size; i++) {
          result_array[i] = array[tid_array[i]];
        }
      }

      //        //ATTENTION: result_array has to have AT LEAST (array_size+1)
      //        bytes!
      //        template<class T>
      //        void serial_prefixsum(T* __restrict__ array, const size_t&
      //        array_size,  T* __restrict__ result_array){
      //                    //std::vector<size_t>
      //                    prefix_sum(number_of_threads+1);
      //                    result_array[0]=0;
      //                    for(unsigned int i=1;i<array_size+1;i++){
      //                        result_array[i]=result_array[i-1]+array[i-1];
      //                    }
      //        }

      inline PositionListPtr gather(PositionListPtr values,
                                    PositionListPtr tids) {
        if (!values || !tids) return PositionListPtr();
        PositionListPtr result = createPositionList();
        result->resize(tids->size());

        parallel_gather(values->data(), tids->data(), tids->size(),
                        result->data(), boost::thread::hardware_concurrency());

        return result;
      }

    }  // end namespace util

    namespace selection {

      template <class T>
      const PositionListPtr serial_selection(
          T* __restrict__ array, size_t array_size,
          const boost::any& value_for_comparison, const ValueComparator comp) {
        T value;
        if (value_for_comparison.type() != typeid(T)) {
          // catch some special cases
          if (typeid(T) == typeid(float) &&
              value_for_comparison.type() == typeid(int)) {
            value = boost::any_cast<int>(value_for_comparison);
          } else {
            std::stringstream str_stream;
            str_stream << "Fatal Error!!! Typemismatch during Selection: "
                       << "Column Type: " << typeid(T).name()
                       << " filter value type: "
                       << value_for_comparison.type().name();
            COGADB_FATAL_ERROR(str_stream.str(), "");
            //                    std::cout << "Fatal Error!!! Typemismatch for
            //                    column " << std::endl;
            //                    std::cout << "Column Type: " <<
            //                    typeid(T).name() << " filter value type: " <<
            //                    value_for_comparison.type().name() <<
            //                    std::endl;
            //                    std::cout << "File: " << __FILE__ << " Line: "
            //                    << __LINE__ << std::endl;
            //                    exit(-1);
          }
        } else {
          // everything fine, filter value matches type of column
          value = boost::any_cast<T>(value_for_comparison);
        }

        PositionListPtr result_tids = createPositionList();
        // calls realloc internally
        result_tids->resize(array_size);
        // get pointer
        TID* array_tids =
            result_tids->data();  // hype::util::begin_ptr(*result_tids);
        assert(array_tids != NULL);
        size_t pos = 0;

        if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

        if (comp == EQUAL) {
          for (TID i = 0; i < array_size; ++i) {
            if (value == array[i]) {
              // result_tids->push_back(i);
              array_tids[pos++] = i;
            }
            //                    array_tids[pos]=i;
            //                    int tmp=(value==array[i]);
            //                    pos+=tmp;
          }
        } else if (comp == LESSER) {
          for (TID i = 0; i < array_size; ++i) {
            if (array[i] < value) {
              // result_table->insert(this->fetchTuple(i));
              // result_tids->push_back(i);
              array_tids[pos++] = i;
            }
            //                    array_tids[pos]=i;
            //                    int tmp=(value<array[i]);
            //                    pos+=tmp;
          }
        } else if (comp == LESSER_EQUAL) {
          for (TID i = 0; i < array_size; ++i) {
            if (array[i] <= value) {
              // result_tids->push_back(i);
              // result_table->insert(this->fetchTuple(i));
              array_tids[pos++] = i;
            }
          }
        } else if (comp == GREATER) {
          for (TID i = 0; i < array_size; ++i) {
            if (array[i] > value) {
              // result_tids->push_back(i);
              // result_table->insert(this->fetchTuple(i));
              array_tids[pos++] = i;
            }
          }
        } else if (comp == GREATER_EQUAL) {
          for (TID i = 0; i < array_size; ++i) {
            if (array[i] >= value) {
              // result_tids->push_back(i);
              // result_table->insert(this->fetchTuple(i));
              array_tids[pos++] = i;
            }
          }
        } else {
        }
        // shrink to actual result size
        result_tids->resize(pos);
        return result_tids;
      }

#ifdef ENABLE_SIMD_ACCELERATION
      template <>
      inline const PositionListPtr serial_selection(
          int* __restrict__ array, size_t array_size,
          const boost::any& value_for_comparison, const ValueComparator comp) {
        // std::cout << "SIMD SCAN" << std::endl;
        int value;

        if (value_for_comparison.type() != typeid(int)) {
          COGADB_FATAL_ERROR(std::string("Typemismatch for column") +
                                 std::string(" Column Type: ") +
                                 typeid(int).name() +
                                 std::string(" filter value type: ") +
                                 value_for_comparison.type().name(),
                             "");
        } else {
          // everything fine, filter value matches type of column
          value = boost::any_cast<int>(value_for_comparison);
        }
        // unsigned int array_size = this->size();
        // int* array=hype::util::begin_ptr(values_);

        return CoGaDB::simd_selection_int(array, array_size, value, comp);
      }
#endif

#ifdef ENABLE_SIMD_ACCELERATION
      template <>

      inline const PositionListPtr serial_selection(
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
#endif

      template <class T>
      const PositionListPtr serial_column_comparison_selection(
          const T* const __restrict__ array,
          const T* const __restrict__ comp_array, size_t array_size,
          const ValueComparator comp) {
        PositionListPtr result_tids = createPositionList();
        // calls realloc internally
        result_tids->resize(array_size);
        // get pointer
        TID* array_tids =
            result_tids->data();  // hype::util::begin_ptr(*result_tids);
        assert(array_tids != NULL);
        size_t pos = 0;

        if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

        if (comp == EQUAL) {
          for (TID i = 0; i < array_size; ++i) {
            if (comp_array[i] == array[i]) {
              // result_tids->push_back(i);
              array_tids[pos++] = i;
            }
            //                    array_tids[pos]=i;
            //                    int tmp=(value==array[i]);
            //                    pos+=tmp;
          }
        } else if (comp == LESSER) {
          for (TID i = 0; i < array_size; ++i) {
            if (array[i] < comp_array[i]) {
              // result_table->insert(this->fetchTuple(i));
              // result_tids->push_back(i);
              array_tids[pos++] = i;
            }
            //                    array_tids[pos]=i;
            //                    int tmp=(value<array[i]);
            //                    pos+=tmp;
          }
        } else if (comp == LESSER_EQUAL) {
          for (TID i = 0; i < array_size; ++i) {
            if (array[i] <= comp_array[i]) {
              // result_tids->push_back(i);
              // result_table->insert(this->fetchTuple(i));
              array_tids[pos++] = i;
            }
          }
        } else if (comp == GREATER) {
          for (TID i = 0; i < array_size; ++i) {
            if (array[i] > comp_array[i]) {
              // result_tids->push_back(i);
              // result_table->insert(this->fetchTuple(i));
              array_tids[pos++] = i;
            }
          }
        } else if (comp == GREATER_EQUAL) {
          for (TID i = 0; i < array_size; ++i) {
            if (array[i] >= comp_array[i]) {
              // result_tids->push_back(i);
              // result_table->insert(this->fetchTuple(i));
              array_tids[pos++] = i;
            }
          }
        } else if (comp == UNEQUAL) {
          for (TID i = 0; i < array_size; ++i) {
            if (array[i] != comp_array[i]) {
              // result_tids->push_back(i);
              // result_table->insert(this->fetchTuple(i));
              array_tids[pos++] = i;
            }
          }
        } else {
          COGADB_FATAL_ERROR("Unknown ComparisonValue!", "");
        }
        // shrink to actual result size
        result_tids->resize(pos);
        return result_tids;
      }

      template <class T>
      void selection_thread(T* array, size_t column_size,
                            unsigned int thread_id,
                            unsigned int number_of_threads, const T& value,
                            const ValueComparator comp, TID* result_tids,
                            size_t* result_size) {
        // std::cout << "Hi I'm thread" << thread_id << std::endl;
        if (!quiet)
          std::cout << "Using CPU for Selection (parallel mode)..."
                    << std::endl;
        // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>&
        // >(*col);
        // number of elements per thread
        size_t chunk_size = column_size / number_of_threads;
        TID begin_index = chunk_size * thread_id;
        TID end_index;
        if (thread_id + 1 == number_of_threads) {
          // process until end of input array
          end_index = column_size;
        } else {
          end_index = (chunk_size) * (thread_id + 1);
        }
        // cout << "Thread " << thread_id << " begin index: " <<  begin_index <<
        // " end index: " << end_index << endl;
        TID pos = begin_index;

        if (comp == EQUAL) {
          //                        for(TID i=begin_index;i<end_index;++i){
          //                            if(column[i]==value){
          //                                    result_tids[pos++]=i;
          //                            }
          //                        }

          TID i;
          for (i = begin_index; i + 7 < end_index; i += 8) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, ==);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                     result_tids, pos, ==);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                     result_tids, pos, ==);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                     result_tids, pos, ==);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                     result_tids, pos, ==);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                     result_tids, pos, ==);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                     result_tids, pos, ==);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                     result_tids, pos, ==);
          }

          for (; i < end_index; ++i) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, ==);
          }

        } else if (comp == LESSER) {
          //                        for(TID i=begin_index;i<end_index;++i){
          //                            if(column[i]<value){
          //                                    result_tids[pos+begin_index]=i;
          //                            }
          //                        }
          TID i;
          for (i = begin_index; i + 7 < end_index; i += 8) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, <);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                     result_tids, pos, <);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                     result_tids, pos, <);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                     result_tids, pos, <);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                     result_tids, pos, <);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                     result_tids, pos, <);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                     result_tids, pos, <);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                     result_tids, pos, <);
          }

          for (; i < end_index; ++i) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, <);
          }
        } else if (comp == LESSER_EQUAL) {
          //                        for(TID i=begin_index;i<end_index;++i){
          //                            if(column[i]<=value){
          //                                    result_tids[pos+begin_index]=i;
          //                            }
          //                        }
          TID i;
          for (i = begin_index; i + 7 < end_index; i += 8) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, <=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                     result_tids, pos, <=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                     result_tids, pos, <=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                     result_tids, pos, <=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                     result_tids, pos, <=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                     result_tids, pos, <=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                     result_tids, pos, <=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                     result_tids, pos, <=);
          }

          for (; i < end_index; ++i) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, <=);
          }
        } else if (comp == GREATER) {
          //                        for(TID i=begin_index;i<end_index;++i){
          //                            if(column[i]>value){
          //                                    result_tids[pos+begin_index]=i;
          //                            }
          //                        }
          TID i;
          for (i = begin_index; i + 7 < end_index; i += 8) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, >);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                     result_tids, pos, >);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                     result_tids, pos, >);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                     result_tids, pos, >);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                     result_tids, pos, >);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                     result_tids, pos, >);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                     result_tids, pos, >);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                     result_tids, pos, >);
          }

          for (; i < end_index; ++i) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, >);
          }
        } else if (comp == GREATER_EQUAL) {
          //                        for(TID i=begin_index;i<end_index;++i){
          //                            if(column[i]>=value){
          //                                    result_tids[pos+begin_index]=i;
          //                            }
          //                        }
          TID i;
          for (i = begin_index; i + 7 < end_index; i += 8) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, >=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                     result_tids, pos, >=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                     result_tids, pos, >=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                     result_tids, pos, >=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                     result_tids, pos, >=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                     result_tids, pos, >=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                     result_tids, pos, >=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                     result_tids, pos, >=);
          }

          for (; i < end_index; ++i) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, >=);
          }
        } else if (comp == UNEQUAL) {
          TID i;
          for (i = begin_index; i + 7 < end_index; i += 8) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, !=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                     result_tids, pos, !=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                     result_tids, pos, !=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                     result_tids, pos, !=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                     result_tids, pos, !=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                     result_tids, pos, !=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                     result_tids, pos, !=);
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                     result_tids, pos, !=);
          }

          for (; i < end_index; ++i) {
            COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                     result_tids, pos, !=);
          }
        } else {
          COGADB_FATAL_ERROR("Unsupported Filter Predicate!", "");
        }
        //}
        // std::cout << "ID: " << thread_id << " SIZE: " << pos-begin_index <<
        // std::endl;
        // write result size to array
        *result_size = pos - begin_index;
      }

      inline void write_selection_result_thread(
          unsigned int thread_id, unsigned int number_of_threads,
          unsigned int column_size, TID* result_tids_array,
          PositionListPtr result_tids, unsigned int result_size,
          unsigned int result_begin_index, unsigned int result_end_index) {
        assert(result_tids->size() == column_size);

        // number of elements per thread
        size_t chunk_size = column_size / number_of_threads;
        TID begin_index = chunk_size * thread_id;
        TID* position_list_output_array = result_tids->data();

        std::memcpy(&position_list_output_array[result_begin_index],
                    &result_tids_array[begin_index], result_size * sizeof(TID));
      }

      inline void resize_PositionListPtr_thread(PositionListPtr tids,
                                                unsigned int new_size) {
        assert(tids != NULL);
        tids->resize(new_size);
      }

      template <class T>
      const PositionListPtr parallel_selection(
          T* array, size_t array_size, const boost::any& value_for_comparison,
          const ValueComparator comp, unsigned int number_of_threads) {
        PositionListPtr result_tids(createPositionList());
        // unsigned int number_of_threads=4;

        T value;
        if (value_for_comparison.type() != typeid(T)) {
          // catch some special cases, cast integer comparison value to float
          // value
          if (typeid(T) == typeid(float) &&
              value_for_comparison.type() == typeid(int)) {
            value = boost::any_cast<int>(value_for_comparison);
          } else {
            std::stringstream str_stream;
            str_stream << "Fatal Error!!! Typemismatch during Selection: "
                       << "Column Type: " << typeid(T).name()
                       << " filter value type: "
                       << value_for_comparison.type().name();
            COGADB_FATAL_ERROR(str_stream.str(), "");
          }
        } else {
          // everything fine, filter value matches type of column
          value = boost::any_cast<T>(value_for_comparison);
        }

        std::vector<TID> result_tids_array(array_size);

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
          threads.add_thread(new boost::thread(
              boost::bind(&CoGaDB::CDK::selection::selection_thread<T>, array,
                          array_size, i, number_of_threads, value, comp,
                          result_tids_array.data(), &result_sizes[i])));
          // selection_thread(i, number_of_threads, value,  comp,
          // this,result_tids_array, &result_sizes[i]);
        }
        threads.join_all();

        std::vector<size_t> prefix_sum(number_of_threads + 1);
        prefix_sum[0] = 0;
        for (unsigned int i = 1; i < number_of_threads + 1; i++) {
          prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
        }

        // copy result chunks in vector
        for (unsigned int i = 0; i < number_of_threads; i++) {
          threads.add_thread(new boost::thread(
              boost::bind(&write_selection_result_thread, i, number_of_threads,
                          array_size, result_tids_array.data(), result_tids,
                          result_sizes[i], prefix_sum[i], prefix_sum[i + 1])));
        }
        threads.join_all();
        // fit positionlist to actual result length
        result_tids->resize(prefix_sum[number_of_threads]);

        return result_tids;
      }

      /* EXPERIMENTAL: Predicate Evaluation with BITMAPS!*/

      template <typename T>
      void bitmap_scan_column_thread(T* __restrict__ array,
                                     const TID& begin_index,
                                     const TID& end_index,
                                     const boost::any& value_for_comparison,
                                     const ValueComparator comp,
                                     char* __restrict__ result_bitmap) {
        T value;
        if (value_for_comparison.type() != typeid(T)) {
          // catch some special cases, cast integer comparison value to float
          // value
          if (typeid(T) == typeid(float) &&
              value_for_comparison.type() == typeid(int)) {
            value = boost::any_cast<int>(value_for_comparison);
          } else {
            std::stringstream str_stream;
            str_stream << "Typemismatch during Selection: "
                       << "Column Type: " << typeid(T).name()
                       << " filter value type: "
                       << value_for_comparison.type().name();
            COGADB_FATAL_ERROR(str_stream.str(), "");
          }
        } else {
          // everything fine, filter value matches type of column
          value = boost::any_cast<T>(value_for_comparison);
        }

        assert(begin_index % 8 == 0);
        if (comp == EQUAL) {
          for (TID i = begin_index; i < end_index; ++i) {
            unsigned int current_bit = i & 7;  // i%8;
            unsigned int comp_val = (array[i] == value);
            unsigned int bitmask = comp_val << current_bit;
            result_bitmap[i / 8] |= bitmask;
          }
        } else if (comp == LESSER) {
          for (TID i = begin_index; i < end_index; ++i) {
            unsigned int current_bit = i & 7;  // i%8;
            unsigned int comp_val = (array[i] < value);
            unsigned int bitmask = comp_val << current_bit;
            result_bitmap[i / 8] |= bitmask;
          }
        } else if (comp == LESSER_EQUAL) {
          for (TID i = begin_index; i < end_index; ++i) {
            unsigned int current_bit = i & 7;  // i%8;
            unsigned int comp_val = (array[i] <= value);
            unsigned int bitmask = comp_val << current_bit;
            result_bitmap[i / 8] |= bitmask;
          }
        } else if (comp == GREATER) {
          for (TID i = begin_index; i < end_index; ++i) {
            unsigned int current_bit = i & 7;  // i%8;
            unsigned int comp_val = (array[i] > value);
            unsigned int bitmask = comp_val << current_bit;
            result_bitmap[i / 8] |= bitmask;
          }
        } else if (comp == GREATER_EQUAL) {
          for (TID i = begin_index; i < end_index; ++i) {
            unsigned int current_bit = i & 7;  // i%8;
            unsigned int comp_val = (array[i] >= value);
            unsigned int bitmask = comp_val << current_bit;
            result_bitmap[i / 8] |= bitmask;
          }
        } else if (comp == UNEQUAL) {
          for (TID i = begin_index; i < end_index; ++i) {
            unsigned int current_bit = i & 7;  // i%8;
            unsigned int comp_val = (array[i] != value);
            unsigned int bitmask = comp_val << current_bit;
            result_bitmap[i / 8] |= bitmask;
          }
        } else {
          COGADB_FATAL_ERROR("Unknown ComparisonValue!", "");
        }
      }

      template <typename T>
      BitmapPtr bitmap_selection(T* __restrict__ array, size_t array_size,
                                 const boost::any& value,
                                 const ValueComparator comp) {
        assert(array != NULL);
        // set non init flag, we will overwrite the values anyway
        BitmapPtr result(new Bitmap(array_size, false, false));
        // do the work
        bitmap_scan_column_thread(array, 0, array_size, value, comp,
                                  result->data());
        return result;
      }

      template <typename T>
      BitmapPtr bitmap_selection(T* __restrict__ array, size_t array_size,
                                 const T& value, const ValueComparator comp) {
        assert(array != NULL);
        // set non init flag, we will overwrite the values anyway
        BitmapPtr result(new Bitmap(array_size, false, false));
        // do the work
        bitmap_scan_column_thread(array, 0, array_size, boost::any(value), comp,
                                  result->data());
        return result;
      }

      template <typename T>
      void scan_column_equal_bitmap_thread(T* __restrict__ array,
                                           const TID& begin_index,
                                           const TID& end_index, const T& value,
                                           char* __restrict__ result_bitmap) {
        //                TID pos=begin_index;
        //                const size_t chunk_size=end_index-begin_index;
        //                for(int i=0;i<chunk_size;++i){
        //                    {COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                    value, i, pos, result_bitmap);}
        //                }

        assert(begin_index % 8 == 0);
        // TID pos=0; //begin_index;
        // const size_t chunk_size=end_index-begin_index;
        for (TID i = begin_index; i < end_index; ++i) {
          //{COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i, pos,
          // result_bitmap);}
          unsigned int current_bit = i & 7;  // i%8;
          unsigned int comp_val = (array[i] == value);
          unsigned int bitmask = comp_val << current_bit;
          result_bitmap[i / 8] |= bitmask;
        }

        //                        TID pos=begin_index;
        //                        const size_t chunk_size=end_index-begin_index;
        //                        int i=0;
        //                        for(;i<chunk_size;i+=8){
        //                            COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                            value, i, pos, result_bitmap);
        //                            COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                            value, i+1, pos, result_bitmap);
        //                            COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                            value, i+2, pos, result_bitmap);
        //                            COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                            value, i+3, pos, result_bitmap);
        //                            COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                            value, i+4, pos, result_bitmap);
        //                            COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                            value, i+5, pos, result_bitmap);
        //                            COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                            value, i+6, pos, result_bitmap);
        //                            COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                            value, i+7, pos, result_bitmap);
        //                        }
        //
        //                        for(;i<chunk_size;++i){
        //                            COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array,
        //                            value, i, pos, result_bitmap);
        //                        }
        // result_size=pos;
      }

      template <typename T>
      void scan_column_equal_bitmap_parallel(T* __restrict__ array,
                                             const size_t& array_size,
                                             const T& value,
                                             char* __restrict__ result_bitmap,
                                             size_t& result_size,
                                             unsigned int number_of_threads) {
        std::vector<size_t> result_sizes(number_of_threads);
        boost::thread_group threads;

        for (unsigned int thread_id = 0; thread_id < number_of_threads;
             ++thread_id) {
          // number of elements per thread
          size_t chunk_size = array_size / number_of_threads;
          TID begin_index = chunk_size * thread_id;
          TID end_index;
          if (thread_id + 1 == number_of_threads) {
            // process until end of input array
            end_index = array_size;
          } else {
            end_index = (chunk_size) * (thread_id + 1);
          }

          // scan_column_equal_bitmap(array, begin_index, end_index, value,
          // result_bitmap, result_size);
          // create a selection thread
          threads.add_thread(new boost::thread(boost::bind(
              &scan_column_equal_bitmap_thread<T>, array, begin_index,
              end_index, value, result_bitmap, result_size)));
        }
        threads.join_all();

        std::vector<size_t> prefix_sum(number_of_threads + 1);
        prefix_sum[0] = 0;
        for (unsigned int i = 1; i < number_of_threads + 1; i++) {
          prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
        }
        result_size = prefix_sum[number_of_threads];
      }

      //        /* Primitives needed for Invisible Join*/
      //        template <typename T>
      //        void filter_column_equal_by_array_bitmap_thread(T* __restrict__
      //        input_array, const TID& begin_index, const TID& end_index, T*
      //        __restrict__  comp_values, size_t num_comp_values, char*
      //        __restrict__ result_bitmap){
      //
      //        //                TID pos=begin_index;
      //        //                const size_t chunk_size=end_index-begin_index;
      //        //                for(int i=0;i<chunk_size;++i){
      //        //
      //        {COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i, pos,
      //        result_bitmap);}
      //        //                }
      //
      //
      //            for(TID i=begin_index;i<end_index;++i){
      //                for(TID j=0;j<num_comp_values;++j){
      //                    //int bitmask=255; //mask everything except the
      //                    first byte, cheap variant of modulo operation!
      //                    bool val = (input_array[i]==comp_values[j]);
      //                    //if(val){
      //                        int current_bit = i % 8;
      //                        val = val << current_bit;
      //                        result_bitmap[i/8]|=val;
      //                    //}
      //                }
      //            }
      //        }

      void bitwise_and_thread(char* left_input_bitmap, char* right_input_bitmap,
                              TID begin_index, TID end_index,
                              char* result_bitmap);
      void bitwise_or_thread(char* left_input_bitmap, char* right_input_bitmap,
                             TID begin_index, TID end_index,
                             char* result_bitmap);

      //        //template <typename T>
      //        inline void bitwise_and_thread(char* left_input_bitmap, char*
      //        right_input_bitmap, TID begin_index, TID end_index, char*
      //        result_bitmap){
      //            assert(begin_index%8==0);
      //            //process (#Bits+7/8) Byte [Rest Bits of the Byte as well]
      //            int new_end_index= (end_index+7)/8;
      //            for(TID i=begin_index;i<new_end_index;i++){
      //                result_bitmap[i] = left_input_bitmap[i] &
      //                right_input_bitmap[i];
      //            }
      ////            for(;i<end_index;++i){
      ////                for(unsigned j=0;j<(end_index%8);++j){
      ////                    result_bitmap[i] = left_input_bitmap[i] &
      /// right_input_bitmap[i];
      ////                }
      ////            }
      //        }

      BitmapPtr bitwise_and(BitmapPtr left_input_bitmap,
                            BitmapPtr right_input_bitmap);
      BitmapPtr bitwise_or(BitmapPtr left_input_bitmap,
                           BitmapPtr right_input_bitmap);

      inline void convertPositionListToBitmap(
          PositionListPtr tids, char* bitmap,
          size_t num_rows_of_indexed_table) {
        assert(tids != NULL);
        for (unsigned int i = 0; i < tids->size(); ++i) {
          TID val = (*tids)[i];
          assert(val < num_rows_of_indexed_table);
          unsigned int current_bit = val % 8;
          unsigned int bitmask = 1 << current_bit;
          bitmap[val / 8] |= bitmask;
        }
      }

      // inline void PositionListToBitmap_create_flag_array_thread(TID* tids,
      // size_t size, size_t sizeof_bitmap, char* bytewise_bitmap){
      inline void PositionListToBitmap_create_flag_array_thread(
          TID* tids, TID begin_index, TID end_index, size_t sizeof_bitmap,
          char* bytewise_bitmap) {
        std::ignore = sizeof_bitmap;
        for (TID i = begin_index; i < end_index; ++i) {
          TID val = tids[i];
          // assert(val<sizeof_bitmap);
          bytewise_bitmap[val] = 1;
        }
      }

      inline void PositionListToBitmap_pack_flag_array_thread(
          char* bytewise_bitmap, size_t number_of_elements,
          char* result_bitmap) {
        //            size_t loop_limit = number_of_elements/8;
        //            loop_limit *= 8;
        // we can run over array bounds in this function, so we allocate 8 byte
        // more than neccessary!
        for (unsigned int i = 0; i < number_of_elements; i += 8) {
          char packed_bits = 0;
          packed_bits |= (bytewise_bitmap[i] != 0);
          packed_bits |= (bytewise_bitmap[i + 1] != 0) << 1;
          packed_bits |= (bytewise_bitmap[i + 2] != 0) << 2;
          packed_bits |= (bytewise_bitmap[i + 3] != 0) << 3;
          packed_bits |= (bytewise_bitmap[i + 4] != 0) << 4;
          packed_bits |= (bytewise_bitmap[i + 5] != 0) << 5;
          packed_bits |= (bytewise_bitmap[i + 6] != 0) << 6;
          packed_bits |= (bytewise_bitmap[i + 7] != 0) << 7;
          result_bitmap[i / 8] = packed_bits;
        }
      }

      inline void parallel_convertPositionListToBitmap(
          PositionListPtr tids, char* bitmap,
          size_t num_rows_of_indexed_table) {
        size_t size_of_bytewise_bitmap =
            num_rows_of_indexed_table +
            8;  // reserve 8 bytes more than neccessary to simplify algorithm
        char* bytewise_bitmap =
            static_cast<char*>(calloc(size_of_bytewise_bitmap, sizeof(char)));
        size_t number_of_threads = 4;
        TID* tids_raw = tids->data();
        boost::thread_group threads;
        uint64_t begin = getTimestamp();
        for (unsigned int thread_id = 0; thread_id < number_of_threads;
             ++thread_id) {
          // number of elements per thread
          size_t chunk_size = tids->size() / number_of_threads;
          TID begin_index = chunk_size * thread_id;
          TID end_index;
          if (thread_id + 1 == number_of_threads) {
            // process until end of input array
            end_index = tids->size();
          } else {
            end_index = (chunk_size) * (thread_id + 1);
          }
          // TID num_elements = end_index - begin_index;
          // gather_thread(array, tid_array, begin_index, end_index,
          // result_array);
          // create a gather thread
          // std::cout << "thread " << thread_id << "   begin index:" <<
          // begin_index << "   end index:" << end_index;
          threads.add_thread(new boost::thread(boost::bind(
              &PositionListToBitmap_create_flag_array_thread, tids_raw,
              begin_index, end_index, num_rows_of_indexed_table,
              bytewise_bitmap)));  // array, tid_array, begin_index, end_index,
                                   // result_array)));
        }
        threads.join_all();
        uint64_t end = getTimestamp();
        std::cout << "First Phase: " << double(end - begin) / (1000 * 1000)
                  << "ms" << std::endl;

        // PositionListToBitmap_create_flag_array_thread(tids->data(),
        // tids->size(),num_rows_of_indexed_table, bytewise_bitmap);
        begin = getTimestamp();
        PositionListToBitmap_pack_flag_array_thread(
            bytewise_bitmap, num_rows_of_indexed_table, bitmap);
        end = getTimestamp();
        std::cout << "Second Phase: " << double(end - begin) / (1000 * 1000)
                  << "ms" << std::endl;
        //            assert(tids!=NULL);
        //            for(unsigned int i=0;i<tids->size();++i){
        //                TID val = (*tids)[i];
        //                assert(val<num_rows_of_indexed_table);
        //                unsigned int current_bit = val%8;
        //                unsigned int bitmask = 1 << current_bit;
        //                bitmap[val/8]|=bitmask;
        //            }
      }

      inline PositionListPtr createPositionListfromBitmap(
          char* __restrict__ input_bitmap, size_t total_number_of_bits) {
        // TID* result_tids = (TID*) malloc(sizeof(TID)*total_number_of_bits);
        PositionListPtr tids = createPositionList();
        tids->resize(total_number_of_bits);
        TID* result_tids = tids->data();
        TID pos = 0;
        TID i = 0;
        for (; i < total_number_of_bits / 8; ++i) {
          for (TID j = 0; j < 8; ++j) {
            result_tids[pos] = i * 8 + j;
            int bitmask = 1 << j;
            pos += (bitmask & input_bitmap[i]) >> j;
          }
        }
        // process remaining byte
        for (; i < (total_number_of_bits + 7) / 8; ++i) {
          for (TID j = 0; j < total_number_of_bits % 8; ++j) {
            result_tids[pos] = i * 8 + j;
            int bitmask = 1 << j;
            pos += (bitmask & input_bitmap[i]) >> j;
          }
        }
        tids->resize(pos);
        return tids;
      }

      inline unsigned int countSetBitsInBitmap(char* __restrict__ input_bitmap,
                                               size_t total_number_of_bits) {
        unsigned int number_of_set_bits = 0;
        for (unsigned int i = 0; i < total_number_of_bits; i += 8) {
          int val = input_bitmap[i / 8];
          for (unsigned int j = 0; j < 8 && i + j < total_number_of_bits; j++) {
            int bitmask = 1 << j;
            int result = (bitmask & val) >> j;
            if (result) number_of_set_bits++;
          }
        }
        //            for(unsigned int i=0;i<(total_number_of_bits+7)/8;++i){
        //                for(TID j=0;j<total_number_of_bits%8;++j){
        //                    //result_tids[pos]=i*8+j;
        //                    int bitmask = 1 << j;
        //                    //number_of_set_bits+=(bitmask & input_bitmap[i])
        //                    >> j;
        //                    if(bitmask & input_bitmap[i])
        //                        number_of_set_bits++;
        //                }
        //            }
        return number_of_set_bits;
      }

      inline void print_bitmap(char* bitmap, size_t bitmap_size) {
        std::cout << "Bitmask:" << std::endl;
        for (unsigned int i = 0; i < bitmap_size; i += 8) {
          int val = bitmap[i / 8];
          for (unsigned int j = 0; j < 8 && i + j < bitmap_size; j++) {
            int bitmask = 1 << j;
            int result = (bitmask & val) >> j;
            std::cout << result << std::endl;
          }
        }
        std::cout << "END Bitmask" << std::endl;
      }

      namespace variants {

        template <class T>
        void unrolled_selection_thread(T* array, size_t column_size,
                                       unsigned int thread_id,
                                       unsigned int number_of_threads,
                                       const T& value,
                                       const ValueComparator comp,
                                       TID* result_tids, size_t* result_size) {
          // std::cout << "Hi I'm thread" << thread_id << std::endl;
          if (!quiet)
            std::cout << "Using CPU for Selection (parallel mode)..."
                      << std::endl;
          // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>&
          // >(*col);
          // number of elements per thread
          size_t chunk_size = column_size / number_of_threads;
          TID begin_index = chunk_size * thread_id;
          TID end_index;
          if (thread_id + 1 == number_of_threads) {
            // process until end of input array
            end_index = column_size;
          } else {
            end_index = (chunk_size) * (thread_id + 1);
          }
          // cout << "Thread " << thread_id << " begin index: " <<  begin_index
          // << " end index: " << end_index << endl;
          TID pos = begin_index;

          if (comp == EQUAL) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     result_tids, pos, ==);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, ==);
            }

          } else if (comp == LESSER) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     result_tids, pos, <);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     result_tids, pos, <=);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     result_tids, pos, >);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     result_tids, pos, >=);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, >=);
            }
          } else {
            COGADB_FATAL_ERROR("Unsupported Filter Predicate!", "");
          }

          //}
          // std::cout << "ID: " << thread_id << " SIZE: " << pos-begin_index <<
          // std::endl;
          // write result size to array
          *result_size = pos - begin_index;
        }

        template <class T>
        void bf_unrolled_selection_thread(
            T* array, size_t column_size, unsigned int thread_id,
            unsigned int number_of_threads, const T& value,
            const ValueComparator comp, TID* result_tids, size_t* result_size) {
          // std::cout << "Hi I'm thread" << thread_id << std::endl;
          if (!quiet)
            std::cout << "Using CPU for Selection (parallel mode)..."
                      << std::endl;
          // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>&
          // >(*col);
          // number of elements per thread
          size_t chunk_size = column_size / number_of_threads;
          TID begin_index = chunk_size * thread_id;
          TID end_index;
          if (thread_id + 1 == number_of_threads) {
            // process until end of input array
            end_index = column_size;
          } else {
            end_index = (chunk_size) * (thread_id + 1);
          }
          // cout << "Thread " << thread_id << " begin index: " <<  begin_index
          // << " end index: " << end_index << endl;
          TID pos = begin_index;

          if (comp == EQUAL) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       result_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       result_tids, pos, ==);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, ==);
            }

          } else if (comp == LESSER) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       result_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       result_tids, pos, <);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       result_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       result_tids, pos, <=);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       result_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       result_tids, pos, >);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = begin_index; i + 7 < end_index; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       result_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       result_tids, pos, >=);
            }
            for (; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, >=);
            }
          } else {
            COGADB_FATAL_ERROR("Unsupported Filter Predicate!", "");
          }

          //}
          // std::cout << "ID: " << thread_id << " SIZE: " << pos-begin_index <<
          // std::endl;
          // write result size to array
          *result_size = pos - begin_index;
        }

        template <class T>
        void selection_thread(T* array, size_t column_size,
                              unsigned int thread_id,
                              unsigned int number_of_threads, const T& value,
                              const ValueComparator comp, TID* result_tids,
                              size_t* result_size) {
          // std::cout << "Hi I'm thread" << thread_id << std::endl;
          if (!quiet)
            std::cout << "Using CPU for Selection (parallel mode)..."
                      << std::endl;
          // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>&
          // >(*col);
          // number of elements per thread
          size_t chunk_size = column_size / number_of_threads;
          TID begin_index = chunk_size * thread_id;
          TID end_index;
          if (thread_id + 1 == number_of_threads) {
            // process until end of input array
            end_index = column_size;
          } else {
            end_index = (chunk_size) * (thread_id + 1);
          }
          // cout << "Thread " << thread_id << " begin index: " <<  begin_index
          // << " end index: " << end_index << endl;
          TID pos = begin_index;

          if (comp == EQUAL) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, ==);
            }

          } else if (comp == LESSER) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     result_tids, pos, >=);
            }
          } else {
            COGADB_FATAL_ERROR("Unsupported Filter Predicate!", "");
          }
          //}
          // std::cout << "ID: " << thread_id << " SIZE: " << pos-begin_index <<
          // std::endl;
          // write result size to array
          *result_size = pos - begin_index;
        }

        template <class T>
        void bf_selection_thread(T* array, size_t column_size,
                                 unsigned int thread_id,
                                 unsigned int number_of_threads, const T& value,
                                 const ValueComparator comp, TID* result_tids,
                                 size_t* result_size) {
          // std::cout << "Hi I'm thread" << thread_id << std::endl;
          if (!quiet)
            std::cout << "Using CPU for Selection (parallel mode)..."
                      << std::endl;
          // ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>&
          // >(*col);
          // number of elements per thread
          size_t chunk_size = column_size / number_of_threads;
          TID begin_index = chunk_size * thread_id;
          TID end_index;
          if (thread_id + 1 == number_of_threads) {
            // process until end of input array
            end_index = column_size;
          } else {
            end_index = (chunk_size) * (thread_id + 1);
          }
          // cout << "Thread " << thread_id << " begin index: " <<  begin_index
          // << " end index: " << end_index << endl;
          TID pos = begin_index;

          if (comp == EQUAL) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, ==);
            }

          } else if (comp == LESSER) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = begin_index; i < end_index; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       result_tids, pos, >=);
            }
          } else {
            COGADB_FATAL_ERROR("Unsupported Filter Predicate!", "");
          }
          //}
          // std::cout << "ID: " << thread_id << " SIZE: " << pos-begin_index <<
          // std::endl;
          // write result size to array
          *result_size = pos - begin_index;
        }

        // serial simple selection
        template <class T>
        const PositionListPtr serial_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
            }
          } else if (comp == UNEQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, !=);
            }
          } else {
            COGADB_FATAL_ERROR("Unknown ComparisonValue!", "");
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial simple selection
        template <class T>
        const PositionListPtr serial_bf_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial simple selection
        template <class T>
        char* serial_bitmap_selection(T* __restrict__ array, size_t array_size,
                                      const boost::any& value_for_comparison,
                                      const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          int res_size = ((array_size - 1) / 8 + 2);
          char* bmp = (char*)malloc(res_size);
          memset(bmp, 0, res_size);
          bmp[res_size - 1] = '\0';
          assert(bmp != NULL);

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_BRANCH(array, value, i, bmp, pos,
                                                     ==);
            }
          } else if (comp == LESSER) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_BRANCH(array, value, i, bmp, pos,
                                                     <);
            }
          } else if (comp == LESSER_EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_BRANCH(array, value, i, bmp, pos,
                                                     <=);
            }
          } else if (comp == GREATER) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_BRANCH(array, value, i, bmp, pos,
                                                     >);
            }
          } else if (comp == GREATER_EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_BRANCH(array, value, i, bmp, pos,
                                                     >=);
            }
          } else {
            COGADB_FATAL_ERROR("Unknown ComparisonValue!", "");
          }

          return bmp;
        }

        // serial simple selection
        template <class T>
        char* serial_bf_bitmap_selection(T* __restrict__ array,
                                         size_t array_size,
                                         const boost::any& value_for_comparison,
                                         const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          int res_size = ((array_size - 1) / 8 + 2);
          char* bmp = (char*)malloc(res_size);
          memset(bmp, 0, res_size);
          bmp[res_size - 1] = '\0';
          assert(bmp != NULL);

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_NOBRANCH(array, value, i, bmp,
                                                       pos, ==);
            }
          } else if (comp == LESSER) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_NOBRANCH(array, value, i, bmp,
                                                       pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_NOBRANCH(array, value, i, bmp,
                                                       pos, <=);
            }
          } else if (comp == GREATER) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_NOBRANCH(array, value, i, bmp,
                                                       pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            for (TID i = 0; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_BMP_NOBRANCH(array, value, i, bmp,
                                                       pos, >=);
            }
          } else {
          }
          return bmp;
        }

        // parallel selection
        template <class T>
        const PositionListPtr parallel_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads) {
          PositionListPtr result_tids = createPositionList();

          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases, cast integer comparison value to float
            // value
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          TID* result_tids_array = (TID*)malloc(array_size * sizeof(TID));

          std::vector<size_t> result_sizes(number_of_threads);
          boost::thread_group threads;
          // create a PositionListPtr of the maximal result size, so
          // that we can write the result tids in parallel to th vector
          // without the default latency
          threads.add_thread(new boost::thread(boost::bind(
              &CoGaDB::CDK::selection::resize_PositionListPtr_thread,
              result_tids, array_size)));
          for (unsigned int i = 0; i < number_of_threads; i++) {
            // create a selection thread
            threads.add_thread(new boost::thread(boost::bind(
                &CoGaDB::CDK::selection::variants::selection_thread<T>, array,
                array_size, i, number_of_threads, value, comp,
                result_tids_array, &result_sizes[i])));
          }
          threads.join_all();

          std::vector<size_t> prefix_sum(number_of_threads + 1);
          prefix_sum[0] = 0;
          for (unsigned int i = 1; i < number_of_threads + 1; i++) {
            prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
          }

          // copy result chunks in vector
          for (unsigned int i = 0; i < number_of_threads; i++) {
            threads.add_thread(new boost::thread(boost::bind(
                &write_selection_result_thread, i, number_of_threads,
                array_size, result_tids_array, result_tids, result_sizes[i],
                prefix_sum[i], prefix_sum[i + 1])));
          }
          threads.join_all();
          // fit positionlist to actual result length
          result_tids->resize(prefix_sum[number_of_threads]);

          free(result_tids_array);

          return result_tids;
        }

        // parallel selection
        template <class T>
        const PositionListPtr parallel_bf_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads) {
          PositionListPtr result_tids = createPositionList();

          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases, cast integer comparison value to float
            // value
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          TID* result_tids_array = (TID*)malloc(array_size * sizeof(TID));

          std::vector<size_t> result_sizes(number_of_threads);
          boost::thread_group threads;
          // create a PositionListPtr of the maximal result size, so
          // that we can write the result tids in parallel to th vector
          // without the default latency
          threads.add_thread(new boost::thread(boost::bind(
              &CoGaDB::CDK::selection::resize_PositionListPtr_thread,
              result_tids, array_size)));
          for (unsigned int i = 0; i < number_of_threads; i++) {
            // create a selection thread
            threads.add_thread(new boost::thread(boost::bind(
                &CoGaDB::CDK::selection::variants::selection_thread<T>, array,
                array_size, i, number_of_threads, value, comp,
                result_tids_array, &result_sizes[i])));
          }
          threads.join_all();

          std::vector<size_t> prefix_sum(number_of_threads + 1);
          prefix_sum[0] = 0;
          for (unsigned int i = 1; i < number_of_threads + 1; i++) {
            prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
          }

          // copy result chunks in vector
          for (unsigned int i = 0; i < number_of_threads; i++) {
            threads.add_thread(new boost::thread(boost::bind(
                &write_selection_result_thread, i, number_of_threads,
                array_size, result_tids_array, result_tids, result_sizes[i],
                prefix_sum[i], prefix_sum[i + 1])));
          }
          threads.join_all();
          // fit positionlist to actual result length
          result_tids->resize(prefix_sum[number_of_threads]);

          free(result_tids_array);

          return result_tids;
        }

#ifdef ENABLE_SIMD_ACCELERATION
        // serial SIMD selection
        template <class T>
        const PositionListPtr serial_SIMD_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);

        template <>
        const PositionListPtr serial_SIMD_selection(
            int* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);

        template <>
        const PositionListPtr serial_SIMD_selection(
            float* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);

        // serial SIMD BF selection
        template <class T>
        const PositionListPtr serial_bf_SIMD_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);

        template <>
        const PositionListPtr serial_bf_SIMD_selection(
            int* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);

        template <>
        const PositionListPtr serial_bf_SIMD_selection(
            float* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);
#endif

        // serial Unrolled2 selection
        template <class T>
        const PositionListPtr serial_unrolled2_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial Unrolled selection
        template <class T>
        const PositionListPtr serial_unrolled3_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial Unrolled selection
        template <class T>
        const PositionListPtr serial_unrolled4_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial Unrolled selection
        template <class T>
        const PositionListPtr serial_unrolled5_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial Unrolled selection
        template <class T>
        const PositionListPtr serial_unrolled6_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial Unrolled selection
        template <class T>
        const PositionListPtr serial_unrolled7_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial Unrolled selection
        template <class T>
        const PositionListPtr serial_unrolled_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 1,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 2,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 3,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 4,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 5,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 6,
                                                     array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i + 7,
                                                     array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i,
                                                     array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        //************** Begin Unrolled-N BF Scan

        // serial branch-free Unrolled2 selection
        template <class T>
        const PositionListPtr serial_bf_unrolled2_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 1 < array_size; i += 2) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial Branch-free Unrolled3 selection
        template <class T>
        const PositionListPtr serial_bf_unrolled3_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 2 < array_size; i += 3) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial branch-free Unrolled4 selection
        template <class T>
        const PositionListPtr serial_bf_unrolled4_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 3 < array_size; i += 4) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial Branch-free Unrolled5 selection
        template <class T>
        const PositionListPtr serial_bf_unrolled5_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 4 < array_size; i += 5) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial branch-free Unrolled selection
        template <class T>
        const PositionListPtr serial_bf_unrolled6_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 5 < array_size; i += 6) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial branch-free Unrolled selection
        template <class T>
        const PositionListPtr serial_bf_unrolled7_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 6 < array_size; i += 7) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

        // serial branch-free Unrolled8 selection
        template <class T>
        const PositionListPtr serial_bf_unrolled_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison,
            const ValueComparator comp) {
          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          PositionListPtr result_tids = createPositionList();
          // calls realloc internally
          result_tids->resize(array_size);
          // get pointer
          TID* array_tids =
              result_tids->data();  // hype::util::begin_ptr(*result_tids);
          assert(array_tids != NULL);
          size_t pos = 0;

          if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

          if (comp == EQUAL) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, ==);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       array_tids, pos, ==);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, ==);
            }
          } else if (comp == LESSER) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, <);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       array_tids, pos, <);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <);
            }
          } else if (comp == LESSER_EQUAL) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, <=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       array_tids, pos, <=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, <=);
            }
          } else if (comp == GREATER) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, >);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       array_tids, pos, >);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >);
            }
          } else if (comp == GREATER_EQUAL) {
            TID i;
            for (i = 0; i + 7 < array_size; i += 8) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 1,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 2,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 3,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 4,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 5,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 6,
                                                       array_tids, pos, >=);
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i + 7,
                                                       array_tids, pos, >=);
            }
            for (; i < array_size; ++i) {
              COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i,
                                                       array_tids, pos, >=);
            }
          } else {
          }
          // shrink to actual result size
          result_tids->resize(pos);
          return result_tids;
        }

//************** End Unrolled-N BF Scan

#ifdef ENABLE_SIMD_ACCELERATION

        template <class T>
        void SIMD_selection_thread(T* array, size_t column_size,
                                   unsigned int thread_id,
                                   unsigned int number_of_threads,
                                   const T value, const ValueComparator comp,
                                   TID* result_tids, size_t* result_size);

        template <>
        void SIMD_selection_thread(int* array, size_t column_size,
                                   unsigned int thread_id,
                                   unsigned int number_of_threads,
                                   const int value_for_comparison,
                                   const ValueComparator comp, TID* result_tids,
                                   size_t* result_size);

        template <>
        void SIMD_selection_thread(float* array, size_t column_size,
                                   unsigned int thread_id,
                                   unsigned int number_of_threads,
                                   const float value_for_comparison,
                                   const ValueComparator comp, TID* result_tids,
                                   size_t* result_size);

        template <class T>
        void bf_SIMD_selection_thread(T* array, size_t column_size,
                                      unsigned int thread_id,
                                      unsigned int number_of_threads,
                                      const T value, const ValueComparator comp,
                                      TID* result_tids, size_t* result_size);

        template <>
        void bf_SIMD_selection_thread(int* array, size_t column_size,
                                      unsigned int thread_id,
                                      unsigned int number_of_threads,
                                      const int value_for_comparison,
                                      const ValueComparator comp,
                                      TID* result_tids, size_t* result_size);

        template <>
        void bf_SIMD_selection_thread(float* array, size_t column_size,
                                      unsigned int thread_id,
                                      unsigned int number_of_threads,
                                      const float value_for_comparison,
                                      const ValueComparator comp,
                                      TID* result_tids, size_t* result_size);

        template <class T>
        void unrolled_SIMD_selection_thread(
            T* array, size_t column_size, unsigned int thread_id,
            unsigned int number_of_threads, const T value,
            const ValueComparator comp, TID* result_tids, size_t* result_size);

        template <>
        void unrolled_SIMD_selection_thread(
            int* array, size_t column_size, unsigned int thread_id,
            unsigned int number_of_threads, const int value_for_comparison,
            const ValueComparator comp, TID* result_tids, size_t* result_size);

        template <>
        void unrolled_SIMD_selection_thread(
            float* array, size_t column_size, unsigned int thread_id,
            unsigned int number_of_threads, const float value_for_comparison,
            const ValueComparator comp, TID* result_tids, size_t* result_size);

        template <class T>
        void bf_unrolled_SIMD_selection_thread(
            T* array, size_t column_size, unsigned int thread_id,
            unsigned int number_of_threads, const T value,
            const ValueComparator comp, TID* result_tids, size_t* result_size);

        template <>
        void bf_unrolled_SIMD_selection_thread(
            int* array, size_t column_size, unsigned int thread_id,
            unsigned int number_of_threads, const int value_for_comparison,
            const ValueComparator comp, TID* result_tids, size_t* result_size);

        template <>
        void bf_unrolled_SIMD_selection_thread(
            float* array, size_t column_size, unsigned int thread_id,
            unsigned int number_of_threads, const float value_for_comparison,
            const ValueComparator comp, TID* result_tids, size_t* result_size);

        // parallel SIMD selection
        template <class T>
        const PositionListPtr parallel_SIMD_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads) {
          PositionListPtr result_tids = createPositionList();
          // unsigned int number_of_threads=4;

          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases, cast integer comparison value to float
            // value
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          TID* result_tids_array = (TID*)malloc(array_size * sizeof(TID));

          std::vector<size_t> result_sizes(number_of_threads);
          boost::thread_group threads;
          // create a PositionListPtr of the maximal result size, so
          // that we can write the result tids in parallel to th vector
          // without the default latency
          threads.add_thread(new boost::thread(boost::bind(
              &CoGaDB::CDK::selection::resize_PositionListPtr_thread,
              result_tids, array_size)));
          for (unsigned int i = 0; i < number_of_threads; i++) {
            // create a selection thread
            threads.add_thread(new boost::thread(boost::bind(
                &CoGaDB::CDK::selection::variants::SIMD_selection_thread<T>,
                array, array_size, i, number_of_threads, value, comp,
                result_tids_array, &result_sizes[i])));
          }
          threads.join_all();

          std::vector<size_t> prefix_sum(number_of_threads + 1);
          prefix_sum[0] = 0;
          for (unsigned int i = 1; i < number_of_threads + 1; i++) {
            prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
          }

          // copy result chunks in vector
          size_t chunk_size = array_size / number_of_threads;
          for (unsigned int i = 0; i < number_of_threads; i++) {
            TID begin_index = chunk_size * i;
            threads.add_thread(new boost::thread(boost::bind(
                &write_selection_result_thread, i, number_of_threads,
                array_size, result_tids_array, result_tids, result_sizes[i],
                prefix_sum[i], prefix_sum[i + 1])));
          }
          threads.join_all();
          // fit positionlist to actual result length
          result_tids->resize(prefix_sum[number_of_threads]);

          free(result_tids_array);

          return result_tids;
        }

        // parallel branch-free SIMD selection
        template <class T>
        const PositionListPtr parallel_bf_SIMD_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads) {
          PositionListPtr result_tids = createPositionList();
          // unsigned int number_of_threads=4;

          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases, cast integer comparison value to float
            // value
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          TID* result_tids_array = (TID*)malloc(array_size * sizeof(TID));

          std::vector<size_t> result_sizes(number_of_threads);
          boost::thread_group threads;
          // create a PositionListPtr of the maximal result size, so
          // that we can write the result tids in parallel to th vector
          // without the default latency
          threads.add_thread(new boost::thread(boost::bind(
              &CoGaDB::CDK::selection::resize_PositionListPtr_thread,
              result_tids, array_size)));
          for (unsigned int i = 0; i < number_of_threads; i++) {
            // create a selection thread
            threads.add_thread(new boost::thread(boost::bind(
                &CoGaDB::CDK::selection::variants::bf_SIMD_selection_thread<T>,
                array, array_size, i, number_of_threads, value, comp,
                result_tids_array, &result_sizes[i])));
          }
          threads.join_all();

          std::vector<size_t> prefix_sum(number_of_threads + 1);
          prefix_sum[0] = 0;
          for (unsigned int i = 1; i < number_of_threads + 1; i++) {
            prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
          }

          // copy result chunks in vector
          size_t chunk_size = array_size / number_of_threads;
          for (unsigned int i = 0; i < number_of_threads; i++) {
            TID begin_index = chunk_size * i;
            threads.add_thread(new boost::thread(boost::bind(
                &write_selection_result_thread, i, number_of_threads,
                array_size, result_tids_array, result_tids, result_sizes[i],
                prefix_sum[i], prefix_sum[i + 1])));
          }
          threads.join_all();
          // fit positionlist to actual result length
          result_tids->resize(prefix_sum[number_of_threads]);

          free(result_tids_array);

          return result_tids;
        }

#endif

        // parallel Unrolled selection
        template <class T>
        const PositionListPtr parallel_unrolled_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads) {
          PositionListPtr result_tids = createPositionList();
          // unsigned int number_of_threads=8;

          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases, cast integer comparison value to float
            // value
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          TID* result_tids_array = (TID*)malloc(array_size * sizeof(TID));

          std::vector<size_t> result_sizes(number_of_threads);
          boost::thread_group threads;
          // create a PositionListPtr of the maximal result size, so
          // that we can write the result tids in parallel to th vector
          // without the default latency
          threads.add_thread(new boost::thread(boost::bind(
              &CoGaDB::CDK::selection::resize_PositionListPtr_thread,
              result_tids, array_size)));
          for (unsigned int i = 0; i < number_of_threads; i++) {
            // create a selection thread
            threads.add_thread(new boost::thread(boost::bind(
                &CoGaDB::CDK::selection::variants::unrolled_selection_thread<T>,
                array, array_size, i, number_of_threads, value, comp,
                result_tids_array, &result_sizes[i])));
          }
          threads.join_all();

          std::vector<size_t> prefix_sum(number_of_threads + 1);
          prefix_sum[0] = 0;
          for (unsigned int i = 1; i < number_of_threads + 1; i++) {
            prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
          }

          // copy result chunks in vector
          for (unsigned int i = 0; i < number_of_threads; i++) {
            threads.add_thread(new boost::thread(boost::bind(
                &write_selection_result_thread, i, number_of_threads,
                array_size, result_tids_array, result_tids, result_sizes[i],
                prefix_sum[i], prefix_sum[i + 1])));
          }
          threads.join_all();
          // fit positionlist to actual result length
          result_tids->resize(prefix_sum[number_of_threads]);

          free(result_tids_array);

          return result_tids;
        }

        // parallel Unrolled selection
        template <class T>
        const PositionListPtr parallel_bf_unrolled_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads) {
          PositionListPtr result_tids = createPositionList();
          // unsigned int number_of_threads=8;

          T value;
          if (value_for_comparison.type() != typeid(T)) {
            // catch some special cases, cast integer comparison value to float
            // value
            if (typeid(T) == typeid(float) &&
                value_for_comparison.type() == typeid(int)) {
              value = boost::any_cast<int>(value_for_comparison);
            } else {
              std::stringstream str_stream;
              str_stream << "Fatal Error!!! Typemismatch during Selection: "
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: "
                         << value_for_comparison.type().name();
              COGADB_FATAL_ERROR(str_stream.str(), "");
            }
          } else {
            // everything fine, filter value matches type of column
            value = boost::any_cast<T>(value_for_comparison);
          }

          TID* result_tids_array = (TID*)malloc(array_size * sizeof(TID));

          std::vector<size_t> result_sizes(number_of_threads);
          boost::thread_group threads;
          // create a PositionListPtr of the maximal result size, so
          // that we can write the result tids in parallel to th vector
          // without the default latency
          threads.add_thread(new boost::thread(boost::bind(
              &CoGaDB::CDK::selection::resize_PositionListPtr_thread,
              result_tids, array_size)));
          for (unsigned int i = 0; i < number_of_threads; i++) {
            // create a selection thread
            threads.add_thread(new boost::thread(boost::bind(
                &CoGaDB::CDK::selection::variants::bf_unrolled_selection_thread<
                    T>,
                array, array_size, i, number_of_threads, value, comp,
                result_tids_array, &result_sizes[i])));
          }
          threads.join_all();

          std::vector<size_t> prefix_sum(number_of_threads + 1);
          prefix_sum[0] = 0;
          for (unsigned int i = 1; i < number_of_threads + 1; i++) {
            prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
          }

          // copy result chunks in vector
          for (unsigned int i = 0; i < number_of_threads; i++) {
            threads.add_thread(new boost::thread(boost::bind(
                &write_selection_result_thread, i, number_of_threads,
                array_size, result_tids_array, result_tids, result_sizes[i],
                prefix_sum[i], prefix_sum[i + 1])));
          }
          threads.join_all();
          // fit positionlist to actual result length
          result_tids->resize(prefix_sum[number_of_threads]);

          free(result_tids_array);

          return result_tids;
        }

#ifdef ENABLE_SIMD_ACCELERATION

        // serial Unrolled SIMD selection
        template <class T>
        const PositionListPtr serial_unrolled_SIMD_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);
        template <>
        const PositionListPtr serial_unrolled_SIMD_selection(
            int* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);

        template <>
        const PositionListPtr serial_unrolled_SIMD_selection(
            float* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);

        template <class T>
        const PositionListPtr parallel_unrolled_SIMD_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads);

        template <>
        const PositionListPtr parallel_unrolled_SIMD_selection(
            int* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads);

        template <>
        const PositionListPtr parallel_unrolled_SIMD_selection(
            float* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads);

        // serial Unrolled SIMD selection
        template <class T>
        const PositionListPtr serial_bf_unrolled_SIMD_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);
        template <>
        const PositionListPtr serial_bf_unrolled_SIMD_selection(
            int* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);

        template <>
        const PositionListPtr serial_bf_unrolled_SIMD_selection(
            float* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp);

        template <class T>
        const PositionListPtr parallel_bf_unrolled_SIMD_selection(
            T* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads);

        template <>
        const PositionListPtr parallel_bf_unrolled_SIMD_selection(
            int* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads);

        template <>
        const PositionListPtr parallel_bf_unrolled_SIMD_selection(
            float* __restrict__ array, size_t array_size,
            const boost::any& value_for_comparison, const ValueComparator comp,
            unsigned int number_of_threads);

#endif

      }  // end namespace variants

      /* end Primitives needed for Invisible Join*/
    }  // end namespace selection

    inline PositionListPtr convertBitmapToPositionList(BitmapPtr bitmap) {
      return CDK::selection::createPositionListfromBitmap(bitmap->data(),
                                                          bitmap->size());
    }

    inline BitmapPtr convertPositionListToBitmap(PositionListPtr tids,
                                                 size_t number_of_rows) {
      // tbb::sort(tids->begin(),tids->end());
      char* matching_rows_fact_table_bitmap =
          (char*)calloc((number_of_rows + 7) / 8, sizeof(char));
      // uint64_t begin = getTimestamp();
      CDK::selection::convertPositionListToBitmap(
          tids, matching_rows_fact_table_bitmap, number_of_rows);
      // uint64_t end = getTimestamp();
      BitmapPtr bitmap(
          new Bitmap(matching_rows_fact_table_bitmap, number_of_rows));
      // std::cout << "Serial PoslistToBitmap: " <<
      // double(end-begin)/(1000*1000) << "ms for " << tids->size() << "rows"
      // << std::endl;

      //            char* matching_rows_fact_table_bitmap2 = (char*)
      //            calloc((number_of_rows+7)/8,sizeof(char));
      //            begin = getTimestamp();
      //            CDK::selection::parallel_convertPositionListToBitmap(tids,
      //            matching_rows_fact_table_bitmap2, number_of_rows);
      //            end = getTimestamp();
      //            std::cout << "Parallel PoslistToBitmap: " <<
      //            double(end-begin)/(1000*1000) << "ms for " << tids->size()
      //            << "rows" << std::endl;
      //            if(memcmp(matching_rows_fact_table_bitmap,
      //            matching_rows_fact_table_bitmap2, (number_of_rows+7)/8)!=0){
      //                COGADB_FATAL_ERROR("Incorrect Result!","");
      //            }
      //            free(matching_rows_fact_table_bitmap2);

      return bitmap;
    }

    namespace join {

      void pk_fk_semi_join_thread(TablePtr filtered_fact_tab,
                                  InvisibleJoinSelection inv_join_sel,
                                  LookupTablePtr* result,
                                  unsigned int thread_id);
      TablePtr invisibleJoin(TablePtr fact_table,
                             InvisibleJoinSelectionList dimensions);

      template <typename T>
      const PositionListPairPtr nested_loop_join(
          T* __restrict__ column1, const size_t& col1_array_size,
          T* __restrict__ column2, const size_t& col2_array_size) {
        assert(column1 != NULL);
        assert(column2 != NULL);

        PositionListPairPtr join_tids(new PositionListPair());
        join_tids->first = createPositionList();
        join_tids->second = createPositionList();

        unsigned int join_column1_size = col1_array_size;
        unsigned int join_column2_size = col2_array_size;

        for (unsigned int i = 0; i < join_column1_size; i++) {
          for (unsigned int j = 0; j < join_column2_size; j++) {
            if (column1[i] == column2[j]) {
              if (debug)
                std::cout << "MATCH: (" << i << "," << j << ")" << std::endl;
              join_tids->first->push_back(i);
              join_tids->second->push_back(j);
            }
          }
        }
        return join_tids;
      }

      //	template<typename T>
      //	const PositionListPairPtr block_nested_loop_join(T* __restrict__
      // column1, const size_t& col1_array_size,T* __restrict__ column2, const
      // size_t& col2_array_size){
      //                assert(column1!=NULL);
      //                assert(column2!=NULL);
      //
      //                PositionListPairPtr join_tids( new PositionListPair());
      //                join_tids->first = PositionListPtr=createPositionList();
      //                join_tids->second =
      //                PositionListPtr=createPositionList();
      //
      //
      //                unsigned int join_column1_size=col1_array_size;
      //                unsigned int join_column2_size=col2_array_size;
      //
      //                unsigned int i=0;
      //                unsigned int j=0;
      //
      //
      //                //getconf LEVEL1_DCACHE_LINESIZE
      //                //sudo dmidecode --type 7
      //                unsigned int level3_cache_size=6144*1024; //on dell
      //                laptop
      //
      //                unsigned int
      //                block_size=(level3_cache_size/2)/sizeof(int);
      //                std::cout << "Block size in KB: " <<
      //                (block_size*sizeof(int))/1024 << std::endl;;
      //
      ////                for(;i<join_column1_size;i++){
      ////                        for(;j<join_column2_size;j++){
      //
      ////                        }
      //                for(;i+block_size<join_column1_size;i+=block_size){
      //                    for(;j<join_column2_size;j++){
      //                         for(unsigned o=0;o<block_size;++o){
      //
      //                                    if(column1[i+o]==column2[j]){
      //                                            //if(debug)
      //                                                std::cout << "MATCH: ("
      //                                                << i+o << "," << j <<
      //                                                ")" << std::endl;
      //                                            join_tids->first->push_back(i+o);
      //                                            join_tids->second->push_back(j);
      //                                    }
      //                            }
      //                        }
      //                }
      //
      //                for(;i<join_column1_size;i++){
      //                        for(;j<join_column2_size;j++){
      //                                std::cout << "COMPARE: (" << i << "," <<
      //                                j << ")" << std::endl;
      //                                if(column1[i]==column2[j]){
      //                                        //if(debug)
      //                                            std::cout << "MATCH: (" << i
      //                                            << "," << j << ")" <<
      //                                            std::endl;
      //                                        join_tids->first->push_back(i);
      //                                        join_tids->second->push_back(j);
      //                                }
      //                        }
      //                }
      //		return join_tids;
      //	}

      template <class T>
      const PositionListPairPtr serial_hash_join(
          T* __restrict__ column1, const size_t& col1_array_size,
          T* __restrict__ column2, const size_t& col2_array_size) {
        typedef boost::unordered_multimap<T, TID, boost::hash<T>,
                                          std::equal_to<T> >
            HashTable;

        PositionListPairPtr join_tids(new PositionListPair());
        join_tids->first = createPositionList();
        join_tids->second = createPositionList();

        Timestamp build_hashtable_begin = getTimestamp();
        // create hash table
        HashTable hashtable;
        unsigned int hash_table_size = col1_array_size;
        unsigned int join_column_size = col2_array_size;

        assert(col2_array_size >= col1_array_size);
        //        unsigned int* join_tids_table1 =  new unsigned
        //        int[join_column_size];
        //        unsigned int* join_tids_table2 =  new unsigned
        //        int[join_column_size];

        //          unsigned int* join_tids_table1 = (unsigned int*)
        //          malloc(join_column_size*sizeof(T));
        //          unsigned int* join_tids_table2 = (unsigned int*)
        //          malloc(join_column_size*sizeof(T));

        //        unsigned int pos1 = 0;
        //        unsigned int pos2 = 0;
        unsigned int pos = 0;

        for (unsigned int i = 0; i < hash_table_size; i++)
          hashtable.insert(std::pair<T, TID>(column1[i], i));
        Timestamp build_hashtable_end = getTimestamp();
        //        std::cout << "Number of Buckets: " << hashtable.bucket_count()
        //        << std::endl;
        //        for(unsigned int i=0;i< hashtable.bucket_count();i++){
        //            std::cout << "Size of Bucket '" << i << "': " <<
        //            hashtable.bucket_size(i) << std::endl;
        //        }

        // probe larger relation
        Timestamp prune_hashtable_begin = getTimestamp();

        std::pair<typename HashTable::iterator, typename HashTable::iterator>
            range;
        typename HashTable::iterator it;
        for (unsigned int i = 0; i < join_column_size; i++) {
          range = hashtable.equal_range(column2[i]);
          for (it = range.first; it != range.second; ++it) {
            if (it->first == column2[i]) {  //(*join_column)[i]){
              //                                  join_tids_table1[pos]=it->second;
              //                                  join_tids_table2[pos]=i;
              //                                  pos++;
              // pos2=++pos1;
              //                                join_tids_table1[pos1++]=it->second;
              //                                join_tids_table2[pos2++]=i;
              join_tids->first->push_back(it->second);
              join_tids->second->push_back(i);

              // cout << "match! " << it->second << ", " << i << "	"  <<
              // it->first << endl;
            }
          }
        }

        //        //copy result in PositionList (vector)
        //        join_tids->first->insert(join_tids->first->end(),join_tids_table1,join_tids_table1+pos);
        //        join_tids->second->insert(join_tids->second->end(),join_tids_table2,join_tids_table2+pos);
        //
        ////        delete join_tids_table1;
        ////        delete join_tids_table2;
        //        free(join_tids_table1);
        //        free(join_tids_table2);
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

        return join_tids;
      }
      ///*
      template <>
      inline const PositionListPairPtr serial_hash_join<int>(
          int* __restrict__ column1, const size_t& col1_array_size,
          int* __restrict__ column2, const size_t& col2_array_size) {
        return CDK::main_memory_joins::serial_hash_join(
            column1, col1_array_size, column2, col2_array_size);
      }
      //*/
      const PositionListPairPtr radix_join(int* __restrict__ column1,
                                           const size_t& col1_array_size,
                                           int* __restrict__ column2,
                                           const size_t& col2_array_size);
      const PositionListPairPtr radix_join(int* __restrict__ column1,
                                           const size_t& col1_array_size,
                                           int* __restrict__ column2,
                                           const size_t& col2_array_size,
                                           int total_number_of_bits,
                                           int number_of_passes);

      //        template<>
      //	inline const PositionListPairPtr serial_hash_join<int>(int*
      //__restrict__ column1, const size_t& col1_array_size,int* __restrict__
      // column2, const size_t& col2_array_size){
      //            return  radix_join(column1, col1_array_size, column2,
      //            col2_array_size, 13, 2);
      //        }

      template <class T>
      int serial_pk_fk_bitmap_hash_join(T* __restrict__ pk_column,
                                        const size_t& pk_col_array_size,
                                        T* __restrict__ fk_column,
                                        const size_t& fk_col_array_size,
                                        char* fk_column_result_bitmap) {
        typedef boost::unordered_map<T, TID, boost::hash<T>, std::equal_to<T> >
            HashTable;

        Timestamp build_hashtable_begin = getTimestamp();
        HashTable hashtable;

        size_t hash_table_size = pk_col_array_size;
        size_t join_column_size = fk_col_array_size;

        for (size_t i = 0; i < hash_table_size; i++) {
          hashtable.insert(std::pair<T, TID>(pk_column[i], i));
        }
        Timestamp build_hashtable_end = getTimestamp();
        // probe larger relation
        Timestamp prune_hashtable_begin = getTimestamp();

        typename HashTable::const_iterator end_it = hashtable.end();
        typename HashTable::const_iterator pk_val_it;
        for (size_t i = 0; i < join_column_size; i++) {
          // lookup hash value
          pk_val_it = hashtable.find(fk_column[i]);
          // set bit corresponding to current fk in bitmap
          unsigned int current_bit = i & 7;  // i%8;
          unsigned int comp_val =
              (pk_val_it != end_it);  // found? //(array[i] == value);
          unsigned int bitmask = comp_val << current_bit;
          fk_column_result_bitmap[i / 8] |= bitmask;
        }

        Timestamp prune_hashtable_end = getTimestamp();

        if (!quiet && verbose) {
          std::cout << "Hash Join: Build Phase: "
                    << double(build_hashtable_end - build_hashtable_begin) /
                           (1000 * 1000)
                    << "ms"
                    << "Pruning Phase: "
                    << double(prune_hashtable_end - prune_hashtable_begin) /
                           (1000 * 1000)
                    << "ms" << std::endl;
        }

        return 0;
      }

    }  // end namespace join

    namespace aggregation {

      PositionListPtr groupby(
          const std::vector<DictionaryCompressedCol*>& dict_compressed_columns);
      PositionListPtr groupby(const std::vector<ColumnPtr>& columns);
      ColumnGroupingKeysPtr computeColumnGroupingKeys(
          const std::vector<ColumnPtr>& columns,
          const ProcessorSpecification& proc_spec);
      unsigned int getGreaterPowerOfTwo(unsigned int val);
      uint32_t createBitmask(unsigned int num_set_bits);
    }  // end namespace aggregation

  }  // end namespace CDK
}  // end namespace CogaDB

//#endif	/* PRIMITIVES_HPP */
