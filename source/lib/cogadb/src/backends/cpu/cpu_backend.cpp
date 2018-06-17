/*
 * File:   cpu_backend.cpp
 * Author: sebastian
 *
 * Created on 27. Dezember 2014, 02:03
 */

#include <backends/cpu/cpu_backend.hpp>

#include <hardware_optimizations/primitives.hpp>
#include <lookup_table/join_index.hpp>

#include <util/column_grouping_keys.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

//#define THRUST_DEVICE_SYSTEM_CUDA    1
//#define THRUST_DEVICE_SYSTEM_OMP     2
//#define THRUST_DEVICE_SYSTEM_TBB     3
//#define THRUST_DEVICE_SYSTEM_CPP     4
#define THRUST_DEVICE_SYSTEM 3
#include <thrust/sequence.h>
#include <thrust/sort.h>

#pragma GCC diagnostic pop

#include <util/reduce_by_keys.hpp>

#include <hardware_optimizations/main_memory_joins/hash_joins.hpp>
#include <util/functions.hpp>
#include <util/utility_functions.hpp>
//#include <hardware_optimizations/main_memory_joins/radix_hash_joins.hpp>
#include <backends/cpu/join.hpp>
#include <util/types.hpp>

namespace CoGaDB {

template <typename T>
CPU_Backend<T>::CPU_Backend() : ProcessorBackend<T>() {}

/***************** relational operations on Columns which return lookup tables
 * *****************/

/* SELECTION */
template <typename T>
const PositionListPtr CPU_Backend<T>::tid_selection(
    T* column, size_t num_elements, const SelectionParam& param) {
  PositionListPtr result;
  if (param.pred_type == ValueConstantPredicate) {
    result = CDK::selection::parallel_selection(
        column, num_elements, param.value, param.comp,
        boost::thread::hardware_concurrency());
    //            result = CDK::selection::serial_selection(column,
    //            num_elements, param.value, param.comp);
  } else if (param.pred_type == ValueValuePredicate) {
    DenseValueColumnPtr comp_column =
        boost::dynamic_pointer_cast<DenseValueColumn>(param.comparison_column);
    assert(comp_column != NULL);
    result = CDK::selection::serial_column_comparison_selection(
        column, comp_column->data(), num_elements, param.comp);
  } else {
    COGADB_FATAL_ERROR("Invalid Predicate Type!", "");
  }
  return result;
}

const PositionListPtr tid_selection_c_string(C_String* array,
                                             size_t num_elements,
                                             C_String comparison_value,
                                             const ValueComparator& comp) {
  PositionListPtr result_tids = createPositionList();
  // calls realloc internally
  result_tids->resize(num_elements);
  // get pointer
  TID* array_tids =
      result_tids->data();  // hype::util::begin_ptr(*result_tids);
  assert(array_tids != NULL);
  size_t pos = 0;

  if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

  if (comp == EQUAL) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comparison_value) == 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == LESSER) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comparison_value) < 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == LESSER_EQUAL) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comparison_value) <= 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == GREATER) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comparison_value) > 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == GREATER_EQUAL) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comparison_value) >= 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == UNEQUAL) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comparison_value) != 0) {
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

static const PositionListPtr tid_selection_c_string(
    C_String* array, size_t num_elements, C_String* comp_array,
    const ValueComparator& comp) {
  PositionListPtr result_tids = createPositionList();
  // calls realloc internally
  result_tids->resize(num_elements);
  // get pointer
  TID* array_tids =
      result_tids->data();  // hype::util::begin_ptr(*result_tids);
  assert(array_tids != NULL);
  size_t pos = 0;

  if (!quiet) std::cout << "Using CPU for Selection..." << std::endl;

  if (comp == EQUAL) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(comp_array[i], array[i]) == 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == LESSER) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comp_array[i]) < 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == LESSER_EQUAL) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comp_array[i]) <= 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == GREATER) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comp_array[i]) > 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == GREATER_EQUAL) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comp_array[i]) >= 0) {
        array_tids[pos++] = i;
      }
    }
  } else if (comp == UNEQUAL) {
    for (TID i = 0; i < num_elements; ++i) {
      if (strcmp(array[i], comp_array[i]) != 0) {
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

template <>
const PositionListPtr CPU_Backend<char*>::tid_selection(
    char** column, size_t num_elements, const SelectionParam& param) {
  PositionListPtr result;
  if (param.pred_type == ValueConstantPredicate) {
    C_String value;
    std::string val;
    if (param.value.type() != typeid(C_String)) {
      // catch some special cases, cast integer comparison value to float value
      if (param.value.type() == typeid(std::string)) {
        std::string val = boost::any_cast<std::string>(param.value);
        value = (char*)val.c_str();
      } else {
        std::stringstream str_stream;
        str_stream << "Fatal Error!!! Typemismatch during Selection: "
                   << "Column Type: " << typeid(C_String).name()
                   << " filter value type: " << param.value.type().name();
        COGADB_FATAL_ERROR(str_stream.str(), "");
      }
    } else {
      // everything fine, filter value matches type of column
      value = boost::any_cast<C_String>(param.value);
    }
    return tid_selection_c_string(column, num_elements, value, param.comp);
  } else if (param.pred_type == ValueValuePredicate) {
    DenseValueColumnPtr comp_column =
        boost::dynamic_pointer_cast<DenseValueColumn>(param.comparison_column);
    assert(comp_column != NULL);
    return tid_selection_c_string(column, num_elements, comp_column->data(),
                                  param.comp);
  } else {
    COGADB_FATAL_ERROR("Invalid Predicate Type!", "");
  }
  return PositionListPtr();
}

template <typename T>
const BitmapPtr CPU_Backend<T>::bitmap_selection(T* column, size_t num_elements,
                                                 const SelectionParam& param) {
  assert(param.pred_type == ValueConstantPredicate);
  BitmapPtr bitmap = CDK::selection::bitmap_selection(column, num_elements,
                                                      param.value, param.comp);
  return bitmap;
}

template <>
const BitmapPtr CPU_Backend<char*>::bitmap_selection(
    char** column, size_t num_elements, const SelectionParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  return BitmapPtr();
}

/* END SELECTION */

/* COLUMN ALGEBRA */
template <typename T>
bool CPU_Backend<T>::column_algebra_operation(
    T* __restrict__ target_column, T* __restrict__ source_column,
    size_t num_elements, const AlgebraOperationParam& param) {
  if (param.alg_op == ADD) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] += source_column[i];
    }
  } else if (param.alg_op == SUB) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] -= source_column[i];
    }
  } else if (param.alg_op == MUL) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] *= source_column[i];
    }
  } else if (param.alg_op == DIV) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] /= source_column[i];
    }
  } else {
    COGADB_FATAL_ERROR("Unknown Algebra Operation!", "");
  }

  return true;
}

template <>
bool CPU_Backend<std::string>::column_algebra_operation(
    std::string* target_column, std::string* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  COGADB_FATAL_ERROR("Called Algebra Operation On Type VARCHAR!", "");
  return false;
}

template <>
bool CPU_Backend<char*>::column_algebra_operation(
    char** target_column, char** source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  COGADB_FATAL_ERROR("Called Algebra Operation On Type VARCHAR!", "");
  return false;
}

template <typename T>
bool CPU_Backend<T>::column_algebra_operation(
    T* target_column, size_t num_elements, T value,
    const AlgebraOperationParam& param) {
  if (param.alg_op == ADD) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] += value;
    }
  } else if (param.alg_op == SUB) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] -= value;
    }
  } else if (param.alg_op == MUL) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] *= value;
    }
  } else if (param.alg_op == DIV) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] /= value;
    }
  } else {
    COGADB_FATAL_ERROR("Unknown Algebra Operation!", "");
  }

  return true;
}

template <>
bool CPU_Backend<std::string>::column_algebra_operation(
    std::string* target_column, size_t num_elements, std::string value,
    const AlgebraOperationParam&) {
  COGADB_FATAL_ERROR("Called Algebra Operation On Type VARCHAR!", "");
  return false;
}

template <>
bool CPU_Backend<char*>::column_algebra_operation(
    char** target_column, size_t num_elements, char* value,
    const AlgebraOperationParam&) {
  COGADB_FATAL_ERROR("Called Algebra Operation On Type VARCHAR!", "");
  return false;
}

template <typename T>
bool CPU_Backend<T>::double_precision_column_algebra_operation(
    double* target_column, T* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  if (param.alg_op == ADD) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] += source_column[i];
    }
  } else if (param.alg_op == SUB) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] -= source_column[i];
    }
  } else if (param.alg_op == MUL) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] *= source_column[i];
    }
  } else if (param.alg_op == DIV) {
#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      target_column[i] /= source_column[i];
    }
  } else {
    COGADB_FATAL_ERROR("Unknown Algebra Operation!", "");
  }

  return true;
}

template <>
bool CPU_Backend<std::string>::double_precision_column_algebra_operation(
    double* target_column, std::string* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  COGADB_FATAL_ERROR("Called Algebra Operation On Type VARCHAR!", "");
  return false;
}

template <>
bool CPU_Backend<char*>::double_precision_column_algebra_operation(
    double* target_column, char** source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  COGADB_FATAL_ERROR("Called Algebra Operation On Type VARCHAR!", "");
  return false;
}

/* END COLUMN ALGEBRA */

/* AGGREGATION */
template <typename T>
const ColumnGroupingKeysPtr CPU_Backend<T>::createColumnGroupingKeys(
    T* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  ColumnGroupingKeysPtr result(
      new ColumnGroupingKeys(hype::util::getMemoryID(proc_spec.proc_id)));
  result->keys->resize(num_elements);
  result->required_number_of_bits =
      this->getNumberOfRequiredBits(column, num_elements, proc_spec);
  // GroupingKeyType needs to be at least as large as T
  assert(sizeof(T) <= sizeof(GroupingKeys::value_type));

  GroupingKeys::value_type* key_array = result->keys->data();

  std::copy(column, column + num_elements, key_array);

  return result;
}

template <>
const ColumnGroupingKeysPtr CPU_Backend<std::string>::createColumnGroupingKeys(
    std::string* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  return ColumnGroupingKeysPtr();
}

template <>
const ColumnGroupingKeysPtr CPU_Backend<char*>::createColumnGroupingKeys(
    char** column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  return ColumnGroupingKeysPtr();
}

template <typename T>
size_t CPU_Backend<T>::getNumberOfRequiredBits(
    T* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  T max = *std::max_element(column, column + num_elements);
  T min = *std::min_element(column, column + num_elements);
  // check if we need all bits (in case negativ bit is set))
  if (min < 0) return sizeof(T) * 8;
  return getGreaterPowerOfTwo(max);
}

template <>
size_t CPU_Backend<std::string>::getNumberOfRequiredBits(
    std::string* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  return 65;
}

template <>
size_t CPU_Backend<char*>::getNumberOfRequiredBits(
    char** column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  return 65;
}

typedef GroupingKeys::value_type GroupingKeyType;

void ArrayLeftShift(ColumnGroupingKeysPtr col_group_keys, size_t num_rows,
                    size_t num_bits_to_shift) {
  if (!col_group_keys) return;
  GroupingKeyType* values = col_group_keys->keys->data();
  assert(values != NULL);
  if (num_bits_to_shift == 0) return;
// shift values
#pragma omp parallel for
  for (TID i = 0; i < num_rows; ++i) {
    values[i] = values[i] << num_bits_to_shift;
  }
}

void ArrayRightShift(ColumnGroupingKeysPtr col_group_keys, size_t num_rows,
                     size_t num_bits_to_shift) {
  if (!col_group_keys) return;
  GroupingKeyType* values = col_group_keys->keys->data();
  assert(values != NULL);
  if (num_bits_to_shift == 0) return;
// shift values
#pragma omp parallel for
  for (TID i = 0; i < num_rows; ++i) {
    values[i] = values[i] >> num_bits_to_shift;
  }
}

void ArrayBitwiseAnd(GroupingKeyType* left_input, GroupingKeyType* right_input,
                     size_t size, GroupingKeyType* result) {
#pragma omp parallel for
  for (TID i = 0; i < size; i++) {
    result[i] = left_input[i] & right_input[i];
  }
}

void ArrayBitwiseOr(GroupingKeyType* left_input, GroupingKeyType* right_input,
                    size_t size, GroupingKeyType* result) {
#pragma omp parallel for
  for (TID i = 0; i < size; i++) {
    result[i] = left_input[i] | right_input[i];
  }
}

template <typename T>
bool CPU_Backend<T>::bit_shift(ColumnGroupingKeysPtr keys,
                               const BitShiftParam& param) {
  if (!keys) return false;
  if (keys->keys->size() == 0) return false;

  if (param.op == SHIFT_BITS_LEFT) {
    ArrayLeftShift(keys, keys->keys->size(), param.number_of_bits);
  } else if (param.op == SHIFT_BITS_RIGHT) {
    ArrayRightShift(keys, keys->keys->size(), param.number_of_bits);
  } else {
    COGADB_FATAL_ERROR("Unknown Bitshift Operation!", "");
  }

  return true;
}

template <typename T>
bool CPU_Backend<T>::bitwise_combination(ColumnGroupingKeysPtr target_keys,
                                         ColumnGroupingKeysPtr source_keys,
                                         const BitwiseCombinationParam& param) {
  if (!target_keys || !source_keys) return false;
  if (target_keys->keys->size() == 0 || source_keys->keys->size() == 0)
    return false;

  if (param.op == BITWISE_AND) {
    ArrayBitwiseAnd(target_keys->keys->data(), source_keys->keys->data(),
                    target_keys->keys->size(), target_keys->keys->data());
  } else if (param.op == BITWISE_OR) {
    ArrayBitwiseOr(target_keys->keys->data(), source_keys->keys->data(),
                   target_keys->keys->size(), target_keys->keys->data());
  } else {
    COGADB_FATAL_ERROR("Unknown Bitwise Operation!", "");
  }

  return true;
}

template <typename T>
const AggregationResult CPU_Backend<T>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, T* aggregation_column,
    size_t num_elements, const AggregationParam& param) {
  AggregationResult result;
  if (param.agg_alg == SORT_BASED_AGGREGATION) {
    result = reduce_by_keys(grouping_keys, aggregation_column, num_elements,
                            param.agg_func);
  } else if (param.agg_alg == HASH_BASED_AGGREGATION) {
    result = hash_aggregation(grouping_keys, aggregation_column, num_elements,
                              param.agg_func, param.write_group_tid_array);
  } else {
    COGADB_FATAL_ERROR("Unsupported Aggregation Algorithm!", "");
    return AggregationResult();
  }
  return result;
}

template <>
const AggregationResult CPU_Backend<char*>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, char** aggregation_column,
    size_t num_elements, const AggregationParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  return AggregationResult();
}

template <typename T>
const AggregationResult CPU_Backend<T>::aggregate(
    T* aggregation_column, size_t num_elements, const AggregationParam& param) {
  T aggregate = aggregation_column[0];
  double sum = 0;
  AggregationResult result;

  if (param.agg_func == COUNT) {
    // create column with a single element, which is the number of elements
    result.second = boost::make_shared<Column<int> >(
        std::string(""), INT, 1, num_elements, hype::PD_Memory_0);
    return result;
  } else if (param.agg_func == MIN) {
    //            #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      aggregate = std::min(aggregate, aggregation_column[i]);
    }
    result.second = boost::make_shared<Column<T> >(
        std::string(""), getAttributeType(typeid(T)), 1, aggregate,
        hype::PD_Memory_0);
    return result;
  } else if (param.agg_func == MAX) {
    //            #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      aggregate = std::max(aggregate, aggregation_column[i]);
    }
    result.second = boost::make_shared<Column<T> >(
        std::string(""), getAttributeType(typeid(T)), 1, aggregate,
        hype::PD_Memory_0);
    return result;
  } else if (param.agg_func == SUM || param.agg_func == AVERAGE) {
    //            #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
      sum += aggregation_column[i];
    }
    if (param.agg_func == AVERAGE) sum = sum / num_elements;
    // create column with a single element, which is the number of elements
    result.second = boost::make_shared<Column<double> >(
        std::string(""), DOUBLE, 1, sum, hype::PD_Memory_0);
    return result;

  } else {
    COGADB_FATAL_ERROR(
        "Unknown or unsupported Algebra Operation: " << param.agg_func, "");
  }

  return AggregationResult();
}

template <>
const AggregationResult CPU_Backend<std::string>::aggregate(
    std::string* aggregation_column, size_t num_elements,
    const AggregationParam& param) {
  AggregationResult result;
  if (param.agg_func == COUNT) {
    // create column with a single element, which is the number of elements
    result.second = boost::make_shared<Column<int> >(
        std::string(""), INT, 1, num_elements, hype::PD_Memory_0);
    return result;
  } else {
    COGADB_FATAL_ERROR(
        "Unknown or unsupported Algebra Operation for string column: "
            << param.agg_func,
        "");
  }

  return AggregationResult();
}

template <>
const AggregationResult CPU_Backend<char*>::aggregate(
    char** aggregation_column, size_t num_elements,
    const AggregationParam& param) {
  AggregationResult result;
  if (param.agg_func == COUNT) {
    // create column with a single element, which is the number of elements
    result.second = boost::make_shared<Column<int> >(
        std::string(""), INT, 1, num_elements, hype::PD_Memory_0);
    return result;
  } else {
    COGADB_FATAL_ERROR(
        "Unknown or unsupported Algebra Operation for string column: "
            << param.agg_func,
        "");
  }

  return AggregationResult();
}

/* END AGGREGATION */

/* FETCH JOIN */
template <typename T>
const PositionListPtr CPU_Backend<T>::tid_fetch_join(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const FetchJoinParam&) {
  return fetchMatchingTIDsFromJoinIndex(join_index, pk_table_tids);
}

template <typename T>
const PositionListPairPtr CPU_Backend<T>::fetch_join(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const FetchJoinParam&) {
  return fetchJoinResultFromJoinIndex(join_index, pk_table_tids);
}

template <typename T>
const BitmapPtr CPU_Backend<T>::bitmap_fetch_join(JoinIndexPtr join_index,
                                                  PositionListPtr pk_table_tids,
                                                  const FetchJoinParam&) {
  return createBitmapOfMatchingTIDsFromJoinIndexParallel(join_index,
                                                         pk_table_tids);
}
/* END FETCH JOIN */

/* JOIN */
template <typename T>
const PositionListPairPtr CPU_Backend<T>::join(T* join_column1,
                                               size_t left_num_elements,
                                               T* join_column2,
                                               size_t right_num_elements,
                                               const JoinParam&) {
  // always build hash table on smaller input table
  if (left_num_elements <= right_num_elements) {
    return CDK::join::serial_hash_join(join_column1, left_num_elements,
                                       join_column2, right_num_elements);
  } else {
    PositionListPairPtr result = CDK::join::serial_hash_join(
        join_column2, right_num_elements, join_column1, left_num_elements);
    if (result) {
      std::swap(result->first, result->second);
    }
    return result;
  }
}

template <>
const PositionListPairPtr CPU_Backend<TID>::join(TID* join_column1,
                                                 size_t left_num_elements,
                                                 TID* join_column2,
                                                 size_t right_num_elements,
                                                 const JoinParam&) {
  // always build hash table on smaller input table
  if (left_num_elements <= right_num_elements) {
    //            if(VariableManager::instance().getVariableValueBoolean("use_radix_hash_join")){
    //                return
    //                CDK::main_memory_joins::parallel_radix_hash_join(join_column1,
    //                left_num_elements, join_column2, right_num_elements);
    //            }else{
    return CDK::main_memory_joins::parallel_hash_join(
        join_column1, left_num_elements, join_column2, right_num_elements);
    //            }
  } else {
    PositionListPairPtr result = CDK::main_memory_joins::parallel_hash_join(
        join_column2, right_num_elements, join_column1, left_num_elements);
    if (result) {
      std::swap(result->first, result->second);
    }
    return result;
  }
}

//    template <>
//    const PositionListPairPtr CPU_Backend<int32_t>::join(int32_t*
//    join_column1,
//                                           size_t left_num_elements,
//                                           int32_t* join_column2,
//                                           size_t right_num_elements,
//                                           const JoinParam&){
//        //always build hash table on smaller input table
//        if(left_num_elements<=right_num_elements){
//            return CDK::main_memory_joins::parallel_hash_join(join_column1,
//            left_num_elements, join_column2, right_num_elements);
//        }else{
//            PositionListPairPtr result =
//            CDK::main_memory_joins::parallel_hash_join(join_column2,
//            right_num_elements, join_column1, left_num_elements);
//            if(result){
//                std::swap(result->first, result->second);
//            }
//            return result;
//        }
//    }

template <typename T>
const PositionListPtr CPU_Backend<T>::tid_semi_join(T* join_column1,
                                                    size_t left_num_elements,
                                                    T* join_column2,
                                                    size_t right_num_elements,
                                                    const JoinParam& param) {
  typedef
      typename CPU_Semi_Join<T>::TIDSemiJoinFunctionPtr TIDSemiJoinFunctionPtr;
  TIDSemiJoinFunctionPtr func =
      CPU_Semi_Join<T>::getTIDSemiJoin(param.join_type);
  assert(func != NULL);
  return (*func)(join_column1, left_num_elements, join_column2,
                 right_num_elements, param);
}

template <typename T>
const BitmapPtr CPU_Backend<T>::bitmap_semi_join(T* join_column1,
                                                 size_t left_num_elements,
                                                 T* join_column2,
                                                 size_t right_num_elements,
                                                 const JoinParam&) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return BitmapPtr();
}

/* SET OPERATIONS */
template <typename T>
const BitmapPtr CPU_Backend<T>::computeBitmapSetOperation(
    BitmapPtr left_bitmap, BitmapPtr right_bitmap,
    const BitmapOperationParam& param) {
  if (param.bitmap_op == BITMAP_AND) {
    return CDK::selection::bitwise_and(left_bitmap, right_bitmap);
  } else if (param.bitmap_op == BITMAP_OR) {
    return CDK::selection::bitwise_or(left_bitmap, right_bitmap);
  } else {
    COGADB_FATAL_ERROR("Unknown Bitmap operation!", "");
  }

  return BitmapPtr();
}

template <typename T>
const ColumnPtr CPU_Backend<T>::computeSetOperation(T* left_column,
                                                    size_t left_num_elements,
                                                    T* right_column,
                                                    size_t right_num_elements,
                                                    const SetOperationParam&) {
  return ColumnPtr();
  //    PositionListPtr computePositionListUnion(PositionListPtr tids1,
  //    PositionListPtr tids2) {
  //        PositionListPtr tmp_tids(createPositionList(tids1->size() +
  //        tids2->size()));
  //        PositionList::iterator it;
  //
  //        it = std::set_union(tids1->begin(), tids1->end(),
  //                tids2->begin(),
  //                tids2->end(), tmp_tids->begin());
  //        //set size to actual result size (union eliminates duplicates)
  //        tmp_tids->resize(it - tmp_tids->begin());
  //        return tmp_tids;
  //    }
  //
  //    PositionListPtr computePositionListIntersection(PositionListPtr tids1,
  //    PositionListPtr tids2) {
  //        PositionListPtr tmp_tids(createPositionList(tids1->size() +
  //        tids2->size()));
  //        PositionList::iterator it;
  //
  //        it = std::set_intersection(tids1->begin(), tids1->end(),
  //                tids2->begin(),
  //                tids2->end(), tmp_tids->begin());
  //        //set size to actual result size (union eliminates duplicates)
  //        tmp_tids->resize(it - tmp_tids->begin());
  //        return tmp_tids;
  //    }

  //
  //        PositionListPtr tids;
  //        if (op_ == POSITIONLIST_INTERSECTION) {
  //            tids = computePositionListIntersection(input_tids_left,
  //            input_tids_right);
  //        } else if (op_ == POSITIONLIST_UNION) {
  //            tids = computePositionListUnion(input_tids_left,
  //            input_tids_right);
  //        }

  //        return tids;
}

template <typename T>
const PositionListPtr CPU_Backend<T>::computePositionListSetOperation(
    PositionListPtr tids1, PositionListPtr tids2,
    const SetOperationParam& param) {
  PositionListPtr tids(createPositionList(tids1->size() + tids2->size()));
  if (param.set_op == INTERSECT) {
    tids = computePositionListIntersection(tids1, tids2);
  } else if (param.set_op == UNION) {
    tids = computePositionListUnion(tids1, tids2);
  }
  return tids;
}
/* END SET OPERATIONS */

/* SORT */
template <typename T>
const PositionListPtr CPU_Backend<T>::sort(T* column, size_t num_elements,
                                           const SortParam& param) {
  PositionListPtr tids = createPositionList();
  tids->resize(num_elements);
  thrust::sequence(tids->begin(), tids->end(), 0);
  if (param.stable) {
    thrust::stable_sort_by_key(column, column + num_elements, tids->data());
  } else {
    thrust::sort_by_key(column, column + num_elements, tids->data());
  }
  // workaround, because sort_by_key does not work for descending sorting
  if (param.order == DESCENDING) {
    thrust::reverse(tids->begin(), tids->end());
  }

  return tids;
}

/* MISC */
template <typename T>
bool CPU_Backend<T>::gather(T* dest_column, T* source_column,
                            PositionListPtr tids, const GatherParam&) {
  if (tids->size() == 0) return true;
  if (tids->size() < 300) {
    CDK::util::serial_gather(source_column, tids->data(), tids->size(),
                             dest_column);
  } else {
    CDK::util::parallel_gather(source_column, tids->data(), tids->size(),
                               dest_column,
                               boost::thread::hardware_concurrency());
  }

  //        CDK::util::serial_gather(source_column, tids->data(), tids->size(),
  //        dest_column);

  return true;
}

template <typename T>
bool CPU_Backend<T>::generateConstantSequence(
    T* dest_column, size_t num_elements, T value,
    const ProcessorSpecification& proc_spec) {
  for (size_t i = 0; i < num_elements; ++i) {
    dest_column[i] = value;
  }
  return true;
}

template <typename T>
bool CPU_Backend<T>::generateAscendingSequence(
    T* dest_column, size_t num_elements, T begin_value,
    const ProcessorSpecification& proc_spec) {
  T current_value = begin_value;
  for (size_t i = 0; i < num_elements; ++i) {
    dest_column[i] = current_value++;
  }
  return true;
}

template <>
bool CPU_Backend<std::string>::generateConstantSequence(
    std::string* dest_column, size_t num_elements, std::string value,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR("Called unimplemented Function!", "");
  return false;
}

template <>
bool CPU_Backend<std::string>::generateAscendingSequence(
    std::string* dest_column, size_t num_elements, std::string begin_value,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR("Called unimplemented Function!", "");
  return false;
}

template <>
bool CPU_Backend<char*>::generateConstantSequence(
    char** dest_column, size_t num_elements, char* value,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR("Called unimplemented Function!", "");
  return false;
}

template <>
bool CPU_Backend<char*>::generateAscendingSequence(
    char** dest_column, size_t num_elements, char* begin_value,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR("Called unimplemented Function!", "");
  return false;
}

template <typename T>
const BitmapPtr CPU_Backend<T>::convertPositionListToBitmap(
    PositionListPtr tids, size_t num_rows_base_table,
    const ProcessorSpecification& proc_spec) {
  char* matching_rows_fact_table_bitmap =
      (char*)calloc((num_rows_base_table + 7) / 8, sizeof(char));
  CDK::selection::convertPositionListToBitmap(
      tids, matching_rows_fact_table_bitmap, num_rows_base_table);
  BitmapPtr bitmap(
      new Bitmap(matching_rows_fact_table_bitmap, num_rows_base_table));
  return bitmap;
}
template <typename T>
const PositionListPtr CPU_Backend<T>::convertBitmapToPositionList(
    BitmapPtr bitmap, const ProcessorSpecification& proc_spec) {
  return CDK::selection::createPositionListfromBitmap(bitmap->data(),
                                                      bitmap->size());
}

template <typename T>
const DoubleDenseValueColumnPtr CPU_Backend<T>::convertToDoubleDenseValueColumn(
    const std::string& column_name, T* array, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  if (typeid(T) == typeid(double)) {
    COGADB_FATAL_ERROR(
        "Meaningless to convert a DOUBLE column into a DOUBLE column!", "");
  }

  DoubleDenseValueColumnPtr double_column(
      new DoubleDenseValueColumn(column_name, DOUBLE, getMemoryID(proc_spec)));
  double_column->resize(num_elements);

  double* double_array = double_column->data();
#pragma omp parallel for
  for (size_t i = 0; i < num_elements; ++i) {
    double_array[i] = double(array[i]);
  }

  return double_column;
}

template <>
const DoubleDenseValueColumnPtr
CPU_Backend<std::string>::convertToDoubleDenseValueColumn(
    const std::string& column_name, std::string* array, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  COGADB_FATAL_ERROR(
      "Called convertToDoubleDenseValueColumn() for dense value VARCHAR "
      "column!",
      "");
  return DoubleDenseValueColumnPtr();
}

template <>
const DoubleDenseValueColumnPtr
CPU_Backend<char*>::convertToDoubleDenseValueColumn(
    const std::string& column_name, char** array, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  COGADB_FATAL_ERROR(
      "Called convertToDoubleDenseValueColumn() for dense value VARCHAR "
      "column!",
      "");
  return DoubleDenseValueColumnPtr();
}

/* END MISC */

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(CPU_Backend)

}  // end namespace CogaDB
