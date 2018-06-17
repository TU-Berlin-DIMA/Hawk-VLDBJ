/*
 * File:   cpu_backend.cpp
 * Author: sebastian
 *
 * Created on 27. Dezember 2014, 02:03
 */

#include <backends/gpu/bit_operations.hpp>
#include <backends/gpu/bitmap_set_operation.hpp>
#include <backends/gpu/column_algebra.hpp>
#include <backends/gpu/conversion_operations.hpp>
#include <backends/gpu/fetch_join.hpp>
#include <backends/gpu/gpu_backend.hpp>
#include <backends/gpu/join.hpp>
#include <backends/gpu/positionlist_set_operation.hpp>
#include <backends/gpu/selection.hpp>
#include <backends/gpu/util.hpp>
#include <new>
#include <util/utility_functions.hpp>
//#include <cxxabi.h>
#include "backends/gpu/aggregation.hpp"
#include "core/column.hpp"

namespace CoGaDB {

template <typename T>
GPU_Backend<T>::GPU_Backend() : ProcessorBackend<T>() {}

/***************** relational operations on Columns which return lookup tables
 * *****************/

/* SELECTION */
template <typename T>
const PositionListPtr GPU_Backend<T>::tid_selection(
    T* column, size_t num_elements, const SelectionParam& param) {
  PositionListPtr result;
  if (param.pred_type == ValueConstantPredicate) {
    T value;
    //            bool ret_success = getValueFromAny("", param.value, value,
    //            this->db_type_);
    //            if(!ret_success) PositionListPtr();

    assert(param.value.type() != typeid(T));
    value = boost::any_cast<T>(param.value);

    return GPU_Selection<T>::tid_selection(
        column, num_elements, value, param.comp,
        hype::util::getMemoryID(param.proc_spec.proc_id));

  } else if (param.pred_type == ValueValuePredicate) {
    DenseValueColumnPtr comp_column =
        boost::dynamic_pointer_cast<DenseValueColumn>(param.comparison_column);
    assert(comp_column != NULL);
    return GPU_Selection<T>::tid_selection(
        column, num_elements, comp_column->data(), param.comp,
        hype::util::getMemoryID(param.proc_spec.proc_id));
  } else {
    COGADB_FATAL_ERROR("Invalid Predicate Type!", "");
  }
  return result;
}

template <typename T>
const BitmapPtr GPU_Backend<T>::bitmap_selection(T* column, size_t num_elements,
                                                 const SelectionParam& param) {
  return BitmapPtr();
}
/* END SELECTION */

/* COLUMN ALGEBRA */
template <typename T>
bool GPU_Backend<T>::column_algebra_operation(
    T* target_column, T* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  return GPU_ColumnAlgebra<T>::column_algebra_operation(
      target_column, source_column, num_elements, param);
}

template <typename T>
bool GPU_Backend<T>::column_algebra_operation(
    T* column, size_t num_elements, T value,
    const AlgebraOperationParam& param) {
  return GPU_ColumnAlgebra<T>::column_algebra_operation(column, num_elements,
                                                        value, param);
}

template <typename T>
bool GPU_Backend<T>::double_precision_column_algebra_operation(
    double* target_column, T* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  return GPU_ColumnAlgebra<T>::double_precision_column_algebra_operation(
      target_column, source_column, num_elements, param);
}

/* END COLUMN ALGEBRA */

/* AGGREGATION */
template <typename T>
const ColumnGroupingKeysPtr GPU_Backend<T>::createColumnGroupingKeys(
    T* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  return GPU_Aggregation<T>::createColumnGroupingKeys(column, num_elements,
                                                      proc_spec);
}

template <typename T>
size_t GPU_Backend<T>::getNumberOfRequiredBits(
    T* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  return GPU_Aggregation<T>::getNumberOfRequiredBits(column, num_elements,
                                                     proc_spec);
}

template <typename T>
bool GPU_Backend<T>::bit_shift(ColumnGroupingKeysPtr keys,
                               const BitShiftParam& param) {
  return GPU_BitOperation::bit_shift(keys, param);
}

template <typename T>
bool GPU_Backend<T>::bitwise_combination(ColumnGroupingKeysPtr target_keys,
                                         ColumnGroupingKeysPtr source_keys,
                                         const BitwiseCombinationParam& param) {
  return GPU_BitOperation::bitwise_combination(target_keys, source_keys, param);
}

template <typename T>
const AggregationResult GPU_Backend<T>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, T* aggregation_column,
    size_t num_elements, const AggregationParam& param) {
  return GPU_Aggregation<T>::aggregateByGroupingKeys(
      grouping_keys, aggregation_column, num_elements, param);
}

template <typename T>
const AggregationResult GPU_Backend<T>::aggregate(
    T* aggregation_column, size_t num_elements, const AggregationParam& param) {
  return GPU_Aggregation<T>::aggregate(aggregation_column, num_elements, param);
}

/* END AGGREGATION */

/* FETCH JOIN */
template <typename T>
const PositionListPtr GPU_Backend<T>::tid_fetch_join(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const FetchJoinParam& param) {
  return GPU_FetchJoin::tid_fetch_join(join_index, pk_table_tids, param);
}

template <typename T>
const PositionListPairPtr GPU_Backend<T>::fetch_join(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const FetchJoinParam& param) {
  return GPU_FetchJoin::fetch_join(join_index, pk_table_tids, param);
}

template <typename T>
const BitmapPtr GPU_Backend<T>::bitmap_fetch_join(JoinIndexPtr join_index,
                                                  PositionListPtr pk_table_tids,
                                                  const FetchJoinParam& param) {
  return GPU_FetchJoin::bitmap_fetch_join(join_index, pk_table_tids, param);
}
/* END FETCH JOIN */

/* JOIN */
template <typename T>
const PositionListPairPtr GPU_Backend<T>::join(T* join_column1,
                                               size_t left_num_elements,
                                               T* join_column2,
                                               size_t right_num_elements,
                                               const JoinParam& param) {
  typedef typename GPU_Join<T>::JoinFunctionPtr JoinFunctionPtr;
  JoinFunctionPtr func = GPU_Join<T>::get(param.join_type);
  assert(func != NULL);
  return (*func)(join_column1, left_num_elements, join_column2,
                 right_num_elements, param);
  //        return GPU_Join<T>::inner_join(join_column1, left_num_elements,
  //                join_column2, right_num_elements,
  //                param);
}

template <typename T>
const PositionListPtr GPU_Backend<T>::tid_semi_join(T* join_column1,
                                                    size_t left_num_elements,
                                                    T* join_column2,
                                                    size_t right_num_elements,
                                                    const JoinParam& param) {
  typedef
      typename GPU_Semi_Join<T>::TIDSemiJoinFunctionPtr TIDSemiJoinFunctionPtr;
  TIDSemiJoinFunctionPtr func =
      GPU_Semi_Join<T>::getTIDSemiJoin(param.join_type);
  assert(func != NULL);
  return (*func)(join_column1, left_num_elements, join_column2,
                 right_num_elements, param);
}

template <typename T>
const BitmapPtr GPU_Backend<T>::bitmap_semi_join(T* join_column1,
                                                 size_t left_num_elements,
                                                 T* join_column2,
                                                 size_t right_num_elements,
                                                 const JoinParam&) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return BitmapPtr();
}

/* SET OPERATIONS */
template <typename T>
const BitmapPtr GPU_Backend<T>::computeBitmapSetOperation(
    BitmapPtr left_bitmap, BitmapPtr right_bitmap,
    const BitmapOperationParam& param) {
  return GPU_BitmapSetOperation::computeBitmapSetOperation(left_bitmap,
                                                           right_bitmap, param);
}

template <typename T>
const ColumnPtr GPU_Backend<T>::computeSetOperation(T* left_column,
                                                    size_t left_num_elements,
                                                    T* right_column,
                                                    size_t right_num_elements,
                                                    const SetOperationParam&) {
  return ColumnPtr();
}

template <typename T>
const PositionListPtr GPU_Backend<T>::computePositionListSetOperation(
    PositionListPtr tids1, PositionListPtr tids2,
    const SetOperationParam& param) {
  return GPU_PositionListSetOperation::computePositionListSetOperation(
      tids1, tids2, param);
}
/* END SET OPERATIONS */

/* SORT */
template <typename T>
const PositionListPtr GPU_Backend<T>::sort(T* column, size_t num_elements,
                                           const SortParam& param) {
  return GPU_Util<T>::sort(column, num_elements, param);
}

/* MISC */
template <typename T>
bool GPU_Backend<T>::gather(T* dest_column, T* source_column,
                            PositionListPtr tids, const GatherParam& param) {
  return GPU_Util<T>::gather(dest_column, source_column, tids, param);
}

template <typename T>
bool GPU_Backend<T>::generateConstantSequence(
    T* dest_column, size_t num_elements, T value,
    const ProcessorSpecification& proc_spec) {
  return GPU_Util<T>::generateConstantSequence(dest_column, num_elements, value,
                                               proc_spec);
}

template <typename T>
bool GPU_Backend<T>::generateAscendingSequence(
    T* dest_column, size_t num_elements, T begin_value,
    const ProcessorSpecification& proc_spec) {
  return GPU_Util<T>::generateAscendingSequence(dest_column, num_elements,
                                                begin_value, proc_spec);
}

template <>
bool GPU_Backend<std::string>::generateConstantSequence(
    std::string* dest_column, size_t num_elements, std::string value,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR("Called unimplemented Function!", "");
  return false;
}

template <>
bool GPU_Backend<std::string>::generateAscendingSequence(
    std::string* dest_column, size_t num_elements, std::string begin_value,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR("Called unimplemented Function!", "");
  return false;
}

template <>
bool GPU_Backend<C_String>::generateConstantSequence(
    C_String* dest_column, size_t num_elements, C_String value,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR("Called unimplemented Function!", "");
  return false;
}

template <>
bool GPU_Backend<C_String>::generateAscendingSequence(
    C_String* dest_column, size_t num_elements, C_String begin_value,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR("Called unimplemented Function!", "");
  return false;
}

template <typename T>
const BitmapPtr GPU_Backend<T>::convertPositionListToBitmap(
    PositionListPtr tids, size_t num_rows_base_table,
    const ProcessorSpecification& proc_spec) {
  return GPU_ConversionOperation<T>::convertToBitmap(tids, num_rows_base_table,
                                                     proc_spec);
}

template <typename T>
const PositionListPtr GPU_Backend<T>::convertBitmapToPositionList(
    BitmapPtr bitmap, const ProcessorSpecification& proc_spec) {
  return GPU_ConversionOperation<T>::convertToPositionList(bitmap, proc_spec);
}

template <typename T>
const DoubleDenseValueColumnPtr GPU_Backend<T>::convertToDoubleDenseValueColumn(
    const std::string& column_name, T* array, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  return GPU_ConversionOperation<T>::convertToDoubleDenseValueColumn(
      column_name, array, num_elements, proc_spec);
}

/* END MISC */

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(GPU_Backend);

}  // end namespace CogaDB
