/*
 * File:   cpu_backend.cpp
 * Author: sebastian
 *
 * Created on 27. Dezember 2014, 02:03
 */

#include <backends/xeonphi/xeonphi_backend.hpp>

namespace CoGaDB {

template <typename T>
XEONPHI_Backend<T>::XEONPHI_Backend() : ProcessorBackend<T>() {}

/***************** relational operations on Columns which return lookup tables
 * *****************/

/* SELECTION */
template <typename T>
const PositionListPtr XEONPHI_Backend<T>::tid_selection(
    T* column, size_t num_elements, const SelectionParam& param) {
  return PositionListPtr();
}
template <typename T>
const BitmapPtr XEONPHI_Backend<T>::bitmap_selection(
    T* column, size_t num_elements, const SelectionParam& param) {
  return BitmapPtr();
}
/* END SELECTION */

/* COLUMN ALGEBRA */
template <typename T>
const bool XEONPHI_Backend<T>::column_algebra_operation(
    T* target_column, T* source_column, size_t num_elements,
    const AlgebraOperationParam&) {
  return false;
}

template <typename T>
const bool XEONPHI_Backend<T>::column_algebra_operation(
    T* column, size_t num_elements, T value, const AlgebraOperationParam&) {
  return false;
}
/* END COLUMN ALGEBRA */

/* AGGREGATION */
template <typename T>
const ColumnGroupingKeysPtr XEONPHI_Backend<T>::createColumnGroupingKeys(
    T* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  return ColumnGroupingKeysPtr();
}

template <typename T>
size_t XEONPHI_Backend<T>::getNumberOfRequiredBits(
    T* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) const {
  return 65;
}

template <typename T>
const AggregationResult XEONPHI_Backend<T>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, T* aggregation_column,
    size_t num_elements, const AggregationParam&) {
  return AggregationResult();
}

template <typename T>
const AggregationResult XEONPHI_Backend<T>::aggregate(T* aggregation_column,
                                                      size_t num_elements,
                                                      const AggregationParam&) {
  return AggregationResult();
}

/* END AGGREGATION */

/* FETCH JOIN */
template <typename T>
const PositionListPtr XEONPHI_Backend<T>::tid_fetch_join(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const FetchJoinParam&) {
  return PositionListPtr();
}

template <typename T>
const BitmapPtr XEONPHI_Backend<T>::bitmap_fetch_join(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const FetchJoinParam&) {
  return BitmapPtr();
}
/* END FETCH JOIN */

/* JOIN */
template <typename T>
const PositionListPairPtr XEONPHI_Backend<T>::join(T* join_column1,
                                                   size_t left_num_elements,
                                                   T* join_column2,
                                                   size_t right_num_elements,
                                                   const JoinParam&) {
  return PositionListPairPtr();
}

/* SET OPERATIONS */
template <typename T>
const BitmapPtr XEONPHI_Backend<T>::computeBitmapSetOperation(
    BitmapPtr left_bitmap, BitmapPtr right_bitmap,
    const ProcessorSpecification& proc_spec) {
  return BitmapPtr();
}

template <typename T>
const ColumnPtr XEONPHI_Backend<T>::computeSetOperation(
    T* left_column, size_t left_num_elements, T* right_column,
    size_t right_num_elements, const SetOperationParam&) {
  return ColumnPtr();
}
/* END SET OPERATIONS */

/* SORT */
template <typename T>
const PositionListPtr XEONPHI_Backend<T>::sort(T* column, size_t num_elements,
                                               const SortParam& param) {
  return PositionListPtr();
}

/* MISC */
template <typename T>
const bool XEONPHI_Backend<T>::gather(T* dest_column, T* source_column,
                                      PositionListPtr tids,
                                      const GatherParam&) {
  return false;
}
template <typename T>
const BitmapPtr XEONPHI_Backend<T>::convertPositionListToBitmap(
    PositionListPtr tids, size_t num_rows_base_table,
    const ProcessorSpecification& proc_spec) {
  return BitmapPtr();
}
template <typename T>
const PositionListPtr XEONPHI_Backend<T>::convertBitmapToPositionList(
    BitmapPtr bitmap, const ProcessorSpecification& proc_spec) {
  return PositionListPtr();
}
/* END MISC */

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(XEONPHI_Backend);

}  // end namespace CogaDB
