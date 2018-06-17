/*
 * File:   processor_backend.hpp
 * Author: sebastian
 *
 * Created on 26. Dezember 2014, 21:59
 */

#pragma once

#ifndef PROCESSOR_BACKEND_HPP
#define PROCESSOR_BACKEND_HPP

#include <core/base_column.hpp>
#include <core/bitmap.hpp>
#include <core/global_definitions.hpp>
#include <hype.hpp>
#include <lookup_table/join_index.hpp>

namespace CoGaDB {

  template <typename T>
  class ProcessorBackend {
   public:
    static ProcessorBackend* get(hype::ProcessingDeviceID);

    /***************** relational operations on Columns which return lookup
     * tables *****************/

    /* SELECTION */
    virtual const PositionListPtr tid_selection(
        T* column, size_t num_elements, const SelectionParam& param) = 0;
    virtual const BitmapPtr bitmap_selection(T* column, size_t num_elements,
                                             const SelectionParam& param) = 0;
    /* END SELECTION */

    /* COLUMN ALGEBRA */
    virtual bool column_algebra_operation(T* target_column, T* source_column,
                                          size_t num_elements,
                                          const AlgebraOperationParam&) = 0;
    virtual bool column_algebra_operation(T* column, size_t num_elements,
                                          T value,
                                          const AlgebraOperationParam&) = 0;

    virtual bool double_precision_column_algebra_operation(
        double* target_column, T* source_column, size_t num_elements,
        const AlgebraOperationParam&) = 0;

    /* END COLUMN ALGEBRA */

    /* AGGREGATION */
    /*! creates a compact version of the values in the column, which is used by
     * groupby*/
    virtual const ColumnGroupingKeysPtr createColumnGroupingKeys(
        T* column, size_t num_elements,
        const ProcessorSpecification& proc_spec) const = 0;
    /*! returns the number of bits we need to represent the values stored in the
     * column*/
    virtual size_t getNumberOfRequiredBits(
        T* column, size_t num_elements,
        const ProcessorSpecification& proc_spec) const = 0;

    /*! \brief computes keys[i] <<= number_of_bits or keys[i] >>= number_of_bits
     */
    virtual bool bit_shift(ColumnGroupingKeysPtr keys,
                           const BitShiftParam& param) = 0;

    /*! \brief computes target_keys[i] = target_keys[i] & source_keys[i] or
     * target_keys[i] = target_keys[i] | source_keys[i] */
    virtual bool bitwise_combination(ColumnGroupingKeysPtr target_keys,
                                     ColumnGroupingKeysPtr source_keys,
                                     const BitwiseCombinationParam& param) = 0;

    /*! \brief aggregates the column values according to grouping keys
     *         and AggregationMethod agg_meth using aggregation algorithm
     * agg_alg
     *  \details using aggregation algorithm sort based requires prior sorting
     * of column!
     */
    virtual const AggregationResult aggregateByGroupingKeys(
        ColumnGroupingKeysPtr grouping_keys, T* aggregation_column,
        size_t num_elements, const AggregationParam&) = 0;
    /*! \brief aggregation without group by clause*/
    virtual const AggregationResult aggregate(T* aggregation_column,
                                              size_t num_elements,
                                              const AggregationParam&) = 0;

    /* END AGGREGATION */

    /* FETCH JOIN */
    virtual const PositionListPtr tid_fetch_join(JoinIndexPtr join_index,
                                                 PositionListPtr pk_table_tids,
                                                 const FetchJoinParam&) = 0;

    virtual const PositionListPairPtr fetch_join(JoinIndexPtr join_index,
                                                 PositionListPtr pk_table_tids,
                                                 const FetchJoinParam&) = 0;

    virtual const BitmapPtr bitmap_fetch_join(JoinIndexPtr join_index,
                                              PositionListPtr pk_table_tids,
                                              const FetchJoinParam&) = 0;
    /* END FETCH JOIN */

    /* JOIN */
    /* The JoinParam object specifies details about the actual join type
     * (e.g., INNER JOIN, LEFT OUTER JOIN, ANTI JOIN) and the used join
     * algorithm (e.g., HASH JOIN or SORT MERGE JOIN) */

    virtual const PositionListPairPtr join(T* join_column1,
                                           size_t left_num_elements,
                                           T* join_column2,
                                           size_t right_num_elements,
                                           const JoinParam&) = 0;

    virtual const PositionListPtr tid_semi_join(T* join_column1,
                                                size_t left_num_elements,
                                                T* join_column2,
                                                size_t right_num_elements,
                                                const JoinParam&) = 0;

    virtual const BitmapPtr bitmap_semi_join(T* join_column1,
                                             size_t left_num_elements,
                                             T* join_column2,
                                             size_t right_num_elements,
                                             const JoinParam&) = 0;

    /* SET OPERATIONS */
    virtual const BitmapPtr computeBitmapSetOperation(
        BitmapPtr left_bitmap, BitmapPtr right_bitmap,
        const BitmapOperationParam& param) = 0;
    virtual const ColumnPtr computeSetOperation(T* left_column,
                                                size_t left_num_elements,
                                                T* right_column,
                                                size_t right_num_elements,
                                                const SetOperationParam&) = 0;

    virtual const PositionListPtr computePositionListSetOperation(
        PositionListPtr tids1, PositionListPtr tids2,
        const SetOperationParam&) = 0;

    /* END SET OPERATIONS */

    /* SORT */
    virtual const PositionListPtr sort(T* column, size_t num_elements,
                                       const SortParam& param) = 0;

    /* MISC */
    virtual bool gather(T* dest_column, T* source_column, PositionListPtr tids,
                        const GatherParam&) = 0;

    virtual bool generateConstantSequence(
        T* dest_column, size_t num_elements, T value,
        const ProcessorSpecification& proc_spec) = 0;
    virtual bool generateAscendingSequence(
        T* dest_column, size_t num_elements, T begin_value,
        const ProcessorSpecification& proc_spec) = 0;

    virtual const BitmapPtr convertPositionListToBitmap(
        PositionListPtr tids, size_t num_rows_base_table,
        const ProcessorSpecification& proc_spec) = 0;
    virtual const PositionListPtr convertBitmapToPositionList(
        BitmapPtr bitmap, const ProcessorSpecification& proc_spec) = 0;

    virtual const DoubleDenseValueColumnPtr convertToDoubleDenseValueColumn(
        const std::string& column_name, T* array, size_t value,
        const ProcessorSpecification& proc_spec) const = 0;

    /* END MISC */
  };

}  // end namespace CogaDB

#endif /* PROCESSOR_BACKEND_HPP */
