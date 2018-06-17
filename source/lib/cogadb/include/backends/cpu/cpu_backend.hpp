/*
 * File:   cpu_backend.hpp
 * Author: sebastian
 *
 * Created on 27. Dezember 2014, 01:54
 */

#pragma once

#ifndef CPU_BACKEND_HPP
#define CPU_BACKEND_HPP

#include <backends/processor_backend.hpp>
#include <core/column_base_typed.hpp>

namespace CoGaDB {

  template <typename T>
  class CPU_Backend : public ProcessorBackend<T> {
   public:
    typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
    typedef
        typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

    CPU_Backend();

    /***************** relational operations on Columns which return lookup
     * tables *****************/

    /* SELECTION */
    virtual const PositionListPtr tid_selection(T* column, size_t num_elements,
                                                const SelectionParam& param);
    virtual const BitmapPtr bitmap_selection(T* column, size_t num_elements,
                                             const SelectionParam& param);
    /* END SELECTION */

    /* COLUMN ALGEBRA */
    virtual bool column_algebra_operation(T* target_column, T* source_column,
                                          size_t num_elements,
                                          const AlgebraOperationParam&);
    virtual bool column_algebra_operation(T* column, size_t num_elements,
                                          T value,
                                          const AlgebraOperationParam&);

    virtual bool double_precision_column_algebra_operation(
        double* target_column, T* source_column, size_t num_elements,
        const AlgebraOperationParam&);
    /* END COLUMN ALGEBRA */

    /* AGGREGATION */
    /*! creates a compact version of the values in the column, which is used by
     * groupby*/
    virtual const ColumnGroupingKeysPtr createColumnGroupingKeys(
        T* column, size_t num_elements,
        const ProcessorSpecification& proc_spec) const;
    /*! returns the number of bits we need to represent the values stored in the
     * column*/
    virtual size_t getNumberOfRequiredBits(
        T* column, size_t num_elements,
        const ProcessorSpecification& proc_spec) const;

    virtual bool bit_shift(ColumnGroupingKeysPtr keys,
                           const BitShiftParam& param);

    virtual bool bitwise_combination(ColumnGroupingKeysPtr target_keys,
                                     ColumnGroupingKeysPtr source_keys,
                                     const BitwiseCombinationParam& param);

    /*! \brief aggregates the column values according to grouping keys
     *         and AggregationMethod agg_meth using aggregation algorithm
     * agg_alg
     *  \details using aggregation algorithm sort based requires prior sorting
     * of column!
     */
    virtual const AggregationResult aggregateByGroupingKeys(
        ColumnGroupingKeysPtr grouping_keys, T* aggregation_column,
        size_t num_elements, const AggregationParam&);
    /*! \brief aggregation without group by clause*/
    virtual const AggregationResult aggregate(T* aggregation_column,
                                              size_t num_elements,
                                              const AggregationParam&);

    /* END AGGREGATION */

    /* FETCH JOIN */
    virtual const PositionListPtr tid_fetch_join(JoinIndexPtr join_index,
                                                 PositionListPtr pk_table_tids,
                                                 const FetchJoinParam&);

    virtual const PositionListPairPtr fetch_join(JoinIndexPtr join_index,
                                                 PositionListPtr pk_table_tids,
                                                 const FetchJoinParam&);

    virtual const BitmapPtr bitmap_fetch_join(JoinIndexPtr join_index,
                                              PositionListPtr pk_table_tids,
                                              const FetchJoinParam&);
    /* END FETCH JOIN */

    /* JOIN */
    virtual const PositionListPairPtr join(T* join_column1,
                                           size_t left_num_elements,
                                           T* join_column2,
                                           size_t right_num_elements,
                                           const JoinParam&);

    virtual const PositionListPtr tid_semi_join(T* join_column1,
                                                size_t left_num_elements,
                                                T* join_column2,
                                                size_t right_num_elements,
                                                const JoinParam&);

    virtual const BitmapPtr bitmap_semi_join(T* join_column1,
                                             size_t left_num_elements,
                                             T* join_column2,
                                             size_t right_num_elements,
                                             const JoinParam&);

    /* SET OPERATIONS */
    virtual const BitmapPtr computeBitmapSetOperation(
        BitmapPtr left_bitmap, BitmapPtr right_bitmap,
        const BitmapOperationParam& param);
    virtual const ColumnPtr computeSetOperation(T* left_column,
                                                size_t left_num_elements,
                                                T* right_column,
                                                size_t right_num_elements,
                                                const SetOperationParam&);

    virtual const PositionListPtr computePositionListSetOperation(
        PositionListPtr tids1, PositionListPtr tids2, const SetOperationParam&);
    /* END SET OPERATIONS */

    /* SORT */
    virtual const PositionListPtr sort(T* column, size_t num_elements,
                                       const SortParam& param);

    /* MISC */
    virtual bool gather(T* dest_column, T* source_column, PositionListPtr tids,
                        const GatherParam&);
    virtual bool generateConstantSequence(
        T* dest_column, size_t num_elements, T value,
        const ProcessorSpecification& proc_spec);
    virtual bool generateAscendingSequence(
        T* dest_column, size_t num_elements, T begin_value,
        const ProcessorSpecification& proc_spec);
    virtual const BitmapPtr convertPositionListToBitmap(
        PositionListPtr tids, size_t num_rows_base_table,
        const ProcessorSpecification& proc_spec);
    virtual const PositionListPtr convertBitmapToPositionList(
        BitmapPtr bitmap, const ProcessorSpecification& proc_spec);

    virtual const DoubleDenseValueColumnPtr convertToDoubleDenseValueColumn(
        const std::string& column_name, T* array, size_t value,
        const ProcessorSpecification& proc_spec) const;
    /* END MISC */
  };

}  // end namespace CogaDB

#endif /* CPU_BACKEND_HPP */
