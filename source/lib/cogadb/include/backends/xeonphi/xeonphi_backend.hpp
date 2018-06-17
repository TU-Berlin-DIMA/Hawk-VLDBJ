/*
 * File:   cpu_backend.hpp
 * Author: sebastian
 *
 * Created on 27. Dezember 2014, 01:54
 */

#pragma once

#ifndef XEONPHI_BACKEND_HPP
#define XEONPHI_BACKEND_HPP

#include <backends/processor_backend.hpp>

namespace CoGaDB {

  template <typename T>
  class XEONPHI_Backend : public ProcessorBackend<T> {
   public:
    XEONPHI_Backend();

    /***************** relational operations on Columns which return lookup
     * tables *****************/

    /* SELECTION */
    virtual const PositionListPtr tid_selection(T* column, size_t num_elements,
                                                const SelectionParam& param);
    virtual const BitmapPtr bitmap_selection(T* column, size_t num_elements,
                                             const SelectionParam& param);
    /* END SELECTION */

    /* COLUMN ALGEBRA */
    virtual const bool column_algebra_operation(T* target_column,
                                                T* source_column,
                                                size_t num_elements,
                                                const AlgebraOperationParam&);
    virtual const bool column_algebra_operation(T* column, size_t num_elements,
                                                T value,
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

    /* SET OPERATIONS */
    virtual const BitmapPtr computeBitmapSetOperation(
        BitmapPtr left_bitmap, BitmapPtr right_bitmap,
        const ProcessorSpecification& proc_spec);
    virtual const ColumnPtr computeSetOperation(T* left_column,
                                                size_t left_num_elements,
                                                T* right_column,
                                                size_t right_num_elements,
                                                const SetOperationParam&);
    /* END SET OPERATIONS */

    /* SORT */
    virtual const PositionListPtr sort(T* column, size_t num_elements,
                                       const SortParam& param);

    /* MISC */
    virtual const bool gather(T* dest_column, T* source_column,
                              PositionListPtr tids, const GatherParam&);
    virtual const BitmapPtr convertPositionListToBitmap(
        PositionListPtr tids, size_t num_rows_base_table,
        const ProcessorSpecification& proc_spec);
    virtual const PositionListPtr convertBitmapToPositionList(
        BitmapPtr bitmap, const ProcessorSpecification& proc_spec);
    /* END MISC */
  };

}  // end namespace CogaDB

#endif /* XEONPHI_BACKEND_HPP */
