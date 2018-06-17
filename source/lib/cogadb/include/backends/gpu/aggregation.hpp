/*
 * File:   aggregation.hpp
 * Author: sebastian
 *
 * Created on 10. Januar 2015, 19:03
 */

#ifndef GPU_AGGREGATION_HPP
#define GPU_AGGREGATION_HPP

#include <core/base_column.hpp>

namespace CoGaDB {

  template <typename T>
  class GPU_Aggregation {
   public:
    static const AggregationResult aggregateByGroupingKeys(
        ColumnGroupingKeysPtr grouping_keys, T* aggregation_column,
        size_t num_elements, const AggregationParam&);

    static const AggregationResult aggregate(T* aggregation_column,
                                             size_t num_elements,
                                             const AggregationParam&);

    static const ColumnGroupingKeysPtr createColumnGroupingKeys(
        T* column, size_t num_elements,
        const ProcessorSpecification& proc_spec);

    static size_t getNumberOfRequiredBits(
        T* column, size_t num_elements,
        const ProcessorSpecification& proc_spec);
  };

}  // end namespace CoGaDB

#endif /* GPU_AGGREGATION_HPP */
