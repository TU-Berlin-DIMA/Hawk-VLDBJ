/*
 * File:   gpu_conversion_operations.hpp
 * Author: sebastian
 *
 * Created on 3. Januar 2015, 11:35
 */

#ifndef GPU_CONVERSION_OPERATIONS_HPP
#define GPU_CONVERSION_OPERATIONS_HPP

#include <core/base_column.hpp>

namespace CoGaDB {

  template <typename T>
  class GPU_ConversionOperation {
   public:
    static const BitmapPtr convertToBitmap(
        PositionListPtr tids, size_t num_rows_base_table,
        const ProcessorSpecification& proc_spec);
    static const PositionListPtr convertToPositionList(
        BitmapPtr bitmap, const ProcessorSpecification& proc_spec);
    static const DoubleDenseValueColumnPtr convertToDoubleDenseValueColumn(
        const std::string& column_name, T* array, size_t value,
        const ProcessorSpecification& proc_spec);
  };

}  // end namespace CoGaDB

#endif /* GPU_CONVERSION_OPERATIONS_HPP */
