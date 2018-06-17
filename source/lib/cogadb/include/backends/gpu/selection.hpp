/*
 * File:   selection.hpp
 * Author: sebastian
 *
 * Created on 31. Dezember 2014, 17:25
 */

#ifndef GPU_BACKEND_SELECTION_HPP
#define GPU_BACKEND_SELECTION_HPP

#include <core/base_column.hpp>

namespace CoGaDB {

  template <typename T>
  class GPU_Selection {
   public:
    static const PositionListPtr tid_selection(
        T* column, size_t num_elements, T comparison_value,
        const ValueComparator& comp,
        const hype::ProcessingDeviceMemoryID& mem_id);

    static const PositionListPtr tid_selection(
        T* column, size_t num_elements, T* comparison_column,
        const ValueComparator& comp,
        const hype::ProcessingDeviceMemoryID& mem_id);

    static const BitmapPtr bitmap_selection(
        T* column, size_t num_elements, T comparison_value,
        const ValueComparator& comp,
        const hype::ProcessingDeviceMemoryID& mem_id);

    static const BitmapPtr bitmap_selection(
        T* column, size_t num_elements, T* comparison_column,
        const ValueComparator& comp,
        const hype::ProcessingDeviceMemoryID& mem_id);
  };

}  // end namespace CoGaDB

#endif /* GPU_BACKEND_SELECTION_HPP */
