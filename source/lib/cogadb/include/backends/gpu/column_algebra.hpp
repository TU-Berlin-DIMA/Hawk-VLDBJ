/*
 * File:   column_algebra.hpp
 * Author: sebastian
 *
 * Created on 15. Januar 2015, 11:47
 */

#ifndef COLUMN_ALGEBRA_HPP
#define COLUMN_ALGEBRA_HPP

#include <core/base_column.hpp>

namespace CoGaDB {

  template <typename T>
  class GPU_ColumnAlgebra {
   public:
    static bool column_algebra_operation(T* target_column, T* source_column,
                                         size_t num_elements,
                                         const AlgebraOperationParam& param);

    static bool column_algebra_operation(T* column, size_t num_elements,
                                         T value,
                                         const AlgebraOperationParam& param);

    static bool double_precision_column_algebra_operation(
        double* target_column, T* source_column, size_t num_elements,
        const AlgebraOperationParam& param);
  };

}  // end namespace CoGaDB

#endif /* COLUMN_ALGEBRA_HPP */
