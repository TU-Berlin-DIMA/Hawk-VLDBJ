/*
 * File:   bit_operations.hpp
 * Author: sebastian
 *
 * Created on 10. Januar 2015, 12:25
 */

#ifndef BIT_OPERATIONS_HPP
#define BIT_OPERATIONS_HPP

#include <core/base_column.hpp>

namespace CoGaDB {

  class GPU_BitOperation {
   public:
    static bool bit_shift(ColumnGroupingKeysPtr keys,
                          const BitShiftParam& param);

    static bool bitwise_combination(ColumnGroupingKeysPtr target_keys,
                                    ColumnGroupingKeysPtr source_keys,
                                    const BitwiseCombinationParam& param);
  };

}  // end namespace CoGaDB

#endif /* BIT_OPERATIONS_HPP */
