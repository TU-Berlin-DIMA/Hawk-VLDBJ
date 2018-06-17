/*
 * File:   positionlist_operation.hpp
 * Author: sebastian
 *
 * Created on 8. Januar 2015, 14:08
 */

#ifndef POSITIONLIST_OPERATION__HPP
#define POSITIONLIST_OPERATION__HPP

#include <core/base_column.hpp>

namespace CoGaDB {

  class GPU_PositionListSetOperation {
   public:
    static const PositionListPtr computePositionListSetOperation(
        PositionListPtr left, PositionListPtr right,
        const SetOperationParam& param);
  };

}  // end namespace CoGaDB

#endif /* POSITIONLIST_OPERATION__HPP */
