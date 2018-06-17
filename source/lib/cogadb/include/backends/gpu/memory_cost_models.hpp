/*
 * File:   memory_cost_models.hpp
 * Author: sebastian
 *
 * Created on 13. März 2015, 11:40
 */

#ifndef MEMORY_COST_MODELS_HPP
#define MEMORY_COST_MODELS_HPP

#include <core/global_definitions.hpp>
#include <hype.hpp>

namespace CoGaDB {
  namespace gpu {

    class GPU_Operators_Memory_Cost_Models {
     public:
      static size_t columnFetchJoin(const hype::Tuple& feature_vector);
      static size_t tableFetchJoin(const hype::Tuple& feature_vector);
      static size_t positionlistToBitmap(const hype::Tuple& feature_vector);
      static size_t bitmapToPositionList(const hype::Tuple& feature_vector);
      static size_t bitwiseAND(const hype::Tuple& feature_vector);
      static size_t columnSelection(const hype::Tuple& feature_vector);
    };

  }  // end namespace gpu
}  // end namespace CogaDB

#endif /* MEMORY_COST_MODELS_HPP */
