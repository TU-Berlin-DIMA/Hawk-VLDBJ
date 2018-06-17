
#pragma once

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

// hype includes
#include <core/recomputation_heuristic.hpp>
#include <core/time_measurement.hpp>

namespace hype {
  namespace core {

    class RelativeErrorBasedRecomputation
        : public RecomputationHeuristic_Internal {
     public:
      RelativeErrorBasedRecomputation();
      explicit RelativeErrorBasedRecomputation(const double& error_threshold);
      /*! returns true, if approximation function has to be recomputed and false
       * otherwise*/
      virtual bool internal_recompute(Algorithm& algortihm);

      static RecomputationHeuristic_Internal* create() {
        return new RelativeErrorBasedRecomputation();
      }

     private:
      unsigned int counter_;
      double error_threshold_;
    };

  }  // end namespace core
}  // end namespace hype
