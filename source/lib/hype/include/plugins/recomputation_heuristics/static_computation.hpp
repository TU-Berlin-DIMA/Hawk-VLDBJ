
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

    class Oneshotcomputation : public RecomputationHeuristic {
     public:
      Oneshotcomputation();
      /*! returns true, if approximation function has to be recomputed and false
       * otherwise*/
      virtual const bool recompute(Algorithm& algortihm);

      static RecomputationHeuristic* create() {
        return new PeriodicRecomputation();
      }

     private:
      unsigned int counter_;
    };

  }  // end namespace core
}  // end namespace hype
