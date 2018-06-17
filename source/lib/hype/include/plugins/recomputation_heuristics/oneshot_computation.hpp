
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

    class Oneshotcomputation : public RecomputationHeuristic_Internal {
     public:
      Oneshotcomputation();
      /*! returns true, if approximation function has to be recomputed and false
       * otherwise*/
      virtual bool internal_recompute(Algorithm& algortihm);

      static RecomputationHeuristic_Internal* create() {
        return new Oneshotcomputation();
      }

     private:
      bool once_;
      unsigned int counter_;
    };

  }  // end namespace core
}  // end namespace hype
