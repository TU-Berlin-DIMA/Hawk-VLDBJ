
#pragma once

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <core/optimization_criterion.hpp>
#include <core/time_measurement.hpp>

namespace hype {
  namespace core {

    class Throughput2 : public OptimizationCriterion_Internal {
     public:
      Throughput2(std::string name_of_operation);

      virtual const SchedulingDecision getOptimalAlgorithm_internal(
          const Tuple& input_values, Operation& op,
          DeviceTypeConstraint dev_constr);
      // OptimizationCriterionFactorySingleton::Instance().

      static OptimizationCriterion_Internal* create() {
        return new Throughput2("");
      }
    };

  }  // end namespace core
}  // end namespace hype
