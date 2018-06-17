
#pragma once

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <core/optimization_criterion.hpp>
#include <core/time_measurement.hpp>

namespace hype {
  namespace core {

    class ResponseTime : public OptimizationCriterion_Internal {
     public:
      ResponseTime(const std::string& name_of_operation);

      virtual const SchedulingDecision getOptimalAlgorithm_internal(
          const Tuple& input_values, Operation& op,
          DeviceTypeConstraint dev_constr);
      // OptimizationCriterionFactorySingleton::Instance().

      static OptimizationCriterion_Internal* create() {
        return new ResponseTime("");
      }
    };

  }  // end namespace core
}  // end namespace hype
