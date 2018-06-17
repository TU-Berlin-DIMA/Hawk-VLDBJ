
#pragma once

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <core/optimization_criterion.hpp>
#include <core/time_measurement.hpp>

namespace hype {
  namespace core {

    class ProbabilityBasedOutsourcing : public OptimizationCriterion_Internal {
     public:
      ProbabilityBasedOutsourcing(const std::string& name_of_operation);

      virtual const SchedulingDecision getOptimalAlgorithm_internal(
          const Tuple& input_values, Operation& op,
          DeviceTypeConstraint dev_constr);
      // OptimizationCriterionFactorySingleton::Instance().

      static OptimizationCriterion_Internal* create() {
        return new ProbabilityBasedOutsourcing("");
      }

     private:
      boost::mt19937 random_number_generator_;
    };

  }  // end namespace core
}  // end namespace hype
