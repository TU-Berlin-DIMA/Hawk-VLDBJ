
#pragma once

#include <config/global_definitions.hpp>
#include <string>

namespace hype {
  namespace util {

    const std::string getName(StatisticalMethod);
    const std::string getName(RecomputationHeuristic);
    const std::string getName(OptimizationCriterion);
    const std::string getName(ProcessingDeviceType);
    const std::string getName(DeviceTypeConstraint);
    const std::string getName(QueryOptimizationHeuristic);

  }  // end namespace util
}  // end namespace hype
