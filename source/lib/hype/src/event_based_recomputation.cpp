
#include <core/algorithm.hpp>
#include <plugins/recomputation_heuristics/event_based_recomputation.hpp>

namespace hype {
namespace core {

RelativeErrorBasedRecomputation::RelativeErrorBasedRecomputation()
    : RecomputationHeuristic_Internal("Event based Recomputation"),
      counter_(0),
      error_threshold_(2.0) {
  RecomputationHeuristicFactorySingleton::Instance().Register(
      "Event based Recomputation", &RelativeErrorBasedRecomputation::create);
}

bool RelativeErrorBasedRecomputation::internal_recompute(Algorithm& algorithm) {
  std::cout << algorithm.getName() << std::endl;
  assert(false == true);
  return false;
}

}  // end namespace core
}  // end namespace hype
