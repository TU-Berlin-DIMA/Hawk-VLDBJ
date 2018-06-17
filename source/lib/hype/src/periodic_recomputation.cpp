
#include <config/configuration.hpp>

#include <core/algorithm.hpp>

#include <plugins/recomputation_heuristics/periodic_recomputation.hpp>

namespace hype {
namespace core {

PeriodicRecomputation::PeriodicRecomputation()
    : RecomputationHeuristic_Internal("Periodic Recomputation"), counter_(0) {
  RecomputationHeuristicFactorySingleton::Instance().Register(
      "Periodic Recomputation", &PeriodicRecomputation::create);
}

bool PeriodicRecomputation::internal_recompute(Algorithm& algorithm) {
  // std::cout << algorithm.getName() << std::endl;
  // algorithm.
  assert(algorithm.getName() != "");
  // static unsigned int counter=0;
  counter_++;
  if (counter_ >=
      Runtime_Configuration::instance()
          .getRecomputationPeriod()) {  // Configuration::period_for_periodic_recomputation){
    counter_ = 0;
    return true;
  }
  return false;
}

}  // end namespace core
}  // end namespace hype
