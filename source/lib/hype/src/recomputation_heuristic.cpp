
#include <config/configuration.hpp>

#include <core/algorithm.hpp>
#include <core/recomputation_heuristic.hpp>

namespace hype {
namespace core {

RecomputationHeuristic_Internal::RecomputationHeuristic_Internal(
    const std::string& name)
    : name_(name),
      length_of_training_phase_(
          Runtime_Configuration::instance().getTrainingLength()),
      is_initial_approximation_function_computed_(false),
      samplecounter_(1) {}

bool RecomputationHeuristic_Internal::recompute(Algorithm& algorithm) {
  if (is_initial_approximation_function_computed_) {
    return this->internal_recompute(algorithm);
  } else {
    if (!quiet && verbose && debug) {
      std::cout << "[DEBUG]: initial Training phase of algortihm '"
                << algorithm.getName() << "' step " << samplecounter_ << "/"
                << length_of_training_phase_ << std::endl;
    }
    // finish training phase either when certain
    // number of iterations completed or the executionHistory
    // if the algorithm has enough training points
    // this can happen because algorithms on the same
    // processor share their execution histories
    if (samplecounter_++ >= length_of_training_phase_ ||
        algorithm.getAlgorithmStatistics().executionHistory_.size() >=
            length_of_training_phase_) {
      is_initial_approximation_function_computed_ = true;
      return true;
    } else {
      return false;
    }
  }
}

const boost::shared_ptr<RecomputationHeuristic_Internal>
getNewRecomputationHeuristicbyName(
    std::string name_of_recomputation_heuristic) {
  // RecomputationHeuristicFactorySingleton::Instance().Register()
  // RecomputationHeuristicFactorySingleton::Instance()
  RecomputationHeuristic_Internal* ptr =
      RecomputationHeuristicFactorySingleton::Instance().CreateObject(
          name_of_recomputation_heuristic);  //.( 1, createProductNull );
  return boost::shared_ptr<RecomputationHeuristic_Internal>(ptr);
}

RecomputationHeuristicFactory&
RecomputationHeuristicFactorySingleton::Instance() {
  static RecomputationHeuristicFactory factory;
  return factory;
}
}  // end namespace core
}  // end namespace hype
