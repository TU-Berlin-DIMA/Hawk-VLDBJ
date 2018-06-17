
#include <iostream>

#include <plugins/pluginloader.hpp>
// Statistical Methods
#include <plugins/statistical_methods/knn_regression.hpp>
#include <plugins/statistical_methods/least_squares.hpp>
#include <plugins/statistical_methods/multi_dim_least_squares.hpp>
// Recomputation Heuristics
#include <plugins/recomputation_heuristics/oneshot_computation.hpp>
#include <plugins/recomputation_heuristics/periodic_recomputation.hpp>
// Optimization Criterias
#include <plugins/optimization_criterias/probability_based_outsourcing.hpp>
#include <plugins/optimization_criterias/response_time.hpp>
#include <plugins/optimization_criterias/response_time_advanced.hpp>
#include <plugins/optimization_criterias/simple_round_robin_throughput.hpp>
#include <plugins/optimization_criterias/throughput.hpp>
#include <plugins/optimization_criterias/throughput2.hpp>

namespace hype {
namespace core {

bool PluginLoader::loadPlugins() {
  // add Statistical Methods
  if (!StatisticalMethodFactorySingleton::Instance().Register(
          "Least Squares 1D", &Least_Squares_Method_1D::create))
    std::cout << "failed to load plugin! "
              << "'Least Squares 1D'" << std::endl;
  if (!StatisticalMethodFactorySingleton::Instance().Register(
          "Least Squares 2D", &Least_Squares_Method_2D::create))
    std::cout << "failed to load plugin! "
              << "'Least Squares 2D'" << std::endl;
  if (!StatisticalMethodFactorySingleton::Instance().Register(
          "KNN_Regression", &KNN_Regression::create))
    std::cout << "failed to load plugin! "
              << "'KNN_Regression'" << std::endl;
  // add Recomputation Heuristics
  RecomputationHeuristicFactorySingleton::Instance().Register(
      "Periodic Recomputation", &PeriodicRecomputation::create);
  RecomputationHeuristicFactorySingleton::Instance().Register(
      "Oneshot Recomputation", &Oneshotcomputation::create);
  // add Optimization Criterias
  OptimizationCriterionFactorySingleton::Instance().Register(
      "Response Time", &ResponseTime::create);
  OptimizationCriterionFactorySingleton::Instance().Register(
      "WaitingTimeAwareResponseTime", &WaitingTimeAwareResponseTime::create);
  OptimizationCriterionFactorySingleton::Instance().Register(
      "Throughput", &Throughput::create);
  // OptimizationCriterionFactorySingleton::Instance().Register("Throughput2",&Throughput2::create);
  OptimizationCriterionFactorySingleton::Instance().Register(
      "Simple Round Robin", &SimpleRoundRobin::create);
  OptimizationCriterionFactorySingleton::Instance().Register(
      "ProbabilityBasedOutsourcing", &ProbabilityBasedOutsourcing::create);
  OptimizationCriterionFactorySingleton::Instance().Register(
      "Throughput2", &ProbabilityBasedOutsourcing::create);

  if (!quiet) std::cout << "[HyPE]: Loading Plugins..." << std::endl;
  return true;
}

}  // end namespace core
}  // end namespace hype
