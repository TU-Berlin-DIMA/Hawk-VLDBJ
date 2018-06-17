
#include <config/configuration.hpp>
#include <core/report.hpp>
#include <core/scheduler.hpp>

#include <cmath>

// using namespace std;

namespace hype {
namespace core {

Report& Report::instance() {
  static Report r;
  return r;
}

double Report::getRelativeEstimationError(
    const std::string& algorithm_name,
    const DeviceSpecification& dev_spec) const throw(std::invalid_argument) {
  std::string internal_algorithm_name =
      hype::util::toInternalAlgName(algorithm_name, dev_spec);

  AlgorithmPtr ptr =
      Scheduler::instance().getAlgorithm(internal_algorithm_name);
  if (!ptr) {
    std::string error_message = "Could not find Algorithm '";
    error_message += algorithm_name + "'!";
    throw new std::invalid_argument(error_message);
  }

  double sum = 0;
  unsigned int length_of_initial_training_phase =
      Runtime_Configuration::instance()
          .getTrainingLength();  // Configuration::period_for_periodic_recomputation;
  for (unsigned int i = length_of_initial_training_phase + 1;
       i < ptr->getAlgorithmStatistics().relative_errors_.size(); i++) {
    sum += std::abs(ptr->getAlgorithmStatistics().relative_errors_[i]);
    // std::cout << "sum " << sum << " " <<
    // ptr->getAlgorithmStatistics().relative_errors_[i] << std::endl;
  }
  return sum / ptr->getAlgorithmStatistics().relative_errors_.size();
  // return ptr->getAlgorithmStatistics().average_relative_error_;
}

Report::Report() {}

}  // end namespace core
}  // end namespace hype
