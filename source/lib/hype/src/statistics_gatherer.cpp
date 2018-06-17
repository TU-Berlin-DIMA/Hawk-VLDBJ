
#include <core/statistics_gatherer.hpp>

namespace hype {
namespace core {

StatisticsGatherer::StatisticsGatherer(const std::string& operation_name)
    : operation_name_(operation_name),
      number_of_right_decisions_(0),
      number_of_total_decisions_(0),
      isolated_execution_time_of_algorithms_(),
      names_of_algorithms_(),
      execution_time_of_ideal_model_(0),
      execution_time_of_cpu_only_model_(0),
      execution_time_of_gpu_only_model_(0),
      execution_time_of_real_model_(0),
      total_time_for_overhead_of_addObservation_(0),
      total_time_for_overhead_of_getOptimalAlgorithm_(0),
      // Feature: inter device parallelism
      inter_device_parallel_time_cpu_(0),
      inter_device_parallel_time_gpu_(0) {}

bool StatisticsGatherer::addStatistics(const WorkloadGenerator& w) {
  assert(this->operation_name_ == w.operation_name_);

  this->number_of_right_decisions_ += w.number_of_right_decisions_;
  this->number_of_total_decisions_ += w.number_of_total_decisions_;
  this->execution_time_of_ideal_model_ += w.execution_time_of_ideal_model_;
  this->execution_time_of_cpu_only_model_ +=
      w.execution_time_of_cpu_only_model_;
  this->execution_time_of_gpu_only_model_ +=
      w.execution_time_of_gpu_only_model_;
  this->execution_time_of_real_model_ += w.execution_time_of_real_model_;
  this->total_time_for_overhead_of_addObservation_ +=
      w.total_time_for_overhead_of_addObservation_;
  this->total_time_for_overhead_of_getOptimalAlgorithm_ +=
      w.total_time_for_overhead_of_getOptimalAlgorithm_;
  // Feature: inter device parallelism
  this->inter_device_parallel_time_cpu_ += w.inter_device_parallel_time_cpu_;
  this->inter_device_parallel_time_gpu_ += w.inter_device_parallel_time_gpu_;

  if (this->isolated_execution_time_of_algorithms_.empty()) {
    this->isolated_execution_time_of_algorithms_ =
        w.isolated_execution_time_of_algorithms_;
    // std::cout << w.offline_algorithms.size() << std::endl;
    for (unsigned int i = 0; i < w.offline_algorithms.size(); i++) {
      // std::cout << w.offline_algorithms[i].getAlgorithmName() << std::endl;
      this->names_of_algorithms_.push_back(
          w.offline_algorithms[i].getAlgorithmName());
    }
    // exit(-1);
    return true;
  }
  assert(this->isolated_execution_time_of_algorithms_.size() ==
         w.isolated_execution_time_of_algorithms_.size());
  assert(this->names_of_algorithms_.size() ==
         this->isolated_execution_time_of_algorithms_.size());
  for (unsigned int i = 0; i < w.offline_algorithms.size(); i++) {
    this->isolated_execution_time_of_algorithms_[i] +=
        w.isolated_execution_time_of_algorithms_[i];
  }
  return true;
}

void StatisticsGatherer::printReport() const throw() {
  // Feature: overhead tracking
  double execution_time_of_real_model_with_overhead =
      execution_time_of_real_model_;
  execution_time_of_real_model_with_overhead +=
      total_time_for_overhead_of_addObservation_;
  execution_time_of_real_model_with_overhead +=
      total_time_for_overhead_of_getOptimalAlgorithm_;

  // Feature: inter device parallelism
  double response_time_with_inter_device_parallelism = std::max(
      inter_device_parallel_time_cpu_, inter_device_parallel_time_gpu_);
  response_time_with_inter_device_parallelism +=
      total_time_for_overhead_of_addObservation_;
  response_time_with_inter_device_parallelism +=
      total_time_for_overhead_of_getOptimalAlgorithm_;

  std::cout
      << "====================================================================="
         "==========="
      << std::endl
      << "Global Report for operation " << operation_name_ << ": " << std::endl
      << "Number of correct decisions: " << number_of_right_decisions_ << "   "
      << "Number of total decisions: " << number_of_total_decisions_
      << std::endl
      << "Precision (Hitrate): "
      << double(number_of_right_decisions_) / number_of_total_decisions_
      << std::endl
      << "---------------------------------------------------------------------"
         "-----------"
      << std::endl
      << "Execution time for workload of ideal model: "
      << execution_time_of_ideal_model_ << "ns" << std::endl
      << "Execution time for workload of real model (without overhead): "
      << execution_time_of_real_model_ << "ns (model quality: "
      << execution_time_of_ideal_model_ / execution_time_of_real_model_ << ")"
      << std::endl
      << "Execution time for workload of real model (with overhead): "
      << execution_time_of_real_model_with_overhead << "ns (model quality: "
      << execution_time_of_ideal_model_ /
             execution_time_of_real_model_with_overhead
      << ")" << std::endl
      << "---------------------------------------------------------------------"
         "-----------"
      << std::endl
      << "Overhead time for workload of real model (addObservation): "
      << total_time_for_overhead_of_addObservation_ << "ns" << std::endl
      << "Overhead time for workload of real model (getOptimalAlgorithm): "
      << total_time_for_overhead_of_getOptimalAlgorithm_ << "ns" << std::endl
      << "Total Overhead time for workload of real model: "
      << total_time_for_overhead_of_addObservation_ +
             total_time_for_overhead_of_getOptimalAlgorithm_
      << "ns" << std::endl
      << "Precentaged Overhead of total time of real model for workload: "
      << ((total_time_for_overhead_of_addObservation_ +
           total_time_for_overhead_of_getOptimalAlgorithm_) /
          execution_time_of_real_model_) *
             100
      << "%" << std::endl

      // Feature: inter device parallelism
      << "---------------------------------------------------------------------"
         "-----------"
      << std::endl
      << "Execution Time spend on CPU: " << inter_device_parallel_time_cpu_
      << "ns" << std::endl
      << "Execution Time spend on GPU: " << inter_device_parallel_time_gpu_
      << "ns" << std::endl
      << "Response Time with Decision Model (including overhead): "
      << response_time_with_inter_device_parallelism << "ns" << std::endl
      /*! \todo check for correctness!*/
      << "(approximative) Ideal Response Time: "
      << ((inter_device_parallel_time_cpu_ + inter_device_parallel_time_gpu_) /
          2)
      << std::endl
      << "---------------------------------------------------------------------"
         "-----------"
      << std::endl
      << "CPU Utilization: "
      << inter_device_parallel_time_cpu_ /
             (inter_device_parallel_time_cpu_ + inter_device_parallel_time_gpu_)
      << "%" << std::endl
      << "GPU Utilization: "
      << inter_device_parallel_time_gpu_ /
             (inter_device_parallel_time_cpu_ + inter_device_parallel_time_gpu_)
      << "%" << std::endl
      << "====================================================================="
         "==========="
      << std::endl;

  assert(this->names_of_algorithms_.size() ==
         this->isolated_execution_time_of_algorithms_.size());
  for (unsigned int i = 0; i < names_of_algorithms_.size(); i++) {
    std::cout
        << "Execution time for workload for model that uses only algorithm "
        << names_of_algorithms_[i] << ": "
        << isolated_execution_time_of_algorithms_[i] << "ns (model quality: "
        << execution_time_of_ideal_model_ /
               isolated_execution_time_of_algorithms_[i]
        << ")" << std::endl;
    // Feature: inter device parallelism
    std::cout
        << "Speedup compared to Algorithm " << names_of_algorithms_[i] << ": "
        << isolated_execution_time_of_algorithms_[i] /
               response_time_with_inter_device_parallelism
        << std::endl
        << "Overall Improvement using decision model (saved time): "
        << ((isolated_execution_time_of_algorithms_[i] -
             response_time_with_inter_device_parallelism) /
            isolated_execution_time_of_algorithms_[i]) *
               100
        << "% ("
        << isolated_execution_time_of_algorithms_[i] -
               response_time_with_inter_device_parallelism
        << "ns)" << std::endl
        //<< "Ideal Speedup compared to Algorithm " << names_of_algorithms_[i]
        //<< ": " <<
        // isolated_execution_time_of_algorithms_[i]/((inter_device_parallel_time_cpu_+inter_device_parallel_time_gpu_)/2)
        //<< std::endl
        << "-------------------------------------------------------------------"
           "-------------"
        << std::endl;
  }
}

}  // end namespace core
}  // end namespace hype
