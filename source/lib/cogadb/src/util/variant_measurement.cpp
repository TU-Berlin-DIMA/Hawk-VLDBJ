#include <core/global_definitions.hpp>
#include <parser/client.hpp>
#include <util/variant_measurement.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <query_compilation/query_context.hpp>

namespace CoGaDB {

const VariantMeasurement createVariantMeasurement(
    double total_elapsed_time_in_s, QueryContextPtr context) {
  double total_compile_time_s = context->getCompileTimeSec();
  double total_host_compile_time_s = context->getHostCompileTimeSec();
  double total_kernel_compile_time_s = context->getKernelCompileTimeSec();
  double total_execution_time_s =
      total_elapsed_time_in_s - total_compile_time_s;
  double pipeline_execution_time_s = context->getExecutionTimeSec();
  double overhead_time_s = total_execution_time_s - pipeline_execution_time_s;

  return VariantMeasurement(
      true, total_execution_time_s, pipeline_execution_time_s,
      total_compile_time_s,
      total_kernel_compile_time_s, /* kernel compilation time */
      total_host_compile_time_s,   /* host compilation time */
      overhead_time_s);
}

void print(ClientPtr client, const VariantMeasurement& vm) {
  std::ostream& out = client->getOutputStream();
  std::string profiling_timer_prefix =
      VariableManager::instance().getVariableValueString(
          "code_gen.opt.profiling_timer_prefix");
  if (vm.success) {
    out << profiling_timer_prefix
        << "Compile Time: " << vm.total_compilation_time_in_s << "s"
        << std::endl;
    out << profiling_timer_prefix << "Total Kernel Compile Time: "
        << vm.total_kernel_compilation_time_in_s << "s" << std::endl;
    out << profiling_timer_prefix
        << "Total Host Compile Time: " << vm.total_host_compilation_time_in_s
        << "s" << std::endl;
    out << profiling_timer_prefix
        << "Execution Time (Real): " << vm.total_execution_time_in_s << "s"
        << std::endl;
    out << profiling_timer_prefix
        << "Pipeline Execution Time: " << vm.total_pipeline_execution_time_in_s
        << "s" << std::endl;
    out << profiling_timer_prefix
        << "Overhead Time: " << vm.total_overhead_time_in_s << "s" << std::endl;
  } else {
    COGADB_ERROR("Variant measurement is invalid! Skip printing...", "");
  }
}

VariantExecutionStatistics::VariantExecutionStatistics(
    std::vector<VariantMeasurement> measurements) {
  std::vector<double> execution_times;

  execution_times.reserve(measurements.size());
  for (const auto& measurement : measurements) {
    execution_times.push_back(measurement.total_pipeline_execution_time_in_s);
  }

  const auto& minmax = std::minmax_element(std::begin(execution_times),
                                           std::end(execution_times));
  min = *minmax.first;
  max = *minmax.second;

  size_t size = execution_times.size();
  sort(execution_times.begin(), execution_times.end());
  if (size % 2 == 0) {
    median = (execution_times[size / 2 - 1] + execution_times[size / 2]) / 2;
  } else {
    median = execution_times[size / 2];
  }

  // http://stackoverflow.com/a/7616783/1531656
  double sum =
      std::accumulate(execution_times.begin(), execution_times.end(), 0.0);
  mean = sum / execution_times.size();
  const double mean_ = mean;
  std::vector<double> diff(execution_times.size());
  std::transform(execution_times.begin(), execution_times.end(), diff.begin(),
                 [mean_](double x) { return x - mean_; });
  double sq_sum =
      std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  standard_deviation =
      std::sqrt(static_cast<double>(sq_sum / execution_times.size()));
  variance = standard_deviation * standard_deviation;
  std::cout << "#####################################################"
            << std::endl;
  std::cout << "Computing Mean and Standard Deviation: " << std::endl;
  std::cout << "Data: ";
  for (auto value : execution_times) {
    std::cout << value << ",";
  }
  std::cout << "Mean: " << mean << std::endl;
  std::cout << "Median: " << median << std::endl;
  std::cout << "Standard Deviation: " << standard_deviation << std::endl;
  std::cout << "#####################################################"
            << std::endl;
}

}  // end namespace CoGaDB
