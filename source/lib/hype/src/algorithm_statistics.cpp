
#include <cmath>
#include <fstream>
#include <iostream>

#include <config/configuration.hpp>
#include <config/global_definitions.hpp>
#include <core/algorithm_statistics.hpp>
#include <util/algorithm_name_conversion.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

namespace hype {
namespace core {

AlgorithmStatistics::AlgorithmStatistics()
    : executionHistory_(),
      relative_errors_(),
      number_of_recomputations_(0),
      number_of_decisions_for_this_algorithm_(0),
      average_relative_error_(0),
      total_execution_time_(0),
      number_of_terminated_executions_of_this_algorithm_(0) {}

bool AlgorithmStatistics::writeToDisc(const std::string& operation_name,
                                      const std::string& algorithm_name) const {
  const std::string output_dir_name = "output";
  if (!boost::filesystem::exists(output_dir_name)) {
    if (!quiet) cout << "create Directory '" << output_dir_name << "'" << endl;
    if (!boost::filesystem::create_directory(output_dir_name)) {
      cout << "HyPE Library: Failed to created Output Directory '"
           << output_dir_name << "' for operation statistics, skipping the "
                                 "write operation for statistical data for "
                                 "operation '"
           << operation_name << "'" << endl;
    }
  }

  std::string dir_name = output_dir_name + "/";
  dir_name += operation_name;
  dir_name += "/";

  if (!boost::filesystem::create_directory(dir_name) && !quiet)
    std::cout << "Directory '" << dir_name << "' already exists!" << std::endl;

  dir_name += algorithm_name;

  if (!boost::filesystem::create_directory(dir_name) && !quiet)
    std::cout << "Directory '" << dir_name << "' already exists!" << std::endl;

  std::string basic_file_name = dir_name + "/";
  // basic_file_name+=algorithm_name;

  std::string file_name_measurement_pairs =
      basic_file_name + "measurement_pairs.data";
  fstream file(file_name_measurement_pairs.c_str(),
               std::ios_base::out | std::ios_base::trunc);
  executionHistory_.store(file);

  std::string file_name_relative_errors =
      basic_file_name + "relative_errors.data";
  fstream file_rel_err(file_name_relative_errors.c_str(),
                       std::ios_base::out | std::ios_base::trunc);

  std::vector<double> average_estimation_errors;

  // TODO: NOTE: here is somewere an ERROR, because for some
  double sum = 0;
  unsigned int length_of_initial_training_phase =
      Runtime_Configuration::instance()
          .getTrainingLength();  // Configuration::period_for_periodic_recomputation;
  for (unsigned int i = length_of_initial_training_phase + 1;
       i < relative_errors_.size(); i++) {
    file_rel_err << (i - length_of_initial_training_phase) << "\t"
                 << relative_errors_[i] << std::endl;
    sum += abs(relative_errors_[i]);
    average_estimation_errors.push_back(
        sum / double(i - length_of_initial_training_phase +
                     1));  // compute average estimation error after each
                           // execution of the algorithm, so we can plot it over
                           // "time"
  }

  std::string file_name_average_relative_errors =
      basic_file_name + "average_relative_errors.data";
  fstream file_average_rel_err(file_name_average_relative_errors.c_str(),
                               std::ios_base::out | std::ios_base::trunc);

  for (unsigned int i = length_of_initial_training_phase;
       i < average_estimation_errors.size(); i++) {
    file_average_rel_err << (i - length_of_initial_training_phase) << "\t"
                         << average_estimation_errors[i] << std::endl;
  }

  // code to compute windowed average relative estimation errors
  std::string file_name_windowed_average_relative_errors =
      basic_file_name + "windowed_average_relative_errors.data";
  if (!quiet)
    cout << "File: " << file_name_windowed_average_relative_errors << endl;
  fstream file_average_windowed_rel_err(
      file_name_windowed_average_relative_errors.c_str(),
      std::ios_base::out | std::ios_base::trunc);

  assert(Runtime_Configuration::instance().getRelativeErrorWindowSize() > 0);
  for (unsigned int i = length_of_initial_training_phase;
       i < relative_errors_.size(); i++) {
    double sum = 0;
    if ((int)i < (int)relative_errors_.size() -
                     (int)Runtime_Configuration::instance()
                         .getRelativeErrorWindowSize()) {
      for (unsigned int j = 0;
           j < Runtime_Configuration::instance().getRelativeErrorWindowSize();
           j++) {
        sum += abs(relative_errors_[i + j]);
      }
    }
    file_average_windowed_rel_err
        << (i - length_of_initial_training_phase) << "\t"
        << sum / double(Runtime_Configuration::instance()
                            .getRelativeErrorWindowSize())
        << std::endl;  // compute average estimation error after each execution
                       // of the algorithm, so we can plot it over "time"
  }

  // output report for algorithm
  //	cout <<
  //"=========================================================================="
  //<< endl
  //		  << "Report for Algorithm '" << algorithm_name << "'" << endl
  //		  << "Average Relative Error: " <<
  // sum/double(relative_errors_.size()) << endl
  //		  << "Number of Recomputations: " << number_of_recomputations_
  //<<
  // endl
  //		  << "Number of Decisions for this Algorithm: " <<
  // number_of_decisions_for_this_algorithm_ << endl
  //		  << "Total Execution Time of this Algorithm: " <<
  // total_execution_time_ << "ns" << endl
  //		  //<<
  //"**************************************************************************"
  //		  << endl;
  cout << this->getReport(operation_name, algorithm_name) << endl;

  return true;
}

double AlgorithmStatistics::getAverageRelativeError() const {
  double sum = 0;
  unsigned int num_valid_samples = 0;
  unsigned int length_of_initial_training_phase =
      Runtime_Configuration::instance()
          .getTrainingLength();  // Configuration::period_for_periodic_recomputation;
  for (unsigned int i = length_of_initial_training_phase + 1;
       i < relative_errors_.size(); i++) {
    if (relative_errors_[i] != -1) {
      sum += abs(relative_errors_[i]);
      // std::cout << relative_errors_[i] << std::endl;
      num_valid_samples++;
    }
  }
  return sum / num_valid_samples;
}

std::string AlgorithmStatistics::getReport(const std::string& operation_name,
                                           const std::string& algorithm_name,
                                           const std::string indent_str) const {
  //	double sum=0;
  //        unsigned int num_valid_samples = 0;
  //	unsigned int
  // length_of_initial_training_phase=Runtime_Configuration::instance().getTrainingLength();
  ////Configuration::period_for_periodic_recomputation;
  //	for(unsigned int
  // i=length_of_initial_training_phase+1;i<relative_errors_.size();i++){
  //            if(relative_errors_[i]!=-1){
  //		sum+=abs(relative_errors_[i]);
  //                //std::cout << relative_errors_[i] << std::endl;
  //                num_valid_samples++;
  //            }
  //	}

  std::vector<MeasuredTime> measured_times =
      this->executionHistory_.getColumnMeasurements();
  double sum_exec_times = 0;
  for (unsigned int i = 0; i < measured_times.size(); i++) {
    sum_exec_times += abs(measured_times[i].getTimeinNanoseconds());
  }
  double avg_exec_time = sum_exec_times / (measured_times.size());

  stringstream ss;
  // report for algorithm
  ss << indent_str << "========================================================"
                      "=================="
     << endl
     << indent_str << "\t"
     << "Report for Algorithm '" << algorithm_name << "'" << endl
     << indent_str << "\t"
     << "Average Relative Error: " << getAverageRelativeError()
     << endl  // sum/num_valid_samples << endl
              // //sum/double(relative_errors_.size()) << endl
     << indent_str << "\t"
     << "Number of Recomputations: " << number_of_recomputations_ << endl
     << indent_str << "\t"
     << "Number of Decisions for this Algorithm: "
     << number_of_decisions_for_this_algorithm_ << endl
     << indent_str << "\t"
     << "Average Execution Time for this Algorithm: "
     << avg_exec_time / (1000 * 1000) << "ms" << endl
     << indent_str << "\t"
     << "Total Execution Time of this Algorithm: "
     << total_execution_time_ / (1000 * 1000 * 1000) << "s" << endl;
  //<<
  //"**************************************************************************"
  //<< endl;
  return ss.str();
}

bool AlgorithmStatistics::storeFeatureVectors(const std::string& path) {
  if (!quiet)
    std::cout << "Store performance model to '" << path << "' ["
              << executionHistory_.feature_values_.size() << " observations]"
              << std::endl;
  std::ofstream outfile(path.c_str(),
                        std::ios_base::binary | std::ios_base::out);

  boost::archive::binary_oarchive oa(outfile);
  std::vector<Tuple> feature_values(executionHistory_.feature_values_.begin(),
                                    executionHistory_.feature_values_.end());
  std::vector<EstimatedTime> estimated_execution_times(
      executionHistory_.estimated_times_.begin(),
      executionHistory_.estimated_times_.end());
  std::vector<MeasuredTime> measured_execution_times(
      executionHistory_.measured_times_.begin(),
      executionHistory_.measured_times_.end());

  oa << feature_values;
  oa << estimated_execution_times;
  oa << measured_execution_times;

  outfile.flush();
  outfile.close();

  return true;
}

bool AlgorithmStatistics::loadFeatureVectors(const std::string& path) {
  if (!boost::filesystem::exists(path)) {
    std::cout << "Error loading performance model: path '" << path
              << "' does not exist!" << std::endl;
    return false;
  }

  std::ifstream infile(path.c_str(), std::ios_base::binary | std::ios_base::in);
  boost::archive::binary_iarchive ia(infile);

  // load vector representation
  std::vector<Tuple> feature_values;
  std::vector<EstimatedTime> estimated_execution_times;
  std::vector<MeasuredTime> measured_execution_times;

  ia >> feature_values;
  ia >> estimated_execution_times;
  ia >> measured_execution_times;

  infile.close();

  executionHistory_.feature_values_.clear();
  executionHistory_.estimated_times_.clear();
  executionHistory_.measured_times_.clear();

  // insert data from vector in ring buffer
  executionHistory_.feature_values_.insert(
      executionHistory_.feature_values_.begin(), feature_values.begin(),
      feature_values.end());
  executionHistory_.estimated_times_.insert(
      executionHistory_.estimated_times_.begin(),
      estimated_execution_times.begin(), estimated_execution_times.end());
  executionHistory_.measured_times_.insert(
      executionHistory_.measured_times_.begin(),
      measured_execution_times.begin(), measured_execution_times.end());

  if (executionHistory_.feature_values_.size() <
      core::Runtime_Configuration::instance().getTrainingLength()) {
    HYPE_WARNING("Insufficient number of data points for '"
                     << path << "', cannot compute cost model!",
                 std::cout);
    return false;
  }

  std::cout << "Successfully load data for cost model '" << path << "' ["
            << executionHistory_.feature_values_.size() << " observations]"
            << std::endl;
  return true;
}

AlgCostModelIdentifier::AlgCostModelIdentifier()
    : external_alg_name(), pd_t(), pd_m() {}

bool operator==(AlgCostModelIdentifier a, AlgCostModelIdentifier b) {
  if (a.external_alg_name == b.external_alg_name && a.pd_m == b.pd_m &&
      a.pd_t == a.pd_t) {
    return true;
  } else {
    return false;
  }
}

bool operator<(AlgCostModelIdentifier a, AlgCostModelIdentifier b) {
  if (a.external_alg_name < b.external_alg_name) {
    return true;
  }
  if (a.external_alg_name == b.external_alg_name && a.pd_m < b.pd_m) {
    return true;
  }
  if (a.external_alg_name == b.external_alg_name && a.pd_m == b.pd_m &&
      a.pd_t < a.pd_t) {
    return true;
  }
  return false;
}

AlgorithmStatisticsManager::AlgorithmStatisticsManager() {}
AlgorithmStatisticsManager& AlgorithmStatisticsManager::instance() {
  static AlgorithmStatisticsManager alg_stat_man;
  return alg_stat_man;
}

AlgorithmStatisticsPtr AlgorithmStatisticsManager::getAlgorithmStatistics(
    const DeviceSpecification& dev_spec, const std::string& alg_name) {
  AlgCostModelIdentifier alg_cost_id;
  alg_cost_id.external_alg_name = util::toExternallAlgName(alg_name);
  alg_cost_id.pd_m = dev_spec.getMemoryID();
  alg_cost_id.pd_t = dev_spec.getDeviceType();
  AlgorithmStatisticsMap::iterator it;
  it = alg_statistics_.find(alg_cost_id);
  if (it != alg_statistics_.end()) {
    // if(!it->second) it->second=AlgorithmStatisticsPtr(new
    // AlgorithmStatistics());
    return it->second;
  } else {
    AlgorithmStatisticsPtr alg_stat(new AlgorithmStatistics());
    alg_statistics_.insert(std::make_pair(alg_cost_id, alg_stat));
    return alg_stat;
  }
}

}  // end namespace core
}  // end namespace hype
