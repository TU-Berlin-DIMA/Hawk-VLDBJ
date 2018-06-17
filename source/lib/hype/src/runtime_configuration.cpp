
#include <config/configuration.hpp>
#include <config/global_definitions.hpp>
#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>

using namespace std;

namespace hype {
namespace core {

std::string map_environment_variable_name_to_option_name(const std::string& s) {
  // STEMOD prefixed variables are deprecated and will be removed in future
  // versions
  if (s == "STEMOD_LENGTH_OF_TRAININGPHASE" ||
      s == "HYPE_LENGTH_OF_TRAININGPHASE") {
    return "length_of_trainingphase";
  } else if (s == "STEMOD_HISTORY_LENGTH" || s == "HYPE_HISTORY_LENGTH") {
    return "history_length";
  } else if (s == "STEMOD_RECOMPUTATION_PERIOD" ||
             s == "HYPE_RECOMPUTATION_PERIOD") {
    return "recomputation_period";
  } else if (s == "STEMOD_ALGORITHM_MAXIMAL_IDLE_TIME" ||
             s == "HYPE_ALGORITHM_MAXIMAL_IDLE_TIME") {
    return "algorithm_maximal_idle_time";
  } else if (s == "STEMOD_RETRAINING_LENGTH" || s == "HYPE_RETRAINING_LENGTH") {
    return "retraining_length";
  } else if (s == "STEMOD_MAXIMAL_SLOWDOWN_OF_NON_OPTIMAL_ALGORITHM_IN_"
                  "PERCENT" ||
             s == "HYPE_MAXIMAL_SLOWDOWN_OF_NON_OPTIMAL_ALGORITHM_IN_PERCENT") {
    return "maximal_slowdown_of_non_optimal_algorithm";
  } else if (s == "STEMOD_READY_QUEUE_LENGTH" ||
             s == "HYPE_READY_QUEUE_LENGTH") {
    return "ready_queue_length";
  } else if (s == "STEMOD_DEFAULT_OPTIMIZATION_CRITERION" ||
             s == "HYPE_DEFAULT_OPTIMIZATION_CRITERION") {
    return "default_optimization_criterion";
  } else if (s == "STEMOD_PRINT_ALGORITHM_STATISTICS" ||
             s == "HYPE_PRINT_ALGORITHM_STATISTICS") {
    return "print_algorithm_statistics";
  } else if (s == "HYPE_REUSE_PERFORMANCE_MODELS") {
    return "reuse_performance_models";
  } else if (s == "HYPE_TRACK_MEMORY_USAGE") {
    return "track_memory_usage";

  } else {
    return std::string();  // return empty string
  }
}
Runtime_Configuration::Runtime_Configuration()
    : length_of_initial_training_phase_(
          Static_Configuration::length_of_initial_training_phase),
      length_of_history_(Static_Configuration::length_of_history),
      period_for_periodic_recomputation_(
          Static_Configuration::period_for_periodic_recomputation),
      maximal_time_where_algorithm_was_not_choosen_(
          Static_Configuration::maximal_time_where_algorithm_was_not_choosen),
      maximal_retraining_length_(
          Static_Configuration::maximal_retraining_length),
      window_size_for_windowed_average_relative_estimation_error_(
          Static_Configuration::
              window_size_for_windowed_average_relative_estimation_error),
      maximal_slowdown_of_non_optimal_algorithm_in_percent_(
          Static_Configuration::
              maximal_slowdown_of_non_optimal_algorithm_in_percent),
      ready_queue_length_of_processing_devices_(
          Static_Configuration::ready_queue_length_of_processing_devices),
      print_algorithm_statistics_report_(
          Static_Configuration::print_algorithm_statistics_report),
      enableQueryChopping_(false),
      defaultOptimizationCriterion_(
          Static_Configuration::default_optimization_criterion),
      store_performance_models_(false),
      output_directory_("output"),
      track_memory_usage_(false),
      enable_dataplacement_driven_query_opt_(false),
      enablePullBasedQueryChopping_(false) {
  // Declare the supported options.
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "length_of_trainingphase", boost::program_options::value<unsigned int>(),
      "set the number algorithms executions to complete training")(
      "history_length", boost::program_options::value<unsigned int>(),
      "set the number of measurement pairs that are kept in the history "
      "(important for precision of approximation functions)")(
      "recomputation_period", boost::program_options::value<unsigned int>(),
      "set the number of algorithm executions to trigger recomputation")(
      "algorithm_maximal_idle_time",
      boost::program_options::value<unsigned int>(),
      "set the number of algorithm executions to trigger recomputation")(
      "retraining_length", boost::program_options::value<unsigned int>(),
      "set the number of algorithm executions to trigger recomputation")(
      "maximal_slowdown_of_non_optimal_algorithm",
      boost::program_options::value<unsigned int>(),
      "set the maximal slowdown in percent, an algorithm may have, and still "
      "is considered for execution (Throughput optimization)")(
      "ready_queue_length", boost::program_options::value<unsigned int>(),
      "set the queue length of operators that may be concurrently scheduled "
      "(then, clients are blocked on a processing device)")(
      "default_optimization_criterion",
      boost::program_options::value<unsigned int>(),
      "set the criterion which should be used for optimization (as long as "
      "none is choosen directly)")(
      "print_algorithm_statistics", boost::program_options::value<bool>(),
      "set the mode for storing algorithm statistics, true means algorithms "
      "statistics are dumped to the output directory of HyPE, false disables "
      "the feature (default)")(
      "reuse_performance_models", boost::program_options::value<bool>(),
      "set the mode for the collected observations that serve as basis for the "
      "performance models, true means collected observations are stored on "
      "disk and are reused at system startup to skip the training phase, false "
      "disables the feature (default)")(
      "track_memory_usage", boost::program_options::value<bool>(),
      "enables or disables book keeping of used memory of processing devices "
      "(disabled by default)");

  boost::program_options::variables_map vm;
  std::fstream config_file("hype.conf");
  if (!config_file.good()) {
    if (!quiet)
      cout << "[HyPE]: No Configuration File 'hype.conf' found! Parsing "
              "environment variables..."
           << endl;
    boost::program_options::store(
        boost::program_options::parse_environment(
            desc, map_environment_variable_name_to_option_name),
        vm);
    // return; //don't parse config file, if stream is not okay
  } else {
    if (!quiet)
      cout << "[HyPE]: Parsing Configuration File 'hype.conf'..." << endl;
    boost::program_options::store(
        boost::program_options::parse_config_file(config_file, desc), vm);
  }

  boost::program_options::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    // return 1;
  }

  if (vm.count("length_of_trainingphase")) {
    if (!quiet)
      cout << "Length of Trainingphase: "
           << vm["length_of_trainingphase"].as<unsigned int>() << "\n";
    length_of_initial_training_phase_ =
        vm["length_of_trainingphase"].as<unsigned int>();
  } else {
    if (!quiet)
      cout << "length_of_trainingphase was not specified, using default "
              "value...\n";
  }

  if (vm.count("history_length")) {
    if (!quiet)
      cout << "History Length: " << vm["history_length"].as<unsigned int>()
           << "\n";
    length_of_history_ = vm["history_length"].as<unsigned int>();
  } else {
    if (!quiet)
      cout << "history_length was not specified, using default value...\n";
  }

  if (vm.count("recomputation_period")) {
    if (!quiet)
      cout << "Recomputation Period: "
           << vm["recomputation_period"].as<unsigned int>() << "\n";
    period_for_periodic_recomputation_ =
        vm["recomputation_period"].as<unsigned int>();
  } else {
    if (!quiet)
      cout
          << "recomputation_period was not specified, using default value...\n";
  }

  if (vm.count("algorithm_maximal_idle_time")) {
    if (!quiet)
      cout << "algorithm_maximal_idle_time: "
           << vm["algorithm_maximal_idle_time"].as<unsigned int>() << "\n";
    maximal_time_where_algorithm_was_not_choosen_ =
        vm["algorithm_maximal_idle_time"].as<unsigned int>();
  } else {
    if (!quiet)
      cout << "algorithm_maximal_idle_time was not specified, using default "
              "value...\n";
  }

  if (vm.count("retraining_length")) {
    if (!quiet)
      cout << "Retraining Length: "
           << vm["retraining_length"].as<unsigned int>() << "\n";
    maximal_retraining_length_ = vm["retraining_length"].as<unsigned int>();
  } else {
    if (!quiet)
      cout << "retraining_length was not specified, using default value...\n";
  }

  if (vm.count("maximal_slowdown_of_non_optimal_algorithm")) {
    if (!quiet)
      cout << "maximal_slowdown_of_non_optimal_algorithm: "
           << vm["maximal_slowdown_of_non_optimal_algorithm"].as<unsigned int>()
           << "\n";
    maximal_slowdown_of_non_optimal_algorithm_in_percent_ =
        vm["maximal_slowdown_of_non_optimal_algorithm"].as<unsigned int>();
  } else {
    if (!quiet)
      cout << "maximal_slowdown_of_non_optimal_algorithm was not specified, "
              "using default value...\n";
  }

  if (vm.count("ready_queue_length")) {
    if (!quiet)
      cout << "Ready Queue Length: "
           << vm["ready_queue_length"].as<unsigned int>() << "\n";
    ready_queue_length_of_processing_devices_ =
        vm["ready_queue_length"].as<unsigned int>();
  } else {
    if (!quiet)
      cout << "ready_queue_length was not specified, using default value...\n";
  }

  if (vm.count("default_optimization_criterion")) {
    if (!quiet)
      cout << "Default Optimization Criterion: "
           << vm["default_optimization_criterion"].as<unsigned int>() << "\n";
    defaultOptimizationCriterion_ =
        vm["default_optimization_criterion"].as<unsigned int>();
  } else {
    if (!quiet)
      cout << "default_optimization_criterion was not specified, using default "
              "value...\n";
  }

  if (vm.count("print_algorithm_statistics")) {
    if (!quiet) cout << "Print Algorithm Statistics: ";
    if (vm["print_algorithm_statistics"].as<bool>()) {
      if (!quiet) cout << "yes";
    } else {
      if (!quiet) cout << "no";
    }
    if (!quiet) cout << endl;
    print_algorithm_statistics_report_ =
        vm["print_algorithm_statistics"].as<bool>();
  } else {
    if (!quiet)
      cout << "print_algorithm_statistics was not specified, using default "
              "value (no)...\n";
  }

  if (vm.count("reuse_performance_models")) {
    if (!quiet) cout << "Reuse Performance Models: ";
    if (vm["reuse_performance_models"].as<bool>()) {
      if (!quiet) cout << "yes";
    } else {
      if (!quiet) cout << "no";
    }
    if (!quiet) cout << endl;
    this->store_performance_models_ = vm["reuse_performance_models"].as<bool>();
  } else {
    if (!quiet)
      cout << "reuse_performance_models was not specified, using default "
              "value...\n";
  }

  if (vm.count("track_memory_usage")) {
    if (!quiet) cout << "Track Memory Usage of Processing Devices: ";
    if (vm["track_memory_usage"].as<bool>()) {
      if (!quiet) cout << "yes";
    } else {
      if (!quiet) cout << "no";
    }
    if (!quiet) cout << endl;
    this->track_memory_usage_ = vm["track_memory_usage"].as<bool>();
  } else {
    if (!quiet)
      cout << "track_memory_usage was not specified, using default value...\n";
  }

  /*
  if (vm.count("")) {
           cout << ": "
   << vm[""].as<unsigned int>() << "\n";
          =vm[""].as<unsigned int>();
  } else {
           cout << " was not specified, using default value...\n";
  }*/
}

Runtime_Configuration& Runtime_Configuration::instance() throw() {
  static Runtime_Configuration run_config;
  return run_config;
}

bool Runtime_Configuration::setHistoryLength(
    unsigned int history_length) throw() {
  this->length_of_history_ = history_length;
  return true;
}

bool Runtime_Configuration::setRecomputationPeriod(
    unsigned int length_of_recomputation_period) throw() {
  this->period_for_periodic_recomputation_ = length_of_recomputation_period;
  return true;
}

bool Runtime_Configuration::setRetrainingLength(
    unsigned int retraining_length) throw() {
  this->maximal_retraining_length_ = retraining_length;
  return true;
}

bool Runtime_Configuration::setAlgorithmMaximalIdleTime(
    unsigned int max_idle_time) throw() {
  this->maximal_time_where_algorithm_was_not_choosen_ = max_idle_time;
  return true;
}

bool Runtime_Configuration::setMaximalReadyQueueLength(
    unsigned int max_ready_queue_length_of_processing_devices) throw() {
  this->ready_queue_length_of_processing_devices_ =
      max_ready_queue_length_of_processing_devices;
  return true;
}

bool Runtime_Configuration::setOutlinerThreshold(double threshold) throw() {
  cout << threshold << endl;
  return false;
}

unsigned int Runtime_Configuration::getTrainingLength() const throw() {
  return length_of_initial_training_phase_;
}

unsigned int Runtime_Configuration::getHistoryLength() const throw() {
  return this->length_of_history_;
}

unsigned int Runtime_Configuration::getRecomputationPeriod() const throw() {
  return this->period_for_periodic_recomputation_;
}

unsigned int Runtime_Configuration::getRetrainingLength() const throw() {
  return this->maximal_retraining_length_;
}

unsigned int Runtime_Configuration::getAlgorithmMaximalIdleTime() const
    throw() {
  return this->maximal_time_where_algorithm_was_not_choosen_;
}

unsigned int Runtime_Configuration::getRelativeErrorWindowSize() const throw() {
  return this->window_size_for_windowed_average_relative_estimation_error_;
}

unsigned int Runtime_Configuration::getMaximalSlowdownOfNonOptimalAlgorithm()
    const throw() {
  return this->maximal_slowdown_of_non_optimal_algorithm_in_percent_;
}

unsigned int Runtime_Configuration::getMaximalReadyQueueLength() const throw() {
  return this->ready_queue_length_of_processing_devices_;
}

double Runtime_Configuration::getOutlinerThreshold() const throw() {
  return double(0);
}

bool Runtime_Configuration::printAlgorithmStatistics() const throw() {
  return this->print_algorithm_statistics_report_;
}

bool Runtime_Configuration::isQueryChoppingEnabled() const throw() {
  return this->enableQueryChopping_;
}

bool Runtime_Configuration::setQueryChoppingEnabled(bool value) throw() {
  return this->enableQueryChopping_ = value;
}

bool Runtime_Configuration::isPullBasedQueryChoppingEnabled() const throw() {
  return this->enablePullBasedQueryChopping_;
}

bool Runtime_Configuration::setPullBasedQueryChoppingEnabled(
    bool value) throw() {
  this->enablePullBasedQueryChopping_ = value;
  return true;
}

bool Runtime_Configuration::getStorePerformanceModels() const throw() {
  return this->store_performance_models_;
}

bool Runtime_Configuration::setStorePerformanceModels(bool value) throw() {
  this->store_performance_models_ = value;
  return true;
}

bool Runtime_Configuration::getTrackMemoryUsage() const throw() {
  return this->track_memory_usage_;
}

bool Runtime_Configuration::setTrackMemoryUsage(bool value) throw() {
  this->track_memory_usage_ = value;
  return true;
}

bool Runtime_Configuration::getDataPlacementDrivenOptimization() const throw() {
  return this->enable_dataplacement_driven_query_opt_;
}

bool Runtime_Configuration::setDataPlacementDrivenOptimization(
    bool value) throw() {
  this->enable_dataplacement_driven_query_opt_ = value;
  return true;
}

std::string Runtime_Configuration::getOutputDirectory() const throw() {
  return this->output_directory_;
}

void Runtime_Configuration::setOutputDirectory(const std::string& path) {
  this->output_directory_ = path;
}

OptimizationCriterion Runtime_Configuration::getDefaultOptimizationCriterion()
    const throw() {
  return static_cast<OptimizationCriterion>(
      this->defaultOptimizationCriterion_);
}

bool Runtime_Configuration::setDefaultOptimizationCriterion(
    unsigned int value) throw() {
  this->defaultOptimizationCriterion_ = value;
  return true;
}

}  // end namespace core
}  // end namespace hype
