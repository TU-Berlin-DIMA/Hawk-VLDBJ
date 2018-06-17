#include "exports.hpp"
#include "global_definitions.hpp"
#pragma once
#include <string>
namespace hype {
  namespace core {

    class Static_Configuration {
     public:
      enum {
        length_of_initial_training_phase = 10,
        length_of_history = 1000,
        period_for_periodic_recomputation = 100,
        maximal_time_where_algorithm_was_not_choosen = 8,
        maximal_retraining_length = 1,
        window_size_for_windowed_average_relative_estimation_error = 100,
        maximal_slowdown_of_non_optimal_algorithm_in_percent = 10,  //%
        ready_queue_length_of_processing_devices = 100,
        print_algorithm_statistics_report = 0,
        default_optimization_criterion = 1
      };
    };

    class HYPE_EXPORT Runtime_Configuration {
     public:
      static Runtime_Configuration& instance() throw();

      bool setHistoryLength(unsigned int history_length) throw();

      bool setRecomputationPeriod(
          unsigned int length_of_recomputation_period) throw();

      bool setRetrainingLength(unsigned int retraining_length) throw();

      bool setAlgorithmMaximalIdleTime(unsigned int max_idle_time) throw();

      bool setMaximalReadyQueueLength(
          unsigned int max_ready_queue_length_of_processing_devices) throw();

      bool setOutlinerThreshold(double threshold) throw();

      unsigned int getTrainingLength() const throw();

      unsigned int getHistoryLength() const throw();

      unsigned int getRecomputationPeriod() const throw();

      unsigned int getRetrainingLength() const throw();

      unsigned int getAlgorithmMaximalIdleTime() const throw();

      unsigned int getRelativeErrorWindowSize() const throw();
      /*! \brief returns maximal slowdown in percent*/
      unsigned int getMaximalSlowdownOfNonOptimalAlgorithm() const throw();

      unsigned int getMaximalReadyQueueLength() const throw();

      double getOutlinerThreshold() const throw();

      bool printAlgorithmStatistics() const throw();

      bool isQueryChoppingEnabled() const throw();

      bool setQueryChoppingEnabled(bool value) throw();

      bool isPullBasedQueryChoppingEnabled() const throw();

      bool setPullBasedQueryChoppingEnabled(bool value) throw();

      bool getStorePerformanceModels() const throw();

      bool setStorePerformanceModels(bool value) throw();

      bool getTrackMemoryUsage() const throw();

      bool setTrackMemoryUsage(bool value) throw();

      bool getDataPlacementDrivenOptimization() const throw();

      bool setDataPlacementDrivenOptimization(bool value) throw();

      std::string getOutputDirectory() const throw();

      void setOutputDirectory(const std::string& path);

      OptimizationCriterion getDefaultOptimizationCriterion() const throw();

      bool setDefaultOptimizationCriterion(unsigned int value) throw();

     private:
      Runtime_Configuration();
      Runtime_Configuration(const Runtime_Configuration&);
      Runtime_Configuration& operator=(const Runtime_Configuration&);

      unsigned int length_of_initial_training_phase_;
      unsigned int length_of_history_;
      unsigned int period_for_periodic_recomputation_;
      unsigned int maximal_time_where_algorithm_was_not_choosen_;
      unsigned int maximal_retraining_length_;
      unsigned int window_size_for_windowed_average_relative_estimation_error_;
      unsigned int maximal_slowdown_of_non_optimal_algorithm_in_percent_;
      unsigned int ready_queue_length_of_processing_devices_;
      bool print_algorithm_statistics_report_;
      bool enableQueryChopping_;
      unsigned int defaultOptimizationCriterion_;
      bool store_performance_models_;
      std::string output_directory_;
      bool track_memory_usage_;
      bool enable_dataplacement_driven_query_opt_;
      bool enablePullBasedQueryChopping_;
      // double outliner_threshold_in_percent_;
    };

  }  // end namespace core
}  // end namespace hype
