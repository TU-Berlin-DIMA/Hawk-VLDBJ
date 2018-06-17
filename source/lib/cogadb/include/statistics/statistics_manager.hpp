#pragma once
#include <core/global_definitions.hpp>
#include <map>

namespace CoGaDB {

  //    enum StatisticMetric {
  //        NUMBER_OF_EXECUTED_GPU_OPERATORS, NUMBER_OF_ABORTED_GPU_OPERATORS
  //    };

  class StatisticsManager {
   public:
    typedef std::map<std::string, double> Statistics;

    static StatisticsManager& instance();

    std::pair<bool, double> getValue(const std::string& metric) const;
    void addToValue(const std::string&, double);
    std::string toString() const;
    void reset();

   private:
    // instantiation only once by instance method
    StatisticsManager();
    // no copying allowed
    StatisticsManager(const StatisticsManager&);
    StatisticsManager& operator=(const StatisticsManager&);
    Statistics statistics_;
  };

#define COGADB_EXECUTE_GPU_OPERATOR(name_of_gpu_operator)                      \
  StatisticsManager::instance().addToValue("NUMBER_OF_EXECUTED_GPU_OPERATORS", \
                                           1);                                 \
  StatisticsManager::instance().addToValue(                                    \
      std::string("NUMBER_OF_EXECUTED_") + name_of_gpu_operator +              \
          "_OPERATORS",                                                        \
      1);                                                                      \
  uint64_t begin_timestamp_aborted_gpu_operator_time = getTimestamp();

#define COGADB_ABORT_GPU_OPERATOR(name_of_gpu_operator)                        \
  {                                                                            \
    COGADB_WARNING("GPU Operator for " << name_of_gpu_operator                 \
                                       << "! Falling back to CPU operator...", \
                   "");                                                        \
    StatisticsManager::instance().addToValue(                                  \
        "NUMBER_OF_ABORTED_GPU_OPERATORS", 1);                                 \
    StatisticsManager::instance().addToValue(                                  \
        std::string("NUMBER_OF_ABORTED_") + name_of_gpu_operator +             \
            "_OPERATORS",                                                      \
        1);                                                                    \
    uint64_t end_timestamp_aborted_gpu_operator_time = getTimestamp();         \
    StatisticsManager::instance().addToValue(                                  \
        "TOTAL_LOST_TIME_IN_NS_DUE_TO_ABORTED_GPU_OPERATORS",                  \
        double(end_timestamp_aborted_gpu_operator_time -                       \
               begin_timestamp_aborted_gpu_operator_time));                    \
    StatisticsManager::instance().addToValue(                                  \
        std::string("LOST_TIME_IN_NS_DUE_TO_ABORTED") + name_of_gpu_operator + \
            "_OPERATOR",                                                       \
        double(end_timestamp_aborted_gpu_operator_time -                       \
               begin_timestamp_aborted_gpu_operator_time));                    \
    has_aborted_ = true;                                                       \
  }

}  // end namespace CoGaDB
