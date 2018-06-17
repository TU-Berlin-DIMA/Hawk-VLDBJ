
#include <boost/thread.hpp>
#include <statistics/statistics_manager.hpp>

namespace CoGaDB {
using namespace std;
boost::mutex global_statistics_manager_mutex;

StatisticsManager::StatisticsManager() : statistics_() {}

StatisticsManager& StatisticsManager::instance() {
  static StatisticsManager sm;
  return sm;
}

std::pair<bool, double> StatisticsManager::getValue(
    const std::string& metric) const {
  // we access this method from a possibly arbitray number of threads,
  // therefore, we have to guard this method with a lock
  boost::lock_guard<boost::mutex> lock(global_statistics_manager_mutex);
  Statistics::const_iterator it = statistics_.find(metric);
  if (it == statistics_.end()) {
    return make_pair(false, 0);
  } else {
    return make_pair(true, it->second);
  }
}
void StatisticsManager::addToValue(const std::string& metric, double value) {
  // we access this method from a possibly arbitray number of threads,
  // therefore, we have to guard this method with a lock
  boost::lock_guard<boost::mutex> lock(global_statistics_manager_mutex);
  Statistics::iterator it = statistics_.find(metric);
  if (it == statistics_.end()) {
    statistics_.insert(
        std::make_pair(metric, value));  // assume initialization with 0
    // if(!quiet) std::cout << "[StatisticsManager]: added variable: " << metric
    // << std::endl;
  } else {
    it->second += value;
    // if(!quiet) std::cout << "[StatisticsManager]: added value '" << value <<
    // "' to variable: " << metric << std::endl;
  }
}

std::string StatisticsManager::toString() const {
  // we access this method from a possibly arbitray number of threads,
  // therefore, we have to guard this method with a lock
  boost::lock_guard<boost::mutex> lock(global_statistics_manager_mutex);
  std::stringstream ss;
  ss << std::string(80, '=') << std::endl;
  ss << "Statistics collected by Statistics Manager:" << std::endl;
  // ss << "NUMBER_OF_EXECUTED_GPU_OPERATORS: " <<
  // getValue(NUMBER_OF_EXECUTED_GPU_OPERATORS).second << std::endl;
  // ss << "NUMBER_OF_ABORTED_GPU_OPERATORS: " <<
  // getValue(NUMBER_OF_ABORTED_GPU_OPERATORS).second << std::endl;
  Statistics::const_iterator it;
  for (it = statistics_.begin(); it != statistics_.end(); ++it) {
    ss << it->first << ": " << it->second << std::endl;
  }
  ss << std::string(80, '=') << std::endl;
  ss << "NOTE: The counters for join indexes are redundant, meaning "
     << "their costs (e.g., " << std::endl
     << "IO) are already included with the positionlists. "
     << "However, it helps us to know" << std::endl
     << "how much cost were caused by fetch joins "
     << "relative to other operators." << std::endl;
  ss << std::string(80, '=');  // << std::endl;
  return ss.str();
}

void StatisticsManager::reset() {
  // we access this method from a possibly arbitray number of threads,
  // therefore, we have to guard this method with a lock
  boost::lock_guard<boost::mutex> lock(global_statistics_manager_mutex);
  statistics_.clear();
}

}  // end namespace CoGaDB
