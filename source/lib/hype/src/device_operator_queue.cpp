/*
 * File:   device_operator_queue.hpp
 * Author: sebastian
 *
 * Created on 21. September 2014, 19:07
 */

#include <iostream>
#include <query_processing/device_operator_queue.hpp>

namespace hype {
namespace queryprocessing {

using namespace std;

DeviceOperatorQueue::DeviceOperatorQueue(const ProcessingDeviceMemoryID mem_id)
    : operators_(),
      new_operator_available_(),
      operator_queue_full_(),
      operator_mutex_(),
      mem_id_(mem_id),
      estimated_execution_time_for_operators_in_queue_to_complete_(0) {}
// Note that this method locks the scoped lock 'lock', and if this function
// returns, 'lock' is locked
OperatorPtr
DeviceOperatorQueue::getNextOperator() {  // boost::mutex::scoped_lock& lock){
  boost::mutex::scoped_lock lock(operator_mutex_);
  while (operators_.empty()) {
    new_operator_available_.wait(lock);
  }
  OperatorPtr op = operators_.front();
  // delete this operator
  operators_.pop_front();
  return op;
}
bool DeviceOperatorQueue::addOperator(OperatorPtr op) {
  assert(op != NULL);
  // boost::mutex::scoped_lock lock(operator_mutex_);
  // boost::lock_guard<boost::mutex> lock(operator_mutex_);
  boost::mutex::scoped_lock lock(operator_mutex_);
  while (operators_.size() >
         hype::core::Runtime_Configuration::instance()
             .getMaximalReadyQueueLength()) {  // 10 //100) {
    operator_queue_full_.wait(lock);
  }
  operators_.push_back(op);
  estimated_execution_time_for_operators_in_queue_to_complete_ +=
      std::max(op->getSchedulingDecision()
                   .getEstimatedExecutionTimeforAlgorithm()
                   .getTimeinNanoseconds(),
               double(0));
  if (!hype::core::quiet) {
    cout << "new waiting time for Algorithm "
         << op->getSchedulingDecision().getNameofChoosenAlgorithm() << ": "
         << estimated_execution_time_for_operators_in_queue_to_complete_ << "ns"
         << endl;
    cout << "number of queued operators: " << this->operators_.size()
         << " for Algorithm "
         << op->getSchedulingDecision().getNameofChoosenAlgorithm() << endl;
  }
  new_operator_available_.notify_all();
  return true;
}

void DeviceOperatorQueue::notify_all() { operator_queue_full_.notify_all(); }

DeviceOperatorQueues::DeviceOperatorQueues() : map_() {}

DeviceOperatorQueues& DeviceOperatorQueues::instance() {
  static DeviceOperatorQueues queues;
  return queues;
}

DeviceOperatorQueuePtr DeviceOperatorQueues::getDeviceOperatorQueue(
    const ProcessingDeviceMemoryID mem_id) {
  DeviceOperatorQueueMap::const_iterator cit = map_.find(mem_id);
  if (cit != map_.end()) {
    return cit->second;
  } else {
    DeviceOperatorQueuePtr queue(new DeviceOperatorQueue(mem_id));
    map_[mem_id] = queue;
    return queue;
  }
}

}  // end namespace queryprocessing
}  // end namespace hype
