
#include <config/configuration.hpp>

#include <query_processing/processing_device.hpp>

#include <core/scheduler.hpp>

#include <boost/bind.hpp>

using namespace std;

namespace hype {
namespace queryprocessing {

void Executor(hype::queryprocessing::ProcessingDevice& pd) { pd.run(); }

ProcessingDevice::ProcessingDevice(const core::DeviceSpecification& dev_spec)
    : operators_(),
      new_operator_available_(),
      operator_queue_full_(),
      operator_mutex_(),
      thread_(),
      start_timestamp_of_currently_executed_operator_(0),
      estimated_execution_time_of_currently_executed_operator_(0),
      operator_in_execution_(false),
      estimated_execution_time_for_operators_in_queue_to_complete_(0),
      total_measured_execution_time_for_all_executed_operators_(0),
      proc_dev_id_(dev_spec.getProcessingDeviceID()),
      device_operator_queue_(
          DeviceOperatorQueues::instance().getDeviceOperatorQueue(
              dev_spec.getMemoryID())) {}

void ProcessingDevice::start() {
  // thread_=boost::thread(  boost::bind( &ProcessingDevice::start,
  // boost::ref(*this) )  );
  thread_ = boost::thread(boost::bind(Executor, boost::ref(*this)));
}

void ProcessingDevice::stop() {
  thread_.interrupt();
  thread_.join();
}

bool ProcessingDevice::addOperator(
    OperatorPtr
        op) { /* \todo this method has to wait, since it is not meaningful to
                 store the whole workload (since we don't have meaningful
                 estimations at the beginning of the application)*/
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

void ProcessingDevice::run() {
  while (true) {
#ifndef HYPE_ALTERNATE_QUERY_CHOPPING
    boost::mutex::scoped_lock lock(operator_mutex_);

    while (operators_.empty()) {
      new_operator_available_.wait(lock);
    }
    if (!hype::core::quiet && hype::core::verbose && hype::core::debug)
      cout << "[Workerthread] Found work" << endl;
    // cout << "Lock: " << lock.owns_lock() << endl;
    OperatorPtr op = operators_.front();
#else
    OperatorPtr op = device_operator_queue_->getNextOperator();
#endif
    // remove estimated count from the estimation sum
    estimated_execution_time_for_operators_in_queue_to_complete_ -=
        std::max(op->getSchedulingDecision()
                     .getEstimatedExecutionTimeforAlgorithm()
                     .getTimeinNanoseconds(),
                 double(0));
    // store the estiamted execution time for current operator
    this->estimated_execution_time_of_currently_executed_operator_ =
        std::max(op->getSchedulingDecision()
                     .getEstimatedExecutionTimeforAlgorithm()
                     .getTimeinNanoseconds(),
                 double(0));
    // store timestamp of starting time for operator
    this->start_timestamp_of_currently_executed_operator_ =
        hype::core::getTimestamp();
    // set flag that processing device is now busy processing an operator
    this->operator_in_execution_ = true;
#ifndef HYPE_ALTERNATE_QUERY_CHOPPING
    lock.unlock();
#endif
    uint64_t timestamp_begin = core::getTimestamp();
    // execute Operator
    (*op)();
    uint64_t timestamp_end = core::getTimestamp();
#ifndef HYPE_ALTERNATE_QUERY_CHOPPING
    lock.lock();
#endif
    // update total processing time on this processing device
    assert(timestamp_end > timestamp_begin);
    total_measured_execution_time_for_all_executed_operators_ +=
        double(timestamp_end - timestamp_begin);
    // set flag that processing device is (at the moment) not busy processing an
    // operator
    this->operator_in_execution_ = false;
#ifndef HYPE_ALTERNATE_QUERY_CHOPPING
    // remove first element (that was the operator, which is still queued!)
    operators_.pop_front();
    // notify that operator has finished and that a new one can be added to the
    // operator queue
    operator_queue_full_.notify_all();
#else  // notify that operator has finished and that a new one can be added to
    // the operator queue
    device_operator_queue_->notify_all();
#endif
    // lock.unlock();
    try {
      if (!hype::core::quiet && hype::core::verbose && hype::core::debug)
        cout << "[Workerthread] Reached interuption point" << endl;
      boost::this_thread::interruption_point();

    } catch (boost::thread_interrupted& e) {
      if (!hype::core::quiet) cout << "Received Termination Signal" << endl;
      return;
    }
  }
}

bool ProcessingDevice::isIdle() {
  // boost::mutex::scoped_lock lock(operator_mutex_);
  boost::lock_guard<boost::mutex> lock(operator_mutex_);
  // cout << "#operators: " << operators_.size() << endl;
  return operators_.empty() && !this->operator_in_execution_;
}

double hype::queryprocessing::ProcessingDevice::getEstimatedTimeUntilIdle() {
  boost::lock_guard<boost::mutex> lock(operator_mutex_);
  return estimated_execution_time_for_operators_in_queue_to_complete_ +
         this->getEstimatedTimeUntilOperatorCompletion();
}

double hype::queryprocessing::ProcessingDevice::
    getEstimatedTimeUntilOperatorCompletion() {
  if (operator_in_execution_) {
    uint64_t current_timestamp = hype::core::getTimestamp();
    uint64_t current_processing_time =
        current_timestamp -
        this->start_timestamp_of_currently_executed_operator_;
    return std::max(
        this->estimated_execution_time_of_currently_executed_operator_ -
            double(current_processing_time),
        double(0));  // return expected remaining processing time
  } else {
    return 0;
  }
}

double hype::queryprocessing::ProcessingDevice::getTotalProcessingTime() const {
  boost::lock_guard<boost::mutex> lock(operator_mutex_);
  return total_measured_execution_time_for_all_executed_operators_;
}

ProcessingDeviceID
hype::queryprocessing::ProcessingDevice::getProcessingDeviceID() const throw() {
  return proc_dev_id_;
}

ProcessingDevicePtr getProcessingDevice(
    const hype::core::DeviceSpecification& dev) {
  //                    core::Scheduler::ProcessingDevices::Devices
  //                    virt_comp_devs_ =
  //                    core::Scheduler::instance().getProcessingDevices().getDevices();
  //                    core::Scheduler::ProcessingDevices::Devices::iterator
  //                    it;
  //
  //                    it=virt_comp_devs_.find(dev.getProcessingDeviceID());
  //                    if(it!=virt_comp_devs_.end()){
  //                         if(!(*it)->second.second.empty()) return
  //                         *((*it)->second.second.begin());
  //                    }
  //                    return ProcessingDevicePtr();

  //                    for(it=virt_comp_devs_.begin();it!=virt_comp_devs_.end();++it){
  //                        core::Scheduler::ProcessingDevices::PhysicalDevices::iterator
  //                        phy_it;
  //                        if(!(*it)->second.second.empty()) return
  //                        *((*it)->second.second.begin());
  ////
  /// for(phy_it=(*it)->second.second.begin();phy_it!=(*it)->second.second.end();++phy_it){
  ////                            (*phy_it)->stop();
  ////                        }
  //                    }

  return core::Scheduler::instance().getProcessingDevices().getProcessingDevice(
      dev.getProcessingDeviceID());

  //			static ProcessingDevice cpu;
  //			static ProcessingDevice gpu;
  //
  //			if(dev.getDeviceType()==hype::CPU){
  //				return cpu;
  //			} else if(dev.getDeviceType()==hype::GPU) {
  //				return gpu;
  //			} else {
  //				//maybe throw illegal argument exception?!
  //				return cpu; //if unkwon, just return cpu
  //			}
}

}  // end namespace queryprocessing
}  // end namespace hype
