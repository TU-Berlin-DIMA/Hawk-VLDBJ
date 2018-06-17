

#include <list>

#include <boost/thread/lock_guard.hpp>
#include <config/global_definitions.hpp>
#include <core/scheduling_decision.hpp>
#include <query_processing/virtual_processing_device.hpp>
#include <util/get_name.hpp>

namespace hype {
namespace queryprocessing {
using namespace std;
using namespace core;
using namespace util;

VirtualProcessingDevice::VirtualProcessingDevice(
    const core::DeviceSpecification& dev_spec)
    : dev_spec_(dev_spec), scheduled_tasks_(), mutex_() {}

bool VirtualProcessingDevice::addRunningOperation(
    const SchedulingDecision& sched_dec) {
  boost::lock_guard<boost::mutex> lock(mutex_);
  // cout  << "VirtualProcessingDevice: add operator" << endl;
  scheduled_tasks_.push_back(sched_dec);
  return true;
}

double VirtualProcessingDevice::getEstimatedFinishingTime() const {
  boost::lock_guard<boost::mutex> lock(mutex_);
  TaskQueue::const_iterator it;
  double result = 0;
  for (it = scheduled_tasks_.begin(); it != scheduled_tasks_.end(); ++it) {
    result +=
        it->getEstimatedExecutionTimeforAlgorithm().getTimeinNanoseconds();
  }
  return result;
}

unsigned int VirtualProcessingDevice::getNumberOfRunningOperations() const {
  boost::lock_guard<boost::mutex> lock(mutex_);
  return scheduled_tasks_.size();
}

bool VirtualProcessingDevice::isIdle() const {
  boost::lock_guard<boost::mutex> lock(mutex_);
  return scheduled_tasks_.empty();
}

bool VirtualProcessingDevice::removeFinishedOperation(
    const SchedulingDecision& sched_dec) {
  // cout << "VirtualProcessingDevice: delete operator" << endl;
  boost::lock_guard<boost::mutex> lock(mutex_);
  scheduled_tasks_.remove(sched_dec);
  return true;
}

const core::DeviceSpecification&
VirtualProcessingDevice::getDeviceSpecification() const throw() {
  boost::lock_guard<boost::mutex> lock(mutex_);
  return this->dev_spec_;
}

void VirtualProcessingDevice::print() const throw() {
  //    boost::lock_guard<boost::mutex> lock(mutex_);
  cout << "Processing Device " << dev_spec_.getProcessingDeviceID() << ":"
       << endl;
  cout << "\t"
       << "Device Type: " << getName(dev_spec_.getDeviceType()) << endl;
  cout << "\t"
       << "MemoryID: " << dev_spec_.getMemoryID() << endl;
  if (dev_spec_.getDeviceType() != CPU) {
    DeviceMemoryPtr dev_mem =
        DeviceMemories::instance().getDeviceMemory(dev_spec_.getMemoryID());
    cout << "\t"
         << "Total Memory Capacity: " << dev_mem->getTotalMemoryInBytes()
         << endl;
    cout << "\t"
         << "Measured Free Memory Capacity: "
         << double(dev_spec_.getAvailableMemoryCapacity()) /
                (1024 * 1024 * 1024)
         << "GB (may or may not include access structures)" << endl;
    size_t available_memory = dev_mem->getEstimatedFreeMemoryInBytes();
    cout << "\t"
         << "Estimated Free Memory Capacity: "
         << double(available_memory) / (1024 * 1024 * 1024)
         << "GB (Note: this is from the bookkeeping mechanism considering "
            "access structures)"
         << endl;
  }
  cout << "\t"
       << "Number of Running Operations: "
       << this->getNumberOfRunningOperations() << endl;
  cout << "\t"
       << "Estimated Finishing Time: " << this->getEstimatedFinishingTime()
       << endl;
  cout << "\t"
       << "Is Idle: " << this->isIdle() << endl;
}

}  // end namespace queryprocessing
}  // end namespace hype
