#include <stdlib.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

class AdmissionControl {
 private:
  size_t availableMemory;

  boost::mutex mutex_;
  boost::condition_variable cond_;

  AdmissionControl();

 public:
  static AdmissionControl& instance();

  void requestMemory(size_t requestDataSize);

  void releaseMemory(size_t releaseDataSize);
};
