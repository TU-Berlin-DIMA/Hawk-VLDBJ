#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <core/gpu_admission_control.hpp>

AdmissionControl::AdmissionControl() {
  size_t avail;
  size_t total;
  cudaMemGetInfo(&avail, &total);
  std::cout << "Device memory available for admission control: " << avail
            << std::endl;
  availableMemory = avail;
}

AdmissionControl& AdmissionControl::instance() {
  static AdmissionControl inst;
  return inst;
}

void AdmissionControl::requestMemory(size_t requestDataSize) {
  boost::unique_lock<boost::mutex> lock(mutex_);
  while (requestDataSize > availableMemory) {
    std::cout << "request: " << requestDataSize
              << " available: " << availableMemory << " go to sleep."
              << std::endl;

    cond_.wait(lock);
  }

  std::cout << "request: " << requestDataSize
            << " available: " << availableMemory << " gets memory."
            << std::endl;

  availableMemory -= requestDataSize;
}

void AdmissionControl::releaseMemory(size_t releaseDataSize) {
  {
    boost::unique_lock<boost::mutex> lock(mutex_);
    std::cout << "free " << releaseDataSize << "." << std::endl;
    availableMemory += releaseDataSize;
  }
  cond_.notify_all();
}
