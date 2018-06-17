#include <assert.h>
#include <backends/gpu/stream_manager.hpp>
#include <boost/thread.hpp>

using namespace std;

namespace CoGaDB {
namespace gpu {

boost::mutex global_stream_manager_mutex;

StreamManager& StreamManager::instance() {
  boost::lock_guard<boost::mutex> lock(global_stream_manager_mutex);
  static StreamManager stream_manager;
  return stream_manager;
}
cudaStream_t* StreamManager::getStream() {
  boost::lock_guard<boost::mutex> lock(global_stream_manager_mutex);
  cudaStream_t* result = &streams_[current_stream_id_++];

  if (current_stream_id_ >= GPU_KERNEL_CONCURRENCY_LEVEL)
    current_stream_id_ = 0;

  return result;
}

GPUContextPtr StreamManager::getCudaContext(unsigned int device_id) {
/*
  boost::lock_guard<boost::mutex> lock(global_stream_manager_mutex);
  cudaStream_t* stream = &streams_[current_stream_id_++];

  if(current_stream_id_>=GPU_KERNEL_CONCURRENCY_LEVEL) current_stream_id_=0;
  gpu_context_->stream=stream;
  return this->gpu_context_;*/
/* avoid call to mgpu library in case GPU-acceleration is disabled*/
#ifdef ENABLE_GPU_ACCELERATION
  return mgpu::CreateCudaDeviceAttachStream(*this->getStream());
#else
  return GPUContextPtr();
#endif
}

StreamManager::StreamManager() : streams_(), current_stream_id_(0) {
  for (unsigned int i = 0; i < GPU_KERNEL_CONCURRENCY_LEVEL; ++i) {
    cudaError_t err = cudaStreamCreate(&streams_[i]);
    assert(err == cudaSuccess);
  }
}

StreamManager::~StreamManager() {
  for (unsigned int i = 0; i < GPU_KERNEL_CONCURRENCY_LEVEL; ++i) {
    cudaError_t err = cudaStreamDestroy(streams_[i]);
    assert(err == cudaSuccess);
  }
}

cudaStream_t* getCUDAStream() { return StreamManager::instance().getStream(); }

}  // end namespace gpu
}  // end namespace CogaDB
