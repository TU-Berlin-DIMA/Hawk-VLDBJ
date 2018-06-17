
#ifndef STREAM_MANAGER_HPP
#define STREAM_MANAGER_HPP

//#ifndef __CUDACC__
//	#error "cannot compile header file __FILE__: you have to use nvcc!"
//#endif

#include <cuda_runtime.h>
#include <vector>
// for moderngpu device reference
#include <util/mgpucontext.h>

namespace CoGaDB {
  namespace gpu {

    const unsigned int GPU_KERNEL_CONCURRENCY_LEVEL =
        16;  // highest concurrency that modern CUDA cards support
    // typedef mgpu::ContextPtr GPUContextPtr;
    typedef mgpu::ContextPtr GPUContextPtr;

    class StreamManager {
     public:
      static StreamManager& instance();
      cudaStream_t* getStream();
      GPUContextPtr getCudaContext(unsigned int device_id);

     private:
      StreamManager();
      StreamManager(const StreamManager&);
      StreamManager& operator=(const StreamManager&);
      ~StreamManager();
      // std::vector<cudaStream_t*> streams_;
      cudaStream_t streams_[GPU_KERNEL_CONCURRENCY_LEVEL];
      unsigned int current_stream_id_;
      GPUContextPtr gpu_context_;
    };
    cudaStream_t* getCUDAStream();

  }  // end namespace gpu
}  // end namespace CogaDB

#endif /* STREAM_MANAGER_HPP */
