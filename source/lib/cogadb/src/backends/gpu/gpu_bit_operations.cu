
#include <backends/gpu/bit_operations.hpp>
#include <backends/gpu/stream_manager.hpp>
#include <backends/gpu/util.hpp>
#include <util/column_grouping_keys.hpp>

namespace CoGaDB {

//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//
//        inline void gpuAssert(cudaError_t code, char *file, int line, bool
//        abort = true) {
//            if (code != cudaSuccess) {
//                fprintf(stderr, "GPUassert: %s %s %d\n",
//                cudaGetErrorString(code), file, line);
//                if (abort) exit(code);
//            }
//        }

template <typename T>
__global__ void bit_shift_left_kernel(T* array, size_t array_size,
                                      size_t num_bits_to_shift) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < array_size) {
    array[tid] = array[tid] << num_bits_to_shift;
    tid += blockDim.x * gridDim.x;
  }
}

template <typename T>
__global__ void bit_shift_right_kernel(T* array, size_t array_size,
                                       size_t num_bits_to_shift) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < array_size) {
    array[tid] = array[tid] >> num_bits_to_shift;
    tid += blockDim.x * gridDim.x;
  }
}

template <typename T>
__global__ void bitwise_and_kernel(T* target_array, T* source_array,
                                   size_t array_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < array_size) {
    target_array[tid] = target_array[tid] & source_array[tid];
    tid += blockDim.x * gridDim.x;
  }
}

template <typename T>
__global__ void bitwise_or_kernel(T* target_array, T* source_array,
                                  size_t array_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < array_size) {
    target_array[tid] = target_array[tid] | source_array[tid];
    tid += blockDim.x * gridDim.x;
  }
}

bool GPU_BitOperation::bit_shift(ColumnGroupingKeysPtr keys,
                                 const BitShiftParam& param) {
  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 512;

  cudaStream_t* stream = gpu::StreamManager::instance().getStream();
  if (param.op == SHIFT_BITS_LEFT) {
    bit_shift_left_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                            *stream>>>(
        keys->keys->data(), keys->keys->size(),
        //                    getPointer(*(keys->keys)),
        //                    getSize(*(keys->keys)),
        param.number_of_bits);

  } else if (param.op == SHIFT_BITS_RIGHT) {
    bit_shift_right_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                             *stream>>>(
        keys->keys->data(), keys->keys->size(),
        //                    getPointer(*(keys->keys)),
        //                    getSize(*(keys->keys)),
        param.number_of_bits);
  } else {
    COGADB_FATAL_ERROR("Unknown Bitshift Operation!", "");
    return false;
  }
  cudaError_t err = cudaStreamSynchronize(*stream);
  if (err == cudaSuccess) {
    return true;
  } else {
    gpuErrchk(err);
    return false;
  }
}

bool GPU_BitOperation::bitwise_combination(
    ColumnGroupingKeysPtr target_keys, ColumnGroupingKeysPtr source_keys,
    const BitwiseCombinationParam& param) {
  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 512;

  cudaStream_t* stream = gpu::StreamManager::instance().getStream();
  if (param.op == BITWISE_AND) {
    bitwise_and_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                         *stream>>>(target_keys->keys->data(),
                                    source_keys->keys->data(),
                                    target_keys->keys->size());
    //                    getPointer(*(target_keys->keys)),
    //                    getPointer(*(source_keys->keys)),
    //                    getSize(*(target_keys->keys))
    //                    );

  } else if (param.op == BITWISE_OR) {
    bitwise_or_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                        *stream>>>(target_keys->keys->data(),
                                   source_keys->keys->data(),
                                   target_keys->keys->size());
    //                    getPointer(*(target_keys->keys)),
    //                    getPointer(*(source_keys->keys)),
    //                    getSize(*(target_keys->keys)));
  } else {
    COGADB_FATAL_ERROR("Unknown Bitshift Operation!", "");
    return false;
  }
  cudaError_t err = cudaStreamSynchronize(*stream);
  if (err == cudaSuccess) {
    return true;
  } else {
    gpuErrchk(err);
    return false;
  }
}

};  // end namespace CoGaDB
