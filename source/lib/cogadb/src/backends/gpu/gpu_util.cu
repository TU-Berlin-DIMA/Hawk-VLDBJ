

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "core/column.hpp"
#include "util/hardware_detector.hpp"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cub/device/device_scan.cuh>
#include <cub/util_allocator.cuh>

#include <backends/gpu/stream_manager.hpp>
#include <backends/gpu/util.hpp>
#include <core/base_column.hpp>
#include <core/global_definitions.hpp>

using namespace cub;

namespace CoGaDB {

template <typename T>
void GPU_Util<T>::print_device_array(T* pointer, size_t length) {
  thrust::device_vector<T> dev_vec(pointer, pointer + length);
  std::cout << dev_vec[0];
  for (int i = 1; i < length; i++) {
    std::cout << ", " << dev_vec[i];
  }
  std::cout << std::endl;
}

template <typename T>
bool GPU_Util<T>::generateConstantSequence(
    T* dest_column, size_t num_elements, T value,
    const ProcessorSpecification& proc_spec) {
  try {
    thrust::fill(thrust::device_pointer_cast(dest_column),
                 thrust::device_pointer_cast(dest_column + num_elements),
                 value);
  } catch (std::bad_alloc& e) {
    COGADB_ERROR(e.what(), "");
    return false;
  }
  return true;
}

template <typename T>
bool GPU_Util<T>::generateAscendingSequence(
    T* dest_column, size_t num_elements, T begin_value,
    const ProcessorSpecification& proc_spec) {
  try {
    thrust::sequence(thrust::device_pointer_cast(dest_column),
                     thrust::device_pointer_cast(dest_column + num_elements),
                     begin_value, T(1));
  } catch (std::bad_alloc& e) {
    COGADB_ERROR(e.what(), "");
    return false;
  }
  return true;
}

template <typename T>
__global__ void gather_kernel(T* dest_column, T* source_column, TID* tids,
                              size_t number_of_tids) {
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < number_of_tids) {
    dest_column[tid] = source_column[tids[tid]];
    tid += blockDim.x * gridDim.x;
  }
}

template <typename T>
bool GPU_Util<T>::gather(T* dest_column, T* source_column, PositionListPtr tids,
                         const GatherParam& param) {
  if (!tids) return false;
  if (tids->size() == 0) return true;

  size_t number_of_blocks = 512;
  size_t number_of_threads_per_block = 1024;

  cudaStream_t* stream = gpu::StreamManager::instance().getStream();

  gather_kernel<<<number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      dest_column, source_column, getPointer(*tids), getSize(*tids));

  cudaError_t err = cudaStreamSynchronize(*stream);
  if (err == cudaSuccess) {
    return true;
  } else {
    gpuErrchk(err);
    return false;
  }
}

template <typename T>
const PositionListPtr GPU_Util<T>::sort(T* column, size_t num_elements,
                                        const SortParam& param) {
  PositionListPtr tids = createPositionList(0, getMemoryID(param.proc_spec));

  if (!tids) PositionListPtr();
  try {
    if (!resize(*tids, num_elements)) return PositionListPtr();
    thrust::sequence(
        thrust::device_pointer_cast(getPointer(*tids)),
        thrust::device_pointer_cast(getPointer(*tids) + num_elements),
        0);  //   tids->begin(), tids->end(), 0);
    // note, that the keys we want to sort are the values in the column
    //-> the order of the "values" is the desired order of tids that
    // describe the new order of tuples
    if (param.stable) {
      thrust::stable_sort_by_key(
          thrust::device_pointer_cast(column),
          thrust::device_pointer_cast(column + num_elements),
          thrust::device_pointer_cast(getPointer(*tids)));
    } else {
      thrust::sort_by_key(
          thrust::device_pointer_cast(column),
          thrust::device_pointer_cast(column + num_elements),
          thrust::device_pointer_cast(getPointer(*tids)));  // tids->data());
    }
    if (param.order == DESCENDING) {
      thrust::reverse(thrust::device_pointer_cast(getPointer(*tids)),
                      thrust::device_pointer_cast(
                          getPointer(*tids) +
                          num_elements));  // tids->begin(), tids->end());
    }
  } catch (std::bad_alloc& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "Ran out of memory during Sort!" << std::endl;
    return PositionListPtr();
  } catch (thrust::system_error& e) {
    std::cerr << e.what() << std::endl;
    return PositionListPtr();
  }

  return tids;
}

template <>
void GPU_Util<std::string>::print_device_array(std::string* pointer,
                                               size_t length) {}

template <>
bool GPU_Util<std::string>::generateConstantSequence(
    std::string* dest_column, size_t num_elements, std::string value,
    const ProcessorSpecification& proc_spec) {
  return false;
}

template <>
bool GPU_Util<std::string>::generateAscendingSequence(
    std::string* dest_column, size_t num_elements, std::string begin_value,
    const ProcessorSpecification& proc_spec) {
  return false;
}

template <>
bool GPU_Util<std::string>::gather(std::string* dest_column,
                                   std::string* source_column,
                                   PositionListPtr tids,
                                   const GatherParam& param) {
  return false;
}

template <>
const PositionListPtr GPU_Util<std::string>::sort(std::string* column,
                                                  size_t num_elements,
                                                  const SortParam& param) {
  return PositionListPtr();
}

template <>
void GPU_Util<C_String>::print_device_array(C_String* pointer, size_t length) {}

template <>
bool GPU_Util<C_String>::generateConstantSequence(
    C_String* dest_column, size_t num_elements, C_String value,
    const ProcessorSpecification& proc_spec) {
  return false;
}

template <>
bool GPU_Util<C_String>::generateAscendingSequence(
    C_String* dest_column, size_t num_elements, C_String begin_value,
    const ProcessorSpecification& proc_spec) {
  return false;
}

template <>
bool GPU_Util<C_String>::gather(C_String* dest_column, C_String* source_column,
                                PositionListPtr tids,
                                const GatherParam& param) {
  return false;
}

template <>
const PositionListPtr GPU_Util<C_String>::sort(C_String* column,
                                               size_t num_elements,
                                               const SortParam& param) {
  return PositionListPtr();
}

template <typename S, typename T>
__global__ void type_cast_gpu(S* data1, T* data2, size_t num) {
  TID id = threadIdx.x + blockIdx.x * blockDim.x;

  while (id < num) {
    data2[id] = data1[id];
    id += blockDim.x * gridDim.x;
  }
}

bool isCuda7Detected() {
  int runtime_version;
  int driver_version;
  cudaRuntimeGetVersion(&runtime_version);
  cudaDriverGetVersion(&driver_version);

  bool cuda_7_detected = false;
  if (runtime_version == 7000 || driver_version == 7000) {
    cuda_7_detected = true;
  }
  return cuda_7_detected;
}

bool canUseCUBPrefixScan() {
  bool can_use_cub_prefix_scan = false;
  cudaError_t err = cudaSuccess;
  /* Detect properties of current GPU device. */
  int device = 0;
  err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    gpuErrchk(err);
    return false;
  }
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    gpuErrchk(err);
    return false;
  }
  /* Check for CUDA GPU with at least compute capability 5.2 (tested) */
  if (prop.major > 5 || (prop.major == 5 && prop.minor >= 2)) {
    can_use_cub_prefix_scan = true;
  }
  return can_use_cub_prefix_scan;
}

std::pair<cudaError_t, size_t> GPU_Prefix_Sum(
    const thrust::device_ptr<char> flag_array,
    thrust::device_ptr<TID> write_positions_array, size_t input_array_size,
    cudaStream_t* stream) {
  TID result_size = 0;
  char last_flag = 0;
  cudaError_t err = cudaSuccess;
  bool cuda_7_detected = isCuda7Detected();
  bool can_use_cub_prefix_scan = canUseCUBPrefixScan();

  if (!quiet) {
    std::cout << "CUDA 7 detected: " << cuda_7_detected << std::endl;
    std::cout << "Can use CUB prefix scan: " << can_use_cub_prefix_scan
              << std::endl;
  }

  /* We use the CUB prefix sum function if possible since it is faster than
   * the prefix sum of thrust. However, the CUB prefix sum does only work
   * when the seleced GPU has compute capability 3.5 or higher. Otherwise,
   * we use thrust. */
  if (can_use_cub_prefix_scan) {
    // Convert input for equal I/O-type
    //        thrust::device_vector<TID> converted_flags(input_array_size);
    // TODO: pass memory id as function parameter!
    PositionListPtr converted_flags =
        createPositionList(input_array_size, hype::PD_Memory_1);
    if (!converted_flags) {
      double free_memory_in_gb =
          double(HardwareDetector::instance().getFreeMemorySizeInByte(
              hype::PD_Memory_1)) /
          (1024 * 1024 * 1024);
      COGADB_ERROR(
          "Ran out of memory during GPU_Prefix_Sum!"
              << std::endl
              << "Attempted to allocate " << input_array_size << " elements "
              << "("
              << double(input_array_size * sizeof(TID)) / (1024 * 1024 * 1024)
              << "GB)" << std::endl
              << "Free Memory: " << free_memory_in_gb << "GB)",
          "");
      return std::make_pair(cudaErrorMemoryAllocation, 0);
    }
    type_cast_gpu<<<128, 128, 0, *stream>>>(
        thrust::raw_pointer_cast(flag_array), converted_flags->data(),
        input_array_size);
    err = cudaStreamSynchronize(*stream);
    if (err != cudaSuccess) {
      gpuErrchk(err);
      return std::make_pair(err, 0);
    }

    // Allocate temporary storage
    // CachingDeviceAllocator  g_allocator(true);
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    err = DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, converted_flags->data(),
        thrust::raw_pointer_cast(write_positions_array), input_array_size,
        *stream);

    err = cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
      if (err == cudaErrorMemoryAllocation) {
        COGADB_ERROR(
            "Ran out of memory during GPU_Prefix_Sum!"
                << std::endl
                << "Attempted to allocate " << input_array_size << " elements "
                << "("
                << double(input_array_size * sizeof(TID)) / (1024 * 1024 * 1024)
                << "GB)",
            "");
      } else {
        gpuErrchk(err);
      }
      return std::make_pair(err, 0);
    }

    // Run
    err = DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, converted_flags->data(),
        thrust::raw_pointer_cast(write_positions_array), input_array_size,
        *stream);
    if (err != cudaSuccess) {
      gpuErrchk(err);
      return std::make_pair(err, 0);
    }

    err = cudaFree(d_temp_storage);
    if (err != cudaSuccess) {
      gpuErrchk(err);
      return std::make_pair(err, 0);
    }
  } else if (cuda_7_detected && !can_use_cub_prefix_scan) {
    /* The thrust prefix sum produces incorrect results in CUDA 7, but if
     we cannot use the CUB prefix scan, we are required to use thrust.
     Thrust prefix sum prodcues incorrect results when the input array has
     a different type as the output array. Thus, we perform a type cast
     operation, and then call thrust. */
    PositionListPtr converted_flags =
        createPositionList(input_array_size, hype::PD_Memory_1);
    if (!converted_flags) {
      double free_memory_in_gb =
          double(HardwareDetector::instance().getFreeMemorySizeInByte(
              hype::PD_Memory_1)) /
          (1024 * 1024 * 1024);
      COGADB_ERROR(
          "Ran out of memory during GPU_Prefix_Sum!"
              << std::endl
              << "Attempted to allocate " << input_array_size << " elements "
              << "("
              << double(input_array_size * sizeof(TID)) / (1024 * 1024 * 1024)
              << "GB)" << std::endl
              << "Free Memory: " << free_memory_in_gb << "GB)",
          "");
      return std::make_pair(cudaErrorMemoryAllocation, 0);
    }
    type_cast_gpu<<<128, 128, 0, *stream>>>(
        thrust::raw_pointer_cast(flag_array), converted_flags->data(),
        input_array_size);
    err = cudaStreamSynchronize(*stream);
    if (err != cudaSuccess) {
      gpuErrchk(err);
      return std::make_pair(err, 0);
    }
    try {
      thrust::exclusive_scan(
          thrust::device_ptr<TID>(converted_flags->data()),
          thrust::device_ptr<TID>(converted_flags->data() + input_array_size),
          write_positions_array);
    } catch (std::bad_alloc& e) {
      COGADB_ERROR("Ran out of memory during GPU_Prefix_Sum!", "");
      return std::make_pair(err, 0);
    }
  } else {
    /* It is save to use standard thrust without any conversions.*/
    try {
      thrust::device_ptr<TID> end_write_positions_array =
          thrust::exclusive_scan(flag_array, flag_array + input_array_size,
                                 write_positions_array);
      assert(end_write_positions_array ==
             write_positions_array + input_array_size);
    } catch (std::bad_alloc& e) {
      COGADB_ERROR("Ran out of memory during GPU_Prefix_Sum!", "");
      return std::make_pair(err, 0);
    }
  }

#ifdef COGADB_VALIDATE_GPU_PREFIX_SUM
  std::vector<char> cpu_flags(input_array_size);
  std::vector<TID> cpu_write_positions(input_array_size);
  thrust::copy(flag_array, flag_array + input_array_size, cpu_flags.begin());
  thrust::copy(write_positions_array, write_positions_array + input_array_size,
               cpu_write_positions.begin());

  TID sum = 0;
  for (size_t i = 0; i < input_array_size; ++i) {
    if (sum != cpu_write_positions[i]) {
      std::cout << i << ". " << cpu_write_positions[i] << std::endl;
      std::cout << "Error: " << sum << "!=" << cpu_write_positions[i]
                << std::endl;
    }
    if (cpu_flags[i] > 1) {
      std::cout << "Error: Invalid flag at position " << i << ": "
                << cpu_flags[i] << std::endl;
    }
    sum += cpu_flags[i];
  }
#endif

  TID* tmp = thrust::raw_pointer_cast(write_positions_array);
  err = cudaMemcpyAsync(&result_size, (void*)&tmp[input_array_size - 1],
                        sizeof(TID), cudaMemcpyDeviceToHost, *stream);
  err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess) {
    gpuErrchk(err);
    return std::make_pair(err, 0);
  }

  char* tmp2 = thrust::raw_pointer_cast(flag_array);
  err = cudaMemcpyAsync(&last_flag, (void*)&tmp2[input_array_size - 1],
                        sizeof(char), cudaMemcpyDeviceToHost, *stream);
  err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess) {
    gpuErrchk(err);
    return std::make_pair(err, 0);
  }
  // since we perform an exclusive scan to get the write positions,
  // we have to check whether the last element in the flag array was a
  // match and adjust the result size accordingly
  // if (last_flag) result_size++;
  result_size += last_flag;
  return std::make_pair(cudaSuccess, result_size);
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(GPU_Util);

};  // end namespace CoGaDB
