/*
 * File:   selection.cu
 * Author: sebastian
 *
 * Created on 31. Dezember 2014, 17:30
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <backends/gpu/selection.hpp>
#include <backends/gpu/stream_manager.hpp>
#include <backends/gpu/util.hpp>

#include <core/variable_manager.hpp>

#ifdef USE_ADMISSION_CONTROL
#include <core/gpu_admission_control.hpp>
#endif

namespace CoGaDB {
namespace gpu {

// mark matching row ids in flag array
template <typename T>
__global__ void selection_set_flag_array_kernel(T* array, size_t array_size,
                                                T comparison_value,
                                                const ValueComparator comp,
                                                char* flags) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (comp == EQUAL) {
    while (tid < array_size) {
      flags[tid] = (array[tid] == comparison_value);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == LESSER) {
    while (tid < array_size) {
      flags[tid] = (array[tid] < comparison_value);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == LESSER_EQUAL) {
    while (tid < array_size) {
      flags[tid] = (array[tid] <= comparison_value);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == GREATER) {
    while (tid < array_size) {
      flags[tid] = (array[tid] > comparison_value);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == GREATER_EQUAL) {
    while (tid < array_size) {
      flags[tid] = (array[tid] >= comparison_value);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == UNEQUAL) {
    while (tid < array_size) {
      flags[tid] = (array[tid] != comparison_value);
      tid += blockDim.x * gridDim.x;
    }
  }
}

template <typename T>
__global__ void column_comparison_set_flag_array_kernel(
    T* left_array, T* right_array, size_t array_size,
    const ValueComparator comp, char* flags) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (comp == EQUAL) {
    while (tid < array_size) {
      flags[tid] = (left_array[tid] == right_array[tid]);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == LESSER) {
    while (tid < array_size) {
      flags[tid] = (left_array[tid] < right_array[tid]);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == LESSER_EQUAL) {
    while (tid < array_size) {
      flags[tid] = (left_array[tid] <= right_array[tid]);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == GREATER) {
    while (tid < array_size) {
      flags[tid] = (left_array[tid] > right_array[tid]);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == GREATER_EQUAL) {
    while (tid < array_size) {
      flags[tid] = (left_array[tid] >= right_array[tid]);
      tid += blockDim.x * gridDim.x;
    }
  } else if (comp == UNEQUAL) {
    while (tid < array_size) {
      flags[tid] = (left_array[tid] != right_array[tid]);
      tid += blockDim.x * gridDim.x;
    }
  }
}

// kernel2: fetch matching row ids to result buffer
__global__ void selection_fetch_tids_in_output_buffer_kernel(
    size_t array_size, char* flags, TID* write_positions_array,
    TID* result_buffer) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < array_size) {
    if (flags[tid]) {
      TID write_index = write_positions_array[tid];
      result_buffer[write_index] = tid;
    }
    tid += blockDim.x * gridDim.x;
  }
}

template <typename T>
const PositionListPtr selection(T* input_array, size_t input_array_size,
                                T value, ValueComparator comp,
                                const hype::ProcessingDeviceMemoryID& mem_id) {
  if (input_array_size == 0) {
    return createPositionList(0, mem_id);
  }

  if (VariableManager::instance().getVariableValueBoolean(
          "unsafe_feature.enable_immediate_selection_abort"))
    return PositionListPtr();

#ifdef USE_ADMISSION_CONTROL
  size_t fullDataSize = 3 * sizeof(T) * input_array_size;
  AdmissionControl::instance().requestMemory(fullDataSize);
#endif

  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 1024;

  cudaStream_t* stream = StreamManager::instance().getStream();

  thrust::device_ptr<char> flag_array;
  thrust::device_ptr<TID> write_positions_array;

  try {
    flag_array = thrust::device_malloc<char>(input_array_size);
  } catch (std::bad_alloc& e) {
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    return PositionListPtr();
  }
  if (!quiet)
    std::cout << "flag array: " << (void*)flag_array.get() << std::endl;

  try {
    write_positions_array = thrust::device_malloc<TID>(input_array_size);
  } catch (std::bad_alloc& e) {
    // free allocated memory
    thrust::device_free(flag_array);
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    return PositionListPtr();
  }

  PositionListPtr result;

  try {
    result = createPositionList(input_array_size, mem_id);
    if (!result) {
      thrust::device_free(flag_array);
      thrust::device_free(write_positions_array);
      COGADB_ERROR("Ran out of memory during gpu_selection!", "");
      return PositionListPtr();
    }

    //                         result_tids_vector.resize(input_array_size);
  } catch (std::bad_alloc& e) {
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    return PositionListPtr();
  }

  selection_set_flag_array_kernel<
      T><<<number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      input_array, input_array_size, value, comp, flag_array.get());

  cudaError_t err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    COGADB_ERROR("Error executing kernel selection_set_flag_array_kernel<T>!",
                 "");
    gpuErrchk(err);
    return PositionListPtr();
  }

  std::pair<cudaError_t, size_t> ret;
  ret = GPU_Prefix_Sum(flag_array, write_positions_array, input_array_size,
                       stream);
  if (ret.first != cudaSuccess) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    return PositionListPtr();
  }
  size_t result_size = ret.second;

  if (!quiet)
    std::cout << "Result size GPU Selection:" << result_size << std::endl;
  // save second kernel call, in case result is empty
  if (result_size == 0) {
    thrust::device_free(write_positions_array);
    thrust::device_free(flag_array);
    return createPositionList(
        0, mem_id);  // GPU_PositionlistPtr(new Impl_GPU_Positionlist());
  }
  // allocate space for result
  try {
    assert(result_size > 0);
    // adjust number of elements to real result size
    if (!resize(*result, result_size)) {
      thrust::device_free(flag_array);
      thrust::device_free(write_positions_array);
      COGADB_ERROR("Ran out of memory during gpu_selection!", "");
      return PositionListPtr();
    }
  } catch (std::bad_alloc& e) {
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    return PositionListPtr();
  }
  // launch kernel that writes the matching TIDs into the output buffer
  selection_fetch_tids_in_output_buffer_kernel<<<
      number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      input_array_size, flag_array.get(), write_positions_array.get(),
      getPointer(*result));
  err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    COGADB_ERROR(
        "Error executing kernel "
        "selection_fetch_tids_in_output_buffer_kernel<T>!",
        "");
    gpuErrchk(err);
    return PositionListPtr();
  }

  // deallocate with device_free
  thrust::device_free(write_positions_array);
  thrust::device_free(flag_array);

#ifdef USE_ADMISSION_CONTROL
  AdmissionControl::instance().releaseMemory(fullDataSize);
#endif

  return result;
}

template <typename T>
const PositionListPtr selection(T* column, size_t num_elements,
                                T* comparison_column,
                                const ValueComparator& comp,
                                const hype::ProcessingDeviceMemoryID& mem_id) {
  if (num_elements == 0) {
    return createPositionList(0, mem_id);
  }

  if (VariableManager::instance().getVariableValueBoolean(
          "unsafe_feature.enable_immediate_selection_abort"))
    return PositionListPtr();

  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 1024;

  cudaStream_t* stream = gpu::StreamManager::instance().getStream();

  thrust::device_ptr<char> flag_array;
  thrust::device_ptr<TID> write_positions_array;

  try {
    flag_array = thrust::device_malloc<char>(num_elements);
  } catch (std::bad_alloc& e) {
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    return PositionListPtr();
  }
  if (!quiet)
    std::cout << "flag array: " << (void*)flag_array.get() << std::endl;

  try {
    write_positions_array = thrust::device_malloc<TID>(num_elements);
  } catch (std::bad_alloc& e) {
    // free allocated memory
    thrust::device_free(flag_array);
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    return PositionListPtr();
  }

  PositionListPtr result;

  try {
    result = createPositionList(num_elements, mem_id);
    if (!result) {
      thrust::device_free(flag_array);
      thrust::device_free(write_positions_array);
      COGADB_ERROR("Ran out of memory during gpu_selection!", "");
      return PositionListPtr();
    }
  } catch (std::bad_alloc& e) {
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    return PositionListPtr();
  }

  column_comparison_set_flag_array_kernel<
      T><<<number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      column, comparison_column, num_elements, comp, flag_array.get());

  cudaError_t err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    COGADB_ERROR(
        "Error executing kernel column_comparison_set_flag_array_kernel<T>!",
        "");
    gpuErrchk(err);
    return PositionListPtr();
  }

  //            size_t result_size=0;
  //            try {
  //                result_size=GPU_Prefix_Sum(flag_array,
  //                write_positions_array, num_elements, stream);
  //            } catch (std::bad_alloc &e) {
  //                //free allocated memory
  //                thrust::device_free(flag_array);
  //                thrust::device_free(write_positions_array);
  //                COGADB_ERROR("Ran out of memory during gpu_selection!", "");
  //                return PositionListPtr();
  //            }

  std::pair<cudaError_t, size_t> ret;
  ret = GPU_Prefix_Sum(flag_array, write_positions_array, num_elements, stream);
  if (ret.first != cudaSuccess) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    return PositionListPtr();
  }
  size_t result_size = ret.second;

  if (!quiet)
    std::cout << "Result size GPU Selection:" << result_size << std::endl;
  // save second kernel call, in case result is empty
  if (result_size == 0) {
    thrust::device_free(write_positions_array);
    thrust::device_free(flag_array);
    return createPositionList(
        0, mem_id);  // GPU_PositionlistPtr(new Impl_GPU_Positionlist());
  }
  // allocate space for result
  try {
    assert(result_size > 0);
    // adjust number of elements to real result size
    if (!resize(*result, result_size)) {
      thrust::device_free(flag_array);
      thrust::device_free(write_positions_array);
      COGADB_ERROR("Ran out of memory during gpu_selection!", "");
      return PositionListPtr();
    }
  } catch (std::bad_alloc& e) {
    COGADB_ERROR("Ran out of memory during gpu_selection!", "");
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    return PositionListPtr();
  }
  // launch kernel that writes the matching TIDs into the output buffer
  selection_fetch_tids_in_output_buffer_kernel<<<
      number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      num_elements, flag_array.get(), write_positions_array.get(),
      getPointer(*result));
  err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess) {
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    COGADB_ERROR(
        "Error executing kernel "
        "selection_fetch_tids_in_output_buffer_kernel<T>!",
        "");
    gpuErrchk(err);
    return PositionListPtr();
  }
  // deallocate with device_free
  thrust::device_free(write_positions_array);
  thrust::device_free(flag_array);

  return result;
}

};  // end namespace gpu

template <typename T>
const PositionListPtr GPU_Selection<T>::tid_selection(
    T* column, size_t num_elements, T comparison_value,
    const ValueComparator& comp, const hype::ProcessingDeviceMemoryID& mem_id) {
  return gpu::selection(column, num_elements, comparison_value, comp, mem_id);
}

template <typename T>
const PositionListPtr GPU_Selection<T>::tid_selection(
    T* column, size_t num_elements, T* comparison_column,
    const ValueComparator& comp, const hype::ProcessingDeviceMemoryID& mem_id) {
  return gpu::selection(column, num_elements, comparison_column, comp, mem_id);
}

template <typename T>
const BitmapPtr GPU_Selection<T>::bitmap_selection(
    T* column, size_t num_elements, T comparison_value,
    const ValueComparator& comp, const hype::ProcessingDeviceMemoryID& mem_id) {
  COGADB_FATAL_ERROR("Called Unimplemented function!", "");
  return BitmapPtr();
}

template <typename T>
const BitmapPtr GPU_Selection<T>::bitmap_selection(
    T* column, size_t num_elements, T* comparison_column,
    const ValueComparator& comp, const hype::ProcessingDeviceMemoryID& mem_id) {
  COGADB_FATAL_ERROR("Called Unimplemented function!", "");
  return BitmapPtr();
}

template <>
const PositionListPtr GPU_Selection<std::string>::tid_selection(
    std::string* column, size_t num_elements, std::string comparison_value,
    const ValueComparator& comp, const hype::ProcessingDeviceMemoryID& mem_id) {
  COGADB_FATAL_ERROR("Called Unimplemented function!", "");
  return PositionListPtr();
}

template <>
const PositionListPtr GPU_Selection<std::string>::tid_selection(
    std::string* column, size_t num_elements, std::string* comparison_column,
    const ValueComparator& comp, const hype::ProcessingDeviceMemoryID& mem_id) {
  COGADB_FATAL_ERROR("Called Unimplemented function!", "");
  return PositionListPtr();
}

template <>
const BitmapPtr GPU_Selection<std::string>::bitmap_selection(
    std::string* column, size_t num_elements, std::string comparison_value,
    const ValueComparator& comp, const hype::ProcessingDeviceMemoryID& mem_id) {
  COGADB_FATAL_ERROR("Called Unimplemented function!", "");
  return BitmapPtr();
}

template <>
const BitmapPtr GPU_Selection<std::string>::bitmap_selection(
    std::string* column, size_t num_elements, std::string* comparison_column,
    const ValueComparator& comp, const hype::ProcessingDeviceMemoryID& mem_id) {
  COGADB_FATAL_ERROR("Called Unimplemented function!", "");
  return BitmapPtr();
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(GPU_Selection);

};  // end namespace CoGaDB
