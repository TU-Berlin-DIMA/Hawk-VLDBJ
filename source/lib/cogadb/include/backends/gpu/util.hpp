/*
 * File:   util.hpp
 * Author: sebastian
 *
 * Created on 8. Januar 2015, 08:17
 */

#pragma once

#ifndef UTIL_HPP
#define UTIL_HPP

#include <core/global_definitions.hpp>

#ifdef __NVCC__
#define COGADB_HOST_DEVICE_DECLARATION __host__ __device__
#else
#define COGADB_HOST_DEVICE_DECLARATION
#endif

#ifdef ENABLE_GPU_ACCELERATION
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

#define gpuAssertCheck(ans) \
  { (gpuAssertionCheck((ans), __FILE__, __LINE__)); }

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <core/base_column.hpp>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

#define gpuAssertCheck(ans) \
  { (gpuAssertionCheck((ans), __FILE__, __LINE__)); }

inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = false) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err == cudaSuccess) {
      std::cerr << "Free Memory in byte: " << free << std::endl;
    } else {
      std::cerr << "Could not retrieve memory information!" << std::endl;
    }
    CoGaDB::printStackTrace(std::cout);
    // treat all cuda errors except cuda memory allocation errors as fatal
    if (abort || code != cudaErrorMemoryAllocation) CoGaDB::exit(code);
  }
}

inline bool gpuAssertionCheck(cudaError_t code, const char* file, int line,
                              bool abort = false) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d (error code: %d)\n",
            cudaGetErrorString(code), file, line, (int)code);
    // treat all cuda errors except cuda memory allocation errors as fatal
    if (abort || code != cudaErrorMemoryAllocation) CoGaDB::exit(code);
    return false;
  }
  return true;
}

namespace CoGaDB {

  template <typename T>
  class GPU_Util {
   public:
    static void print_device_array(T* pointer, size_t length);

    static bool generateConstantSequence(
        T* dest_column, size_t num_elements, T value,
        const ProcessorSpecification& proc_spec);

    static bool generateAscendingSequence(
        T* dest_column, size_t num_elements, T begin_value,
        const ProcessorSpecification& proc_spec);

    static bool gather(T* dest_column, T* source_column, PositionListPtr tids,
                       const GatherParam& param);

    static const PositionListPtr sort(T* column, size_t num_elements,
                                      const SortParam& param);
  };

  std::pair<cudaError_t, size_t> GPU_Prefix_Sum(
      const thrust::device_ptr<char> flag_array,
      thrust::device_ptr<TID> write_positions_array, size_t input_array_size,
      cudaStream_t* stream);

}  // end namespace CoGaDB

#endif

// device procedures for different binary searches

// Finds any occurence of needle in sorted sequence haystack
// works for N > 0
template <typename T>
COGADB_HOST_DEVICE_DECLARATION TID binary_search(T* haystack, size_t N,
                                                 T needle) {
  TID high = N - 1;
  TID mid;
  TID low = 0;
  T sample;
  while (low <= high) {
    mid = (high + low) / 2;
    sample = haystack[mid];
    if (sample == needle) {
      return mid;
    } else if (needle < sample) {
      if (mid == 0) break;
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return N;
}

// Finds the first occurence of needle in sorted sequence haystack
// works for N > 0
template <typename T>
COGADB_HOST_DEVICE_DECLARATION TID binary_search_first_occurence(T* haystack,
                                                                 size_t N,
                                                                 T needle) {
  TID high = N - 1;
  TID mid;
  TID low = 0;
  TID index = N;
  T sample;
  while (low <= high) {
    mid = (high + low) / 2;
    sample = haystack[mid];
    if (sample == needle) {
      index = mid;
      if (mid == 0) break;
      high = mid - 1;
    } else if (needle < sample) {
      if (mid == 0) break;
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return index;
}

// Finds the first occurence of needle in sorted sequence haystack
// works for N > 0
template <typename T>
COGADB_HOST_DEVICE_DECLARATION TID binary_search_last_occurrence(T* haystack,
                                                                 size_t N,
                                                                 T needle) {
  TID high = N - 1;
  TID mid;
  TID low = 0;
  TID index = N;
  T sample;
  while (low <= high) {
    mid = (high + low) / 2;
    sample = haystack[mid];
    if (sample == needle) {
      index = mid;
      low = mid + 1;
    } else if (needle > sample) {
      low = mid + 1;
    } else {
      if (mid == 0) break;
      high = mid - 1;
    }
  }
  return index;
}

// Finds the nearest occurence that is greater than the needle in a sorted
// sequence haystack
// works for N > 0
template <typename T>
COGADB_HOST_DEVICE_DECLARATION TID
binary_search_find_nearest_greater(T* haystack, size_t N, T needle) {
  TID high = N - 1;
  TID mid;
  TID low = 0;
  T sample;
  while (low <= high) {
    mid = (high + low) / 2;
    sample = haystack[mid];
    if (needle >= sample) {
      low = mid + 1;
    } else if (needle < sample) {
      if (mid == 0) return mid;
      if (needle >= haystack[mid - 1]) return mid;
      high = mid - 1;
    }
  }
  return N;
}

#endif /* UTIL_HPP */
