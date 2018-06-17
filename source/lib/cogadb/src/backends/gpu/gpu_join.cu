/*
 * File:   gpu_join.cu
 * Author: henning
 *
 * Created on 3. Februar 2015, 10:00
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <backends/gpu/hashtable/hash_table.h>
#include <backends/gpu/join.hpp>
#include <backends/gpu/stream_manager.hpp>
#include <backends/gpu/util.hpp>
#include <core/base_column.hpp>
#include <core/column.hpp>
#include <core/global_definitions.hpp>
#include <ctime>
#include <moderngpu.cuh>
#include <util/getname.hpp>
#include <util/utility_functions.hpp>

#define SIZE_1MB (1000 * 1000)

namespace CoGaDB {

using namespace std;
using namespace thrust;

template <typename T>
typename GPU_Join<T>::JoinFunctionPtr GPU_Join<T>::get(
    const JoinType& join_type) {
  typedef GPU_Join<T>::JoinFunctionPtr JoinFunctionPtr;
  typedef std::map<JoinType, JoinFunctionPtr> JoinTypeMap;
  static JoinTypeMap map;
  static bool initialized = false;
  if (!initialized) {
    map.insert(std::make_pair(INNER_JOIN, &GPU_Join<T>::inner_join));
    map.insert(std::make_pair(LEFT_OUTER_JOIN, &GPU_Join<T>::left_outer_join));
    map.insert(
        std::make_pair(RIGHT_OUTER_JOIN, &GPU_Join<T>::right_outer_join));
    map.insert(std::make_pair(FULL_OUTER_JOIN, &GPU_Join<T>::full_outer_join));
    initialized = true;
  }
  if (map.find(join_type) != map.end()) {
    return map.find(join_type)->second;
  } else {
    COGADB_FATAL_ERROR("Detected use of unsupported Join Type!", "");
    return JoinFunctionPtr();
  }
}

template <typename T>
const PositionListPairPtr GPU_Join<T>::inner_join(T* join_column1,
                                                  size_t left_num_elements,
                                                  T* join_column2,
                                                  size_t right_num_elements,
                                                  const JoinParam& param) {
  PositionListPairPtr result;
  if (left_num_elements > right_num_elements) {
    result = binary_search_join(join_column1, join_column2, left_num_elements,
                                right_num_elements, param);
  } else {
    result = binary_search_join(join_column2, join_column1, right_num_elements,
                                left_num_elements, param);
    if (!result) return result;
    std::swap(result->first, result->second);
  }

  return result;
}

// dont write place_found but search again in write kernel
template <typename T>
__global__ void join_and_count_kernel(T* column_left, size_t count_column_left,
                                      T* column_right,
                                      size_t count_column_right,
                                      TID* times_found) {
  // threads align to entries in large join column
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < count_column_left) {
    // try to find first occurence of key
    TID index = binary_search_first_occurence(column_right, count_column_right,
                                              column_left[tid]);
    TID count = 0;

    if (index < count_column_right) {
      T value = column_right[index];
      count++;

      while (index + count < count_column_right) {
        if (column_right[index + count] == value) {
          count++;
        } else {
          break;
        }
      }
    }
    times_found[tid] = count;
    tid += blockDim.x * gridDim.x;
  }
}

template <typename T>
__global__ void join_and_write_kernel(T* fact_table, size_t count_fact_table,
                                      T* dimension_table,
                                      size_t count_dimension_table,
                                      TID* scan_times_found, TID* result_fact,
                                      TID* result_dimension) {
  // threads align to join index entries
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < count_fact_table) {
    // try to find first occurence of key
    TID index = binary_search_first_occurence(
        dimension_table, count_dimension_table, fact_table[tid]);

    if (index < count_dimension_table) {
      T value = dimension_table[index];
      TID idxStartWrite = scan_times_found[tid];
      TID writeCount = 0;

      while (index + writeCount < count_dimension_table) {
        if (value == dimension_table[index + writeCount]) {
          result_dimension[idxStartWrite + writeCount] = index + writeCount;
          result_fact[idxStartWrite + writeCount] = tid;
          writeCount++;
        } else {
          break;
        }
      }
    }
    tid += blockDim.x * gridDim.x;
  }
}

// left is bigger, right is small
template <typename T>
const PositionListPairPtr GPU_Join<T>::binary_search_join(
    T* join_column1, T* join_column2, size_t left_num_elements,
    size_t right_num_elements, const JoinParam& param) {
  vector<std::pair<string, double>> time_measurement;
  PositionListPairPtr result;

  //------------------- Init device objects ------------------------------
  cudaStream_t* stream = gpu::StreamManager::instance().getStream();
  gpu::GPUContextPtr contextptr =
      gpu::StreamManager::instance().getCudaContext(0);
  const hype::ProcessingDeviceMemoryID mem_id =
      hype::util::getMemoryID(param.proc_spec.proc_id);

  //------------------- Empty result ----------------
  PositionListPairPtr emptyPair(new PositionListPair());
  emptyPair->first = createPositionList(0, mem_id);
  emptyPair->second = createPositionList(0, mem_id);
  if (left_num_elements == 0 || right_num_elements == 0) {
    cout << "WARNING!! Join input with size zero (" << left_num_elements << ", "
         << right_num_elements << ")" << endl;
    return emptyPair;
  }

  //---------- Make sure that smaller table is sorted (unsort later) -----
  bool right_column_is_sorted = false;
  // function wide scope required
  thrust::device_vector<T> dev_vec_column2_sorted;
  PositionListPtr unsort_lookup;
  if (!right_column_is_sorted) {
    try {
      dev_vec_column2_sorted = thrust::device_vector<T>(
          thrust::device_pointer_cast(join_column2),
          thrust::device_pointer_cast(join_column2 + right_num_elements));
    } catch (std::bad_alloc& e) {
      COGADB_ERROR("Ran out of memory during GPU_Join<T>::binary_search_join!",
                   "");
      return result;
    }
    // overwrite pointer, original column is not needed anymore
    join_column2 = thrust::raw_pointer_cast(&(dev_vec_column2_sorted[0]));
    SortParam sort_param(param.proc_spec, ASCENDING);
    unsort_lookup =
        GPU_Util<T>::sort(join_column2, right_num_elements, sort_param);
    if (!unsort_lookup) return PositionListPairPtr();
  }

  //-------------- Get memory with known size -------------------
  thrust::device_vector<TID> dev_vec_times_found;
  thrust::device_vector<TID> dev_vec_scan;
  try {
    dev_vec_times_found.resize(left_num_elements);
    dev_vec_scan.resize(left_num_elements);
  } catch (std::bad_alloc& e) {
    COGADB_ERROR(
        "Ran out of memory during GPU_Join<T>::binary_search_join!"
            << std::endl
            << "Attempted to allocate " << left_num_elements << " elements "
            << "("
            << double(left_num_elements * sizeof(TID)) / (1024 * 1024 * 1024)
            << "GB)",
        "");
    return result;
  }
  TID* devptr_times_found = thrust::raw_pointer_cast(&(dev_vec_times_found[0]));

  //------------------- Perform join -------------------------------------
  int number_of_blocks = 512;
  int number_of_threads = 1024;
  // find and join equal values in both tables
  join_and_count_kernel<<<number_of_blocks, number_of_threads, 0, *stream>>>(
      join_column1, left_num_elements, join_column2, right_num_elements,
      devptr_times_found);
  cudaError_t err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess) {
    COGADB_ERROR(
        "Error executing kernel join_and_count_kernel<T>!"
            << std::endl
            << "\tSize Left Table: " << left_num_elements << "rows" << std::endl
            << "\tSize Right Table: " << right_num_elements << "rows\n",
        "");
    gpuErrchk(err);
    return PositionListPairPtr();
  }

  // out of place exclusive scan gives write positions and result size
  // times_found is needed for length of incrementing index sequence in write
  thrust::exclusive_scan(dev_vec_times_found.begin(), dev_vec_times_found.end(),
                         dev_vec_scan.begin());
  size_t count_join_output = dev_vec_scan[left_num_elements - 1] +
                             dev_vec_times_found[left_num_elements - 1];
  if (count_join_output == 0) return emptyPair;

  // allocate memory for join result
  PositionListPtr first, second;
  first = createPositionList(count_join_output, mem_id);
  second = createPositionList(count_join_output, mem_id);
  if (!first || !second) {
    COGADB_ERROR("Ran out of memory during GPU_Join<T>::binary_search_join!",
                 "");
    return PositionListPairPtr();
  }

  TID* devptr_left_out = getPointer(*first);
  TID* devptr_right_out = getPointer(*second);
  TID* devptr_scan = thrust::raw_pointer_cast(&(dev_vec_scan[0]));
  // write output
  join_and_write_kernel<<<number_of_blocks, number_of_threads, 0, *stream>>>(
      join_column1, left_num_elements, join_column2, right_num_elements,
      devptr_scan, devptr_left_out, devptr_right_out);
  err = cudaStreamSynchronize(*stream);
  gpuErrchk(err);

  //---------- undo sorting of right column using gather -----------------
  if (!right_column_is_sorted) {
    GatherParam gather_param(param.proc_spec);
    GPU_Util<TID>::gather(getPointer(*second), getPointer(*unsort_lookup),
                          second, gather_param);
  }

  result = PositionListPairPtr(new PositionListPair(first, second));
  return result;
}

template <>
const PositionListPairPtr GPU_Join<std::string>::binary_search_join(
    std::string* join_column1, std::string* join_column2,
    size_t left_num_elements, size_t right_num_elements,
    const JoinParam& param) {
  COGADB_FATAL_ERROR("Called Unimplemented function!", "");
  return PositionListPairPtr();
}

template <>
const PositionListPairPtr GPU_Join<C_String>::binary_search_join(
    C_String* join_column1, C_String* join_column2, size_t left_num_elements,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called Unimplemented function!", "");
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr GPU_Join<T>::left_outer_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr GPU_Join<T>::right_outer_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr GPU_Join<T>::full_outer_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return PositionListPairPtr();
}

template <typename T>
typename GPU_Semi_Join<T>::TIDSemiJoinFunctionPtr
GPU_Semi_Join<T>::getTIDSemiJoin(const JoinType& join_type) {
  typedef GPU_Semi_Join<T>::TIDSemiJoinFunctionPtr TIDSemiJoinFunctionPtr;

  typedef std::map<JoinType, TIDSemiJoinFunctionPtr> JoinTypeMap;
  static JoinTypeMap map;
  static bool initialized = false;
  if (!initialized) {
    map.insert(
        std::make_pair(LEFT_SEMI_JOIN, &GPU_Semi_Join<T>::tid_left_semi_join));
    map.insert(std::make_pair(RIGHT_SEMI_JOIN,
                              &GPU_Semi_Join<T>::tid_right_semi_join));
    map.insert(std::make_pair(LEFT_ANTI_SEMI_JOIN,
                              &GPU_Semi_Join<T>::tid_left_anti_semi_join));
    map.insert(std::make_pair(RIGHT_ANTI_SEMI_JOIN,
                              &GPU_Semi_Join<T>::tid_right_anti_semi_join));
    initialized = true;
  }
  if (map.find(join_type) != map.end()) {
    return map.find(join_type)->second;
  } else {
    COGADB_FATAL_ERROR(
        "Detected use of unsupported Join Type: " << util::getName(join_type),
        "");
    return TIDSemiJoinFunctionPtr();
  }
}

template <typename T>
typename GPU_Semi_Join<T>::BitmapSemiJoinFunctionPtr
GPU_Semi_Join<T>::getBitmapSemiJoin(const JoinType& join_type) {
  typedef GPU_Semi_Join<T>::BitmapSemiJoinFunctionPtr BitmapSemiJoinFunctionPtr;
  COGADB_FATAL_ERROR("Called unimplemented Method!", "");
  return BitmapSemiJoinFunctionPtr();
}

template <typename T>
const PositionListPtr GPU_Semi_Join<T>::tid_left_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  PositionListPtr result =
      d_SemiJoin(join_column2, right_num_elements, join_column1,
                 left_num_elements, false, param);

  if (result) cout << "--- TID LEFT SEMI JOIN result is valid" << endl;
  if (!result) cout << "--- TID LEFT SEMI JOIN result is INVALID" << endl;
  return result;
}

template <typename T>
const PositionListPtr GPU_Semi_Join<T>::tid_right_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return d_SemiJoin(join_column1, left_num_elements, join_column2,
                    right_num_elements, false, param);
}

template <typename T>
const PositionListPtr GPU_Semi_Join<T>::tid_left_anti_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return d_SemiJoin(join_column2, right_num_elements, join_column1,
                    left_num_elements, true, param);
}

template <typename T>
const PositionListPtr GPU_Semi_Join<T>::tid_right_anti_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return d_SemiJoin(join_column1, left_num_elements, join_column2,
                    right_num_elements, true, param);
}

template <typename T>
const BitmapPtr GPU_Semi_Join<T>::bitmap_left_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return d_BitmapSemiJoin(join_column2, right_num_elements, join_column1,
                          left_num_elements, false, param);
}

template <typename T>
const BitmapPtr GPU_Semi_Join<T>::bitmap_right_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return d_BitmapSemiJoin(join_column1, left_num_elements, join_column2,
                          right_num_elements, false, param);
}

template <typename T>
const BitmapPtr GPU_Semi_Join<T>::bitmap_left_anti_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return d_BitmapSemiJoin(join_column2, right_num_elements, join_column1,
                          left_num_elements, true, param);
}

template <typename T>
const BitmapPtr GPU_Semi_Join<T>::bitmap_right_anti_semi_join(
    T* join_column1, size_t left_num_elements, T* join_column2,
    size_t right_num_elements, const JoinParam& param) {
  return d_BitmapSemiJoin(join_column1, left_num_elements, join_column2,
                          right_num_elements, true, param);
}

template <typename T>
const PositionListPtr GPU_Semi_Join<T>::d_SemiJoin(T* d_build, size_t numBuild,
                                                   T* d_probe, size_t numProbe,
                                                   bool anti,
                                                   const JoinParam& param) {
  std::clock_t start;
  double duration;
  start = std::clock();
  cout << "--- semi join input sizes - probe: "
       << ((numProbe * sizeof(T)) / (double)SIZE_1MB)
       << "mb, build: " << ((numBuild * sizeof(T)) / (double)SIZE_1MB) << "mb"
       << endl;

  // -------- gpu hashtable setup --------
  cudaStream_t stream = *gpu::StreamManager::instance().getStream();
  CudaHT::CuckooHashing::HashTable ht;
  if (!ht.Initialize(numBuild, stream)) {
    return PositionListPtr();
  }

  // -------- hashjoin build --------
  if (!ht.Build(numBuild, (unsigned*)d_build, NULL)) return PositionListPtr();

  // -------- hashjoin probe --------
  TID* d_semiJoin;
  if (!(gpuAssertCheck(
          cudaMalloc((void**)&d_semiJoin, sizeof(TID) * numProbe))))
    return PositionListPtr();

  unsigned int resultSize = 0;
  bool success = ht.RetrieveSemiJoin(numProbe, (unsigned*)d_probe, resultSize,
                                     (unsigned*)d_semiJoin, anti);
  if (!success) {
    cudaFree(d_semiJoin);
    return PositionListPtr();
  }

  cout << "--- semi join result number: " << resultSize << endl;

  // ------- prepare result ---------
  const hype::ProcessingDeviceMemoryID mem_id =
      hype::util::getMemoryID(param.proc_spec.proc_id);
  PositionListPtr result = createPositionList(resultSize, mem_id);
  if (!result) {
    cudaFree(d_semiJoin);
    return PositionListPtr();
  }

  if (!(gpuAssertCheck(cudaMemcpy(result->data(), d_semiJoin,
                                  resultSize * sizeof(TID),
                                  cudaMemcpyDeviceToDevice)))) {
    cudaFree(d_semiJoin);
    return PositionListPtr();
  }
  cudaFree(d_semiJoin);

  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cudaDeviceSynchronize();
  std::cout << "--- SEMI JOIN computation time: " << duration * 1000 << "ms\n";

  return result;
}

__global__ void pack_flagarray_kernel(char* flag_array,
                                      size_t num_rows_base_table,
                                      char* result_bitmap) {
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < (num_rows_base_table + 7) / 8) {
    TID flag_array_tid = tid * 8;
    char packed_bits = 0;
    packed_bits |= (flag_array[flag_array_tid] != 0);
    packed_bits |= (flag_array[flag_array_tid + 1] != 0) << 1;
    packed_bits |= (flag_array[flag_array_tid + 2] != 0) << 2;
    packed_bits |= (flag_array[flag_array_tid + 3] != 0) << 3;
    packed_bits |= (flag_array[flag_array_tid + 4] != 0) << 4;
    packed_bits |= (flag_array[flag_array_tid + 5] != 0) << 5;
    packed_bits |= (flag_array[flag_array_tid + 6] != 0) << 6;
    packed_bits |= (flag_array[flag_array_tid + 7] != 0) << 7;

    result_bitmap[tid] = packed_bits;
    tid += blockDim.x * gridDim.x;
  }
}

template <typename T>
const BitmapPtr GPU_Semi_Join<T>::d_BitmapSemiJoin(T* d_build, size_t numBuild,
                                                   T* d_probe, size_t numProbe,
                                                   bool anti,
                                                   const JoinParam& param) {
  // -------- gpu hashtable setup --------
  cudaStream_t stream = *gpu::StreamManager::instance().getStream();
  CudaHT::CuckooHashing::HashTable ht;
  if (!ht.Initialize(numBuild, stream)) {
    return BitmapPtr();
  }

  // -------- hashjoin build --------
  if (!ht.Build(numBuild, (unsigned*)d_build, NULL)) return BitmapPtr();

  // -------- hashjoin probe --------
  char* d_bitmap = NULL;
  if (!(gpuAssertCheck(cudaMalloc((void**)&d_bitmap, sizeof(char) * numProbe))))
    return BitmapPtr();

  bool success =
      ht.RetrieveBitmapSemiJoin(numProbe, (unsigned*)d_probe, d_bitmap, anti);
  if (!success) {
    cudaFree(d_bitmap);
    return BitmapPtr();
  }

  // -------- prepare result --------
  const hype::ProcessingDeviceMemoryID mem_id =
      hype::util::getMemoryID(param.proc_spec.proc_id);
  BitmapPtr result = createBitmap(numProbe, mem_id);
  if (!result) {
    cudaFree(d_bitmap);
    return BitmapPtr();
  }

  pack_flagarray_kernel<<<128, 128, 0, stream>>>(d_bitmap, numProbe,
                                                 result->data());
  if (!(gpuAssertCheck(cudaStreamSynchronize(stream)))) {
    cudaFree(d_bitmap);
    return BitmapPtr();
  }

  return BitmapPtr();
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(GPU_Join);
COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(GPU_Semi_Join);

};  // end namespace CoGaDB
