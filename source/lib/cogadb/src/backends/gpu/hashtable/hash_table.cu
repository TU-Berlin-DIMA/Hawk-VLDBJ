// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision:$
// $Date:$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file hash_table.cu
 *
 * @brief Hides all of the CUDA calls from the actual CPP file.
 */

#include <backends/gpu/hashtable/definitions.h>
#include <core/global_definitions.hpp>
#include "cuda_util.h"
#include "debugging.h"
#include "hash_table.cuh"

#include <cuda.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <iostream>
//#include <backends/gpu/util.hpp>

using namespace std;

namespace CudaHT {
namespace CuckooHashing {

namespace CUDAWrapper {
void ClearTable(const unsigned slots_in_table, const Entry fill_value,
                Entry *d_contents, cudaStream_t computeStream) {
  clear_table<<<ComputeGridDim(slots_in_table), kBlockSize, 0, computeStream>>>(
      slots_in_table, fill_value, d_contents);
  cudaError_t err = cudaStreamSynchronize(computeStream);
  if (err != cudaSuccess) {
    COGADB_ERROR("Could not clean up hash table!", "");
    gpuErrchk(err);
    return;
  }
}

cudaError_t CallCuckooHash(
    const unsigned n, const unsigned num_hash_functions, const unsigned *d_keys,
    const unsigned *d_values, const unsigned table_size,
    const Functions<2> constants_2, const Functions<3> constants_3,
    const Functions<4> constants_4, const Functions<5> constants_5,
    const unsigned max_iterations, Entry *d_contents, uint2 stash_constants,
    unsigned *d_stash_count, unsigned *d_failures, unsigned *d_iterations_taken,
    cudaStream_t computeStream) {
  // Build the table.
  cudaError_t err = cudaMemset(d_failures, 0, sizeof(unsigned));
  if (err != cudaSuccess) {
    gpuErrchk(err);
    return err;
  }

  if (num_hash_functions == 2) {
    CuckooHash<<<ComputeGridDim(n), kBlockSize, 0, computeStream>>>(
        n, d_keys, d_values, table_size, constants_2, max_iterations,
        d_contents, stash_constants, d_stash_count, d_failures,
        d_iterations_taken);
  } else if (num_hash_functions == 3) {
    CuckooHash<<<ComputeGridDim(n), kBlockSize, 0, computeStream>>>(
        n, d_keys, d_values, table_size, constants_3, max_iterations,
        d_contents, stash_constants, d_stash_count, d_failures,
        d_iterations_taken);
  } else if (num_hash_functions == 4) {
    CuckooHash<<<ComputeGridDim(n), kBlockSize, 0, computeStream>>>(
        n, d_keys, d_values, table_size, constants_4, max_iterations,
        d_contents, stash_constants, d_stash_count, d_failures,
        d_iterations_taken);
  } else {
    CuckooHash<<<ComputeGridDim(n), kBlockSize, 0, computeStream>>>(
        n, d_keys, d_values, table_size, constants_5, max_iterations,
        d_contents, stash_constants, d_stash_count, d_failures,
        d_iterations_taken);
  }

  err = cudaStreamSynchronize(computeStream);
  if (err != cudaSuccess) {
    COGADB_ERROR("Could not build hash table!", "");
    gpuErrchk(err);
    return err;
  }
  return cudaSuccess;
}

bool CallHashRetrieve(const unsigned n_queries,
                      const unsigned num_hash_functions, const unsigned *d_keys,
                      const unsigned table_size, const Entry *d_contents,
                      const Functions<2> constants_2,
                      const Functions<3> constants_3,
                      const Functions<4> constants_4,
                      const Functions<5> constants_5,
                      const uint2 stash_constants, const unsigned stash_count,
                      unsigned *d_values, cudaStream_t computeStream) {
  unsigned *d_retrieval_probes = NULL;
#ifdef TRACK_ITERATIONS
  CUDA_SAFE_MALLOC(
      cudaMalloc((void **)&d_retrieval_probes, sizeof(unsigned) * n_queries));
#endif

  if (num_hash_functions == 2) {
    hash_retrieve<<<ComputeGridDim(n_queries), kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_2, stash_constants,
        stash_count, d_values, d_retrieval_probes);
  } else if (num_hash_functions == 3) {
    hash_retrieve<<<ComputeGridDim(n_queries), kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_3, stash_constants,
        stash_count, d_values, d_retrieval_probes);
  } else if (num_hash_functions == 4) {
    hash_retrieve<<<ComputeGridDim(n_queries), kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_4, stash_constants,
        stash_count, d_values, d_retrieval_probes);
  } else {
    hash_retrieve<<<ComputeGridDim(n_queries), kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_5, stash_constants,
        stash_count, d_values, d_retrieval_probes);
  }
  cudaError_t err = cudaStreamSynchronize(computeStream);
  if (err != cudaSuccess) {
    COGADB_ERROR("Could not retrieve semi join result from hash table!", "");
    gpuErrchk(err);
    return false;
  }

#ifdef TRACK_ITERATIONS
  OutputRetrievalStatistics(n_queries, d_retrieval_probes, num_hash_functions);
  CUDA_SAFE_CALL(cudaFree(d_retrieval_probes));
#endif
  return true;
}

template <typename T>
thrust::device_ptr<T> devPtr(T *data) {
  return thrust::device_pointer_cast(data);
}

template <typename T>
T *rawPtr(thrust::device_vector<T> &vec) {
  return thrust::raw_pointer_cast(&(vec[0]));
}

bool CallRetrieveHashJoin(
    const unsigned n_queries, const unsigned num_hash_functions,
    const unsigned *d_keys, const unsigned table_size, const Entry *d_contents,
    const Functions<2> constants_2, const Functions<3> constants_3,
    const Functions<4> constants_4, const Functions<5> constants_5,
    const uint2 stash_constants, const unsigned stash_count,
    unsigned &result_size, unsigned *d_temp_query, unsigned *d_temp_table,
    unsigned *d_out_query, unsigned *d_out_table, bool anti,
    cudaStream_t computeStream) {
  unsigned *d_retrieval_probes = NULL;
#ifdef TRACK_ITERATIONS
  CUDA_SAFE_MALLOC(
      cudaMalloc((void **)&d_retrieval_probes, sizeof(unsigned) * n_queries));
#endif

  dim3 gridDim = ComputeGridDim(n_queries);
  int block_count = gridDim.x * gridDim.y;
  if (block_count < 4) block_count = 4;

  thrust::device_vector<unsigned> d_block_result_sizes, d_block_scans;
  try {
    d_block_result_sizes.resize(block_count);
    d_block_scans.resize(block_count);
  } catch (exception e) {
    cout << "Error: out of memory at " << __FILE__ << ", " << __LINE__ << endl;
    return false;
  }

  if (num_hash_functions == 2) {
    retrieve_hash_join<<<gridDim, kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_2, stash_constants,
        stash_count, d_temp_query, d_temp_table, rawPtr(d_block_result_sizes),
        d_retrieval_probes, anti);
  } else if (num_hash_functions == 3) {
    retrieve_hash_join<<<gridDim, kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_3, stash_constants,
        stash_count, d_temp_query, d_temp_table, rawPtr(d_block_result_sizes),
        d_retrieval_probes, anti);
  } else if (num_hash_functions == 4) {
    retrieve_hash_join<<<gridDim, kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_4, stash_constants,
        stash_count, d_temp_query, d_temp_table, rawPtr(d_block_result_sizes),
        d_retrieval_probes, anti);
  } else {
    retrieve_hash_join<<<gridDim, kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_5, stash_constants,
        stash_count, d_temp_query, d_temp_table, rawPtr(d_block_result_sizes),
        d_retrieval_probes, anti);
  }
  cudaError_t err = cudaStreamSynchronize(computeStream);
  if (err != cudaSuccess) {
    gpuErrchk(err);
    COGADB_ERROR("Could not retrieve HashJoin result from hash table! "
                     << "(retrieve_hash_join<<<gridDim, kBlockSize, 0, "
                        "computeStream>>>)",
                 "");
    return false;
  }

  // compaction to positionlist
  try {
    thrust::exclusive_scan(d_block_result_sizes.begin(),
                           d_block_result_sizes.end(), d_block_scans.begin());
  } catch (std::bad_alloc &e) {
    COGADB_WARNING(e.what(), "");
    return false;
  } catch (thrust::system_error &e) {
    COGADB_ERROR(e.what(), "");
    return false;
  }

  unsigned count = d_block_scans[gridDim.x * gridDim.y - 1] +
                   d_block_result_sizes[gridDim.x * gridDim.y - 1];
  block_compaction<<<gridDim, kBlockSize, 0, computeStream>>>(
      n_queries, rawPtr(d_block_result_sizes), rawPtr(d_block_scans),
      d_temp_query, d_temp_table, d_out_query, d_out_table);
  err = cudaStreamSynchronize(computeStream);
  if (err != cudaSuccess) {
    gpuErrchk(err);
    COGADB_ERROR(
        "Could not retrieve HashJoin result from hash table! "
            << "(block_compaction<<<gridDim, kBlockSize, 0, computeStream>>>)",
        "");
    return false;
  }

#ifdef TRACK_ITERATIONS
  OutputRetrievalStatistics(n_queries, d_retrieval_probes, num_hash_functions);
  CUDA_CHECKED_CALL(cudaFree(d_retrieval_probes));
#endif
  result_size = count;
  return true;
}

bool CallBitmapSemiJoin(
    const unsigned n_queries, const unsigned num_hash_functions,
    const unsigned *d_keys, const unsigned table_size, const Entry *d_contents,
    const Functions<2> constants_2, const Functions<3> constants_3,
    const Functions<4> constants_4, const Functions<5> constants_5,
    const uint2 stash_constants, const unsigned stash_count, char *d_flag_array,
    bool anti, cudaStream_t computeStream) {
  unsigned *d_retrieval_probes = NULL;
#ifdef TRACK_ITERATIONS
  CUDA_SAFE_MALLOC(
      cudaMalloc((void **)&d_retrieval_probes, sizeof(unsigned) * n_queries));
#endif

  dim3 gridDim = ComputeGridDim(n_queries);

  if (num_hash_functions == 2) {
    retrieve_bitmap_semi_join<<<gridDim, kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_2, stash_constants,
        stash_count, d_flag_array, anti, d_retrieval_probes);
  } else if (num_hash_functions == 3) {
    retrieve_bitmap_semi_join<<<gridDim, kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_3, stash_constants,
        stash_count, d_flag_array, anti, d_retrieval_probes);
  } else if (num_hash_functions == 4) {
    retrieve_bitmap_semi_join<<<gridDim, kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_4, stash_constants,
        stash_count, d_flag_array, anti, d_retrieval_probes);
  } else {
    retrieve_bitmap_semi_join<<<gridDim, kBlockSize, 0, computeStream>>>(
        n_queries, d_keys, table_size, d_contents, constants_5, stash_constants,
        stash_count, d_flag_array, anti, d_retrieval_probes);
  }

  cudaStreamSynchronize(computeStream);

  CUDA_CHECK_ERROR("Retrieval failed.\n");

#ifdef TRACK_ITERATIONS
  OutputRetrievalStatistics(n_queries, d_retrieval_probes, num_hash_functions);

  CUDA_CHECKED_CALL(cudaFree(d_retrieval_probes));
#endif
  return true;
}

};  // namespace CUDAWrapper

};  // namespace CuckooHashing
};  // namespace CudaHT
