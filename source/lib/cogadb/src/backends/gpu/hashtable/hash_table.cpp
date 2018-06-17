// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision:$
// $Date:$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/*
 * @file hash_table.cpp
 *
 * @brief Implements a basic hash table that stores one value per key.
 */

#include <backends/gpu/hashtable/hash_table.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include "cuda_util.h"
#include "debugging.h"
#include "mt19937ar.h"

using namespace std;

namespace CudaHT {
namespace CuckooHashing {
char buffer[256];

// ! @name Internal
// / @{
dim3 ComputeGridDim(unsigned n) {
  // Round up in order to make sure all items are hashed in.
  dim3 grid((n + kBlockSize - 1) / kBlockSize);

  if (grid.x > kGridSize) {
    grid.y = (grid.x + kGridSize - 1) / kGridSize;
    grid.x = kGridSize;
  }

  return grid;
}

unsigned ComputeMaxIterations(const unsigned n, const unsigned table_size,
                              const unsigned num_functions) {
  float lg_input_size = (float)(log((double)n) / log(2.0));

// #define CONSTANT_ITERATIONS

#ifdef CONSTANT_ITERATIONS

  // Set the maximum number of iterations to 7lg(N).
  const unsigned MAX_ITERATION_CONSTANT = 7;
  unsigned max_iterations = MAX_ITERATION_CONSTANT * lg_input_size;
#else

  // Use an empirical formula for determining what the maximum number of
  // iterations should be.  Works OK in most situations.
  float load_factor = float(n) / table_size;
  float ln_load_factor = (float)(log(load_factor) / log(2.71828183));
  unsigned max_iterations =
      (unsigned)(4.0 * ceil(-1.0 / (0.028255 + 1.1594772 * ln_load_factor) *
                            lg_input_size));
#endif

// Debugging: Print out how many iterations are allowed.

#ifdef _DEBUG
  sprintf(buffer, "Maximum iterations: %u (%f lg (N))", max_iterations,
          max_iterations / lg_input_size);
  PrintMessage(buffer, false);
  sprintf(buffer, "Number of functions: %u", num_functions);
  PrintMessage(buffer, false);
#endif

  return max_iterations;
}

// / @}

HashTable::HashTable()
    : table_size_(0),
      d_contents_(NULL),
      stash_count_(0),
      d_failures_(NULL),
      table_temp_size(0),
      query_temp_size(0),
      d_table_temp(NULL),
      d_query_temp(NULL) {}

bool HashTable::Initialize(const unsigned max_table_entries,
                           const cudaStream_t stream, const float space_usage,
                           const unsigned num_functions) {
  Release();

  computeStream = stream;

  // Determine the minimum amount of slots the table requires,
  // and whether the space_usage is within range.
  float minimum_space_usage;

  if ((num_functions < 2) || (num_functions > 5)) {
    char message[256] =
        "Number of hash functions must be from 2 to 5; "
        "others are unimplemented.";

    PrintMessage(message, true);

    return false;
  } else {
    minimum_space_usage = kMinimumSpaceUsages[num_functions];
  }

  if (space_usage < minimum_space_usage) {
    sprintf(buffer, "Minimum possible space usage for %u functions is %f.",
            num_functions, minimum_space_usage);
    PrintMessage(buffer);

    return false;
  }

  num_hash_functions_ = num_functions;
  table_size_ = unsigned(ceil(max_table_entries * space_usage));

#ifdef _DEBUG
  sprintf(buffer, "Table size: %u slots", table_size_);
  PrintMessage(buffer);
#endif

  // Allocate memory.
  const unsigned slots_to_allocate = table_size_ + kStashSize;

  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&d_contents_, sizeof(Entry) * slots_to_allocate);
  if (err != cudaSuccess) {
    COGADB_ERROR("Could not initialize hash table!", "");
    gpuErrchk(err);
    return false;
  }
  err = cudaMalloc((void **)&d_failures_, sizeof(unsigned));
  if (err != cudaSuccess) {
    COGADB_ERROR("Could not initialize hash table!", "");
    gpuErrchk(err);
    /* cleanup already allocated memory*/
    CUDA_CHECKED_CALL(cudaFree(d_contents_));
    return false;
  }
  return true;
}

void HashTable::Release() {
  if (d_contents_ != NULL) CUDA_CHECKED_CALL(cudaFree(d_contents_));
  if (d_failures_ != NULL) CUDA_CHECKED_CALL(cudaFree(d_failures_));
  if (d_table_temp != NULL) CUDA_CHECKED_CALL(cudaFree(d_table_temp));
  if (d_query_temp != NULL) CUDA_CHECKED_CALL(cudaFree(d_query_temp));

  d_table_temp = NULL;
  d_query_temp = NULL;
  d_contents_ = NULL;
  d_failures_ = NULL;

  table_size_ = 0;
  query_temp_size = 0;
  table_temp_size = 0;
}

bool HashTable::Build(const unsigned n, const unsigned *d_keys,
                      const unsigned *d_values) {
  unsigned max_iterations =
      ComputeMaxIterations(n, table_size_, num_hash_functions_);
  unsigned num_failures = 1;
  unsigned num_attempts = 0;

  // Storage for statistics collection.
  unsigned *d_iterations_taken = NULL;

#ifdef TRACK_ITERATIONS
  CUDA_SAFE_MALLOC(
      cudaMalloc((void **)&d_iterations_taken, sizeof(unsigned) * n));
#endif

  // Track how many items ended up in the stash.
  unsigned *d_stash_count = NULL;

  CUDA_SAFE_MALLOC(cudaMalloc((void **)&d_stash_count, sizeof(unsigned)));

  // Main build loop.
  while (num_failures && (++num_attempts < kMaxRestartAttempts)) {
    CUDA_SAFE_CALL(cudaMemset(d_stash_count, 0, sizeof(unsigned)));

    // Generate new hash functions.
    if (num_hash_functions_ == 2) {
      constants_2_.Generate(n, d_keys, table_size_);
    } else if (num_hash_functions_ == 3) {
      constants_3_.Generate(n, d_keys, table_size_);
    } else if (num_hash_functions_ == 4) {
      constants_4_.Generate(n, d_keys, table_size_);
    } else {
      constants_5_.Generate(n, d_keys, table_size_);
    }

    stash_constants_.x = std::max(1lu, genrand_int32()) % kPrimeDivisor;
    stash_constants_.y = genrand_int32() % kPrimeDivisor;
    stash_count_ = 0;

    // Initialize memory.
    unsigned slots_in_table = table_size_ + kStashSize;

    CUDAWrapper::ClearTable(slots_in_table, kEntryEmpty, d_contents_,
                            computeStream);

    num_failures = 0;

    cudaError_t err = CUDAWrapper::CallCuckooHash(
        n, num_hash_functions_, d_keys, d_values, table_size_, constants_2_,
        constants_3_, constants_4_, constants_5_, max_iterations, d_contents_,
        stash_constants_, d_stash_count, d_failures_, d_iterations_taken,
        computeStream);
    if (err != cudaSuccess) {
      COGADB_ERROR("Could not create cuckoo hash table!", "");
      gpuErrchk(err);
      return false;
    }

    // Check if successful.
    CUDA_SAFE_CALL(cudaMemcpy(&num_failures, d_failures_, sizeof(unsigned),
                              cudaMemcpyDeviceToHost));

#ifdef COUNT_UNINSERTED
    if (num_failures || err != cudaSuccess) {
      COGADB_ERROR("Failed to insert " << num_failures << " items!", "");
    }
#endif
  }

  // Copy out the stash size.
  CUDA_SAFE_CALL(cudaMemcpy(&stash_count_, d_stash_count, sizeof(unsigned),
                            cudaMemcpyDeviceToHost));

  if (stash_count_ && (num_failures == 0)) {
    sprintf(buffer, "Stash size: %u", stash_count_);
    PrintMessage(buffer, true);

#ifdef _DEBUG
    PrintStashContents(d_contents_ + table_size_);
#endif
  }

  CUDA_CHECKED_CALL(cudaFree(d_stash_count));

#ifdef TRACK_ITERATIONS
  if (num_failures == 0) {
    OutputBuildStatistics(n, d_iterations_taken);
  }

  CUDA_CHECKED_CALL(cudaFree(d_iterations_taken));
#endif

  // Dump some info if a restart was required.
  if (num_attempts >= kMaxRestartAttempts) {
    sprintf(buffer, "Completely failed to build");
    PrintMessage(buffer, true);
  } else if (num_attempts > 1) {
    sprintf(buffer, "Needed %u attempts to build", num_attempts);
    PrintMessage(buffer, true);
  }

  //            CUDA_CHECK_ERROR("Error occurred during hash table build.\n");

  return num_failures == 0;
}

void HashTable::Retrieve(const unsigned n_queries, const unsigned *d_keys,
                         unsigned *d_values) {
  CUDAWrapper::CallHashRetrieve(
      n_queries, num_hash_functions_, d_keys, table_size_, d_contents_,
      constants_2_, constants_3_, constants_4_, constants_5_, stash_constants_,
      stash_count_, d_values, computeStream);
}

bool HashTable::RetrieveJoin(const unsigned n_queries, const unsigned *d_keys,
                             unsigned &result_size, unsigned *d_out_query,
                             unsigned *d_out_table) {
  if (d_out_table != NULL) {
    CUDA_SAFE_MALLOC(
        cudaMalloc((void **)&d_table_temp, sizeof(unsigned) * n_queries));
    table_temp_size = n_queries;
  }
  CUDA_SAFE_MALLOC(
      cudaMalloc((void **)&d_query_temp, sizeof(unsigned) * n_queries));
  query_temp_size = n_queries;

  bool success = CUDAWrapper::CallRetrieveHashJoin(
      n_queries, num_hash_functions_, d_keys, table_size_, d_contents_,
      constants_2_, constants_3_, constants_4_, constants_5_, stash_constants_,
      stash_count_, result_size, d_query_temp, d_table_temp, d_out_query,
      d_out_table, false, computeStream);

  if (d_table_temp != NULL) {
    CUDA_CHECKED_CALL(cudaFree(d_table_temp));
    d_table_temp = NULL;
  }
  CUDA_CHECKED_CALL(cudaFree(d_query_temp));
  d_query_temp = NULL;

  return success;
}

bool HashTable::RetrieveSemiJoin(const unsigned n_queries,
                                 const unsigned *d_keys, unsigned &result_size,
                                 unsigned *d_out_query, bool anti) {
  unsigned *d_query_temp;

  CUDA_SAFE_MALLOC(
      cudaMalloc((void **)&d_query_temp, sizeof(unsigned) * n_queries));

  bool success;
  success = CUDAWrapper::CallRetrieveHashJoin(
      n_queries, num_hash_functions_, d_keys, table_size_, d_contents_,
      constants_2_, constants_3_, constants_4_, constants_5_, stash_constants_,
      stash_count_, result_size, d_query_temp, NULL, d_out_query, NULL, anti,
      computeStream);

  CUDA_CHECKED_CALL(cudaFree(d_query_temp));
  return success;
}

bool HashTable::RetrieveBitmapSemiJoin(const unsigned n_queries,
                                       const unsigned *d_query_keys,
                                       char *d_flag_array, bool anti) {
  CUDAWrapper::CallBitmapSemiJoin(
      n_queries, num_hash_functions_, d_query_keys, table_size_, d_contents_,
      constants_2_, constants_3_, constants_4_, constants_5_, stash_constants_,
      stash_count_, d_flag_array, anti, computeStream);
  return true;
}
};  // namesapce CuckooHashing
};  // namespace CudaHT

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

//~ Formatted by Jindent --- http://www.jindent.com
