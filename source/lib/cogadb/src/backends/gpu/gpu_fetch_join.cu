
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <backends/gpu/fetch_join.hpp>
#include <backends/gpu/stream_manager.hpp>
#include <backends/gpu/util.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <core/column.hpp>
#include <lookup_table/join_index.hpp>
#include <moderngpu.cuh>
#include <util/time_measurement.hpp>

namespace CoGaDB {

using namespace std;
using namespace mgpu;
// using namespace CoGaDB::CDK::selection;

namespace gpu {
//#define VALIDATE_GPU_MEMORY_COST_MODELS

typedef int64_t SIGNED_INTEGER;

//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//        inline void gpuAssert(cudaError_t code, const char *file, int line,
//        bool abort = true) {
//            if (code != cudaSuccess) {
//                fprintf(stderr, "GPUassert: %s %s %d\n",
//                cudaGetErrorString(code), file, line);
//                if (abort) exit(code);
//            }
//        }

// FIXME: MAke sure this is still correct after port to 64 bit!!!
// assumes that matching tids is unique!

__device__ SIGNED_INTEGER binary_search(TID* matching_tids,
                                        TID number_of_matching_tids,
                                        TID search_val) {
  SIGNED_INTEGER low = 0;
  SIGNED_INTEGER high = number_of_matching_tids - 1;
  SIGNED_INTEGER mid = low + ((high - low) / 2);

  while (low <= high) {  //&& !(matching_tids[mid - 1] <= search_val &&
    // matching_tids[mid] > search_val))
    if (mid < 0 || mid >= number_of_matching_tids)
      return number_of_matching_tids;

    if (matching_tids[mid] > search_val) {
      high = mid - 1;
    } else if (matching_tids[mid] < search_val) {
      low = mid + 1;
    } else {
      return (TID)mid;  // found
    }
    mid = low + ((high - low) / 2);
  }

  return number_of_matching_tids;  // not found
}

// template<typename T> __device__ inline TID my_binary_index_search(T * in, T
// elem, long startIdx, long endIdx){
//	long from = startIdx;
//	long to = endIdx;
//	long mid = (long) (from + to) / 2;
//
//	while (from <= to && !(in[mid - 1] <= elem && in[mid] > elem))
//	{
//		if (in[mid] > elem)
//			to = mid - 1;
//		else
//			from = mid + 1;
//		mid = (long) (from + to) / 2;
//	}
//	return (mid == endIdx && in[endIdx] <= elem) ? mid + 1 : mid;
//}

__global__ void fetch_join_kernel(TID* matching_tids,
                                  size_t number_of_matching_tids,
                                  TID* pk_column_tids, size_t input_array_size,
                                  char* flags) {
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < input_array_size) {
    TID index = binary_search(matching_tids, number_of_matching_tids,
                              pk_column_tids[tid]);
    flags[tid] = (index < number_of_matching_tids);
    tid += blockDim.x * gridDim.x;
  }
}

// kernel2: fetch matching row ids to result buffer

__global__ void fetch_marked_tids_in_output_buffer_kernel(
    TID* fk_column_tids, size_t input_array_size, char* flags,
    TID* write_positions_array, TID* result_buffer) {
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < input_array_size) {
    if (flags[tid]) {
      TID write_index = write_positions_array[tid];
      result_buffer[write_index] = fk_column_tids[tid];
    }
    tid += blockDim.x * gridDim.x;
  }
}

// kernel 3: works similar to fetch_marked_tids_in_output_buffer_kernel, but
// this kernel writes the tids from the primary key table as well

__global__ void fetch_marked_tid_pairs_in_output_buffer_kernel(
    TID* pk_column_tids, TID* fk_column_tids, size_t input_array_size,
    char* flags, TID* write_positions_array,
    TID* result_buffer_for_tids_of_primary_key_table,
    TID* result_buffer_for_tids_of_foreign_key_table) {
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < input_array_size) {
    if (flags[tid]) {
      TID write_index = write_positions_array[tid];
      result_buffer_for_tids_of_primary_key_table[write_index] =
          pk_column_tids[tid];
      result_buffer_for_tids_of_foreign_key_table[write_index] =
          fk_column_tids[tid];
    }
    tid += blockDim.x * gridDim.x;
  }
}

//

__global__ void fetch_join_bitmap_kernel(TID* matching_tids,
                                         size_t number_of_matching_tids,
                                         TID* pk_column_tids,
                                         TID* fk_column_tids,
                                         size_t join_index_size, char* flags) {
  // threads align to join index entries
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < join_index_size) {
    // try to find primary key for this thread in matching tids
    TID index = binary_search(matching_tids, number_of_matching_tids,
                              pk_column_tids[tid]);

    // get foreign key for this thread
    TID fk = fk_column_tids[tid];

    // scatter to flag array at foreign key index
    if (index < number_of_matching_tids) flags[fk] = 1;

    // flags[fk] = index < number_of_matching_tids;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void scatter_fk_bitmap(size_t join_index_size, char* flags_in,
                                  TID* fk_column_tids, char* flags_out) {
  // threads align to join index entries
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < join_index_size) {
    // get foreign key for this thread
    TID fk = fk_column_tids[tid];

    // scatter to flag array at foreign key index
    if (flags_in[tid]) flags_out[fk] = 1;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void pack_flagarray_kernel(char* flag_array,
                                      size_t num_rows_base_table,
                                      char* result_bitmap) {
  // TODO: flag_array als long integer lesen!!!
  // uint64_t* long_flag_array = (uint64_t*) flag_array;
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

typedef mgpu::ContextPtr GPUContextPtr;

/*
const GPU_BitmapPtr
GPU_Operators::createBitmapOfMatchingTIDsFromJoinIndex(GPU_JoinIndexPtr
join_index, GPU_PositionlistPtr pk_table_tids) {

    bool measure = false;

    //-------- Get device pointers and sizes --------

    GPU_BitmapPtr result;
    if(!join_index || !pk_table_tids) return result;

    TID* query_tids = getRawPointerToGPU_Positionlist( pk_table_tids );
    size_t num_query_tids = pk_table_tids->size();

    TID* primary_keys = getRawPointerToGPU_Positionlist(
join_index->pos_list_pair->first );
    TID* foreign_keys = getRawPointerToGPU_Positionlist(
join_index->pos_list_pair->second );
    size_t join_index_size = join_index->pos_list_pair->first->size();
    size_t bitmap_size = ((join_index_size + 7) / 8) * 8;

    size_t number_of_blocks = 512;
    size_t number_of_threads = 1024;

    //-------- Init cuda objects --------

    cudaStream_t* stream = StreamManager::instance().getStream();
    GPUContextPtr contextptr = StreamManager::instance().getCudaContext(0);
    mgpu::CudaContext *context = contextptr.get();

    //-------- Reserve space for output data --------

    //Flags for primary keys and foreign keys from join-index
    //MGPU_MEM(char) flags_primary = context->Malloc<char>(bitmap_size);
    thrust::device_ptr<char> flags_primary;
    thrust::device_ptr<char> flags_foreign;
    //allocate memory for flags_primary
    try{
        flags_primary = thrust::device_malloc<char>(bitmap_size);
    }catch(std::bad_alloc &e){
        std::cerr << "Ran out of memory during
createBitmapOfMatchingTIDsFromJoinIndex!" << std::endl;
        return GPU_BitmapPtr();
    }
    //allocate memory for flags_primary
    try{
        flags_foreign = thrust::device_malloc<char>(bitmap_size);
    }catch(std::bad_alloc &e){
        std::cerr << "Ran out of memory during
createBitmapOfMatchingTIDsFromJoinIndex!" << std::endl;
        if(flags_primary.get()) thrust::device_free(flags_primary);
        return GPU_BitmapPtr();
    }
    gpuErrchk(cudaMemsetAsync(flags_foreign.get(), 0, bitmap_size));
    cudaStreamSynchronize ( *stream );

    //Compressed bitmap
    GPU_BitmapPtr gpu_bitmap=createGPU_Bitmap(join_index_size,false,true);
    if(!gpu_bitmap){
          if(flags_primary.get()) thrust::device_free(flags_primary);
          if(flags_foreign.get()) thrust::device_free(flags_foreign);
          //thrust::device_free(flags_foreign);
          std::cerr << "Ran out of memory during
createBitmapOfMatchingTIDsFromJoinIndex!" << std::endl;
          return GPU_BitmapPtr();
    }

    //--------------- Perform fetch join ----------------
    context->Start();

    //Vectorized Sorted Search of query tids in primary keys (generate flags for
join index rows)
    SortedSearch<MgpuBoundsLower, MgpuSearchTypeMatch, MgpuSearchTypeNone>
        (primary_keys, join_index_size, query_tids, num_query_tids,
flags_primary.get(), (int*)0, *context);

    //For every match: Scatter bit with foreign key as index
    scatter_fk_bitmap<<<number_of_blocks,
number_of_threads,0,*stream>>>(join_index_size, flags_primary.get(),
foreign_keys, flags_foreign.get());
    cudaStreamSynchronize ( *stream );


    //Pack flags to binary bitmap
    pack_flagarray_kernel<<<number_of_blocks,number_of_threads,0,*stream>>>(flags_foreign.get(),
join_index_size, gpu_bitmap->data());
    cudaStreamSynchronize ( *stream );

    if(flags_primary.get()) thrust::device_free(flags_primary);
    if(flags_foreign.get()) thrust::device_free(flags_foreign);


    double time = context->Split();
    if(measure)
        cout << "ModernGPU SortedSearch + scatter-kernel + packing-kernel took "
<< time*1000.0f << "ms" << endl;

    return gpu_bitmap;
}
//*/

///*

const BitmapPtr createBitmapOfMatchingTIDsFromJoinIndex(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const ProcessorSpecification& proc_spec) {
  if (!join_index || !pk_table_tids) return BitmapPtr();

  hype::ProcessingDeviceMemoryID mem_id = getMemoryID(proc_spec);

  TID* matching_tids = pk_table_tids->data();
  size_t number_of_matching_tids = pk_table_tids->size();
  TID* pk_column_tids = join_index->first->getPositionList()->data();
  TID* fk_column_tids = join_index->second->getPositionList()->data();
  size_t join_index_size = join_index->first->getPositionList()->size();

  cudaStream_t* stream = StreamManager::instance().getStream();
  size_t number_of_blocks = 512;
  size_t number_of_threads_per_block = 1024;

  if (!quiet) {
    std::cout << "matching_tids: " << (void*)matching_tids << std::endl;
    std::cout << "#PK_INPUT_TIDS: " << number_of_matching_tids << std::endl;
    std::cout << "pk_column_tids: " << (void*)pk_column_tids << std::endl;
    std::cout << "#Rows in Join Index: "
              << join_index->first->getPositionList()->size() << std::endl;
  }

  thrust::device_ptr<char> flag_array;
  size_t number_of_flags = ((join_index_size + 7) / 8) * 8;
  try {
    flag_array = thrust::device_malloc<char>(number_of_flags);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during primary key fetch!" << std::endl;
    return BitmapPtr();
  }
  gpuErrchk(cudaMemsetAsync(flag_array.get(), 0, number_of_flags));
  cudaStreamSynchronize(*stream);

  fetch_join_bitmap_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                             *stream>>>(matching_tids, number_of_matching_tids,
                                        pk_column_tids, fk_column_tids,
                                        join_index_size, flag_array.get());

  cudaStreamSynchronize(*stream);

  BitmapPtr bitmap = createBitmap(join_index_size, false, true, mem_id);
  if (!bitmap) {
    thrust::device_free(flag_array);
    std::cerr << "Ran out of memory during BitmapFetchJoin!" << std::endl;
    return BitmapPtr();
  }

  pack_flagarray_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                          *stream>>>(flag_array.get(), join_index_size,
                                     bitmap->data());
  cudaStreamSynchronize(*stream);
  if (flag_array.get()) thrust::device_free(flag_array);
  return bitmap;
}
//*/

const PositionListPtr fetchMatchingTIDsFromJoinIndex_Param(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const ProcessorSpecification& proc_spec, size_t number_of_blocks,
    size_t number_of_threads_per_block) {
  Timestamp start, end;
  start = getTimestamp();

#ifdef VALIDATE_GPU_MEMORY_COST_MODELS
  size_t gpu_memory_start =
      getTotalGPUMemorySizeInByte() - getFreeGPUMemorySizeInByte();
  size_t gpu_memory_start_without_input_column;
  // if(gpu_memory_start>(pk_table_tids->size()*sizeof(TID)))
  gpu_memory_start_without_input_column =
      gpu_memory_start - (pk_table_tids->size() * sizeof(TID));
  size_t gpu_memory_start_without_input_column_and_join_index =
      gpu_memory_start_without_input_column -
      join_index->cpu_join_index->first->getPositionList()->size() * 2 *
          sizeof(TID);
#endif

  cudaStream_t* stream = StreamManager::instance().getStream();
  hype::ProcessingDeviceMemoryID mem_id = getMemoryID(proc_spec);

  TID* matching_tids = pk_table_tids->data();
  size_t number_of_matching_tids = pk_table_tids->size();
  TID* pk_column_tids = join_index->first->getPositionList()->data();
  TID* fk_column_tids = join_index->second->getPositionList()->data();
  size_t input_array_size = join_index->first->getPositionList()->size();

  if (!quiet) {
    std::cout << "matching_tids: " << (void*)matching_tids << std::endl;
    std::cout << "#PK_INPUT_TIDS: " << number_of_matching_tids << std::endl;
    std::cout << "pk_column_tids: " << (void*)pk_column_tids << std::endl;
    std::cout << "fk_column_tids: " << (void*)fk_column_tids << std::endl;
    std::cout << "#Rows in Join Index: "
              << join_index->first->getPositionList()->size() << std::endl;
  }

  thrust::device_ptr<char> flag_array;
  thrust::device_ptr<TID> write_positions_array;

  try {
    flag_array = thrust::device_malloc<char>(input_array_size);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during FetchJoin!" << std::endl;
    return PositionListPtr();
  }

  try {
    write_positions_array = thrust::device_malloc<TID>(input_array_size);
  } catch (std::bad_alloc& e) {
    // free allocated memory
    thrust::device_free(flag_array);
    std::cerr << "Ran out of memory during FetchJoin!" << std::endl;
    return PositionListPtr();
  }

  // flag
  fetch_join_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                      *stream>>>(matching_tids, number_of_matching_tids,
                                 pk_column_tids, input_array_size,
                                 flag_array.get());

  cudaStreamSynchronize(*stream);

  // scan
  try {
    thrust::exclusive_scan(flag_array, flag_array + input_array_size,
                           write_positions_array);
  } catch (std::bad_alloc& e) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    std::cerr << "Ran out of memory during FetchJoin!" << std::endl;
    return PositionListPtr();
  }

  size_t result_size;  // = write_positions_array[input_array_size-1];
  char last_flag;

  TID* tmp = thrust::raw_pointer_cast(write_positions_array);
  cudaError_t err =
      cudaMemcpyAsync(&result_size, (void*)&tmp[input_array_size - 1],
                      sizeof(TID), cudaMemcpyDeviceToHost, *stream);
  gpuErrchk(err);
  cudaStreamSynchronize(*stream);
  // gpuErrchk( cudaDeviceSynchronize() )

  char* tmp2 = thrust::raw_pointer_cast(flag_array);
  err = cudaMemcpyAsync(&last_flag, (void*)&tmp2[input_array_size - 1],
                        sizeof(char), cudaMemcpyDeviceToHost, *stream);
  gpuErrchk(err);
  cudaStreamSynchronize(*stream);

  // since we perform an exclusive scan, to get the write positions,
  // we have to check whether the last element in the flag array was a
  // match and adjust the result size accordingly
  if (last_flag) result_size++;

  PositionListPtr result_tids = createPositionList(result_size, mem_id);
  if (!result_tids) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    std::cerr << "Ran out of memory during FetchJoin!" << std::endl;
    return PositionListPtr();
  }

  // fetch
  fetch_marked_tids_in_output_buffer_kernel<<<
      number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      fk_column_tids, input_array_size, flag_array.get(),
      write_positions_array.get(), result_tids->data());

#ifdef VALIDATE_GPU_MEMORY_COST_MODELS
  size_t gpu_memory_end =
      getTotalGPUMemorySizeInByte() -
      getFreeGPUMemorySizeInByte();  // getFreeGPUMemorySizeInByte();

  // if(gpu_memory_start>gpu_memory_start_without_input_column){
  //    COGADB_FATAL_ERROR("Assertion Failed!","");
  //}
  if (gpu_memory_end > gpu_memory_start_without_input_column_and_join_index) {
    COGADB_FATAL_ERROR("Assertion Failed!", "");
  }
  size_t used_memory =
      gpu_memory_end -
      gpu_memory_start_without_input_column_and_join_index;  // gpu_memory_start_without_input_column;

  hype::Tuple feature_vector;
  feature_vector.push_back(
      double(pk_table_tids->size()));  // size of primary key tableb
  feature_vector.push_back(
      double(join_index->cpu_join_index->first->getPositionList()
                 ->size()));  // size of foreign key table

  size_t estimated_gpu_memory_footprint =
      GPU_Operators_Memory_Cost_Models::columnFetchJoin(feature_vector);
  std::cout << "Memory Measurement," << gpu_memory_start << ","
            << gpu_memory_start_without_input_column << "," << gpu_memory_end
            << "," << used_memory << "," << estimated_gpu_memory_footprint
            << std::endl;
#endif
  cudaStreamSynchronize(*stream);

  // deallocate with device_free
  thrust::device_free(write_positions_array);
  thrust::device_free(flag_array);

  // result=GPU_PositionlistPtr(new Impl_GPU_Positionlist(result_tids));
  //            result = GPU_PositionlistPtr(result_tids);

  // Timer
  end = getTimestamp();
  std::cout << "Fetch Matching TIDs from Join Index took "
            << double(end - start) / (1000 * 1000) << endl;

  return result_tids;
}

const PositionListPairPtr fetchJoinResultFromJoinIndex_Param(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const ProcessorSpecification& proc_spec, size_t number_of_blocks,
    size_t number_of_threads_per_block) {
  PositionListPairPtr result;

  cudaStream_t* stream = StreamManager::instance().getStream();
  hype::ProcessingDeviceMemoryID mem_id = getMemoryID(proc_spec);

  TID* matching_tids = pk_table_tids->data();
  size_t number_of_matching_tids = pk_table_tids->size();
  TID* pk_column_tids = join_index->first->getPositionList()->data();
  TID* fk_column_tids = join_index->second->getPositionList()->data();
  size_t input_array_size = join_index->first->getPositionList()->size();

  thrust::device_ptr<char> flag_array;
  thrust::device_ptr<TID> write_positions_array;

  try {
    flag_array = thrust::device_malloc<char>(input_array_size);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during FetchJoin!" << std::endl;
    return PositionListPairPtr();
  }

  try {
    write_positions_array = thrust::device_malloc<TID>(input_array_size);
  } catch (std::bad_alloc& e) {
    // free allocated memory
    thrust::device_free(flag_array);
    std::cerr << "Ran out of memory during FetchJoin!" << std::endl;
    return PositionListPairPtr();
  }

  fetch_join_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                      *stream>>>(matching_tids, number_of_matching_tids,
                                 pk_column_tids, input_array_size,
                                 flag_array.get());
  cudaStreamSynchronize(*stream);

  try {
    thrust::exclusive_scan(flag_array, flag_array + input_array_size,
                           write_positions_array);
  } catch (std::bad_alloc& e) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    std::cerr << "Ran out of memory during FetchJoin!" << std::endl;
    return PositionListPairPtr();
  }

  size_t result_size = 0;  // = write_positions_array[input_array_size-1];
  char last_flag;

  TID* tmp = thrust::raw_pointer_cast(write_positions_array);
  cudaError_t err =
      cudaMemcpyAsync(&result_size, (void*)&tmp[input_array_size - 1],
                      sizeof(TID), cudaMemcpyDeviceToHost, *stream);
  gpuErrchk(err);
  err = cudaStreamSynchronize(*stream);
  gpuErrchk(err);
  // gpuErrchk( cudaDeviceSynchronize() )

  char* tmp2 = thrust::raw_pointer_cast(flag_array);
  err = cudaMemcpyAsync(&last_flag, (void*)&tmp2[input_array_size - 1],
                        sizeof(char), cudaMemcpyDeviceToHost, *stream);
  gpuErrchk(err);
  err = cudaStreamSynchronize(*stream);
  gpuErrchk(err);

  // since we perform an exclusive scan, to get the write positions,
  // we have to check whether the last element in the flag array was a
  // match and adjust the result size accordingly
  if (last_flag) result_size++;

  PositionListPtr result_tids_pk_table =
      createPositionList(result_size, mem_id);
  PositionListPtr result_tids_fk_table =
      createPositionList(result_size, mem_id);

  if (!result_tids_pk_table || !result_tids_fk_table) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    std::cerr << "Ran out of memory during FetchJoin!" << std::endl;
    return PositionListPairPtr();
  }

  fetch_marked_tid_pairs_in_output_buffer_kernel<<<
      number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      pk_column_tids, fk_column_tids, input_array_size, flag_array.get(),
      write_positions_array.get(), result_tids_pk_table->data(),
      result_tids_fk_table->data());

  cudaStreamSynchronize(*stream);
  // deallocate with device_free
  thrust::device_free(write_positions_array);
  thrust::device_free(flag_array);

  result = PositionListPairPtr(new PositionListPair());
  result->first = result_tids_pk_table;
  result->second = result_tids_fk_table;

  return result;
}

};  // end namespace gpu

const PositionListPtr GPU_FetchJoin::tid_fetch_join(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const FetchJoinParam& param) {
  PositionListPtr result;
  if (!join_index || !pk_table_tids) return result;
  result = gpu::fetchMatchingTIDsFromJoinIndex_Param(
      join_index, pk_table_tids, param.proc_spec, 512, 1024);
  //  The following Code is currently used to derive the optimal number of
  //  blocks and blocks per threads for a GPU
  //                       for(size_t
  //                       num_blocks=64;num_blocks<64000;num_blocks*=2){
  //                          for(size_t
  //                          num_thread_per_block=64;num_thread_per_block<=1024;num_thread_per_block*=2){

  //                              std::cout << "Blocks: " << num_blocks << "\t"
  //                              << "Threads per Block: " <<
  //                              num_thread_per_block << std::endl;
  //                              Timestamp begin = getTimestamp();
  //                              result =
  //                              fetchMatchingTIDsFromJoinIndex_Param(join_index,
  //                              pk_table_tids, param.proc_spec,
  //                              num_blocks,num_thread_per_block);
  //                              Timestamp end = getTimestamp();
  //                              std::cout << "Execution Time: " <<
  //                              double(end-begin)/(1000*1000) << "ms" <<
  //                              std::endl;
  //                          }
  //                       }
  return result;
}

const PositionListPairPtr GPU_FetchJoin::fetch_join(
    JoinIndexPtr join_index, PositionListPtr pk_table_tids,
    const FetchJoinParam& param) {
  return gpu::fetchJoinResultFromJoinIndex_Param(join_index, pk_table_tids,
                                                 param.proc_spec, 512, 1024);
}

const BitmapPtr GPU_FetchJoin::bitmap_fetch_join(JoinIndexPtr join_index,
                                                 PositionListPtr pk_table_tids,
                                                 const FetchJoinParam& param) {
  if (!join_index || !pk_table_tids) return BitmapPtr();
  return gpu::createBitmapOfMatchingTIDsFromJoinIndex(join_index, pk_table_tids,
                                                      param.proc_spec);
}

};  // end namespace CoGaDB
