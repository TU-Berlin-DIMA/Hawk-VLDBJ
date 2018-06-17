
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <backends/gpu/conversion_operations.hpp>
#include <core/column.hpp>

#include <backends/gpu/stream_manager.hpp>
#include <backends/gpu/util.hpp>
#include <statistics/statistics_manager.hpp>
#include <util/time_measurement.hpp>

//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//
// inline void gpuAssert(cudaError_t code, char *file, int line, bool abort =
// true) {
//    if (code != cudaSuccess) {
//        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
//        file, line);
//        if (abort) exit(code);
//    }
//}

namespace CoGaDB {

// set memory to 0 beforehand!

__global__ void convertPositionListToBitmap_create_flagarray_kernel(
    TID* tids, size_t num_tids, size_t num_rows_base_table, char* flag_array) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < num_tids) {
    flag_array[tids[tid]] = 1;
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void convertPositionListToBitmap_pack_flagarray_kernel(
    char* flag_array, size_t num_rows_base_table, char* result_bitmap) {
  // TODO: flag_array als long integer lesen!!!
  // uint64_t* long_flag_array = (uint64_t*) flag_array;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < (num_rows_base_table + 7) / 8) {
    int flag_array_tid = tid * 8;
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

__global__ void convertBitmapToPositionList_setFlagArray_kernel(
    char* bitmap, size_t num_of_bits, char* flag_array) {
  int tid = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
  int flag_array_size = (num_of_bits + 7) / 8;
  flag_array_size *= 8;
  while (tid < flag_array_size) {  // num_of_bits-8){
    flag_array[tid] = ((bitmap[tid / 8] & 1) != 0);
    flag_array[tid + 1] = ((bitmap[tid / 8] & 2) != 0);
    flag_array[tid + 2] = ((bitmap[tid / 8] & 4) != 0);
    flag_array[tid + 3] = ((bitmap[tid / 8] & 8) != 0);
    flag_array[tid + 4] = ((bitmap[tid / 8] & 16) != 0);
    flag_array[tid + 5] = ((bitmap[tid / 8] & 32) != 0);
    flag_array[tid + 6] = ((bitmap[tid / 8] & 64) != 0);
    flag_array[tid + 7] = ((bitmap[tid / 8] & 128) != 0);
    tid += (blockDim.x * gridDim.x) * 8;
  }
}

__global__ void convertBitmapToPositionList_createPositionList_kernel(
    char* flag_array, size_t flag_array_size, TID* write_positions,
    TID* result_tids) {
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < flag_array_size) {
    if (flag_array[tid]) {
      TID write_index = write_positions[tid];
      result_tids[write_index] = tid;
    }
    tid += blockDim.x * gridDim.x;
  }
}

const BitmapPtr convertPositionListToBitmap(
    PositionListPtr gpu_pos_list, size_t num_rows_base_table,
    const ProcessorSpecification& proc_spec) {
  // assert(gpu_pos_list!=NULL);
  if (!gpu_pos_list) return BitmapPtr();

  TID* gpu_tids = gpu_pos_list->data();
  assert(gpu_tids != NULL);

  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 1024;

  //    convertPositionListToBitmap_create_flagarray_kernel(TID* tids, size_t
  //    num_tids, size_t num_rows_base_table, char* flag_array);
  //    convertPositionListToBitmap_pack_flagarray_kernel(char* flag_array,
  //    size_t num_rows_base_table, char* result_bitmap);
  unsigned int number_of_bytes_for_flag_array = (num_rows_base_table + 7) / 8;
  number_of_bytes_for_flag_array *= 8;
  thrust::device_ptr<char> flag_array;
  try {
    flag_array = thrust::device_malloc<char>(number_of_bytes_for_flag_array);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during convertPositionListToBitmap!"
              << std::endl;
    return BitmapPtr();
  }

  cudaStream_t* stream = gpu::StreamManager::instance().getStream();
  assert(flag_array.get() != NULL);
  cudaError_t err =
      cudaMemsetAsync(flag_array.get(), 0, num_rows_base_table, *stream);
  gpuErrchk(err);
  // cudaStreamSynchronize ( *stream );

  convertPositionListToBitmap_create_flagarray_kernel<<<
      number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      gpu_tids, gpu_pos_list->size(), num_rows_base_table, flag_array.get());
  cudaStreamSynchronize(*stream);

  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  // create gpu bitmap and init bitmap with 0 (false)
  BitmapPtr gpu_bitmap = createBitmap(
      num_rows_base_table, false, true,
      getMemoryID(
          proc_spec));  //(new GPU_Bitmap(num_rows_base_table,false,true));
  if (!gpu_bitmap) {
    thrust::device_free(flag_array);
    std::cerr << "Ran out of memory during convertPositionListToBitmap!"
              << std::endl;
    return BitmapPtr();
  }
  convertPositionListToBitmap_pack_flagarray_kernel<<<
      number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      flag_array.get(), num_rows_base_table, gpu_bitmap->data());
  cudaStreamSynchronize(*stream);
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  thrust::device_free(flag_array);

  return gpu_bitmap;
}

const PositionListPtr convertBitmapToPositionList(
    BitmapPtr gpu_bitmap, const ProcessorSpecification& proc_spec) {
  if (!gpu_bitmap) return PositionListPtr();
  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 1024;

  unsigned int input_array_size = gpu_bitmap->size();
  unsigned int flag_array_size = (gpu_bitmap->size() + 7) / 8;
  flag_array_size *= 8;

  thrust::device_ptr<char> flag_array;
  try {
    flag_array = thrust::device_malloc<char>(flag_array_size);
    assert(flag_array.get() != NULL);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during convertBitMapToPositionList!"
              << std::endl;
    return PositionListPtr();
  }
  cudaStream_t* stream = gpu::StreamManager::instance().getStream();
  // cudaError_t err = cudaMemset(flag_array.get(), 0, flag_array_size);
  cudaError_t err =
      cudaMemsetAsync(flag_array.get(), 0, flag_array_size, *stream);
  gpuErrchk(err);
  cudaStreamSynchronize(*stream);

  //        std::cout << "flag array: " << (void*) flag_array.get() <<
  //        std::endl;
  thrust::device_ptr<TID> write_positions_array;
  try {
    write_positions_array = thrust::device_malloc<TID>(input_array_size);
    assert(write_positions_array.get() != NULL);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during convertBitMapToPositionList!"
              << std::endl;
    thrust::device_free(flag_array);
    return PositionListPtr();
  }

  convertBitmapToPositionList_setFlagArray_kernel<<<
      number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      gpu_bitmap->data(), gpu_bitmap->size(), flag_array.get());
  cudaStreamSynchronize(*stream);
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  try {
    thrust::exclusive_scan(flag_array, flag_array + input_array_size,
                           write_positions_array);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during convertBitMapToPositionList!"
              << std::endl;
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    return PositionListPtr();
  }

  unsigned int result_size;  // = write_positions_array[input_array_size-1];
  char last_flag;

  TID* tmp = thrust::raw_pointer_cast(write_positions_array);
  err = cudaMemcpyAsync(&result_size, (void*)&tmp[input_array_size - 1],
                        sizeof(TID), cudaMemcpyDeviceToHost, *stream);
  gpuErrchk(err);
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() )
  char* tmp2 = thrust::raw_pointer_cast(flag_array);
  err = cudaMemcpyAsync(&last_flag, (void*)&tmp2[input_array_size - 1],
                        sizeof(char), cudaMemcpyDeviceToHost, *stream);
  gpuErrchk(err);
  // wair for both copy operations to complete
  cudaStreamSynchronize(*stream);
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  // since we perform an exclusive scan, to get the write positions,
  // we have to check whether the last element in the flag array was a
  // match and adjust the result size accordingly
  if (last_flag) result_size++;
  // std::cout << "Bitmap to PositionList: #tids: " << result_size << std::endl;

  PositionListPtr result_tids =
      createPositionList(result_size, getMemoryID(proc_spec));
  if (!result_tids) {
    // free allocated memory
    thrust::device_free(flag_array);
    thrust::device_free(write_positions_array);
    std::cerr << "Ran out of memory during FetchJoin!" << std::endl;
    return PositionListPtr();
  }

  convertBitmapToPositionList_createPositionList_kernel<<<
      number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      flag_array.get(), input_array_size, write_positions_array.get(),
      result_tids->data());
  cudaStreamSynchronize(*stream);

  // deallocate with device_free
  thrust::device_free(write_positions_array);
  thrust::device_free(flag_array);

  return result_tids;
}

template <typename T>
const BitmapPtr GPU_ConversionOperation<T>::convertToBitmap(
    PositionListPtr tids, size_t num_rows_base_table,
    const ProcessorSpecification& proc_spec) {
  return convertPositionListToBitmap(tids, num_rows_base_table, proc_spec);
}

template <typename T>
const PositionListPtr GPU_ConversionOperation<T>::convertToPositionList(
    BitmapPtr bitmap, const ProcessorSpecification& proc_spec) {
  return convertBitmapToPositionList(bitmap, proc_spec);
}

template <typename T>
__global__ void convert_to_double_array_kernel(T* array, size_t array_size,
                                               double* output_array) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < array_size) {
    output_array[tid] = double(array[tid]);
    tid += blockDim.x * gridDim.x;
  }
}

template <typename T>
const DoubleDenseValueColumnPtr
GPU_ConversionOperation<T>::convertToDoubleDenseValueColumn(
    const std::string& column_name, T* array, size_t num_elements,
    const ProcessorSpecification& proc_spec) {
  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 1024;

  cudaStream_t* stream = gpu::StreamManager::instance().getStream();

  DoubleDenseValueColumnPtr double_column(
      new DoubleDenseValueColumn(column_name, DOUBLE, getMemoryID(proc_spec)));
  try {
    double_column->resize(num_elements);
  } catch (const std::bad_alloc& e) {
    std::cerr << "Ran out of memory during convertToDoubleDenseValueColumn()!"
              << std::endl;
    return DoubleDenseValueColumnPtr();
  }

  convert_to_double_array_kernel<
      T><<<number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      array, num_elements, double_column->data());

  cudaError_t err = cudaStreamSynchronize(*stream);
  if (err != cudaSuccess) {
    COGADB_ERROR("Error executing kernel convert_to_double_array_kernel<T>!",
                 "");
    gpuErrchk(err);
    return DoubleDenseValueColumnPtr();
  }

  return double_column;
}

template <>
const DoubleDenseValueColumnPtr
GPU_ConversionOperation<std::string>::convertToDoubleDenseValueColumn(
    const std::string& column_name, std::string* array, size_t num_elements,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR(
      "Called convertToDoubleDenseValueColumn() for dense value VARCHAR "
      "column!",
      "");
  return DoubleDenseValueColumnPtr();
}

template <>
const DoubleDenseValueColumnPtr
GPU_ConversionOperation<C_String>::convertToDoubleDenseValueColumn(
    const std::string& column_name, C_String* array, size_t num_elements,
    const ProcessorSpecification& proc_spec) {
  COGADB_FATAL_ERROR(
      "Called convertToDoubleDenseValueColumn() for dense value VARCHAR "
      "column!",
      "");
  return DoubleDenseValueColumnPtr();
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(GPU_ConversionOperation);

};  // end namespace CoGaDB
