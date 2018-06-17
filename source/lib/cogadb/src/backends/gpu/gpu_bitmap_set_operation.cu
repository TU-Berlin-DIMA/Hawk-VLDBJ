
#include <backends/gpu/bitmap_set_operation.hpp>
#include <backends/gpu/stream_manager.hpp>

namespace CoGaDB {

__global__ void bitmap_and_kernel(char* left_bitmap, char* right_bitmap,
                                  size_t num_of_bits, char* result_bitmap) {
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < (num_of_bits + 7) / 8) {
    result_bitmap[tid] = left_bitmap[tid] & right_bitmap[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void bitmap_or_kernel(char* left_bitmap, char* right_bitmap,
                                 size_t num_of_bits, char* result_bitmap) {
  TID tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < (num_of_bits + 7) / 8) {
    result_bitmap[tid] = left_bitmap[tid] | right_bitmap[tid];
    tid += blockDim.x * gridDim.x;
  }
}

const BitmapPtr GPU_BitmapSetOperation::computeBitmapSetOperation(
    BitmapPtr left_bitmap, BitmapPtr right_bitmap,
    const BitmapOperationParam& param) {
  if (!left_bitmap || !right_bitmap) {
    return BitmapPtr();
  }

  assert(left_bitmap->size() == right_bitmap->size());

  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 512;
  size_t num_of_bits = left_bitmap->size();

  BitmapPtr gpu_bitmap = createBitmap(
      num_of_bits, false, false,
      getMemoryID(
          param
              .proc_spec));  //(new GPU_Bitmap(num_rows_base_table,false,true));
  if (!gpu_bitmap) {
    std::cerr << "Ran out of memory during computeBitmapOperationAnd!"
              << std::endl;
    return BitmapPtr();
  }
  cudaStream_t* stream = gpu::StreamManager::instance().getStream();
  if (param.bitmap_op == BITMAP_AND) {
    bitmap_and_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                        *stream>>>(left_bitmap->data(), right_bitmap->data(),
                                   num_of_bits, gpu_bitmap->data());
  } else if (param.bitmap_op == BITMAP_OR) {
    bitmap_or_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                       *stream>>>(left_bitmap->data(), right_bitmap->data(),
                                  num_of_bits, gpu_bitmap->data());
  } else {
    COGADB_FATAL_ERROR("Unkwon BitOperation!", "");
  }
  cudaStreamSynchronize(*stream);

  return gpu_bitmap;
}

};  // end namespace CoGaDB