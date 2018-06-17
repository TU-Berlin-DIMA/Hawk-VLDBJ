#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/detail/stable_radix_sort.h>
#include <backends/gpu/stream_manager.hpp>
#include <backends/gpu/util.hpp>
#include <core/base_column.hpp>
#include <core/positionlist.hpp>
#include <cstring>
#include <unittests/unittests.hpp>
#include <util/time_measurement.hpp>
//#include <cub/cub.cuh>
//#include <moderngpu/moderngpu.cuh>

//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool
// abort=true)
//{
//   if (code != cudaSuccess)
//   {
//      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
//      line);
//      if (abort) exit(code);
//   }
//}

#define N 3

namespace CoGaDB {

namespace unit_tests {

using namespace std;
using namespace CoGaDB::gpu;

// planned experiments:
// thrust partition for binning
// thrust sort
// shared memory

// many duplicates (> bin size), need to remove

// set memory to 0 beforehand!
__global__ void convertPositionListToBitmap_create_flagarray_kernel(
    TID* tids, size_t num_tids, size_t num_rows_base_table, char* flag_array) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < num_tids) {
    flag_array[tids[tid]] = 1;
    tid += blockDim.x * gridDim.x;
  }
}

// set memory to 0 beforehand!
__global__ void convert_multi_pass(TID* tids, size_t num_tids,
                                   size_t num_rows_base_table, char* flag_array,
                                   int num_passes) {
  /*
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      int tids_per_pass = (num_rows_base_table + num_passes - 1) / num_passes;

      while(tid<num_tids){
          int flag_index;
          //multi_pass
          for(int i = 0; i < num_passes; i++) {
              flag_index = tids[tid];
              if(flag_index > tids_per_pass*i & flag_index <
     tids_per_pass*(i+1))
                  flag_array[flag_index]=1;
          }
          tid+=blockDim.x * gridDim.x;
      }
  */

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tids_per_pass = (num_rows_base_table + num_passes - 1) / num_passes;

  while (tid < num_tids) {
    int flag_index = tids[tid];
    // multi_pass
    for (int i = 0; i < num_passes; i++) {
      if (flag_index > tids_per_pass * i & flag_index < tids_per_pass * (i + 1))
        flag_array[flag_index] = 1;
    }
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void convert_to_bitmap_kernel(TID* tids, size_t num_tids,
                                         size_t num_rows_base_table,
                                         char* result_bitmap) {}

/*
typedef struct {
    int start;
    int size;
} bin;
*/

__global__ void convertPositionListToBitmap_create_flagarray_bins_kernel(
    TID* tids, size_t num_tids, size_t num_rows_base_table, char* flag_array) {
  __shared__ char s[512];
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int tid;

  while (index < num_rows_base_table) {
    s[threadIdx.x] = 0;
    tid = tids[index];
    s[tid % 512] = 1;
    __syncthreads();
    flag_array[index] = s[threadIdx.x];
    index += blockDim.x * gridDim.x;
  }
}

// Bin size has to be 512 and there need to be 256 elements in each bin
__global__ void convert_bins_kernel(TID* tids, size_t num_tids,
                                    size_t num_rows_base_table,
                                    char* flag_array) {
  __shared__ char s[512];
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int tid;

  while (index < num_rows_base_table) {
    s[threadIdx.x] = 0;
    if (threadIdx.x < 256) {
      tid = tids[threadIdx.x + blockIdx.x * 256];
      s[tid % 512] = 1;
    }
    __syncthreads();
    flag_array[index] = s[threadIdx.x];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void convertPositionListToBitmap_pack_flagarray_kernel(
    char* flag_array, size_t num_rows_base_table, char* result_bitmap) {
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

// Max bins per kernel-call shared memory size 48kb / cache line size 128bytes =
// 375

__global__ void binning_kernel(TID* data, int binsize) {}

/*
__global__ void computeBins_kernel(TID* tids, unsigned int* bins, size_t
num_tids) {


    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid;

    while(index < num_rows_base_table){

        index += blockIdx.x * blockDim.x;
    }
}
*/

// unique
vector<TID> generate_tids(unsigned long int num_rows_base_table,
                          unsigned long int positionlist_size) {
  vector<TID> full_data(num_rows_base_table);

  for (unsigned long int j = 0; j < num_rows_base_table; j++) {
    full_data[j] = j;
  }

  std::random_shuffle(full_data.begin(), full_data.end());

  vector<TID> result_data(&full_data[0], &full_data[positionlist_size]);
  return result_data;
}

// non-unique
vector<TID> generate_duplicate_tids(unsigned long int num_rows_base_table,
                                    unsigned long int positionlist_size) {
  vector<TID> data(positionlist_size);

  for (unsigned long int j = 0; j < positionlist_size; j++) {
    data[j] = rand() % num_rows_base_table;
  }

  return data;
}

// unique
vector<TID> generate_binned_tids(unsigned long int num_rows_base_table,
                                 unsigned long int positionlist_size,
                                 unsigned long int elements_per_bin,
                                 unsigned long int binsize) {
  vector<TID> data(num_rows_base_table);
  vector<TID> result_data;

  for (unsigned long int j = 0; j < num_rows_base_table; j++) {
    data[j] = j;
  }

  unsigned int num_bins = num_rows_base_table / binsize;
  result_data.reserve(elements_per_bin * num_bins);

  for (int i = 0; i < num_bins; i++) {
    TID* first_from_bin = &data[i * binsize];

    std::random_shuffle(first_from_bin, first_from_bin + binsize);
    result_data.insert(result_data.end(), first_from_bin,
                       first_from_bin + elements_per_bin);
  }

  return result_data;
}

// non-unique
vector<TID> generate_binned_duplicate_tids(
    unsigned long int num_rows_base_table, unsigned long int positionlist_size,
    unsigned long int binsize) {
  vector<TID> data(positionlist_size);

  for (unsigned long int j = 0; j < positionlist_size; j++) {
    data[j] = (rand() % binsize) + j - (j % binsize);
  }

  return data;
}

vector<char> generate_flag_array(unsigned long int size) {
  vector<char> data(size);

  for (unsigned long int i = 0; i < size; i++) {
    data[i] = rand() % 2;
  }

  return data;
}

template <typename T>
thrust::device_ptr<T> toDevice(vector<T> data, unsigned long int n) {
  thrust::device_ptr<T> device_data;

  try {
    device_data = thrust::device_malloc<T>(n);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during gpu memory allocation for device "
                 "tid vector"
              << std::endl;
    return device_data;
  }

  cudaError_t err = cudaMemcpy(thrust::raw_pointer_cast(device_data), &data[0],
                               n * sizeof(T), cudaMemcpyHostToDevice);
  gpuErrchk(err);

  return device_data;
}

template <typename T>
std::vector<T> fromDevice(thrust::device_ptr<T> device_data,
                          unsigned long int n) {
  std::vector<T> data(n);

  cudaError_t err = cudaMemcpy(&data[0], thrust::raw_pointer_cast(device_data),
                               n * sizeof(T), cudaMemcpyDeviceToHost);
  gpuErrchk(err);

  return data;
}

struct in_bin {
  __host__ __device__ bool operator()(const int& x) { return (x > (2 << 25)); }
};

double measureCubRadixSort(std::vector<TID> data) {
  /*
    // Create a set of DoubleBuffers to wrap pairs of device pointers
    thrust::device_vector<TID> device_data(data);
    thrust::device_vector<TID> device_data_alt(data);
    cub::DoubleBuffer<TID> d_keys(thrust::raw_pointer_cast(device_data.data()),
    thrust::raw_pointer_cast(device_data_alt.data()));



    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
    data.size());
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation

    Timestamp begin;
    Timestamp end;
    double time = 0.0;
    begin=getTimestamp();
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
    data.size());
    end=getTimestamp();
    time += double(end-begin)/(1000*1000);
    return time; // / n;*/
  return 0;
}

double measureThrustPartition(unsigned int num_rows_base_table,
                              std::vector<TID> data) {
  thrust::device_vector<TID> d_data(data);
  Timestamp begin;
  Timestamp end;
  double time = 0.0;
  begin = getTimestamp();
  thrust::partition(d_data.begin(), d_data.end(), in_bin());
  end = getTimestamp();
  time += double(end - begin) / (1000 * 1000);
  return time;  // / n;
}

double measureThrustSort(unsigned int num_rows_base_table,
                         std::vector<TID> data) {
  thrust::device_vector<TID> device_data(data);

  Timestamp begin;
  Timestamp end;
  double time = 0.0;

  // int n = N;
  // for( int i = 0; i <= n; i++ ) {
  thrust::device_vector<int> d_values(data);
  // thrust::device_vector<int> d_map(data);

  thrust::device_vector<unsigned short> d_keys(data);

  // thrust::device_vector<char>

  begin = getTimestamp();

  // mark even indices with a 1; odd indices with a 0

  // thrust::scatter(d_values.begin(), d_values.end(), d_map.begin(),
  // d_output.begin());

  // thrust::system::cuda::detail::detail::stable_radix_sort( thrust::device,
  // device_data.begin(), device_data.end());

  // thrust::sort(device_data.begin(), device_data.end());

  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
  //
  // cudaStreamSynchronize ( );
  end = getTimestamp();

  // if(i > 0)
  time += double(end - begin) / (1000 * 1000);
  //}

  // thrust::device_free(device_data);

  return time;  // / n;
}

double measurePackingKernel(unsigned long int num_rows_base_table,
                            std::vector<char> data) {
  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 1024;

  thrust::device_ptr<char> device_data =
      toDevice<char>(data, num_rows_base_table);

  unsigned int bitmap_size = (num_rows_base_table + 7) / 8;

  thrust::device_ptr<char> bitmap;
  try {
    bitmap = thrust::device_malloc<char>(bitmap_size);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during gpu memory allocation for bitmap"
              << std::endl;
    return false;
  }
  cudaStream_t* stream = StreamManager::instance().getStream();
  assert(bitmap.get() != NULL);

  Timestamp begin;
  Timestamp end;

  double time = 0.0;
  int n = N;

  for (int i = 0; i <= n; i++) {
    cudaError_t err = cudaMemsetAsync(bitmap.get(), 0, bitmap_size, *stream);
    gpuErrchk(err);
    cudaStreamSynchronize(*stream);

    begin = getTimestamp();
    convertPositionListToBitmap_pack_flagarray_kernel<<<
        number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
        thrust::raw_pointer_cast(device_data), num_rows_base_table,
        thrust::raw_pointer_cast(bitmap));

    cudaStreamSynchronize(*stream);
    end = getTimestamp();

    if (i > 0) time += double(end - begin) / (1000 * 1000);
  }

  thrust::device_free(bitmap);
  thrust::device_free(device_data);

  return time / n;
}

double measureConversionMultipassKernel(unsigned long int num_rows_base_table,
                                        unsigned long int positionlist_size,
                                        std::vector<TID> data,
                                        unsigned int passes) {
  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 1024;

  thrust::device_ptr<TID> device_data = toDevice<TID>(data, positionlist_size);

  unsigned int number_of_bytes_for_flag_array = (num_rows_base_table + 7) / 8;
  number_of_bytes_for_flag_array *= 8;

  thrust::device_ptr<char> flag_array;
  try {
    flag_array = thrust::device_malloc<char>(number_of_bytes_for_flag_array);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during gpu memory allocation for flag array"
              << std::endl;
    return false;
  }
  cudaStream_t* stream = StreamManager::instance().getStream();
  assert(flag_array.get() != NULL);

  Timestamp begin;
  Timestamp end;

  double time = 0.0;
  int n = N;

  for (int i = 0; i <= n; i++) {
    cudaError_t err =
        cudaMemsetAsync(flag_array.get(), 0, num_rows_base_table, *stream);
    gpuErrchk(err);
    cudaStreamSynchronize(*stream);

    begin = getTimestamp();
    convert_multi_pass<<<number_of_blocks, number_of_threads_per_block, 0,
                         *stream>>>(thrust::raw_pointer_cast(device_data),
                                    positionlist_size, num_rows_base_table,
                                    flag_array.get(), passes);

    cudaStreamSynchronize(*stream);
    end = getTimestamp();

    if (i > 0) time += double(end - begin) / (1000 * 1000);
  }

  thrust::device_free(flag_array);
  thrust::device_free(device_data);

  return time / n;
}

double measureConversionKernel(unsigned long int num_rows_base_table,
                               unsigned long int positionlist_size,
                               std::vector<TID> data) {
  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 1024;

  thrust::device_ptr<TID> device_data = toDevice<TID>(data, positionlist_size);

  unsigned int number_of_bytes_for_flag_array = (num_rows_base_table + 7) / 8;
  number_of_bytes_for_flag_array *= 8;

  thrust::device_ptr<char> flag_array;
  try {
    flag_array = thrust::device_malloc<char>(number_of_bytes_for_flag_array);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during gpu memory allocation for flag array"
              << std::endl;
    return false;
  }
  cudaStream_t* stream = StreamManager::instance().getStream();
  assert(flag_array.get() != NULL);

  Timestamp begin;
  Timestamp end;

  double time = 0.0;
  int n = N;

  for (int i = 0; i <= n; i++) {
    cudaError_t err =
        cudaMemsetAsync(flag_array.get(), 0, num_rows_base_table, *stream);
    gpuErrchk(err);
    cudaStreamSynchronize(*stream);

    begin = getTimestamp();
    convertPositionListToBitmap_create_flagarray_kernel<<<
        number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
        thrust::raw_pointer_cast(device_data), positionlist_size,
        num_rows_base_table, flag_array.get());

    cudaStreamSynchronize(*stream);
    end = getTimestamp();

    if (i > 0) time += double(end - begin) / (1000 * 1000);
  }

  thrust::device_free(flag_array);
  thrust::device_free(device_data);

  return time / n;
}

double measureConversionBitmapKernel(unsigned long int num_rows_base_table,
                                     unsigned long int positionlist_size,
                                     std::vector<TID> data) {
  const int number_of_blocks = 512;
  const int number_of_threads_per_block = 1024;

  thrust::device_ptr<TID> device_data = toDevice<TID>(data, positionlist_size);

  unsigned int number_of_bytes_for_flag_array = (num_rows_base_table + 7) / 8;
  // number_of_bytes_for_flag_array *=8;

  thrust::device_ptr<char> bitmap;
  try {
    bitmap = thrust::device_malloc<char>(number_of_bytes_for_flag_array);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during gpu memory allocation for flag array"
              << std::endl;
    return false;
  }
  cudaStream_t* stream = StreamManager::instance().getStream();
  assert(bitmap.get() != NULL);

  Timestamp begin;
  Timestamp end;

  double time = 0.0;
  int n = N;

  for (int i = 0; i <= n; i++) {
    cudaError_t err = cudaMemsetAsync(bitmap.get(), 0,
                                      number_of_bytes_for_flag_array, *stream);
    gpuErrchk(err);
    cudaStreamSynchronize(*stream);

    begin = getTimestamp();
    convert_to_bitmap_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                               *stream>>>(thrust::raw_pointer_cast(device_data),
                                          positionlist_size,
                                          num_rows_base_table, bitmap.get());

    cudaStreamSynchronize(*stream);
    end = getTimestamp();

    if (i > 0) time += double(end - begin) / (1000 * 1000);
  }

  thrust::device_free(bitmap);
  thrust::device_free(device_data);

  return time / n;
}

double measureBinnedConversionKernel(unsigned long int num_rows_base_table,
                                     unsigned long int positionlist_size,
                                     unsigned int binsize,
                                     std::vector<TID> data) {
  const int number_of_blocks = 1024;
  const int number_of_threads_per_block = 512;

  thrust::device_ptr<TID> device_data = toDevice<TID>(data, positionlist_size);

  unsigned int number_of_bytes_for_flag_array = (num_rows_base_table + 7) / 8;
  number_of_bytes_for_flag_array *= 8;

  thrust::device_ptr<char> flag_array;
  try {
    flag_array = thrust::device_malloc<char>(number_of_bytes_for_flag_array);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during gpu memory allocation for flag array"
              << std::endl;
    return false;
  }
  cudaStream_t* stream = StreamManager::instance().getStream();
  assert(flag_array.get() != NULL);

  Timestamp begin;
  Timestamp end;

  double time = 0.0;
  int n = N;

  for (int i = 0; i <= n; i++) {
    cudaError_t err =
        cudaMemsetAsync(flag_array.get(), 0, num_rows_base_table, *stream);
    gpuErrchk(err);
    cudaStreamSynchronize(*stream);

    begin = getTimestamp();
    convertPositionListToBitmap_create_flagarray_bins_kernel<<<
        number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
        thrust::raw_pointer_cast(device_data), positionlist_size,
        num_rows_base_table, flag_array.get());

    cudaStreamSynchronize(*stream);
    end = getTimestamp();

    if (i > 0) time += double(end - begin) / (1000 * 1000);
  }

  thrust::device_free(flag_array);
  thrust::device_free(device_data);

  return time / n;
}

double measureBinnedConversionHalfKernel(unsigned long int num_rows_base_table,
                                         unsigned long int positionlist_size,
                                         unsigned int binsize,
                                         std::vector<TID> data) {
  const int number_of_blocks = 1024;
  const int number_of_threads_per_block = 512;

  thrust::device_ptr<TID> device_data = toDevice<TID>(data, positionlist_size);

  unsigned int number_of_bytes_for_flag_array = (num_rows_base_table + 7) / 8;
  number_of_bytes_for_flag_array *= 8;

  thrust::device_ptr<char> flag_array;
  try {
    flag_array = thrust::device_malloc<char>(number_of_bytes_for_flag_array);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during gpu memory allocation for flag array"
              << std::endl;
    return false;
  }
  cudaStream_t* stream = StreamManager::instance().getStream();
  assert(flag_array.get() != NULL);

  Timestamp begin;
  Timestamp end;

  double time = 0.0;
  int n = N;

  for (int i = 0; i <= n; i++) {
    cudaError_t err =
        cudaMemsetAsync(flag_array.get(), 0, num_rows_base_table, *stream);
    gpuErrchk(err);
    cudaStreamSynchronize(*stream);

    begin = getTimestamp();
    convert_bins_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                          *stream>>>(thrust::raw_pointer_cast(device_data),
                                     positionlist_size, num_rows_base_table,
                                     flag_array.get());

    cudaStreamSynchronize(*stream);
    end = getTimestamp();

    if (i > 0) time += double(end - begin) / (1000 * 1000);
  }

  thrust::device_free(flag_array);
  thrust::device_free(device_data);

  return time / n;
}

bool testAllocationAndTransfer() {
  unsigned long int n = 2 << 18;
  bool result = true;

  for (unsigned long int i = 0; i < 9; i++) {
    cout << "Data size: " << (sizeof(TID) * n) / (2 << 20) << "mb";
    n *= 2;
    std::vector<TID> data = generate_binned_tids(n, n, 256, 512);
    thrust::device_ptr<TID> device_data = toDevice<TID>(data, n);
    std::vector<TID> check = fromDevice<TID>(device_data, n);

    // fails beyond 32mb, why?
    // result = std::memcmp(&data[0], &check[0], n * sizeof(TID));

    for (unsigned long int i = 0; i < n; i++) {
      if (data[i] != check[i]) result = false;
    }
    thrust::device_free(device_data);
    if (result) cout << ", successfull";
    cout << endl;
  }

  if (result == false) {
    COGADB_FATAL_ERROR(
        "Unittests for loading generated tid-data to gpu failed!", "");
  }

  return result;
}

bool testGenerateBinnedTids() {
  vector<TID> tids = generate_binned_tids(128, 64, 16, 32);

  for (int i = 0; i < 64; i++) {
    cout << tids[i] << ", ";
  }
  cout << endl;

  return true;
}

bool testBinningKernel() {
  unsigned long int tablesize = 2 << 20;
  unsigned long int listsize = 2 << 19;
  unsigned long int binsize = 512;

  std::vector<TID> data;
  data = generate_binned_tids(tablesize, listsize, binsize / 2, binsize);

  const int number_of_blocks = 1024;
  const int number_of_threads_per_block = 512;

  thrust::device_ptr<TID> device_data = toDevice<TID>(data, listsize);
  unsigned int number_flag_array_elements = (tablesize);

  thrust::device_ptr<char> flag_array_a, flag_array_b;
  try {
    flag_array_a =
        thrust::device_malloc<char>(number_flag_array_elements * sizeof(char));
    flag_array_b =
        thrust::device_malloc<char>(number_flag_array_elements * sizeof(char));
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during gpu memory allocation for flag array"
              << std::endl;
    return false;
  }
  cudaStream_t* stream = StreamManager::instance().getStream();
  assert(flag_array_a.get() != NULL);
  assert(flag_array_b.get() != NULL);

  cudaError_t err =
      cudaMemsetAsync(flag_array_b.get(), 0,
                      number_flag_array_elements * sizeof(char), *stream);
  gpuErrchk(err);
  cudaStreamSynchronize(*stream);

  convert_bins_kernel<<<number_of_blocks, number_of_threads_per_block, 0,
                        *stream>>>(thrust::raw_pointer_cast(device_data),
                                   listsize, tablesize, flag_array_a.get());

  convertPositionListToBitmap_create_flagarray_kernel<<<
      number_of_blocks, number_of_threads_per_block, 0, *stream>>>(
      thrust::raw_pointer_cast(device_data), listsize, tablesize,
      flag_array_b.get());

  vector<char> result_a, result_b;
  result_a = fromDevice<char>(flag_array_a, number_flag_array_elements);
  result_b = fromDevice<char>(flag_array_b, number_flag_array_elements);

  bool result = true;
  for (int i = 0; i < number_flag_array_elements; i++) {
    if (result_a[i] != result_b[i]) {
      cout << "Kernel failed at flag " << i << endl;
      if (std::find(data.begin(), data.end(), i) != data.end()) {
        if (result_a[i] == 0) cout << "new kernel is wrong" << endl;
        if (result_b[i] == 0) cout << "original kernel is wrong" << endl;
      } else {
        if (result_a[i] == 1) cout << "new kernel is wrong" << endl;
        if (result_b[i] == 1) cout << "original kernel is wrong" << endl;
      }
      result = false;
      break;
    }
  }

  thrust::device_free(flag_array_a);
  thrust::device_free(flag_array_b);
  thrust::device_free(device_data);

  return result;
}

void printSetup(unsigned long int tablesize, unsigned long int listsize = 0,
                unsigned long int binsize = 0) {
  std::cout << "\nStarting GPU-work test...\n";
  std::cout << "table size is " << tablesize / (2 << 20)
            << "M Elements, positionlist size is "
            << listsize * sizeof(TID) / (2 << 20) << "mb";
  if (binsize > 0) std::cout << ", bin size is " << binsize;
  std::cout << endl;
}

bool experiment1() {
  unsigned long int tablesize = 2 << 26;
  unsigned long int listsize = 2 << 26;
  unsigned long int binsize = 512;

  printSetup(tablesize, listsize, binsize);
  std::cout << "binning kernel - unique vs duplicates" << endl;
  std::cout << "results:" << endl;

  std::vector<TID> data;
  data = generate_binned_duplicate_tids(tablesize, listsize, binsize);
  double time1 =
      measureBinnedConversionKernel(tablesize, listsize, binsize, data);
  std::cout << "convertPositionListToBitmap_create_flagarray_bins_kernel with "
               "duplicates took "
            << time1 << "ms" << endl;

  data = generate_binned_tids(tablesize, listsize, binsize, binsize);
  double time2 =
      measureBinnedConversionKernel(tablesize, listsize, binsize, data);
  std::cout << "convertPositionListToBitmap_create_flagarray_bins_kernel with "
               "unique data took "
            << time2 << "ms" << endl;

  std::cout << "experiment finished" << endl;

  return true;
}

bool experiment2() {
  unsigned int tablesize = 2 << 26;
  unsigned int listsize = 2 << 25;

  printSetup(tablesize, listsize);
  std::cout << "original kernel - unique vs duplicates" << endl;
  std::cout << "results:" << endl;

  std::vector<TID> data;
  data = generate_tids(tablesize, listsize);
  double time2 = measureConversionKernel(tablesize, listsize, data);
  std::cout << "convertPositionListToBitmap_create_flagarray_kernel with "
               "unique data took "
            << time2 << "ms" << endl;

  data = generate_duplicate_tids(tablesize, listsize);
  double time1 = measureConversionKernel(tablesize, listsize, data);
  std::cout << "convertPositionListToBitmap_create_flagarray_kernel with "
               "duplicates took "
            << time1 << "ms" << endl;

  std::cout << "experiment finished" << endl;

  return true;
}

bool experiment3() {
  unsigned long int tablesize = 2 << 26;
  unsigned long int listsize = 2 << 25;
  unsigned long int binsize = 512;

  printSetup(tablesize, listsize, binsize);
  std::cout << "original kernel vs binning kernel - unique data" << endl;
  std::cout << "results:" << endl;

  std::vector<TID> data;
  data = generate_tids(tablesize, listsize);
  double time1 = measureConversionKernel(tablesize, listsize, data);
  std::cout << "convertPositionListToBitmap_create_flagarray_kernel took "
            << time1 << "ms" << endl;

  data = generate_binned_tids(tablesize, listsize, binsize, binsize);
  double time2 =
      measureBinnedConversionKernel(tablesize, listsize, binsize, data);
  std::cout << "convertPositionListToBitmap_create_flagarray_bins_kernel took "
            << time2 << "ms" << endl;

  std::cout << "experiment finished" << endl;

  return true;
}

bool experiment4() {
  unsigned long int tablesize = 2 << 26;
  unsigned long int listsize = 2 << 25;

  printSetup(tablesize, listsize);
  std::cout << "original kernels - conversion vs packing (duplicates allowed)"
            << endl;
  std::cout << "results:" << endl;

  std::vector<TID> conversion_data = generate_tids(tablesize, listsize);
  double time1 = measureConversionKernel(tablesize, listsize, conversion_data);
  std::cout << "convertPositionListToBitmap_create_flagarray_kernel took "
            << time1 << "ms" << endl;

  std::vector<char> packing_data = generate_flag_array(tablesize);
  double time2 = measurePackingKernel(tablesize, packing_data);
  std::cout << "convertPositionListToBitmap_create_flagarray_kernel took "
            << time2 << "ms" << endl;

  conversion_data = generate_tids(tablesize, listsize);
  double time3 =
      measureConversionBitmapKernel(tablesize, listsize, conversion_data);
  std::cout << "convert_to_bitmap_kernel took " << time3 << "ms" << endl;

  std::cout << "experiment finished" << endl;

  return true;
}

bool experiment5() {
  unsigned long int tablesize = 2 << 26;
  unsigned long int listsize = 2 << 25;

  printSetup(tablesize, listsize);
  std::cout << "thrust sort, thrust partition" << endl;
  std::cout << "results:" << endl;

  std::vector<TID> list_data = generate_tids(tablesize, listsize);
  double time0 = measureCubRadixSort(list_data);
  std::cout << "cub::RadixSort took " << time0 << "ms" << endl;

  list_data = generate_tids(tablesize, listsize);
  double time1 = measureThrustSort(tablesize, list_data);
  std::cout << "thrust::sort took " << time1 << "ms" << endl;

  std::vector<TID> data = generate_tids(tablesize, listsize);
  double time2 = measureThrustPartition(tablesize, data);
  std::cout << "thrust::partition took " << time2 << "ms" << endl;

  std::cout << "experiments finished" << endl;

  return true;
}

bool experiment6() {
  unsigned long int tablesize = 2 << 26;
  unsigned long int listsize = 2 << 25;
  unsigned long int binsize = 512;

  printSetup(tablesize, listsize, binsize);
  std::cout << "original kernel vs binning kernel - listsize = 0.5 tablesize"
            << endl;
  std::cout << "results:" << endl;

  std::vector<TID> data;
  data = generate_tids(tablesize, listsize);
  double time1 = measureConversionKernel(tablesize, listsize, data);
  std::cout << "convertPositionListToBitmap_create_flagarray_kernel took "
            << time1 << "ms" << endl;

  data = generate_binned_tids(tablesize, listsize, 256, 512);
  double time2 =
      measureBinnedConversionHalfKernel(tablesize, listsize, binsize, data);
  std::cout << "convert_bins_kernel took " << time2 << "ms" << endl;

  /*
      data = generate_tids( tablesize, listsize );
      double time3 = measureConversionMultipassKernel(tablesize, listsize, data,
     1024);
      std::cout << "Multi pass conversion with 1024 passes took " << time3 <<
     "ms" << endl;
  */

  std::cout << "experiment finished" << endl;

  return true;
}

/*
bool experiment7() {
        unsigned long int tablesize = 2 << 26;
        unsigned long int listsize = 2 << 25;
        unsigned long int binsize = 512;

        printSetup(tablesize, listsize, binsize);
        std::cout << "original kernel vs binning kernel - listsize = 0.5
tablesize" << endl;
        std::cout << "results:" << endl;

        std::vector<TID> data;
        data = generate_tids( tablesize, listsize );
        double time1 = measureConversionKernel(tablesize, listsize, data);
        std::cout << "convertPositionListToBitmap_create_flagarray_kernel took "
<< time1 << "ms" << endl;

        data = generate_binned_tids( tablesize, listsize, 256, 512 );
        double time2 = measureBinnedConversionHalfKernel(tablesize, listsize,
binsize, data);
        std::cout << "convert_bins_kernel took " << time2 << "ms" << endl;


            data = generate_tids( tablesize, listsize );
            double time3 = measureConversionMultipassKernel(tablesize, listsize,
data, 2);
            std::cout << "Multi pass conversion with 4 passes took " << time3 <<
"ms" << endl;


        std::cout << "experiment finished" << endl;

        return true;



}*/

bool gpu_work_kernels_test() { return true; }
}
}