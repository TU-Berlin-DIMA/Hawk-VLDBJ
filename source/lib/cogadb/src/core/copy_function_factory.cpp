
#include <boost/thread.hpp>
#include <core/copy_function_factory.hpp>
#include <core/global_definitions.hpp>
#include <statistics/statistics_manager.hpp>
#include <util/time_measurement.hpp>

#ifdef ENABLE_GPU_ACCELERATION
#include <cuda_runtime.h>
#include <backends/gpu/stream_manager.hpp>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrorMessage(ans) \
  { gpuAssert((ans), __FILE__, __LINE__, false); }
inline void gpuAssert(cudaError_t code, const char* const file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
#endif

#define COGADB_TIME_COPY_DELAYS

namespace CoGaDB {

template <typename T>
bool CopyFunctionFactory<T>::copyCPU2CPU(T* dest, const T* source,
                                         size_t number_of_bytes) {
  if (!dest || !source || number_of_bytes == 0) return false;

  if (typeid(T) != typeid(std::string)) {
    std::memcpy(dest, source, number_of_bytes);
  } else {
    size_t num_elements = number_of_bytes / sizeof(T);
    // make proper copies of objects
    std::copy(source, source + num_elements, dest);
  }

  return true;
}

#ifdef ENABLE_GPU_ACCELERATION
template <typename T>
bool CopyFunctionFactory<T>::copyCPU2GPU(T* dest, const T* source,
                                         size_t number_of_bytes) {
  if (!dest || !source || number_of_bytes == 0) return false;

  cudaStream_t* stream = gpu::StreamManager::instance().getStream();

  uint64_t begin_copy = getTimestamp();

  //          try {
  const T* host_ptr = source;
  T* device_ptr = dest;
  // copy from cpu to  co-processor
  hype::Tuple t;
  t.push_back(number_of_bytes);
  hype::OperatorSpecification op_spec("COPY_CPU_CP", t, hype::PD_Memory_0,
                                      hype::PD_Memory_0);
  hype::DeviceConstraint dev_constr;
  hype::SchedulingDecision sched_dec =
      hype::Scheduler::instance().getOptimalAlgorithm(op_spec, dev_constr);
  hype::AlgorithmMeasurement alg_measure(sched_dec);
  // copy data into pinned memory first, so it can be faster transferred
  // to the GPU
  T* pinned_host_memory = NULL;
  cudaError_t err =
      cudaMallocHost((void**)&pinned_host_memory, number_of_bytes);
  if (err != cudaSuccess) {
    return false;
  }
  //                gpuErrchk( err );
  // copy host data to pinned memory
  memcpy(pinned_host_memory, host_ptr, number_of_bytes);
#ifdef COGADB_TIME_COPY_DELAYS
  // measure copy time using cuda API
  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  gpuErrchk(err);
  err = cudaEventCreate(&stop);
  gpuErrchk(err);
  Timestamp begin_queued_async_memcpy = getTimestamp();
  err = cudaEventRecord(start, *stream);
  gpuErrchk(err);
#endif
  // use asynchronous cudaMemcpy call
  err = cudaMemcpyAsync(device_ptr, pinned_host_memory, number_of_bytes,
                        cudaMemcpyHostToDevice, *stream);
  gpuErrchk(err);

#ifdef COGADB_TIME_COPY_DELAYS
  err = cudaEventRecord(stop, *stream);
  gpuErrchk(err);
  err = cudaEventSynchronize(stop);
  gpuErrchk(err);
  Timestamp end_queued_async_memcpy = getTimestamp();
  double host_response_time_copy_operation_sec =
      double(end_queued_async_memcpy - begin_queued_async_memcpy) /
      (1000 * 1000 * 1000);
  float device_response_time_copy_operation_ms = 0;

  err = cudaEventElapsedTime(&device_response_time_copy_operation_ms, start,
                             stop);
  gpuErrchk(err);
  assert(host_response_time_copy_operation_sec >=
         (device_response_time_copy_operation_ms / 1000));
  double delay_time_ins_sec = host_response_time_copy_operation_sec -
                              (device_response_time_copy_operation_ms / 1000);
  std::cout << "Host Measured Copy Time: "
            << host_response_time_copy_operation_sec << " seconds" << std::endl;
  std::cout << "CUDA Measured Copy Time: "
            << device_response_time_copy_operation_ms / 1000 << " seconds"
            << std::endl;
  std::cout << "Delay Time: " << delay_time_ins_sec << " seconds" << std::endl;
  std::cout << "Copy Bandwidth CPU to GPU: "
            << (double(number_of_bytes) / (1024 * 1024 * 1024)) /
                   (device_response_time_copy_operation_ms / 1000)
            << std::endl;
  assert(delay_time_ins_sec >= 0);
  StatisticsManager::instance().addToValue(
      "DELAY_TIME_OF_COPY_OPERATIONS_HOST_TO_DEVICE_IN_SECS",
      delay_time_ins_sec);
  StatisticsManager::instance().addToValue(
      "TOTAL_DELAY_TIME_OF_COPY_OPERATIONS_IN_SECS", delay_time_ins_sec);
#else
  err = cudaStreamSynchronize(*stream);
  gpuErrchk(err);
#endif
  err = cudaFreeHost(pinned_host_memory);
  gpuErrchk(err);
  // give HyPE feedback about operator cost
  alg_measure.afterAlgorithmExecution();

  StatisticsManager::instance().addToValue(
      "TOTAL_TRANSFERRED_BYTES_HOST_TO_DEVICE", number_of_bytes);
  StatisticsManager::instance().addToValue(
      "COLUMN_TRANSFERRED_BYTES_HOST_TO_DEVICE", number_of_bytes);

  //            } catch (thrust::system::detail::bad_alloc& e) {
  //                cout << e.what() << endl;
  //                printGPUStatus();
  //                return false;
  //            }
  uint64_t end_copy = getTimestamp();
  StatisticsManager::instance().addToValue(
      "TOTAL_COPY_TIME_HOST_TO_DEVICE_IN_SEC",
      double(end_copy - begin_copy) / (1000 * 1000 * 1000));
  StatisticsManager::instance().addToValue(
      "COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC",
      double(end_copy - begin_copy) / (1000 * 1000 * 1000));

  StatisticsManager::instance().addToValue(
      "NUMBER_OF_COPY_OPERATIONS_HOST_TO_DEVICE_TOTAL", double(1));
  StatisticsManager::instance().addToValue(
      "NUMBER_OF_COPY_OPERATIONS_HOST_TO_DEVICE_COLUMN", double(1));

  return true;
}

template <typename T>
bool CopyFunctionFactory<T>::copyGPU2CPU(T* dest, const T* source,
                                         size_t number_of_bytes) {
  if (!dest || !source || number_of_bytes == 0) return false;

  const T* src_array_on_device = source;

  T* dest_array_on_host = dest;

  T* pinned_host_memory = NULL;
  cudaError_t err =
      cudaMallocHost((void**)&pinned_host_memory, number_of_bytes);
  gpuErrorMessage(err);
  if (err != cudaSuccess) return false;

  cudaStream_t* stream = gpu::StreamManager::instance().getStream();

  uint64_t begin_copy = getTimestamp();

  hype::Tuple t;
  t.push_back(number_of_bytes);
  // copy from co-processor to CPU
  hype::OperatorSpecification op_spec("COPY_CP_CPU", t, hype::PD_Memory_0,
                                      hype::PD_Memory_0);
  hype::DeviceConstraint dev_constr;
  hype::SchedulingDecision sched_dec =
      hype::Scheduler::instance().getOptimalAlgorithm(op_spec, dev_constr);
  hype::AlgorithmMeasurement alg_measure(sched_dec);
#ifdef COGADB_TIME_COPY_DELAYS
  // measure copy time using cuda API
  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  gpuErrchk(err);
  err = cudaEventCreate(&stop);
  gpuErrchk(err);
  Timestamp begin_queued_async_memcpy = getTimestamp();
  err = cudaEventRecord(start, *stream);
  gpuErrchk(err);
#endif
  err = cudaMemcpyAsync(pinned_host_memory, src_array_on_device,
                        number_of_bytes, cudaMemcpyDeviceToHost, *stream);
  gpuErrorMessage(err);
  if (err != cudaSuccess) return false;
#ifdef COGADB_TIME_COPY_DELAYS
  err = cudaEventRecord(stop, *stream);
  gpuErrchk(err);
  err = cudaEventSynchronize(stop);
  gpuErrchk(err);
  Timestamp end_queued_async_memcpy = getTimestamp();
  double host_response_time_copy_operation_sec =
      double(end_queued_async_memcpy - begin_queued_async_memcpy) /
      (1000 * 1000 * 1000);
  float device_response_time_copy_operation_ms = 0;

  err = cudaEventElapsedTime(&device_response_time_copy_operation_ms, start,
                             stop);
  gpuErrchk(err);
  assert(host_response_time_copy_operation_sec >=
         (device_response_time_copy_operation_ms / 1000));
  double delay_time_ins_sec = host_response_time_copy_operation_sec -
                              (device_response_time_copy_operation_ms / 1000);
  std::cout << "Host Measured Copy Time: "
            << host_response_time_copy_operation_sec << " seconds" << std::endl;
  std::cout << "CUDA Measured Copy Time: "
            << device_response_time_copy_operation_ms / 1000 << " seconds"
            << std::endl;
  std::cout << "Delay Time: " << delay_time_ins_sec << " seconds" << std::endl;
  std::cout << "Copy Bandwidth GPU to CPU: "
            << (double(number_of_bytes) / (1024 * 1024 * 1024)) /
                   (device_response_time_copy_operation_ms / 1000)
            << std::endl;
  assert(delay_time_ins_sec >= 0);
  StatisticsManager::instance().addToValue(
      "DELAY_TIME_OF_COPY_OPERATIONS_DEVICE_TO_HOST_IN_SECS",
      delay_time_ins_sec);
  StatisticsManager::instance().addToValue(
      "TOTAL_DELAY_TIME_OF_COPY_OPERATIONS_IN_SECS", delay_time_ins_sec);
#else
  err = cudaStreamSynchronize(*stream);
  gpuErrchk(err);
#endif
  memcpy(dest_array_on_host, pinned_host_memory, number_of_bytes);
  err = cudaFreeHost(pinned_host_memory);
  gpuErrchk(err);
  alg_measure.afterAlgorithmExecution();

  StatisticsManager::instance().addToValue(
      "TOTAL_TRANSFERRED_BYTES_DEVICE_TO_HOST", number_of_bytes);
  StatisticsManager::instance().addToValue(
      "COLUMN_TRANSFERRED_BYTES_DEVICE_TO_HOST", number_of_bytes);
  uint64_t end_copy = getTimestamp();
  StatisticsManager::instance().addToValue(
      "TOTAL_COPY_TIME_DEVICE_TO_HOST_IN_SEC",
      double(end_copy - begin_copy) / (1000 * 1000 * 1000));
  StatisticsManager::instance().addToValue(
      "COLUMN_COPY_TIME_DEVICE_TO_HOST_IN_SEC",
      double(end_copy - begin_copy) / (1000 * 1000 * 1000));

  StatisticsManager::instance().addToValue(
      "NUMBER_OF_COPY_OPERATIONS_DEVICE_TO_HOST_TOTAL", double(1));
  StatisticsManager::instance().addToValue(
      "NUMBER_OF_COPY_OPERATIONS_DEVICE_TO_HOST_COLUMN", double(1));

  return true;
}

template <typename T>
bool CopyFunctionFactory<T>::copyGPU2GPU(T* dest, const T* source,
                                         size_t number_of_bytes) {
  if (!dest || !source || number_of_bytes == 0) return false;

  // cudaMemcpyDeviceToDevice

  cudaStream_t* stream = gpu::StreamManager::instance().getStream();
  cudaError_t err;
  err = cudaMemcpyAsync(dest, source, number_of_bytes, cudaMemcpyDeviceToDevice,
                        *stream);

  gpuErrorMessage(err);
  if (err != cudaSuccess) return false;
  cudaStreamSynchronize(*stream);

  return true;
}

#else
template <typename T>
bool CopyFunctionFactory<T>::copyCPU2GPU(T* dest, const T* source,
                                         size_t number_of_bytes) {
  COGADB_FATAL_ERROR("Called GPU function, but GPU acceleration is disabled!",
                     "");
  return false;
}

template <typename T>
bool CopyFunctionFactory<T>::copyGPU2CPU(T* dest, const T* source,
                                         size_t number_of_bytes) {
  COGADB_FATAL_ERROR("Called GPU function, but GPU acceleration is disabled!",
                     "");
  return false;
}

template <typename T>
bool CopyFunctionFactory<T>::copyGPU2GPU(T* dest, const T* source,
                                         size_t number_of_bytes) {
  COGADB_FATAL_ERROR("Called GPU function, but GPU acceleration is disabled!",
                     "");
  return false;
}
#endif

template <>
bool CopyFunctionFactory<std::string>::copyCPU2GPU(std::string* dest,
                                                   const std::string* source,
                                                   size_t number_of_bytes) {
  COGADB_FATAL_ERROR("VARCHAR columns not supported on GPU!", "");
  return false;
}

template <>
bool CopyFunctionFactory<std::string>::copyGPU2CPU(std::string* dest,
                                                   const std::string* source,
                                                   size_t number_of_bytes) {
  COGADB_FATAL_ERROR("VARCHAR columns not supported on GPU!", "");
  return false;
}

template <>
bool CopyFunctionFactory<std::string>::copyGPU2GPU(std::string* dest,
                                                   const std::string* source,
                                                   size_t number_of_bytes) {
  COGADB_FATAL_ERROR("VARCHAR columns not supported on GPU!", "");
  return false;
}

bool isCPUMemory(const hype::ProcessingDeviceMemoryID& mem_id) {
  // FIXME: query this information from HyPE!
  if (mem_id == hype::PD_Memory_0) {
    return true;
  } else {
    return false;
  }
}

bool isGPUMemory(const hype::ProcessingDeviceMemoryID& mem_id) {
  // FIXME: query this information from HyPE!
  if (mem_id == hype::PD_Memory_1) {
    return true;
  } else {
    return false;
  }
}

boost::mutex copy_function_factory_mutex;
template <typename T>
CopyFunctionFactory<T>& CopyFunctionFactory<T>::instance() {
  copy_function_factory_mutex.lock();
  static CopyFunctionFactory<T> factory;
  copy_function_factory_mutex.unlock();
  return factory;
}

template <typename T>
CopyFunctionFactory<T>::CopyFunctionFactory() : map() {
  map.insert(std::make_pair(std::make_pair(true, true),
                            &CopyFunctionFactory<T>::copyCPU2CPU));
  map.insert(std::make_pair(std::make_pair(false, true),
                            &CopyFunctionFactory<T>::copyCPU2GPU));
  map.insert(std::make_pair(std::make_pair(true, false),
                            &CopyFunctionFactory<T>::copyGPU2CPU));
  map.insert(std::make_pair(std::make_pair(false, false),
                            &CopyFunctionFactory<T>::copyGPU2GPU));
}

template <typename T>
typename CopyFunctionFactory<T>::CopyFunctionPtr
CopyFunctionFactory<T>::getCopyFunction_internal(
    const hype::ProcessingDeviceMemoryID& mem_id_dest,
    const hype::ProcessingDeviceMemoryID& mem_id_source) {
  MemoryLocationPair key =
      std::make_pair(isCPUMemory(mem_id_dest), isCPUMemory(mem_id_source));

  typename CopyFunctionMap::const_iterator it = map.find(key);
  if (it == map.end()) {
    COGADB_FATAL_ERROR("Impossible Situation, cannot determine copy function.",
                       "");
    return NULL;
  } else {
    return it->second;
  }
}

template <typename T>
typename CopyFunctionFactory<T>::CopyFunctionPtr CopyFunctionFactory<
    T>::getCopyFunction(const hype::ProcessingDeviceMemoryID& mem_id_dest,
                        const hype::ProcessingDeviceMemoryID& mem_id_source) {
  return CopyFunctionFactory<T>::instance().getCopyFunction_internal(
      mem_id_dest, mem_id_source);
}

bool MemsetFunctionFactory::memsetCPU(void* array, int value,
                                      size_t number_of_bytes) {
  memset(array, value, number_of_bytes);
  return true;
}

#ifdef ENABLE_GPU_ACCELERATION
bool MemsetFunctionFactory::memsetGPU(void* array, int value,
                                      size_t number_of_bytes) {
  cudaError_t err = cudaMemset(array, value, number_of_bytes);
  if (err == cudaSuccess) {
    return true;
  } else {
    gpuErrchk(err);
    return false;
  }
}
#else
bool MemsetFunctionFactory::memsetGPU(void* array, int value,
                                      size_t number_of_bytes) {
  COGADB_FATAL_ERROR("Called GPU function, but GPU acceleration is disabled!",
                     "");
  return false;
}
#endif

MemsetFunctionFactory::MemsetFunctionPtr
MemsetFunctionFactory::getMemsetFunction(
    const hype::ProcessingDeviceMemoryID& mem_id) {
  if (isCPUMemory(mem_id)) return &MemsetFunctionFactory::memsetCPU;
  if (isGPUMemory(mem_id)) return &MemsetFunctionFactory::memsetGPU;
  COGADB_FATAL_ERROR(
      "Could not determine memset function for memory id " << (int)mem_id, "");
  return NULL;
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(CopyFunctionFactory)

}  // end namespae CoGaDB
