
#include <core/global_definitions.hpp>
#include <query_compilation/ocl_api.hpp>
#include <query_compilation/ocl_data_cache.hpp>
#include <util/opencl/prefix_sum.hpp>
#include <util/time_measurement.hpp>

#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/gather.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/algorithm/reduce_by_key.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>

#include <boost/thread.hpp>

#include <query_compilation/ocl_api.h>
#include <limits>
#include <thread>

OCL_Execution_Context::OCL_Execution_Context(
    const boost::compute::context& _context,
    const std::vector<boost::compute::command_queue>& _compute_queues,
    const std::vector<boost::compute::command_queue>&
        _copy_host_to_device_queues,
    const std::vector<boost::compute::command_queue>&
        _copy_device_to_host_queues,
    const boost::compute::program& _program)
    : context(_context),
      compute_queues(_compute_queues),
      copy_host_to_device_queues(_copy_host_to_device_queues),
      copy_device_to_host_queues(_copy_device_to_host_queues),
      thread_access_index_counter(0),
      program(_program) {}

OCL_Execution_Context::~OCL_Execution_Context() {}

cl_context ocl_getContext(OCL_Execution_Context* context) {
  if (!context) {
    return nullptr;
  }

  return context->context;
}

thread_local cl_command_queue compute_queue;
thread_local std::uint_fast32_t thread_access_index =
    std::numeric_limits<std::uint_fast32_t>::max();

boost::mutex global_command_queue_mutex;

cl_command_queue ocl_getComputeCommandQueue(OCL_Execution_Context* context) {
  if (!context) {
    return nullptr;
  }

  if (compute_queue) {
    return compute_queue;
  }
  boost::lock_guard<boost::mutex> lock(global_command_queue_mutex);
  if (thread_access_index == std::numeric_limits<std::uint_fast32_t>::max()) {
    thread_access_index = context->thread_access_index_counter++;
  }

  compute_queue = context->compute_queues[thread_access_index];

  return compute_queue;
}

thread_local cl_command_queue to_device_queue;

cl_command_queue ocl_getTransferHostToDeviceCommandQueue(
    OCL_Execution_Context* context) {
  if (!context) {
    return nullptr;
  }

  if (to_device_queue) {
    return to_device_queue;
  }

  if (thread_access_index == std::numeric_limits<std::uint_fast32_t>::max()) {
    thread_access_index = context->thread_access_index_counter++;
  }

  to_device_queue = context->copy_host_to_device_queues[thread_access_index];

  return to_device_queue;
}

thread_local cl_command_queue to_host_queue;

cl_command_queue ocl_getTransferDeviceToHostCommandQueue(
    OCL_Execution_Context* context) {
  if (!context) {
    return nullptr;
  }

  if (to_host_queue) {
    return to_host_queue;
  }

  if (thread_access_index == std::numeric_limits<std::uint_fast32_t>::max()) {
    thread_access_index = context->thread_access_index_counter++;
  }

  to_host_queue = context->copy_device_to_host_queues[thread_access_index];

  return to_host_queue;
}

void ocl_api_reset_thread_local_variables() {
  compute_queue = to_device_queue = to_host_queue = nullptr;
  thread_access_index = std::numeric_limits<std::uint_fast32_t>::max();
}

cl_program ocl_getProgram(OCL_Execution_Context* context) {
  if (!context) {
    return nullptr;
  }

  return context->program;
}

cl_device_id ocl_getDeviceID(OCL_Execution_Context* context) {
  if (!context) {
    return nullptr;
  }

  cl_device_id dev_id = nullptr;
  cl_int err =
      clGetCommandQueueInfo(ocl_getComputeCommandQueue(context),
                            CL_QUEUE_DEVICE, sizeof(dev_id), &dev_id, nullptr);

  if (err != CL_SUCCESS) {
    COGADB_FATAL_ERROR(
        "Could not retrieve command queue info "
        "CL_QUEUE_DEVICE from command queue!",
        "");
  }

  return dev_id;
}

boost::mutex ocl_data_transfer_mutex;

void ocl_enter_critical_section_copy_host_to_device() {
  ocl_data_transfer_mutex.lock();
}

void ocl_leave_critical_section_copy_host_to_device() {
  ocl_data_transfer_mutex.unlock();
}

boost::mutex ocl_compute_kernel_mutex;

void ocl_enter_critical_section_schedule_computation() {
  ocl_compute_kernel_mutex.lock();
}

void ocl_leave_critical_section_schedule_computation() {
  ocl_compute_kernel_mutex.unlock();
}

thread_local uint64_t start_ = 0;

void ocl_start_timer() {
  assert(start_ == 0);
  start_ = CoGaDB::getTimestamp();
}

void ocl_stop_timer(const char* string) {
  assert(start_ != 0);
  uint64_t end = CoGaDB::getTimestamp();
  std::cout << string << ": " << double(end - start_) / (1000 * 1000 * 1000)
            << "s" << std::endl;
  start_ = 0;
}

void ocl_print_device(cl_device_id device_id) {
  assert(device_id != NULL);

  char buffer[10240];
  cl_uint buf_uint;
  cl_ulong buf_ulong;
  CL_CHECK(
      clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
  printf("  DEVICE_NAME = %s\n", buffer);
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(buffer), buffer,
                           NULL));
  printf("  DEVICE_VENDOR = %s\n", buffer);
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(buffer), buffer,
                           NULL));
  printf("  DEVICE_VERSION = %s\n", buffer);
  CL_CHECK(clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(buffer), buffer,
                           NULL));
  printf("  DRIVER_VERSION = %s\n", buffer);
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                           sizeof(buf_uint), &buf_uint, NULL));
  printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", buf_uint);
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                           sizeof(buf_uint), &buf_uint, NULL));
  printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", buf_uint);
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE,
                           sizeof(buf_ulong), &buf_ulong, NULL));
  printf("  DEVICE_GLOBAL_MEM_SIZE = %lu\n", buf_ulong);
}

size_t ocl_prefix_sum(cl_command_queue queue, cl_program program,
                      cl_mem cl_output_mem_flags, cl_mem cl_output_prefix_sum,
                      size_t num_elements) {
  return CoGaDB::ocl_prefix_sum(queue, program, cl_output_mem_flags,
                                cl_output_prefix_sum, num_elements);
}

#define REDUCE(function)                                                   \
  boost::compute::reduce(                                                  \
      boost::compute::make_buffer_iterator<ResultType>(buffer_wrapper, 0), \
      boost::compute::make_buffer_iterator<ResultType>(buffer_wrapper,     \
                                                       num_elements),      \
      &result, function<ResultType>(), queue_wrapper);

template <typename ResultType>
ResultType ocl_reduce(cl_command_queue queue, cl_mem buffer,
                      uint64_t num_elements, unsigned char aggr_method) {
  boost::compute::buffer buffer_wrapper(buffer);
  boost::compute::command_queue queue_wrapper(queue);
  ResultType result = 0;

  switch (aggr_method) {
    case CoGaDB::COUNT:
    case CoGaDB::SUM:
      REDUCE(boost::compute::plus);
      break;
    case CoGaDB::MIN:
      REDUCE(boost::compute::min);
      break;
    case CoGaDB::MAX:
      REDUCE(boost::compute::max);
      break;
    default:
      COGADB_FATAL_ERROR("Unknown aggregation method!", "");
  }

  return result;
}

uint64_t ocl_reduce_uint64_t(cl_command_queue queue, cl_mem buffer,
                             uint64_t num_elements, unsigned char aggr_method) {
  return ocl_reduce<uint64_t>(queue, buffer, num_elements, aggr_method);
}

double ocl_reduce_double(cl_command_queue queue, cl_mem buffer,
                         uint64_t num_elements, unsigned char aggr_method) {
  return ocl_reduce<double>(queue, buffer, num_elements, aggr_method);
}

double ocl_reduce_float(cl_command_queue queue, cl_mem buffer,
                        uint64_t num_elements, unsigned char aggr_method) {
  return ocl_reduce<float>(queue, buffer, num_elements, aggr_method);
}

template <typename ValueType>
void ocl_print(cl_command_queue queue, int num_elements, cl_mem mem) {
  boost::compute::buffer mem_wrapper(mem);
  boost::compute::command_queue queue_wrapper(queue);
  auto start_print =
      boost::compute::make_buffer_iterator<ValueType>(mem_wrapper, 0);
  auto end_print = boost::compute::make_buffer_iterator<ValueType>(
      mem_wrapper, num_elements);
  std::cout << "vector: [ ";
  boost::compute::copy(start_print, end_print,
                       std::ostream_iterator<ValueType>(std::cout, ", "),
                       queue_wrapper);
  std::cout << "]" << std::endl;
  std::cout << std::endl;
}

void ocl_print_float(cl_command_queue queue, int num_elements, cl_mem mem) {
  ocl_print<float>(queue, num_elements, mem);
}

void ocl_print_uint64_t(cl_command_queue queue, int num_elements, cl_mem mem) {
  ocl_print<uint64_t>(queue, num_elements, mem);
}

BOOST_COMPUTE_FUNCTION(bool, my_comparator, (uint64_t a, uint64_t b),
                       { return a < b; });

void ocl_sort_by_key_uint64_t(cl_command_queue queue, uint64_t num_elements,
                              cl_mem keys, cl_mem vals) {
  boost::compute::buffer key_buffer_wrapper(keys);
  boost::compute::buffer vals_buffer_wrapper(vals);
  boost::compute::command_queue queue_wrapper(queue);

  boost::compute::sort_by_key(
      boost::compute::make_buffer_iterator<uint64_t>(key_buffer_wrapper, 0),
      boost::compute::make_buffer_iterator<uint64_t>(key_buffer_wrapper,
                                                     num_elements),
      boost::compute::make_buffer_iterator<uint64_t>(vals_buffer_wrapper, 0),
      queue_wrapper);
}

// todo: default reduction operation sum
template <typename ValueType>
uint64_t ocl_reduce_by_key(cl_command_queue queue, uint64_t num_elements,
                           cl_mem keys_in, cl_mem vals_in, cl_mem keys_out,
                           cl_mem vals_out, bool take_any_value) {
  boost::compute::buffer keys_buffer_wrapper(keys_in);
  boost::compute::buffer vals_buffer_wrapper(vals_in);
  boost::compute::buffer keys_out_buffer_wrapper(keys_out);
  boost::compute::buffer vals_out_buffer_wrapper(vals_out);
  boost::compute::command_queue queue_wrapper(queue);

  uint64_t result_size;
  if (take_any_value) {
    auto end_iterators = boost::compute::reduce_by_key(
        boost::compute::make_buffer_iterator<uint64_t>(keys_buffer_wrapper, 0),
        boost::compute::make_buffer_iterator<uint64_t>(keys_buffer_wrapper,
                                                       num_elements),
        boost::compute::make_buffer_iterator<ValueType>(vals_buffer_wrapper, 0),
        boost::compute::make_buffer_iterator<uint64_t>(keys_out_buffer_wrapper,
                                                       0),
        boost::compute::make_buffer_iterator<ValueType>(vals_out_buffer_wrapper,
                                                        0),
        boost::compute::max<ValueType>(), boost::compute::equal_to<uint64_t>(),
        queue_wrapper);
    // retrieve result size
    auto out_it =
        boost::compute::make_buffer_iterator<uint64_t>(keys_out_buffer_wrapper);
    result_size = std::distance(out_it, end_iterators.first);
  } else {
    auto end_iterators = boost::compute::reduce_by_key(
        boost::compute::make_buffer_iterator<uint64_t>(keys_buffer_wrapper, 0),
        boost::compute::make_buffer_iterator<uint64_t>(keys_buffer_wrapper,
                                                       num_elements),
        boost::compute::make_buffer_iterator<ValueType>(vals_buffer_wrapper, 0),
        boost::compute::make_buffer_iterator<uint64_t>(keys_out_buffer_wrapper,
                                                       0),
        boost::compute::make_buffer_iterator<ValueType>(vals_out_buffer_wrapper,
                                                        0),
        boost::compute::plus<ValueType>(), boost::compute::equal_to<uint64_t>(),
        queue_wrapper);
    // retrieve result size
    auto out_it =
        boost::compute::make_buffer_iterator<uint64_t>(keys_out_buffer_wrapper);
    result_size = std::distance(out_it, end_iterators.first);
  }

  return result_size;
}

uint64_t ocl_reduce_by_key_float(cl_command_queue queue, uint64_t num_elements,
                                 cl_mem keys_in, cl_mem vals_in,
                                 cl_mem keys_out, cl_mem vals_out,
                                 bool take_any_value) {
  return ocl_reduce_by_key<float>(queue, num_elements, keys_in, vals_in,
                                  keys_out, vals_out, take_any_value);
}

uint64_t ocl_reduce_by_key_double(cl_command_queue queue, uint64_t num_elements,
                                  cl_mem keys_in, cl_mem vals_in,
                                  cl_mem keys_out, cl_mem vals_out,
                                  bool take_any_value) {
  return ocl_reduce_by_key<double>(queue, num_elements, keys_in, vals_in,
                                   keys_out, vals_out, take_any_value);
}

uint64_t ocl_reduce_by_key_int32_t(cl_command_queue queue,
                                   uint64_t num_elements, cl_mem keys_in,
                                   cl_mem vals_in, cl_mem keys_out,
                                   cl_mem vals_out, bool take_any_value) {
  return ocl_reduce_by_key<int32_t>(queue, num_elements, keys_in, vals_in,
                                    keys_out, vals_out, take_any_value);
}

uint64_t ocl_reduce_by_key_uint64_t(cl_command_queue queue,
                                    uint64_t num_elements, cl_mem keys_in,
                                    cl_mem vals_in, cl_mem keys_out,
                                    cl_mem vals_out, bool take_any_value) {
  return ocl_reduce_by_key<uint64_t>(queue, num_elements, keys_in, vals_in,
                                     keys_out, vals_out, take_any_value);
}

uint64_t ocl_reduce_by_key_int(cl_command_queue queue, uint64_t num_elements,
                               cl_mem keys_in, cl_mem vals_in, cl_mem keys_out,
                               cl_mem vals_out, bool take_any_value) {
  return ocl_reduce_by_key<int>(queue, num_elements, keys_in, vals_in, keys_out,
                                vals_out, take_any_value);
}

template <typename ValueType>
void ocl_gather(cl_command_queue queue, uint64_t num_elements,
                cl_mem source_positions, cl_mem values, cl_mem values_out) {
  boost::compute::buffer source_wrapper(source_positions);
  boost::compute::buffer values_wrapper(values);
  boost::compute::buffer values_out_wrapper(values_out);
  boost::compute::command_queue queue_wrapper(queue);

  boost::compute::gather(
      boost::compute::make_buffer_iterator<uint64_t>(source_wrapper, 0),
      boost::compute::make_buffer_iterator<uint64_t>(source_wrapper,
                                                     num_elements),
      boost::compute::make_buffer_iterator<ValueType>(values_wrapper, 0),
      boost::compute::make_buffer_iterator<ValueType>(values_out_wrapper, 0),
      queue_wrapper);
}

void ocl_gather_float(cl_command_queue queue, uint64_t num_elements,
                      cl_mem source_positions, cl_mem values,
                      cl_mem values_out) {
  ocl_gather<float>(queue, num_elements, source_positions, values, values_out);
}

void ocl_gather_double(cl_command_queue queue, uint64_t num_elements,
                       cl_mem source_positions, cl_mem values,
                       cl_mem values_out) {
  ocl_gather<double>(queue, num_elements, source_positions, values, values_out);
}

void ocl_gather_uint64_t(cl_command_queue queue, uint64_t num_elements,
                         cl_mem source_positions, cl_mem values,
                         cl_mem values_out) {
  ocl_gather<uint64_t>(queue, num_elements, source_positions, values,
                       values_out);
}

void ocl_gather_uint32_t(cl_command_queue queue, uint64_t num_elements,
                         cl_mem source_positions, cl_mem values,
                         cl_mem values_out) {
  ocl_gather<uint32_t>(queue, num_elements, source_positions, values,
                       values_out);
}

void ocl_gather_int32_t(cl_command_queue queue, uint64_t num_elements,
                        cl_mem source_positions, cl_mem values,
                        cl_mem values_out) {
  ocl_gather<int>(queue, num_elements, source_positions, values, values_out);
}

void ocl_gather_char(cl_command_queue queue, uint64_t num_elements,
                     cl_mem source_positions, cl_mem values,
                     cl_mem values_out) {
  ocl_gather<char>(queue, num_elements, source_positions, values, values_out);
}

cl_int wrapper_memcpy(void* __restrict dest, const void* __restrict src,
                      size_t num_bytes) {
  std::cout << "MemCpy" << std::endl;
  memcpy(dest, src, num_bytes);
  return CL_SUCCESS;
}

void CL_CALLBACK cleanup_buffer_callback(cl_event event,
                                         cl_int event_command_exec_status,
                                         void* user_data) {
  std::cout << "Called Cleanup Callback" << std::endl;
  cl_mem pinned_host_memory = (cl_mem)user_data;
  if (pinned_host_memory) clReleaseMemObject(pinned_host_memory);
  clReleaseEvent(event);
}

typedef boost::function<void()> MemCopyFunction;
typedef boost::function<cl_int()> OCLEnqueueFunction;

void CL_CALLBACK async_host_memcpy_callback(cl_event event,
                                            cl_int event_command_exec_status,
                                            void* user_data) {
  std::cout << "MemCpy" << std::endl;
  MemCopyFunction* memcpy_func = static_cast<MemCopyFunction*>(user_data);
  /*perform memcpy*/
  (*memcpy_func)();
  delete memcpy_func;
  clReleaseEvent(event);
}

void CL_CALLBACK execute_enqueue_commands_callback(
    cl_event event, cl_int event_command_exec_status, void* user_data) {
  std::cout << "Enqueue New Commands" << std::endl;
  std::vector<OCLEnqueueFunction>* functions =
      static_cast<std::vector<OCLEnqueueFunction>*>(user_data);
  for (size_t i = 0; i < functions->size(); ++i) {
    std::cout << "Command: " << i << std::endl;
    cl_int err = (*functions)[i]();
    CL_CHECK(err);
  }
  delete functions;
  clReleaseEvent(event);
}

cl_int oclCopyHostToDevice(const void* data, size_t num_bytes,
                           cl_mem input_buffer, cl_context context,
                           cl_command_queue command_queue,
                           cl_int blocking_write, cl_event* event) {
  cl_int err = CL_SUCCESS;

  //  ocl_start_timer();
  if (blocking_write) {
    err = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                               num_bytes, data, 0, NULL, event);
  } else {
    cl_mem mem_pinned_host_memory = NULL;
    /* allocate pinned host memory */
    mem_pinned_host_memory =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                       num_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Allocation Failed!" << std::endl;
      return err;
    }
    cl_event map_buffer_event;
    unsigned char* pinned_host_memory = (unsigned char*)clEnqueueMapBuffer(
        command_queue, mem_pinned_host_memory, CL_TRUE, CL_MAP_WRITE, 0,
        num_bytes, 0, NULL, &map_buffer_event, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Error! Map Buffer Failed!" << std::endl;
      return err;
    }

    /* copy data to pinned host memory, as soon as the map buffer function is
     * complete*/
    MemCopyFunction* memcpy_func = new MemCopyFunction(
        boost::bind(memcpy, pinned_host_memory, data, num_bytes));
    err = clSetEventCallback(map_buffer_event, CL_COMPLETE,
                             &async_host_memcpy_callback, memcpy_func);
    if (err != CL_SUCCESS) {
      return err;
    }
    /* copy pinned host memory to device */
    cl_event* new_event = NULL;
    cl_event event_placeholder;
    if (event)
      new_event = event;
    else
      new_event = &event_placeholder;
    err = clEnqueueWriteBuffer(command_queue, input_buffer, CL_FALSE, 0,
                               num_bytes, pinned_host_memory, 1,
                               &map_buffer_event, new_event);
    if (err != CL_SUCCESS) {
      return err;
    }
    err = clSetEventCallback(*new_event, CL_COMPLETE, &cleanup_buffer_callback,
                             (void*)mem_pinned_host_memory);
    if (err != CL_SUCCESS) {
      return err;
    }
  }
  //  ocl_stop_timer("Copy Time");
  return CL_SUCCESS;
}

cl_int oclCopyDeviceToHost(cl_mem output_buffer, size_t num_bytes, void* data,
                           cl_context context, cl_command_queue command_queue,
                           cl_int blocking_write, cl_event* event) {
  cl_int err = CL_SUCCESS;

  err = clEnqueueReadBuffer(command_queue, output_buffer, blocking_write, 0,
                            num_bytes, data, 0, NULL, event);

  return err;
}

cl_mem oclCachedCopyHostToDevice(const void* data, size_t num_bytes,
                                 cl_device_id dev_id, cl_int* error_code) {
  cl_mem mem =
      CoGaDB::OCL_DataCaches::instance().getMemBuffer(dev_id, data, num_bytes);
  if (error_code) {
    if (mem) {
      *error_code = CL_SUCCESS;
    } else {
      *error_code = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }
  }
  return mem;
}

void freeOCLCuckooHT(void* ptr) {
  OCLCuckooHT* ht = static_cast<OCLCuckooHT*>(ptr);

  CoGaDB::OCL_DataCaches::instance().uncacheMemoryArea(
      ht->hash_table, sizeof(uint64_t) * ht->hash_table_size);

  CoGaDB::OCL_DataCaches::instance().uncacheMemoryArea(
      ht->overflow, sizeof(uint64_t) * ht->overflow_size);

  free(ht->hash_table);
  free(ht->overflow);
  free(ht);
}

void freeOCLLinearProbingHT(void* ptr) {
  OCLLinearProbingHT* ht = static_cast<OCLLinearProbingHT*>(ptr);

  CoGaDB::OCL_DataCaches::instance().uncacheMemoryArea(
      ht->hash_table, sizeof(uint64_t) * ht->hash_table_size);

  free(ht->hash_table);
  free(ht);
}

void freeOCLCuckoo2HashesHT(void* ptr) {
  OCLCuckoo2HashesHT* ht = static_cast<OCLCuckoo2HashesHT*>(ptr);

  CoGaDB::OCL_DataCaches::instance().uncacheMemoryArea(
      ht->hash_table, sizeof(uint64_t) * (ht->hash_table_size));

  free(ht->hash_table);
  free(ht);
}

#define FILL_TEMPLATE()                                                      \
  auto buffer = boost::compute::buffer(mem);                                 \
  auto bqueue = boost::compute::command_queue(queue);                        \
  auto begin =                                                               \
      boost::compute::buffer_iterator<decltype(init_value)>(buffer, offset); \
  auto end = boost::compute::buffer_iterator<decltype(init_value)>(          \
      buffer, offset + count);                                               \
  boost::compute::fill_async(begin, end, init_value, bqueue).wait();

void oclFillBuffer_char(cl_command_queue queue, cl_mem mem, char init_value,
                        uint64_t offset, uint64_t count) {
  FILL_TEMPLATE();
}

void oclFillBuffer_uint64_t(cl_command_queue queue, cl_mem mem,
                            uint64_t init_value, uint64_t offset,
                            uint64_t count) {
  FILL_TEMPLATE();
}

void oclFillBuffer_uint32_t(cl_command_queue queue, cl_mem mem,
                            uint32_t init_value, uint64_t offset,
                            uint64_t count) {
  FILL_TEMPLATE();
}

void oclFillBuffer_float(cl_command_queue queue, cl_mem mem, float init_value,
                         uint64_t offset, uint64_t count) {
  FILL_TEMPLATE();
}

void oclFillBuffer_double(cl_command_queue queue, cl_mem mem, double init_value,
                          uint64_t offset, uint64_t count) {
  FILL_TEMPLATE();
}
