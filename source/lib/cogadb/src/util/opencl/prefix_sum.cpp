
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/thread.hpp>
#include <iostream>

#include <core/global_definitions.hpp>
#include <query_compilation/ocl_api.hpp>
#include <util/opencl/prefix_sum.hpp>
#include <util/time_measurement.hpp>

//#define ENABLE_PREFIX_SUM_DEBUG_OUTPUT

namespace CoGaDB {

//__kernel void prefix_sum_sequential_kernel(ulong n, ulong initial_value,
//__global char* _buf0, __global ulong* _buf1)
//{
// const ulong start_idx = 0;
// ulong sum = initial_value;
// for(ulong i = start_idx; i < n; i++){
// const char x = _buf0[i];
//_buf1[i] = sum;
//    sum = ((sum)+(x));
//}

//}

//#define __kernel
//#define __global

//__kernel void prefix_sum_sequential_kernel(const ulong num_elements,
//                                                     const ulong num_threads,
//                                                     const ulong
//                                                     initial_value,
//                                                     __global const char*
//                                                     flags,
//                                                     __global const ulong*
//                                                     prefix_sum){
//  ulong tuple_id=(num_elements/num_threads)*get_global_id(0);
//  ulong tmp = tuple_id+(num_elements/num_threads);
//  ulong end_index;
//  if(num_elements > tmp){
//      end_index = tmp;
//  }else{
//      end_index = num_elements;
//  }
//  ulong sum = initial_value;
//  for(;tuple_id<end_index;++tuple_id){
//    const char x = flags[i];
//    prefix_sum[i] = sum;
//    sum = sum+x;
//  }
//}

//__kernel void prefix_blocksum_sequential_uint8_flag_kernel(const ulong
// num_elements,
//                                           const ulong num_threads,
//                                           __global const char* flags,
//                                           __global const ulong* block_sums,
//                                           __global const ulong* prefix_sum){
//  ulong tuple_id=(num_elements/num_threads)*get_global_id(0);
//  ulong tmp = tuple_id+(num_elements/num_threads);
//  ulong end_index;
//  if(num_elements > tmp){
//      end_index = tmp;
//  }else{
//      end_index = num_elements;
//  }
//  ulong sum = 0;
//  for(;tuple_id<end_index;++tuple_id){
//    const char x = flags[i];
//    prefix_sum[tuple_id]=sum;
//    sum = sum+x;
//  }
//  block_sums[get_global_id(0)]=sum;
//}

//__kernel void prefix_blocksum_sequential_uint64_flag_kernel(const ulong
// num_elements,
//                                           const ulong num_threads,
//                                           __global const ulong* flags,
//                                           __global const ulong* block_sums,
//                                           __global const ulong* prefix_sum){
//  ulong tuple_id=(num_elements/num_threads)*get_global_id(0);
//  ulong tmp = tuple_id+(num_elements/num_threads);
//  ulong end_index;
//  if(num_elements > tmp){
//      end_index = tmp;
//  }else{
//      end_index = num_elements;
//  }
//  ulong sum = 0;
//  for(;tuple_id<end_index;++tuple_id){
//    const char x = flags[i];
//    prefix_sum[tuple_id]=sum;
//    sum = sum+x;
//  }
//  block_sums[get_global_id(0)]=sum;
//}

//__kernel void write_scanned_output_kernel(const ulong num_elements,
//                                           const ulong num_threads,
//                                           __global const ulong* block_sums,
//                                           __global const ulong* prefix_sum){
//  ulong tuple_id=(num_elements/num_threads)*get_global_id(0);
//  ulong tmp = tuple_id+(num_elements/num_threads);
//  ulong end_index;
//  if(num_elements > tmp){
//      end_index = tmp;
//  }else{
//      end_index = num_elements;
//  }
//  for(;tuple_id<end_index;++tuple_id){
//    prefix_sum[tuple_id] =
//    ((block_sums[get_global_id(0)])+(prefix_sum[tuple_id]));
//  }
//}

// const char* c_prog = ""
//                     "__kernel void prefix_sum_sequential_kernel(const ulong
//                     num_elements,"
//                     " const ulong num_threads,"
//                     " const ulong initial_value,"
//                     " __global const char* flags,"
//                     " __global const ulong* prefix_sum){"
//                     "  ulong
//                     tuple_id=(num_elements/num_threads)*get_global_id(0);"
//                     "  ulong tmp = tuple_id+(num_elements/num_threads);"
//                     "  ulong end_index;"
//                     "  if(num_elements > tmp){"
//                     "      end_index = tmp;"
//                     "  }else{"
//                     "      end_index = num_elements;"
//                     "  }"
//                     "  ulong sum = initial_value;"
//                     "  for(;tuple_id<end_index;++tuple_id){"
//                     "    const char x = flags[i];"
//                     "    prefix_sum[i] = sum;"
//                     "    sum = sum+x;"
//                     "  }"
//                     "}"
//                     ""
//                     "__kernel void
//                     prefix_blocksum_sequential_uint8_flag_kernel(const ulong
//                     num_elements,"
//                     "                                           const ulong
//                     num_threads,"
//                     "                                           __global
//                     const char* flags,"
//                     "                                           __global
//                     const ulong* block_sums,"
//                     "                                           __global
//                     const ulong* prefix_sum){"
//                     "  ulong
//                     tuple_id=(num_elements/num_threads)*get_global_id(0);"
//                     "  ulong tmp = tuple_id+(num_elements/num_threads);"
//                     "  ulong end_index;"
//                     "  if(num_elements > tmp){"
//                     "      end_index = tmp;"
//                     "  }else{"
//                     "      end_index = num_elements;"
//                     "  }"
//                     "  ulong sum = 0;"
//                     "  for(;tuple_id<end_index;++tuple_id){"
//                     "    const char x = flags[i];"
//                     "    prefix_sum[tuple_id]=sum;"
//                     "    sum = sum+x;"
//                     "  }"
//                     "  block_sums[get_global_id(0)]=sum;"
//                     "}"
//                     ""
//                     "__kernel void
//                     prefix_blocksum_sequential_uint64_flag_kernel(const ulong
//                     num_elements,"
//                     "                                           const ulong
//                     num_threads,"
//                     "                                           __global
//                     const ulong* flags,"
//                     "                                           __global
//                     const ulong* block_sums,"
//                     "                                           __global
//                     const ulong* prefix_sum){"
//                     "  ulong
//                     tuple_id=(num_elements/num_threads)*get_global_id(0);"
//                     "  ulong tmp = tuple_id+(num_elements/num_threads);"
//                     "  ulong end_index;"
//                     "  if(num_elements > tmp){"
//                     "      end_index = tmp;"
//                     "  }else{"
//                     "      end_index = num_elements;"
//                     "  }"
//                     "  ulong sum = 0;"
//                     "  for(;tuple_id<end_index;++tuple_id){"
//                     "    const char x = flags[i];"
//                     "    prefix_sum[tuple_id]=sum;"
//                     "    sum = sum+x;"
//                     "  }"
//                     "  block_sums[get_global_id(0)]=sum;"
//                     "}"
//                     ""
//                     "__kernel void write_scanned_output_kernel(const ulong
//                     num_elements,"
//                     "                                           const ulong
//                     num_threads,"
//                     "                                           __global
//                     const ulong* block_sums,"
//                     "                                           __global
//                     const ulong* prefix_sum){"
//                     "  ulong
//                     tuple_id=(num_elements/num_threads)*get_global_id(0);"
//                     "  ulong tmp = tuple_id+(num_elements/num_threads);"
//                     "  ulong end_index;"
//                     "  if(num_elements > tmp){"
//                     "      end_index = tmp;"
//                     "  }else{"
//                     "      end_index = num_elements;"
//                     "  }"
//                     "  for(;tuple_id<end_index;++tuple_id){"
//                     "    prefix_sum[tuple_id] =
//                     ((block_sums[get_global_id(0)])+(prefix_sum[tuple_id]));"
//                     "  }"
//                     "}";

static const char* c_prog =
    "__kernel void prefix_sum_sequential_kernel(const ulong num_elements,\n"
    "                                                     const ulong "
    "initial_value,\n"
    "                                                     __global const "
    "ulong* flags,\n"
    "                                                     __global ulong* "
    "prefix_sum){\n"
    "  ulong sum = initial_value;\n"
    "  for(ulong tuple_id=0;tuple_id<num_elements;++tuple_id){\n"
    "    const ulong x = flags[tuple_id];\n"
    "    prefix_sum[tuple_id] = sum;\n"
    "    sum = sum+x;\n"
    "  }\n"
    "}\n"
    "\n"
    "__kernel void prefix_blocksum_sequential_uint8_flag_kernel(const ulong "
    "num_elements,\n"
    "                                           const ulong num_threads,\n"
    "                                           __global const char* flags,\n"
    "                                           __global ulong* block_sums,\n"
    "                                           __global ulong* prefix_sum){\n"
    "  ulong block_size=(num_elements+num_threads-1)/num_threads;\n"
    "  ulong tuple_id=block_size*get_global_id(0);\n"
    "  ulong tmp = tuple_id+block_size;\n"
    "  ulong end_index;\n"
    "  if(num_elements > tmp){\n"
    "      end_index = tmp;\n"
    "  }else{\n"
    "      end_index = num_elements;\n"
    "  }\n"
    "  ulong sum = 0;\n"
    "  for(;tuple_id<end_index;++tuple_id){\n"
    "    const char x = flags[tuple_id];\n"
    "    prefix_sum[tuple_id]=sum;\n"
    "    sum = sum+x;\n"
    "  }\n"
    "  block_sums[get_global_id(0)]=sum;\n"
    "}\n"
    "\n"
    "__kernel void prefix_blocksum_sequential_uint64_flag_kernel(const ulong "
    "num_elements,\n"
    "                                           const ulong num_threads,\n"
    "                                           __global const ulong* flags,\n"
    "                                           __global ulong* block_sums,\n"
    "                                           __global ulong* prefix_sum){\n"
    "  ulong block_size=(num_elements+num_threads-1)/num_threads;\n"
    "  ulong tuple_id=block_size*get_global_id(0);\n"
    "  ulong tmp = tuple_id+block_size;\n"
    "  ulong end_index;\n"
    "  if(num_elements > tmp){\n"
    "      end_index = tmp;\n"
    "  }else{\n"
    "      end_index = num_elements;\n"
    "  }\n"
    "  ulong sum = 0;\n"
    "  for(;tuple_id<end_index;++tuple_id){\n"
    "    const char x = flags[tuple_id];\n"
    "    prefix_sum[tuple_id]=sum;\n"
    "    sum = sum+x;\n"
    "  }\n"
    "  block_sums[get_global_id(0)]=sum;\n"
    "}\n"
    "\n"
    "__kernel void write_scanned_output_kernel(const ulong num_elements,\n"
    "                                           const ulong num_threads,\n"
    "                                           __global const ulong* "
    "block_sums,\n"
    "                                           __global ulong* prefix_sum){\n"
    "  ulong block_size=(num_elements+num_threads-1)/num_threads;\n"
    "  ulong tuple_id=block_size*get_global_id(0);\n"
    "  ulong tmp = tuple_id+block_size;\n"
    "  ulong end_index;\n"
    "  if(num_elements > tmp){\n"
    "      end_index = tmp;\n"
    "  }else{\n"
    "      end_index = num_elements;\n"
    "  }\n"
    "  for(;tuple_id<end_index;++tuple_id){\n"
    "    prefix_sum[tuple_id] = "
    "((block_sums[get_global_id(0)])+(prefix_sum[tuple_id]));\n"
    "  }\n"
    "}\n";

OCL_Kernels& OCL_Kernels::instance() {
  static OCL_Kernels kernels;
  return kernels;
}

OCL_Kernels::OCL_Kernels() : programs_(), mutex_() {}

cl_kernel OCL_Kernels::getKernel(const std::string& kernel_name,
                                 cl_device_id device_id, cl_context context) {
  boost::lock_guard<boost::mutex> lock(mutex_);

  cl_int err = CL_SUCCESS;
  Programs::iterator it = programs_.find(device_id);
  if (it == programs_.end()) {
    cl_program prefix_sum_program;
    prefix_sum_program =
        clCreateProgramWithSource(context, 1, &c_prog, NULL, &err);
    assert(prefix_sum_program != NULL && err == CL_SUCCESS);
    if (clBuildProgram(prefix_sum_program, 1, &device_id, "", NULL, NULL) !=
        CL_SUCCESS) {
      char buffer[10240];
      clGetProgramBuildInfo(prefix_sum_program, device_id, CL_PROGRAM_BUILD_LOG,
                            sizeof(buffer), buffer, NULL);
      fprintf(stderr, "CL Compilation failed:\n%s", buffer);
      abort();
    } else {
      std::cout << "Kernel Compilation Successfull!" << std::endl;
    }
    std::pair<Programs::iterator, bool> ret =
        programs_.insert(std::make_pair(device_id, prefix_sum_program));
    assert(ret.second == true);
    it = ret.first;
    assert(it != programs_.end());
  }

  cl_kernel kernel;
  kernel = clCreateKernel(it->second, kernel_name.c_str(), &err);

  if (err != CL_SUCCESS) {
    COGADB_FATAL_ERROR("Create Kernel Failed! Error code: " << err, "");
  }
  assert(kernel != NULL);
  return kernel;
}

void OCL_Kernels::resetCache() {
  boost::lock_guard<boost::mutex> lock(mutex_);
  for (auto& program : programs_) {
    clReleaseProgram(program.second);
  }

  programs_.clear();
}

size_t getBlockCount(size_t num_elements, size_t num_threads) {
  size_t elements_per_thread = (num_elements + num_threads - 1) / num_threads;
  if (elements_per_thread == 0) {
    elements_per_thread = 1;
  }
  size_t num_blocks =
      (num_elements + elements_per_thread - 1) / elements_per_thread;
  if (num_elements < 1000) {
    return 1;
  } else {
    return num_blocks;
  }
}

void run_and_wait_for_kernel(cl_command_queue queue, cl_kernel kernel,
                             const size_t global_work_size) {
  cl_event kernel_completion;
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size,
                                  NULL, 0, NULL, &kernel_completion));
  CL_CHECK(clWaitForEvents(1, &kernel_completion));
  CL_CHECK(clReleaseEvent(kernel_completion));
}

cl_int ocl_serial_prefix_sum(cl_command_queue queue, cl_program program,
                             cl_mem cl_output_mem_flags,
                             cl_mem cl_output_prefix_sum, size_t num_elements) {
  cl_context context;
  CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context),
                                 &context, NULL));
  assert(context != NULL);
  cl_device_id device_id;
  CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device_id),
                                 &device_id, NULL));
  assert(device_id != NULL);

  cl_ulong initial_value = 0;

  cl_kernel scan_kernel;
  scan_kernel = OCL_Kernels::instance().getKernel(
      "prefix_sum_sequential_kernel", device_id, context);
  CL_CHECK(clSetKernelArg(scan_kernel, 0, sizeof(num_elements), &num_elements));
  CL_CHECK(
      clSetKernelArg(scan_kernel, 1, sizeof(initial_value), &initial_value));

  CL_CHECK(clSetKernelArg(scan_kernel, 2, sizeof(cl_output_mem_flags),
                          &cl_output_mem_flags));
  CL_CHECK(clSetKernelArg(scan_kernel, 3, sizeof(cl_output_prefix_sum),
                          &cl_output_prefix_sum));
  run_and_wait_for_kernel(queue, scan_kernel, 1);
  clReleaseKernel(scan_kernel);

  return CL_SUCCESS;
}

cl_int ocl_prefix_sum_xeon_phi_coprocessor_impl(cl_command_queue queue,
                                                cl_program program,
                                                cl_mem cl_output_mem_flags,
                                                cl_mem cl_output_prefix_sum,
                                                size_t num_elements) {
  cl_int err = CL_SUCCESS;

  cl_context context;
  CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context),
                                 &context, NULL));
  assert(context != NULL);
  cl_device_id device_id;
  CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device_id),
                                 &device_id, NULL));
  assert(device_id != NULL);

  cl_device_type device_type;

  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type),
                           &device_type, NULL));

  cl_ulong num_threads = 1;
  if (device_type == CL_DEVICE_TYPE_CPU) {
    num_threads = boost::thread::hardware_concurrency();
  } else if (device_type == CL_DEVICE_TYPE_ACCELERATOR) {
    num_threads = 61;
  } else {
    COGADB_FATAL_ERROR(
        "Unsupported opencl device type for parallel prefix sum: "
            << device_type,
        "");
  }

  cl_ulong block_count = num_threads;
  if (block_count >= num_elements) {
    block_count = 1;
  }
  cl_ulong block_size = (num_elements + block_count - 1) / block_count;
  if (block_size == 0) {
    block_size = 1;
  }

  if (num_elements < 1000) {
    return ocl_serial_prefix_sum(queue, program, cl_output_mem_flags,
                                 cl_output_prefix_sum, num_elements);
  }

  cl_mem block_sum = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(uint64_t) * block_count, NULL, &err);
  assert(err == CL_SUCCESS);
  oclFillBuffer_uint64_t(queue, block_sum, 0, 0, block_count);

  cl_kernel blockscan_kernel;
  blockscan_kernel = OCL_Kernels::instance().getKernel(
      "prefix_blocksum_sequential_uint64_flag_kernel", device_id, context);
  CL_CHECK(
      clSetKernelArg(blockscan_kernel, 0, sizeof(num_elements), &num_elements));
  CL_CHECK(
      clSetKernelArg(blockscan_kernel, 1, sizeof(block_count), &block_count));

  CL_CHECK(clSetKernelArg(blockscan_kernel, 2, sizeof(cl_output_mem_flags),
                          &cl_output_mem_flags));

  CL_CHECK(clSetKernelArg(blockscan_kernel, 3, sizeof(block_sum), &block_sum));

  CL_CHECK(clSetKernelArg(blockscan_kernel, 4, sizeof(cl_output_prefix_sum),
                          &cl_output_prefix_sum));
  run_and_wait_for_kernel(queue, blockscan_kernel, block_count);

  if (block_count > 1) {
    ocl_prefix_sum_xeon_phi_coprocessor_impl(queue, program, block_sum,
                                             block_sum, block_count);
  }

  if (block_count > 1) {
    cl_kernel write_output_kernel;
    write_output_kernel = OCL_Kernels::instance().getKernel(
        "write_scanned_output_kernel", device_id, context);
    //          clCreateKernel(prefix_sum_program,
    //          "write_scanned_output_kernel", &err);
    CL_CHECK(clSetKernelArg(write_output_kernel, 0, sizeof(num_elements),
                            &num_elements));
    CL_CHECK(clSetKernelArg(write_output_kernel, 1, sizeof(block_count),
                            &block_count));

    CL_CHECK(
        clSetKernelArg(write_output_kernel, 2, sizeof(block_sum), &block_sum));

    CL_CHECK(clSetKernelArg(write_output_kernel, 3,
                            sizeof(cl_output_prefix_sum),
                            &cl_output_prefix_sum));

    run_and_wait_for_kernel(queue, write_output_kernel, block_count);

    CL_CHECK(clReleaseKernel(write_output_kernel));
  }

  CL_CHECK(clReleaseMemObject(block_sum));
  CL_CHECK(clReleaseKernel(blockscan_kernel));

  return CL_SUCCESS;
}

cl_int ocl_prefix_sum_xeon_phi_coprocessor(cl_command_queue queue,
                                           cl_program program,
                                           cl_mem cl_output_mem_flags,
                                           cl_mem cl_output_prefix_sum,
                                           size_t num_elements) {
  /* workaround: convert flags from char in uint64_t array
   */
  boost::compute::buffer buf_flags(cl_output_mem_flags);
  boost::compute::command_queue queue2(queue, true);
  auto context = queue2.get_context();

  boost::compute::vector<uint64_t> flags_(num_elements, context);

  boost::compute::copy(
      boost::compute::make_buffer_iterator<char>(buf_flags, 0),
      boost::compute::make_buffer_iterator<char>(buf_flags, num_elements),
      flags_.begin(), queue2);

  return ocl_prefix_sum_xeon_phi_coprocessor_impl(
      queue, program, flags_.get_buffer().get(), cl_output_prefix_sum,
      num_elements);
}

size_t ocl_prefix_sum(cl_command_queue queue, cl_program program,
                      cl_mem cl_output_mem_flags, cl_mem cl_output_prefix_sum,
                      size_t num_elements) {
  if (num_elements == 0) {
    return 0;
  }

  cl_device_id device_id;
  cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device_id),
                                     &device_id, NULL);
  if (err != CL_SUCCESS)
    COGADB_FATAL_ERROR("Failed to get Command Queue Info!", "");

#ifdef ENABLE_PREFIX_SUM_DEBUG_OUTPUT
  ocl_print_device(device_id);

  CoGaDB::Timestamp begin = CoGaDB::getTimestamp();
#endif

  boost::compute::buffer buf_flags(cl_output_mem_flags);
  boost::compute::buffer buf_prefix_sum(cl_output_prefix_sum);

  boost::compute::command_queue queue2(queue, true);

  cl_device_type device_type;
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type),
                           &device_type, NULL));

#ifdef ENABLE_PREFIX_SUM_DEBUG_OUTPUT
  CoGaDB::Timestamp end_buffer_creation = CoGaDB::getTimestamp();
#endif

  if (device_type == CL_DEVICE_TYPE_CPU) {
    cl_int err =
        ocl_prefix_sum_xeon_phi_coprocessor(queue, program, cl_output_mem_flags,
                                            cl_output_prefix_sum, num_elements);
    assert(err == CL_SUCCESS);
  } else if (device_type == CL_DEVICE_TYPE_GPU) {
    /* boost compute GPU workaround: convert flags from char in uint64_t array
     */
    cl_context context;
    CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context),
                                   &context, NULL));
    assert(context != NULL);

    boost::compute::context context2(context, true);

    boost::compute::vector<uint64_t> flags_(num_elements, context2);

    boost::compute::copy(
        boost::compute::make_buffer_iterator<char>(buf_flags, 0),
        boost::compute::make_buffer_iterator<char>(buf_flags, num_elements),
        flags_.begin(), queue2);

    boost::compute::exclusive_scan(
        flags_.begin(), flags_.end(),
        boost::compute::make_buffer_iterator<uint64_t>(buf_prefix_sum, 0),
        queue2);

    clFinish(queue);
  } else if (device_type == CL_DEVICE_TYPE_ACCELERATOR) {
    cl_int err =
        ocl_prefix_sum_xeon_phi_coprocessor(queue, program, cl_output_mem_flags,
                                            cl_output_prefix_sum, num_elements);
    assert(err == CL_SUCCESS);
  } else {
    COGADB_FATAL_ERROR("Unknown Device Type!", "");
  }

#ifdef ENABLE_PREFIX_SUM_DEBUG_OUTPUT
  CoGaDB::Timestamp end = CoGaDB::getTimestamp();

  std::cout << "Time Prefix Sum: " << double(end - begin) / (1000 * 1000 * 1000)
            << "s" << std::endl;
  std::cout << "Time Buffer Creation: "
            << double(end_buffer_creation - begin) / (1000 * 1000 * 1000) << "s"
            << std::endl;
  std::cout << "Tupels: " << num_elements << std::endl;
#endif

  std::vector<char> flags_cpu;
  std::vector<uint64_t> prefix_sum_cpu;

  flags_cpu.resize(1);
  prefix_sum_cpu.resize(1);

  boost::compute::copy(
      boost::compute::make_buffer_iterator<char>(buf_flags, num_elements - 1),
      boost::compute::make_buffer_iterator<char>(buf_flags, num_elements),
      flags_cpu.begin(), queue2);

  boost::compute::copy(boost::compute::make_buffer_iterator<uint64_t>(
                           buf_prefix_sum, num_elements - 1),
                       boost::compute::make_buffer_iterator<uint64_t>(
                           buf_prefix_sum, num_elements),
                       prefix_sum_cpu.begin(), queue2);

  char last_flag = flags_cpu.front();
  uint64_t last_prefix_sum = prefix_sum_cpu.front();

  uint64_t result_sum = last_prefix_sum + static_cast<uint64_t>(last_flag);

#ifdef ENABLE_PREFIX_SUM_DEBUG_OUTPUT
  std::cout << "Last Prefix Sum: " << last_prefix_sum << std::endl;
  std::cout << "Last Flag: " << last_flag << std::endl;
  std::cout << "Sum: " << result_sum << std::endl;
#endif

  return result_sum;
}

}  // end namespace CoGaDB
