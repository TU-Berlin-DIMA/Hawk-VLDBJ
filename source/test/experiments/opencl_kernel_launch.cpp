/*
 * File:   opencl_kernel_launch.cpp
 * Author: sebastian
 *
 * Created on 20. Februar 2016, 12:09
 */

#include <cstdlib>
#include <iostream>
#include <persistence/storage_manager.hpp>
#include <util/getname.hpp>
#include <util/tests.hpp>

#include <query_compilation/minimal_api_c.h>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/thread/lock_guard.hpp>
#include <core/block_iterator.hpp>
#include <parser/commandline_interpreter.hpp>
#include <query_compilation/algebraic_aggregate_specification.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/minimal_api_c_internal.hpp>
#include <query_compilation/pipeline_info.hpp>
#include <query_compilation/predicate_expression.hpp>
#include <query_compilation/query_context.hpp>

#include <query_compilation/ocl_api.hpp>
#include <util/query_processing.hpp>
//#include <boost/compute/algorithm/exclusive_scan.hpp>
//#include <boost/compute/algorithm/reduce.hpp>

//#define USE_COALESCED_MEMORY_ACCESS

#define NUM_DATA 100

void pfn_notify(const char* errinfo, const void* private_info, size_t cb,
                void* user_data) {
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

cl_platform_id* platforms = NULL;
cl_uint platforms_n = 0;
cl_context context;
cl_command_queue queue;
cl_program program = NULL;
cl_device_id* devices = NULL;

cl_device_id* selected_device = NULL;
int device_type = CL_DEVICE_TYPE_CPU;
// int device_type = CL_DEVICE_TYPE_ALL;

const std::string readFileContent(const std::string& path_to_file) {
  std::ifstream in(path_to_file.c_str());
  std::stringstream ss_in;
  ss_in << in.rdbuf();
  return ss_in.str();
}

int init_ocl() {
  CL_CHECK(clGetPlatformIDs(0, NULL, &platforms_n));

  if (!platforms_n) {
    return -1;
  }

  platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id*) * platforms_n);

  CL_CHECK(clGetPlatformIDs(platforms_n, platforms, NULL));

  std::cout << " we  detect " << platforms_n << " platforms " << std::endl;

  printf("=== %d OpenCL platform(s) found: ===\n", platforms_n);
  for (cl_uint i = 0; i < platforms_n; i++) {
    char buffer[10240];
    printf("  -- %d --\n", i);
    CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer,
                               NULL));
    printf("  PROFILE = %s\n", buffer);
    CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer,
                               NULL));
    printf("  VERSION = %s\n", buffer);
    CL_CHECK(
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
    printf("  NAME = %s\n", buffer);
    CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer,
                               NULL));
    printf("  VENDOR = %s\n", buffer);
    CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240,
                               buffer, NULL));
    printf("  EXTENSIONS = %s\n", buffer);

    cl_uint devices_n = 0;
    if (clGetDeviceIDs(platforms[i], device_type, 0, NULL, &devices_n) !=
        CL_SUCCESS) {
      continue;
    }
    if (devices_n == 0) {
      continue;
    }

    devices = (cl_device_id*)malloc(sizeof(cl_device_id*) * devices_n);
    assert(devices != NULL);
    std::cout << "Found Devices: " << devices_n << std::endl;

    CL_CHECK(
        clGetDeviceIDs(platforms[i], device_type, devices_n, devices, NULL));

    if (!selected_device) {
      selected_device = (cl_device_id*)malloc(sizeof(cl_device_id*));
      CL_CHECK(
          clGetDeviceIDs(platforms[i], device_type, 1, selected_device, NULL));
    }

    printf("=== %d OpenCL device(s) found on platform:\n", platforms_n);
    for (cl_uint j = 0; j < devices_n; j++) {
      char buffer[10240];
      cl_uint buf_uint;
      cl_ulong buf_ulong;
      printf("  -- %d --\n", j);
      CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer),
                               buffer, NULL));
      printf("  DEVICE_NAME = %s\n", buffer);
      CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buffer),
                               buffer, NULL));
      printf("  DEVICE_VENDOR = %s\n", buffer);
      CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(buffer),
                               buffer, NULL));
      printf("  DEVICE_VERSION = %s\n", buffer);
      CL_CHECK(clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(buffer),
                               buffer, NULL));
      printf("  DRIVER_VERSION = %s\n", buffer);
      CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                               sizeof(buf_uint), &buf_uint, NULL));
      printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
      CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                               sizeof(buf_uint), &buf_uint, NULL));
      printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
      CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                               sizeof(buf_ulong), &buf_ulong, NULL));
      printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n",
             (unsigned long long)buf_ulong);
    }

    free(devices);
  }

  cl_int _err;
  context = clCreateContext(NULL, 1, selected_device, &pfn_notify, NULL, &_err);
  assert(_err == CL_SUCCESS);
  if (!context) {
    return -1;
  }
  CoGaDB::Timestamp begin = CoGaDB::getTimestamp();
  std::string program_source =
      readFileContent(std::string(PATH_TO_COGADB_SOURCE_CODE) +
                      "/test/experiments/select_kernel.cl");
  std::cout << "Source: " << std::endl
            << "'" << program_source << "'" << std::endl;
  const char* c_prog = program_source.c_str();
  program = clCreateProgramWithSource(context, 1, &c_prog, NULL, &_err);
  assert(program != NULL && _err == CL_SUCCESS);
  if (clBuildProgram(program, 1, selected_device, "", NULL, NULL) !=
      CL_SUCCESS) {
    char buffer[10240];
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, NULL);
    fprintf(stderr, "CL Compilation failed:\n%s", buffer);
    abort();
  } else {
    std::cout << "Kernel Compilation Successfull!" << std::endl;
  }
  CoGaDB::Timestamp end = CoGaDB::getTimestamp();
  std::cout << "Kernel Compile Time: "
            << double(end - begin) / (1000 * 1000 * 1000) << "s" << std::endl;
  queue = clCreateCommandQueue(context, *selected_device, 0, &_err);
  assert(_err == CL_SUCCESS);
  return 0;
}

void cleanup_ocl() {
  CL_CHECK(clReleaseProgram(program));
  CL_CHECK(clReleaseContext(context));
  CL_CHECK(clReleaseCommandQueue(queue));
  //    clReleaseDevice();
  //    free(devices);
  free(platforms);
}

// class OCL_PinnedMemoryManagerImpl{

// public:
//  OCL_PinnedMemoryManagerImpl(cl_context context):
//  void* allocate(size_t bytes);
//  void deallocate(void*);

//  struct MemoryRegion{
//    size_t offset;
//    size_t size;
//    bool is_free;
//  };

//  typedef std::list<MemoryRegion> MemoryMap;
//  cl_context context_;
//  MemoryMap mem_map_;
//};

class OCL_PinnedMemoryManagerImpl;
typedef boost::shared_ptr<OCL_PinnedMemoryManagerImpl>
    OCL_PinnedMemoryManagerImplPtr;

class OCL_PinnedMemoryManager {
 public:
  void* allocate(size_t bytes, cl_context context, cl_command_queue queue);
  void deallocate(void*, cl_context context, cl_command_queue queue);

  static OCL_PinnedMemoryManager& instance();

 private:
  OCL_PinnedMemoryManager();
  OCL_PinnedMemoryManager(const OCL_PinnedMemoryManager&);
  void allocatePinnedMemory() {}

  boost::mutex mutex_;
  typedef std::map<cl_context, OCL_PinnedMemoryManagerImplPtr> ContextMap;
  typedef std::map<void*, cl_mem> MemoryMap;
  MemoryMap mem_map_;
};

OCL_PinnedMemoryManager::OCL_PinnedMemoryManager() : mutex_(), mem_map_() {}

void* OCL_PinnedMemoryManager::allocate(size_t bytes, cl_context context,
                                        cl_command_queue queue) {
  cl_int err = CL_SUCCESS;
  cl_mem mem_pinned_host_memory = clCreateBuffer(
      context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bytes, NULL, &err);
  if (err != CL_SUCCESS) {
    std::cout << "Allocation Failed!" << std::endl;
    return NULL;
  }
  unsigned char* pinned_host_memory = (unsigned char*)clEnqueueMapBuffer(
      queue, mem_pinned_host_memory, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL,
      NULL, &err);
  if (err != CL_SUCCESS) {
    std::cout << "Error! Map Buffer Failed!" << std::endl;
    return NULL;
  }

  mem_map_.insert(std::make_pair(pinned_host_memory, mem_pinned_host_memory));

  return pinned_host_memory;
}

void OCL_PinnedMemoryManager::deallocate(void* mem, cl_context context,
                                         cl_command_queue queue) {
  auto it = mem_map_.find(mem);
  assert(it != mem_map_.end());

  // TODO: unmap memory
  clReleaseMemObject(it->second);
}

OCL_PinnedMemoryManager& OCL_PinnedMemoryManager::instance() {
  static OCL_PinnedMemoryManager instance;
  return instance;
}

C_Table* ocl_cpu_selection_compiled_query(C_Table** c_tables) {
  std::cout << "Launching CPU Selection" << std::endl;
  C_Table* table_LINEORDER1 = c_tables[0];
  assert(table_LINEORDER1 != NULL);
  C_Column* col_LINEORDER1_LINEORDER_LO_QUANTITY1 =
      getColumnById(table_LINEORDER1, 8);
  if (!col_LINEORDER1_LINEORDER_LO_QUANTITY1) {
    printf("Column 'col_LINEORDER1_LINEORDER_LO_QUANTITY1' not found!\n");
    return NULL;
  }
  const int32_t* array_LINEORDER1_LINEORDER_LO_QUANTITY1 =
      getArrayFromColumn_int32_t(col_LINEORDER1_LINEORDER_LO_QUANTITY1);
  C_Column* col_LINEORDER1_LINEORDER_LO_DISCOUNT1 =
      getColumnById(table_LINEORDER1, 11);
  if (!col_LINEORDER1_LINEORDER_LO_DISCOUNT1) {
    printf("Column 'col_LINEORDER1_LINEORDER_LO_DISCOUNT1' not found!\n");
    return NULL;
  }
  const float* array_LINEORDER1_LINEORDER_LO_DISCOUNT1 =
      getArrayFromColumn_float(col_LINEORDER1_LINEORDER_LO_DISCOUNT1);
  C_Column* col_LINEORDER1_LINEORDER_LO_REVENUE1 =
      getColumnById(table_LINEORDER1, 12);
  if (!col_LINEORDER1_LINEORDER_LO_REVENUE1) {
    printf("Column 'col_LINEORDER1_LINEORDER_LO_REVENUE1' not found!\n");
    return NULL;
  }
  const float* array_LINEORDER1_LINEORDER_LO_REVENUE1 =
      getArrayFromColumn_float(col_LINEORDER1_LINEORDER_LO_REVENUE1);

  size_t num_elements_tuple_id_LINEORDER1 = getNumberOfRows(table_LINEORDER1);

  char* tempory_flag_array =
      (char*)realloc(NULL, num_elements_tuple_id_LINEORDER1 * sizeof(char));
  TID* tempory_prefix_sum_array =
      (TID*)realloc(NULL, num_elements_tuple_id_LINEORDER1 * sizeof(TID));

  cl_int _err = 0;

  cl_mem cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(uint32_t) * num_elements_tuple_id_LINEORDER1,
                     (void*)array_LINEORDER1_LINEORDER_LO_QUANTITY1, &_err);
  assert(_err == CL_SUCCESS);
  cl_mem cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(float) * num_elements_tuple_id_LINEORDER1,
                     (void*)array_LINEORDER1_LINEORDER_LO_DISCOUNT1, &_err);
  assert(_err == CL_SUCCESS);
  cl_mem cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1 =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(float) * num_elements_tuple_id_LINEORDER1,
                     (void*)array_LINEORDER1_LINEORDER_LO_REVENUE1, &_err);
  assert(_err == CL_SUCCESS);

  cl_mem cl_output_mem_flags =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                     sizeof(char) * num_elements_tuple_id_LINEORDER1,
                     tempory_flag_array, &_err);
  assert(_err == CL_SUCCESS);

  cl_mem cl_output_prefix_sum =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                     sizeof(TID) * num_elements_tuple_id_LINEORDER1,
                     tempory_prefix_sum_array, &_err);
  assert(_err == CL_SUCCESS);

  CoGaDB::Timestamp begin_filter_kernel = CoGaDB::getTimestamp();
  /* first phase: pass over data and compute flag array */
  cl_kernel select_hashprobe_kernel;
  select_hashprobe_kernel =
      clCreateKernel(program, "select_and_hashprobe_kernel", &_err);
  CL_CHECK(clSetKernelArg(select_hashprobe_kernel, 0,
                          sizeof(num_elements_tuple_id_LINEORDER1),
                          &num_elements_tuple_id_LINEORDER1));
  CL_CHECK(
      clSetKernelArg(select_hashprobe_kernel, 1,
                     sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1),
                     &cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(
      clSetKernelArg(select_hashprobe_kernel, 2,
                     sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1),
                     &cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(clSetKernelArg(select_hashprobe_kernel, 3,
                          sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1),
                          &cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1));
  CL_CHECK(clSetKernelArg(select_hashprobe_kernel, 4,
                          sizeof(cl_output_mem_flags), &cl_output_mem_flags));

  {
    cl_event kernel_completion;
    size_t global_work_size[1] = {4};
    CL_CHECK(clEnqueueNDRangeKernel(queue, select_hashprobe_kernel, 1, NULL,
                                    global_work_size, NULL, 0, NULL,
                                    &kernel_completion));
    CL_CHECK(clWaitForEvents(1, &kernel_completion));
    CL_CHECK(clReleaseEvent(kernel_completion));
  }

  CoGaDB::Timestamp end_filter_kernel = CoGaDB::getTimestamp();

  /* prefix sum */
  size_t result_size = 0;
  result_size =
      ocl_prefix_sum(queue, program, cl_output_mem_flags, cl_output_prefix_sum,
                     num_elements_tuple_id_LINEORDER1);

  /* second phase: pass over data and write result */
  int32_t* result_array_LINEORDER_LO_QUANTITY_1 =
      (int32_t*)realloc(NULL, result_size * sizeof(int32_t));
  float* result_array_LINEORDER_LO_DISCOUNT_1 =
      (float*)realloc(NULL, result_size * sizeof(float));
  float* result_array_LINEORDER_LO_REVENUE_1 =
      (float*)realloc(NULL, result_size * sizeof(float));

  cl_mem cl_output_mem_LINEORDER1_LINEORDER_LO_QUANTITY1 =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(uint32_t) * result_size,
                     result_array_LINEORDER_LO_QUANTITY_1, &_err);
  assert(_err == CL_SUCCESS);
  cl_mem cl_output_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1 = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(float) * result_size, result_array_LINEORDER_LO_DISCOUNT_1, &_err);
  assert(_err == CL_SUCCESS);
  cl_mem cl_output_mem_LINEORDER1_LINEORDER_LO_REVENUE1 = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(float) * result_size, result_array_LINEORDER_LO_REVENUE_1, &_err);
  assert(_err == CL_SUCCESS);
  CoGaDB::Timestamp begin_projection_kernel = CoGaDB::getTimestamp();
  cl_kernel hashprobe_aggregate_and_project_kernel;
  hashprobe_aggregate_and_project_kernel =
      clCreateKernel(program, "hashprobe_aggregate_and_project_kernel", &_err);
  CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 0,
                          sizeof(num_elements_tuple_id_LINEORDER1),
                          &num_elements_tuple_id_LINEORDER1));
  CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 1,
                          sizeof(cl_output_mem_flags), &cl_output_mem_flags));
  CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 2,
                          sizeof(cl_output_prefix_sum), &cl_output_prefix_sum));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 3,
                     sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1),
                     &cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 4,
                     sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1),
                     &cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 5,
                          sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1),
                          &cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 6,
                     sizeof(cl_output_mem_LINEORDER1_LINEORDER_LO_QUANTITY1),
                     &cl_output_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 7,
                     sizeof(cl_output_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1),
                     &cl_output_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 8,
                     sizeof(cl_output_mem_LINEORDER1_LINEORDER_LO_REVENUE1),
                     &cl_output_mem_LINEORDER1_LINEORDER_LO_REVENUE1));

  {
    cl_event kernel_completion;
    size_t global_work_size[1] = {4};
    CL_CHECK(clEnqueueNDRangeKernel(
        queue, hashprobe_aggregate_and_project_kernel, 1, NULL,
        global_work_size, NULL, 0, NULL, &kernel_completion));
    CL_CHECK(clWaitForEvents(1, &kernel_completion));
    CL_CHECK(clReleaseEvent(kernel_completion));
  }
  CoGaDB::Timestamp end_projection_kernel = CoGaDB::getTimestamp();

  C_Column* result_columns[] = {
      createResultArray_int32_t("LINEORDER.LO_QUANTITY.1",
                                result_array_LINEORDER_LO_QUANTITY_1,
                                result_size),
      createResultArray_float("LINEORDER.LO_DISCOUNT.1",
                              result_array_LINEORDER_LO_DISCOUNT_1,
                              result_size),
      createResultArray_float("LINEORDER.LO_REVENUE.1",
                              result_array_LINEORDER_LO_REVENUE_1,
                              result_size)};
  C_Table* result_table =
      createTableFromColumns("LINEORDER", result_columns, 3);

  free(tempory_flag_array);
  free(tempory_prefix_sum_array);

  CL_CHECK(clReleaseMemObject(cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(clReleaseMemObject(cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(clReleaseMemObject(cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1));
  CL_CHECK(clReleaseMemObject(cl_output_mem_flags));
  CL_CHECK(clReleaseMemObject(cl_output_prefix_sum));
  CL_CHECK(clReleaseMemObject(cl_output_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(clReleaseMemObject(cl_output_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(clReleaseMemObject(cl_output_mem_LINEORDER1_LINEORDER_LO_REVENUE1));

  CL_CHECK(clReleaseKernel(select_hashprobe_kernel));
  CL_CHECK(clReleaseKernel(hashprobe_aggregate_and_project_kernel));

  std::cout << "Filter Kernel:"
            << double(end_filter_kernel - begin_filter_kernel) /
                   (1000 * 1000 * 1000)
            << "s" << std::endl;
  std::cout << "Projection Kernel:"
            << double(end_projection_kernel - begin_projection_kernel) /
                   (1000 * 1000 * 1000)
            << "s" << std::endl;

  return result_table;
}

C_Table* ocl_gpu_selection_compiled_query(C_Table** c_tables) {
  std::cout << "Launching GPU Selection" << std::endl;
  C_Table* table_LINEORDER1 = c_tables[0];
  assert(table_LINEORDER1 != NULL);
  C_Column* col_LINEORDER1_LINEORDER_LO_QUANTITY1 =
      getColumnById(table_LINEORDER1, 8);
  if (!col_LINEORDER1_LINEORDER_LO_QUANTITY1) {
    printf("Column 'col_LINEORDER1_LINEORDER_LO_QUANTITY1' not found!\n");
    return NULL;
  }
  const int32_t* array_LINEORDER1_LINEORDER_LO_QUANTITY1 =
      getArrayFromColumn_int32_t(col_LINEORDER1_LINEORDER_LO_QUANTITY1);
  C_Column* col_LINEORDER1_LINEORDER_LO_DISCOUNT1 =
      getColumnById(table_LINEORDER1, 11);
  if (!col_LINEORDER1_LINEORDER_LO_DISCOUNT1) {
    printf("Column 'col_LINEORDER1_LINEORDER_LO_DISCOUNT1' not found!\n");
    return NULL;
  }
  const float* array_LINEORDER1_LINEORDER_LO_DISCOUNT1 =
      getArrayFromColumn_float(col_LINEORDER1_LINEORDER_LO_DISCOUNT1);
  C_Column* col_LINEORDER1_LINEORDER_LO_REVENUE1 =
      getColumnById(table_LINEORDER1, 12);
  if (!col_LINEORDER1_LINEORDER_LO_REVENUE1) {
    printf("Column 'col_LINEORDER1_LINEORDER_LO_REVENUE1' not found!\n");
    return NULL;
  }
  const float* array_LINEORDER1_LINEORDER_LO_REVENUE1 =
      getArrayFromColumn_float(col_LINEORDER1_LINEORDER_LO_REVENUE1);

  size_t num_elements_tuple_id_LINEORDER1 = getNumberOfRows(table_LINEORDER1);

  cl_int _err = CL_SUCCESS;
  /* allocate device memory */
  cl_mem cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1 = clCreateBuffer(
      context, CL_MEM_READ_ONLY,
      sizeof(int32_t) * num_elements_tuple_id_LINEORDER1, NULL, &_err);
  assert(_err == CL_SUCCESS);
  cl_mem cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1 = clCreateBuffer(
      context, CL_MEM_READ_ONLY,
      sizeof(float) * num_elements_tuple_id_LINEORDER1, NULL, &_err);
  assert(_err == CL_SUCCESS);
  cl_mem cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1 = clCreateBuffer(
      context, CL_MEM_READ_ONLY,
      sizeof(float) * num_elements_tuple_id_LINEORDER1, NULL, &_err);
  assert(_err == CL_SUCCESS);

  cl_mem cl_output_mem_flags = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      sizeof(char) * num_elements_tuple_id_LINEORDER1, NULL, &_err);
  assert(_err == CL_SUCCESS);

  cl_mem cl_output_prefix_sum = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      sizeof(TID) * num_elements_tuple_id_LINEORDER1, NULL, &_err);
  assert(_err == CL_SUCCESS);

  //    {
  //    size_t s = 1;
  //    s=s*1024*1024*1024;
  //    int* tmp = (int*) malloc(s);
  //    memset(tmp, 0, s);
  //    free(tmp);
  //    }

  // Write our data set into the input array in device memory
  //#define USE_NEW_OPENCL_TRANSFER

  //#ifndef USE_NEW_OPENCL_TRANSFER
  //    _err = clEnqueueWriteBuffer(queue,
  //            cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1, CL_TRUE, 0,
  //            sizeof (uint32_t) * num_elements_tuple_id_LINEORDER1,
  //            array_LINEORDER1_LINEORDER_LO_QUANTITY1,
  //            0, NULL, NULL);

  //    _err = clEnqueueWriteBuffer(queue,
  //            cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1, CL_TRUE, 0,
  //            sizeof (float) * num_elements_tuple_id_LINEORDER1,
  //            array_LINEORDER1_LINEORDER_LO_DISCOUNT1,
  //            0, NULL, NULL);

  //    _err = clEnqueueWriteBuffer(queue,
  //            cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1, CL_TRUE, 0,
  //            sizeof (float) * num_elements_tuple_id_LINEORDER1,
  //            array_LINEORDER1_LINEORDER_LO_REVENUE1,
  //            0, NULL, NULL);
  //#else
  cl_int blocking_write = 0;
  cl_event last_data_transfer_event;
  CL_CHECK(
      oclCopyHostToDevice(array_LINEORDER1_LINEORDER_LO_QUANTITY1,
                          sizeof(uint32_t) * num_elements_tuple_id_LINEORDER1,
                          cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1,
                          context, queue, blocking_write, NULL));
  CL_CHECK(oclCopyHostToDevice(array_LINEORDER1_LINEORDER_LO_DISCOUNT1,
                               sizeof(float) * num_elements_tuple_id_LINEORDER1,
                               cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1,
                               context, queue, blocking_write, NULL));
  CL_CHECK(oclCopyHostToDevice(array_LINEORDER1_LINEORDER_LO_REVENUE1,
                               sizeof(float) * num_elements_tuple_id_LINEORDER1,
                               cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1,
                               context, queue, blocking_write,
                               &last_data_transfer_event));
  //#endif

  // Wait for the command queue to get serviced before reading back results
  //    clFinish(queue);

  /* first phase: pass over data and compute flag array */
  cl_kernel select_hashprobe_kernel;
  select_hashprobe_kernel =
      clCreateKernel(program, "select_and_hashprobe_kernel", &_err);
  CL_CHECK(clSetKernelArg(select_hashprobe_kernel, 0,
                          sizeof(num_elements_tuple_id_LINEORDER1),
                          &num_elements_tuple_id_LINEORDER1));
  CL_CHECK(
      clSetKernelArg(select_hashprobe_kernel, 1,
                     sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1),
                     &cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(
      clSetKernelArg(select_hashprobe_kernel, 2,
                     sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1),
                     &cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(clSetKernelArg(select_hashprobe_kernel, 3,
                          sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1),
                          &cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1));
  CL_CHECK(clSetKernelArg(select_hashprobe_kernel, 4,
                          sizeof(cl_output_mem_flags), &cl_output_mem_flags));

  {
    cl_event kernel_completion;
    size_t global_work_size[1] = {num_elements_tuple_id_LINEORDER1};
    CL_CHECK(clEnqueueNDRangeKernel(
        queue, select_hashprobe_kernel, 1, NULL, global_work_size, NULL, 1,
        &last_data_transfer_event, &kernel_completion));
    CL_CHECK(clWaitForEvents(1, &kernel_completion));
    CL_CHECK(clReleaseEvent(kernel_completion));
    //        CL_CHECK(clReleaseEvent(last_data_transfer_event));
  }

  /* prefix sum */
  size_t result_size = 0;
  result_size =
      ocl_prefix_sum(queue, program, cl_output_mem_flags, cl_output_prefix_sum,
                     num_elements_tuple_id_LINEORDER1);

  //    tempory_prefix_sum_array[0] = 0;
  //    for (size_t i = 1; i < num_elements_tuple_id_LINEORDER1; i++) {
  //        tempory_prefix_sum_array[i] = tempory_prefix_sum_array[i - 1] +
  //        tempory_flag_array[i - 1];
  //    }
  //    result_size = tempory_prefix_sum_array[num_elements_tuple_id_LINEORDER1
  //    - 1] + tempory_flag_array[num_elements_tuple_id_LINEORDER1 - 1];
  std::cout << "Result size: " << result_size << std::endl;

  /* second phase: pass over data and write result */
  int32_t* result_array_LINEORDER_LO_QUANTITY_1 =
      (int32_t*)realloc(NULL, result_size * sizeof(int32_t));
  float* result_array_LINEORDER_LO_DISCOUNT_1 =
      (float*)realloc(NULL, result_size * sizeof(float));
  float* result_array_LINEORDER_LO_REVENUE_1 =
      (float*)realloc(NULL, result_size * sizeof(float));

  cl_mem cl_output_mem_LINEORDER1_LINEORDER_LO_QUANTITY1 = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, sizeof(uint32_t) * result_size, NULL, &_err);
  assert(_err == CL_SUCCESS);
  cl_mem cl_output_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1 = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, sizeof(float) * result_size, NULL, &_err);
  assert(_err == CL_SUCCESS);
  cl_mem cl_output_mem_LINEORDER1_LINEORDER_LO_REVENUE1 = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, sizeof(float) * result_size, NULL, &_err);
  assert(_err == CL_SUCCESS);

  cl_kernel hashprobe_aggregate_and_project_kernel;
  hashprobe_aggregate_and_project_kernel =
      clCreateKernel(program, "hashprobe_aggregate_and_project_kernel", &_err);
  CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 0,
                          sizeof(num_elements_tuple_id_LINEORDER1),
                          &num_elements_tuple_id_LINEORDER1));
  CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 1,
                          sizeof(cl_output_mem_flags), &cl_output_mem_flags));
  CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 2,
                          sizeof(cl_output_prefix_sum), &cl_output_prefix_sum));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 3,
                     sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1),
                     &cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 4,
                     sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1),
                     &cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 5,
                          sizeof(cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1),
                          &cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 6,
                     sizeof(cl_output_mem_LINEORDER1_LINEORDER_LO_QUANTITY1),
                     &cl_output_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 7,
                     sizeof(cl_output_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1),
                     &cl_output_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(
      clSetKernelArg(hashprobe_aggregate_and_project_kernel, 8,
                     sizeof(cl_output_mem_LINEORDER1_LINEORDER_LO_REVENUE1),
                     &cl_output_mem_LINEORDER1_LINEORDER_LO_REVENUE1));

  {
    cl_event kernel_completion;
    size_t global_work_size[1] = {num_elements_tuple_id_LINEORDER1};
    //        size_t global_work_size[1] = {4};
    CL_CHECK(clEnqueueNDRangeKernel(
        queue, hashprobe_aggregate_and_project_kernel, 1, NULL,
        global_work_size, NULL, 0, NULL, &kernel_completion));
    CL_CHECK(clWaitForEvents(1, &kernel_completion));
    CL_CHECK(clReleaseEvent(kernel_completion));
  }

  // Wait for the command queue to get serviced before reading back results
  clFinish(queue);

  // Read the results from the device
  _err = clEnqueueReadBuffer(
      queue, cl_output_mem_LINEORDER1_LINEORDER_LO_QUANTITY1, CL_TRUE, 0,
      sizeof(uint32_t) * result_size, result_array_LINEORDER_LO_QUANTITY_1, 0,
      NULL, NULL);
  assert(_err == CL_SUCCESS);

  _err = clEnqueueReadBuffer(
      queue, cl_output_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1, CL_TRUE, 0,
      sizeof(float) * result_size, result_array_LINEORDER_LO_DISCOUNT_1, 0,
      NULL, NULL);
  assert(_err == CL_SUCCESS);

  _err =
      clEnqueueReadBuffer(queue, cl_output_mem_LINEORDER1_LINEORDER_LO_REVENUE1,
                          CL_TRUE, 0, sizeof(float) * result_size,
                          result_array_LINEORDER_LO_REVENUE_1, 0, NULL, NULL);
  assert(_err == CL_SUCCESS);

  clFinish(queue);

  C_Column* result_columns[] = {
      createResultArray_int32_t("LINEORDER.LO_QUANTITY.1",
                                result_array_LINEORDER_LO_QUANTITY_1,
                                result_size),
      createResultArray_float("LINEORDER.LO_DISCOUNT.1",
                              result_array_LINEORDER_LO_DISCOUNT_1,
                              result_size),
      createResultArray_float("LINEORDER.LO_REVENUE.1",
                              result_array_LINEORDER_LO_REVENUE_1,
                              result_size)};
  C_Table* result_table =
      createTableFromColumns("LINEORDER", result_columns, 3);

  //    free(tempory_flag_array);
  //    free(tempory_prefix_sum_array);

  CL_CHECK(clReleaseMemObject(cl_input_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(clReleaseMemObject(cl_input_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(clReleaseMemObject(cl_input_mem_LINEORDER1_LINEORDER_LO_REVENUE1));
  CL_CHECK(clReleaseMemObject(cl_output_mem_flags));
  CL_CHECK(clReleaseMemObject(cl_output_prefix_sum));
  CL_CHECK(clReleaseMemObject(cl_output_mem_LINEORDER1_LINEORDER_LO_QUANTITY1));
  CL_CHECK(clReleaseMemObject(cl_output_mem_LINEORDER1_LINEORDER_LO_DISCOUNT1));
  CL_CHECK(clReleaseMemObject(cl_output_mem_LINEORDER1_LINEORDER_LO_REVENUE1));

  CL_CHECK(clReleaseKernel(select_hashprobe_kernel));
  CL_CHECK(clReleaseKernel(hashprobe_aggregate_and_project_kernel));

  return result_table;
}

C_Table* selection_compiled_query(C_Table** c_tables) {
  C_Table* table_LINEORDER1 = c_tables[0];
  assert(table_LINEORDER1 != NULL);
  C_Column* col_LINEORDER1_LINEORDER_LO_QUANTITY1 =
      getColumnById(table_LINEORDER1, 8);
  if (!col_LINEORDER1_LINEORDER_LO_QUANTITY1) {
    printf("Column 'col_LINEORDER1_LINEORDER_LO_QUANTITY1' not found!\n");
    return NULL;
  }
  const int32_t* array_LINEORDER1_LINEORDER_LO_QUANTITY1 =
      getArrayFromColumn_int32_t(col_LINEORDER1_LINEORDER_LO_QUANTITY1);
  C_Column* col_LINEORDER1_LINEORDER_LO_DISCOUNT1 =
      getColumnById(table_LINEORDER1, 11);
  if (!col_LINEORDER1_LINEORDER_LO_DISCOUNT1) {
    printf("Column 'col_LINEORDER1_LINEORDER_LO_DISCOUNT1' not found!\n");
    return NULL;
  }
  const float* array_LINEORDER1_LINEORDER_LO_DISCOUNT1 =
      getArrayFromColumn_float(col_LINEORDER1_LINEORDER_LO_DISCOUNT1);
  C_Column* col_LINEORDER1_LINEORDER_LO_REVENUE1 =
      getColumnById(table_LINEORDER1, 12);
  if (!col_LINEORDER1_LINEORDER_LO_REVENUE1) {
    printf("Column 'col_LINEORDER1_LINEORDER_LO_REVENUE1' not found!\n");
    return NULL;
  }
  const float* array_LINEORDER1_LINEORDER_LO_REVENUE1 =
      getArrayFromColumn_float(col_LINEORDER1_LINEORDER_LO_REVENUE1);

  size_t current_result_size = 0;
  size_t allocated_result_elements = 10000;
  int32_t* result_array_LINEORDER_LO_QUANTITY_1 =
      (int32_t*)realloc(NULL, allocated_result_elements * sizeof(int32_t));
  float* result_array_LINEORDER_LO_DISCOUNT_1 =
      (float*)realloc(NULL, allocated_result_elements * sizeof(float));
  float* result_array_LINEORDER_LO_REVENUE_1 =
      (float*)realloc(NULL, allocated_result_elements * sizeof(float));

  size_t num_elements_tuple_id_LINEORDER1 = getNumberOfRows(table_LINEORDER1);
  size_t tuple_id_LINEORDER1;
  for (tuple_id_LINEORDER1 = 0;
       tuple_id_LINEORDER1 < num_elements_tuple_id_LINEORDER1;
       ++tuple_id_LINEORDER1) {
    if (((array_LINEORDER1_LINEORDER_LO_REVENUE1[tuple_id_LINEORDER1] >
          4900000.0f) &&
         (array_LINEORDER1_LINEORDER_LO_DISCOUNT1[tuple_id_LINEORDER1] >=
          1.0f) &&
         (array_LINEORDER1_LINEORDER_LO_DISCOUNT1[tuple_id_LINEORDER1] <=
          3.0f) &&
         (array_LINEORDER1_LINEORDER_LO_QUANTITY1[tuple_id_LINEORDER1] < 25))) {
      if (current_result_size >= allocated_result_elements) {
        allocated_result_elements *= 1.4;
        result_array_LINEORDER_LO_QUANTITY_1 =
            (int32_t*)realloc(result_array_LINEORDER_LO_QUANTITY_1,
                              allocated_result_elements * sizeof(int32_t));
        result_array_LINEORDER_LO_DISCOUNT_1 =
            (float*)realloc(result_array_LINEORDER_LO_DISCOUNT_1,
                            allocated_result_elements * sizeof(float));
        result_array_LINEORDER_LO_REVENUE_1 =
            (float*)realloc(result_array_LINEORDER_LO_REVENUE_1,
                            allocated_result_elements * sizeof(float));
      }
      result_array_LINEORDER_LO_QUANTITY_1[current_result_size] =
          array_LINEORDER1_LINEORDER_LO_QUANTITY1[tuple_id_LINEORDER1];
      result_array_LINEORDER_LO_DISCOUNT_1[current_result_size] =
          array_LINEORDER1_LINEORDER_LO_DISCOUNT1[tuple_id_LINEORDER1];
      result_array_LINEORDER_LO_REVENUE_1[current_result_size] =
          array_LINEORDER1_LINEORDER_LO_REVENUE1[tuple_id_LINEORDER1];
      ++current_result_size;
    }
  }
  C_Column* result_columns[] = {
      createResultArray_int32_t("LINEORDER.LO_QUANTITY.1",
                                result_array_LINEORDER_LO_QUANTITY_1,
                                current_result_size),
      createResultArray_float("LINEORDER.LO_DISCOUNT.1",
                              result_array_LINEORDER_LO_DISCOUNT_1,
                              current_result_size),
      createResultArray_float("LINEORDER.LO_REVENUE.1",
                              result_array_LINEORDER_LO_REVENUE_1,
                              current_result_size)};
  C_Table* result_table =
      createTableFromColumns("LINEORDER", result_columns, 3);

  return result_table;
}

namespace CoGaDB {

TablePtr launch_query(SharedCLibPipelineQueryPtr query, ScanParam param) {
  std::vector<C_Table*> c_tables(param.size());

  for (unsigned int i = 0; i < param.size(); ++i) {
    c_tables[i] = getCTableFromTablePtr(param[i].getTable());
  }

  C_Table* c_table = (*query)(c_tables.data());

  if (c_table == NULL) {
    return TablePtr();
  }

  TablePtr table = getTablePtrFromCTable(c_table);
  releaseTable(c_table);

  for (unsigned int i = 0; i < param.size(); ++i) {
    releaseTable(c_tables[i]);
  }

  return table;
}

void launch_selection_query(ClientPtr client) {
  ScanParam param;
  AttributeReferencePtr lo_quantity = boost::make_shared<AttributeReference>(
      getTablebyName("LINEORDER"), "LINEORDER.LO_QUANTITY");
  AttributeReferencePtr lo_discount = boost::make_shared<AttributeReference>(
      getTablebyName("LINEORDER"), "LINEORDER.LO_DISCOUNT");
  AttributeReferencePtr lo_revenue = boost::make_shared<AttributeReference>(
      getTablebyName("LINEORDER"), "LINEORDER.LO_REVENUE");

  param.push_back(*lo_quantity);
  param.push_back(*lo_discount);
  param.push_back(*lo_revenue);

  Timestamp begin = getTimestamp();
  //        TablePtr result = launch_query(&selection_compiled_query, param);
  //        TablePtr result = launch_query(&ocl_cpu_selection_compiled_query,
  //        param);
  TablePtr result = launch_query(&ocl_gpu_selection_compiled_query, param);
  Timestamp end = getTimestamp();
  if (!result) {
    COGADB_FATAL_ERROR("", "");
  }
  printResult(result, client, double(end - begin) / (1000 * 1000));
}
}

int main(int argc, char** argv) {
  //    device_type = CL_DEVICE_TYPE_CPU;
  device_type = CL_DEVICE_TYPE_GPU;

  if (init_ocl()) {
    std::cout << "Failed to set up OpenCL!" << std::endl;
    return -1;
  }

  CoGaDB::ClientPtr client(new CoGaDB::LocalClient());
  if (!CoGaDB::loadReferenceDatabaseStarSchemaScaleFactor1(client)) {
    COGADB_FATAL_ERROR("Failed to load database!", "");
  }

  for (size_t i = 0; i < 1; ++i) CoGaDB::launch_selection_query(client);

  cleanup_ocl();

  return 0;
}
