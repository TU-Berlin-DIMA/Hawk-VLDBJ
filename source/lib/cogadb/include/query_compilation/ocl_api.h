/*
 * File:   ocl_api.h
 * Author: sebastian
 *
 * Created on 22. Februar 2016, 08:41
 */

#ifndef OCL_API_H
#define OCL_API_H

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct OCL_Execution_Context;
typedef struct OCL_Execution_Context OCL_Execution_Context;

cl_context ocl_getContext(OCL_Execution_Context*);
cl_command_queue ocl_getComputeCommandQueue(OCL_Execution_Context*);
cl_command_queue ocl_getTransferHostToDeviceCommandQueue(
    OCL_Execution_Context*);
cl_command_queue ocl_getTransferDeviceToHostCommandQueue(
    OCL_Execution_Context*);
cl_program ocl_getProgram(OCL_Execution_Context*);
cl_device_id ocl_getDeviceID(OCL_Execution_Context*);

void ocl_enter_critical_section_copy_host_to_device();
void ocl_leave_critical_section_copy_host_to_device();

void ocl_enter_critical_section_schedule_computation();
void ocl_leave_critical_section_schedule_computation();

void ocl_start_timer();
void ocl_stop_timer(const char* string);

void ocl_print_device(cl_device_id device_id);

size_t ocl_prefix_sum(cl_command_queue queue, cl_program program,
                      cl_mem cl_output_mem_flags, cl_mem cl_output_prefix_sum,
                      size_t num_elements);

uint64_t ocl_reduce_uint64_t(cl_command_queue queue, cl_mem buffer,
                             uint64_t num_elements, unsigned char aggr_method);

double ocl_reduce_double(cl_command_queue queue, cl_mem buffer,
                         uint64_t num_elements, unsigned char aggr_method);

double ocl_reduce_float(cl_command_queue queue, cl_mem buffer,
                        uint64_t num_elements, unsigned char aggr_method);

void ocl_sort_by_key_uint64_t(cl_command_queue queue, uint64_t num_elements,
                              cl_mem keys, cl_mem vals);

uint64_t ocl_reduce_by_key_float(cl_command_queue queue, uint64_t num_elements,
                                 cl_mem keys_in, cl_mem vals_in,
                                 cl_mem keys_out, cl_mem vals_out,
                                 bool take_any_value);

uint64_t ocl_reduce_by_key_double(cl_command_queue queue, uint64_t num_elements,
                                  cl_mem keys_in, cl_mem vals_in,
                                  cl_mem keys_out, cl_mem vals_out,
                                  bool take_any_value);

uint64_t ocl_reduce_by_key_uint64_t(cl_command_queue queue,
                                    uint64_t num_elements, cl_mem keys_in,
                                    cl_mem vals_in, cl_mem keys_out,
                                    cl_mem vals_out, bool take_any_value);

uint64_t ocl_reduce_by_key_int32_t(cl_command_queue queue,
                                   uint64_t num_elements, cl_mem keys_in,
                                   cl_mem vals_in, cl_mem keys_out,
                                   cl_mem vals_out, bool take_any_value);

uint64_t ocl_reduce_by_key_int(cl_command_queue queue, uint64_t num_elements,
                               cl_mem keys_in, cl_mem vals_in, cl_mem keys_out,
                               cl_mem vals_out, bool take_any_value);

void ocl_gather_float(cl_command_queue queue, uint64_t num_elements,
                      cl_mem source_positions, cl_mem values,
                      cl_mem values_out);

void ocl_gather_double(cl_command_queue queue, uint64_t num_elements,
                       cl_mem source_positions, cl_mem values,
                       cl_mem values_out);

void ocl_gather_uint64_t(cl_command_queue queue, uint64_t num_elements,
                         cl_mem source_positions, cl_mem values,
                         cl_mem values_out);

void ocl_gather_int32_t(cl_command_queue queue, uint64_t num_elements,
                        cl_mem source_positions, cl_mem values,
                        cl_mem values_out);

void ocl_gather_uint32_t(cl_command_queue queue, uint64_t num_elements,
                         cl_mem source_positions, cl_mem values,
                         cl_mem values_out);

void ocl_gather_char(cl_command_queue queue, uint64_t num_elements,
                     cl_mem source_positions, cl_mem values, cl_mem values_out);

void ocl_print_float(cl_command_queue queue, int num_elements, cl_mem mem);

cl_int oclCopyHostToDevice(const void* data, size_t num_bytes,
                           cl_mem input_buffer, cl_context context,
                           cl_command_queue command_queue,
                           cl_int blocking_write, cl_event* event);

cl_int oclCopyDeviceToHost(cl_mem output_buffer, size_t num_bytes, void* data,
                           cl_context context, cl_command_queue command_queue,
                           cl_int blocking_write, cl_event* event);

cl_mem oclCachedCopyHostToDevice(const void* data, size_t num_bytes,
                                 cl_device_id dev_id, cl_int* error_code);

void oclFillBuffer_char(cl_command_queue queue, cl_mem mem, char init_value,
                        uint64_t offset, uint64_t count);

void oclFillBuffer_uint64_t(cl_command_queue queue, cl_mem mem,
                            uint64_t init_value, uint64_t offset,
                            uint64_t count);

void oclFillBuffer_uint32_t(cl_command_queue queue, cl_mem mem,
                            uint32_t init_value, uint64_t offset,
                            uint64_t count);

void oclFillBuffer_float(cl_command_queue queue, cl_mem mem, float init_value,
                         uint64_t offset, uint64_t count);

void oclFillBuffer_double(cl_command_queue queue, cl_mem mem, double init_value,
                          uint64_t offset, uint64_t count);

#ifdef __cplusplus
#define CL_ERROR(_expr, _err)                                           \
  fprintf(stderr, "OpenCL Error: '%s' returned %d! Line: %d\n", #_expr, \
          static_cast<int>(_err), __LINE__)
#else
#define CL_ERROR(_expr, _err)                                           \
  fprintf(stderr, "OpenCL Error: '%s' returned %d! Line: %d\n", #_expr, \
          (int)_err, __LINE__)
#endif

#define CL_CHECK(_expr)       \
  do {                        \
    cl_int _err = _expr;      \
    if (_err == CL_SUCCESS) { \
      break;                  \
    }                         \
    CL_ERROR(_expr, _err);    \
    abort();                  \
  } while (0)

typedef struct {
  uint64_t* hash_table;
  uint64_t hash_table_size;
  uint64_t* overflow;
  uint64_t overflow_size;
} OCLCuckooHT;

void freeOCLCuckooHT(void* ptr);

typedef struct {
  uint32_t* hash_table;
  uint32_t hash_table_size;
} OCLLinearProbingHT;

void freeOCLLinearProbingHT(void* ptr);

typedef struct {
  uint64_t* hash_table;
  uint64_t hash_table_size;  // includes 101 elements stash at the end
} OCLCuckoo2HashesHT;

void freeOCLCuckoo2HashesHT(void* ptr);

#ifdef __cplusplus
}
#endif

#endif /* OCL_API_H */
