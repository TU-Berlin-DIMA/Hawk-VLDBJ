
//#include <query_compilation/code_generators/multi_stage_code_generator.hpp>

#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/execution_strategy/ocl.hpp>

#include <iomanip>
#include <list>
#include <set>
#include <sstream>

#include <core/attribute_reference.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>

#include <util/functions.hpp>
#include <util/opencl_runtime.hpp>
#include <util/time_measurement.hpp>

//#define GENERATE_PROFILING_CODE

#define LOCK_OPENCL_CALLS

namespace CoGaDB {

std::string removePointer(std::string type) {
  auto find = type.find_first_of("*");

  if (find == std::string::npos) {
    return type;
  } else {
    if (find != type.find_last_of("*")) {
      COGADB_FATAL_ERROR("We don't support '" << type << "**'", "");
    }

    return type.erase(find, 1);
  }
}

bool isPointerType(const std::string& type) {
  return type.find("*") != std::string::npos;
}

const std::string getCodeKernelSignature(
    const std::string& name, const std::string& extra_vars,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars) {
  std::stringstream str;
  str << "__kernel void " << name << "( " << extra_vars;
  for (const auto& var : input_vars) {
    str << ", ";

    if (isPointerType(var.second)) {
      str << "__global const ";
    }

    str << var.second << " ";

    if (isPointerType(var.second)) {
      str << " const ";
    }

    str << var.first;
  }

  for (const auto& var : output_vars) {
    str << ", __global ";
    str << var.second;
    str << " ";
    str << var.first;
  }

  str << ")";

  return str.str();
}

const std::string getCodeFilterKernelSignature(
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars) {
  return getCodeKernelSignature("select_and_hashprobe_kernel",
                                "uint64_t num_elements, __global char* flags",
                                input_vars, output_vars);
}

const std::string getCodeProjectionKernelSignature(
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars) {
  std::stringstream str;
  return getCodeKernelSignature(
      "hashprobe_aggregate_and_project_kernel",
      "uint64_t num_elements, __global const char* flags, "
      "__global const uint64_t* write_positions",
      input_vars, output_vars);
}

const std::string getCodeAggregationKernelSignature(
    const std::map<std::string, std::string>& input_vars,
    std::map<std::string, std::string> output_vars,
    const std::map<std::string, std::string>& aggr_vars) {
  for (const auto& var : aggr_vars) {
    auto find = output_vars.find(var.first);
    if (find != output_vars.end()) {
      find->second = removePointer(find->second) + "*";
    }
  }

  std::stringstream str;
  return getCodeKernelSignature("aggregation_kernel", "uint64_t num_elements",
                                input_vars, output_vars);
}

const std::string getCLBufferInputVarName(const std::string& varname) {
  std::stringstream str;
  str << "cl_input_mem_" << varname;
  return str.str();
}

const std::string getCLBufferInputVarName(const AttributeReference& attr) {
  return getCLBufferInputVarName(getVarName(attr));
}

const std::string getCLBufferResultVarName(const std::string& varname) {
  std::stringstream str;
  str << "cl_result_mem_" << varname;
  return str.str();
}

const std::string getCLBufferResultVarName(const AttributeReference& attr) {
  return getCLBufferResultVarName(getVarName(attr));
}

const std::string getKernelHeadersAndTypes() {
  std::stringstream kernel;
  kernel << "typedef unsigned char		uint8_t;" << std::endl;
  kernel << "typedef unsigned short int	uint16_t;" << std::endl;
  kernel << "typedef unsigned int		uint32_t;" << std::endl;
  kernel << "typedef unsigned long int	uint64_t;" << std::endl;
  kernel << "typedef unsigned long int	TID;" << std::endl;

  kernel << "typedef char		int8_t;" << std::endl;
  kernel << "typedef short int	int16_t;" << std::endl;
  kernel << "typedef int		int32_t;" << std::endl;
  kernel << "typedef long int	int64_t;" << std::endl;

  kernel << "#define C_MIN(a, b) (a = (a < b ? a : b))" << std::endl
         << "#define C_MIN_uint64t(a, b) C_MIN(a, b)" << std::endl
         << "#define C_MIN_double(a, b) C_MIN(a, b)" << std::endl;

  kernel << "#define C_MAX(a, b) (a = (a > b ? a : b))" << std::endl
         << "#define C_MAX_uint64_t(a, b) C_MAX(a, b)" << std::endl
         << "#define C_MAX_double(a, b) C_MAX(a, b)" << std::endl;

  kernel << "#define C_SUM(a, b) (a += b)" << std::endl
         << "#define C_SUM_uint64_t(a, b) C_SUM(a, b)" << std::endl
         << "#define C_SUM_double(a, b) C_SUM(a, b)" << std::endl;

  kernel << "#ifndef NULL" << std::endl
         << "#define NULL 0" << std::endl
         << "#endif" << std::endl;

  // enable double precision for OpenCL 1.1
  kernel << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl;

  return kernel.str();
}

const std::pair<std::string, std::string> getCodeOCLKernelMemoryTraversal(
    const std::string& loop_variable_name, MemoryAccessPattern mem_access) {
  std::stringstream begin_loop;
  std::stringstream end_loop;
  begin_loop << "uint32_t result_increment=1;" << std::endl;
  if (mem_access == COALESCED_MEMORY_ACCESS) {
    begin_loop << "uint64_t " << loop_variable_name << " = get_global_id(0);"
               << std::endl;
    begin_loop << "const uint64_t number_of_threads = get_global_size(0);"
               << std::endl;
    begin_loop << "while(" << loop_variable_name << " < num_elements){"
               << std::endl;
  } else if (mem_access == BLOCK_MEMORY_ACCESS) {
    begin_loop << "uint64_t block_size = (num_elements + get_global_size(0)"
               << " - 1) / get_global_size(0);" << std::endl;
    begin_loop << "uint64_t " << loop_variable_name << " = block_size "
               << " * get_global_id(0);" << std::endl;
    begin_loop << "uint64_t tmp =  " << loop_variable_name << " + block_size;"
               << std::endl;
    begin_loop << "uint64_t end_index;" << std::endl;
    begin_loop << "if (num_elements > tmp) {" << std::endl;
    begin_loop << "    end_index = tmp;" << std::endl;
    begin_loop << "} else {" << std::endl;
    begin_loop << "    end_index = num_elements;" << std::endl;
    begin_loop << "}" << std::endl;
    begin_loop << "for(;" << loop_variable_name << "<end_index;++"
               << loop_variable_name << "){" << std::endl;

  } else {
    COGADB_FATAL_ERROR("", "");
  }

  if (mem_access == COALESCED_MEMORY_ACCESS) {
    end_loop << loop_variable_name << " += number_of_threads;" << std::endl;
  } else if (mem_access == BLOCK_MEMORY_ACCESS) {
  }
  end_loop << " }" << std::endl << std::endl;

  return std::make_pair(begin_loop.str(), end_loop.str());
}

const std::string getCodeFilterKernel(
    ExecutionStrategy::GeneratedKernelPtr filter_kernel_,
    const std::string& loop_variable_name, size_t global_worksize,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars,
    MemoryAccessPattern mem_access_) {
  std::stringstream kernel;
  kernel << getCodeFilterKernelSignature(input_vars, output_vars) << "{"
         << std::endl;

  std::pair<std::string, std::string> loop_code =
      getCodeOCLKernelMemoryTraversal(loop_variable_name, mem_access_);

  kernel << loop_code.first;

  for (size_t i = 0; i < filter_kernel_->kernel_code_blocks.size(); ++i) {
    std::list<std::string>::const_iterator cit;
    for (cit = filter_kernel_->kernel_code_blocks[i]->upper_block.begin();
         cit != filter_kernel_->kernel_code_blocks[i]->upper_block.end();
         ++cit) {
      kernel << *cit << std::endl;
    }
    kernel << "flags[" << loop_variable_name << "]=result_increment;"
           << std::endl;
    for (cit = filter_kernel_->kernel_code_blocks[i]->lower_block.begin();
         cit != filter_kernel_->kernel_code_blocks[i]->lower_block.end();
         ++cit) {
      kernel << *cit << std::endl;
    }
    //            kernel << "else { flags[" <<
    //            first_loop_->getLoopVariableName() << "]=0; }";
  }

  kernel << loop_code.second;

  kernel << "}" << std::endl << std::endl;
  return kernel.str();
}

const std::string getCodeProjectionKernel(
    ExecutionStrategy::GeneratedKernelPtr projection_kernel,
    const std::string& loop_variable_name, size_t global_worksize,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars,
    MemoryAccessPattern mem_access) {
  std::stringstream kernel;

  kernel << getCodeProjectionKernelSignature(input_vars, output_vars) << "{"
         << std::endl;
  std::pair<std::string, std::string> loop_code =
      getCodeOCLKernelMemoryTraversal(loop_variable_name, mem_access);

  kernel << loop_code.first;

  kernel << "if(flags[" << loop_variable_name << "]){" << std::endl;
  for (size_t i = 0; i < projection_kernel->kernel_code_blocks.size(); ++i) {
    std::list<std::string>::const_iterator cit;

    kernel << "uint64_t write_pos = write_positions[" << loop_variable_name
           << "];" << std::endl;

    for (cit = projection_kernel->kernel_code_blocks[i]->upper_block.begin();
         cit != projection_kernel->kernel_code_blocks[i]->upper_block.end();
         ++cit) {
      kernel << *cit << std::endl;
    }

    for (cit = projection_kernel->kernel_code_blocks[i]->lower_block.begin();
         cit != projection_kernel->kernel_code_blocks[i]->lower_block.end();
         ++cit) {
      kernel << *cit << std::endl;
    }
  }
  kernel << "}" << std::endl << std::endl;

  kernel << loop_code.second;

  kernel << "}" << std::endl << std::endl;

  return kernel.str();
}

const std::string getCodeSerialSinglePassKernel(
    ExecutionStrategy::GeneratedKernelPtr kernel_,
    const std::string& loop_variable_name,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars,
    MemoryAccessPattern mem_access) {
  std::stringstream kernel;

  kernel << getCodeKernelSignature(
                "serial_single_pass_kernel",
                "uint64_t num_elements, __global uint64_t* result_size",
                input_vars, output_vars)
         << "{" << std::endl;
  std::pair<std::string, std::string> loop_code =
      getCodeOCLKernelMemoryTraversal(loop_variable_name, mem_access);

  kernel << "uint64_t write_pos = 0;" << std::endl;
  kernel << loop_code.first;

  for (const auto& code_block : kernel_->kernel_code_blocks) {
    for (const auto& upper : code_block->upper_block) {
      kernel << upper << std::endl;
    }

    kernel << "write_pos+=result_increment;";
    for (const auto& lower : code_block->lower_block) {
      kernel << lower << std::endl;
    }
  }

  kernel << loop_code.second;

  kernel << "result_size[0]=write_pos;" << std::endl;

  kernel << "}" << std::endl << std::endl;

  return kernel.str();
}

const std::string getCodeParallelGlobalAtomicSinglePassKernel(
    ExecutionStrategy::GeneratedKernelPtr kernel_,
    const std::string& loop_variable_name,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars,
    MemoryAccessPattern mem_access) {
  std::stringstream kernel;

  kernel << getCodeKernelSignature("serial_single_pass_kernel",
                                   "uint64_t num_elements, "
                                   "__global uint32_t* global_write_pos",
                                   input_vars, output_vars)
         << "{" << std::endl;
  std::pair<std::string, std::string> loop_code =
      getCodeOCLKernelMemoryTraversal(loop_variable_name, mem_access);

  kernel << "uint32_t write_pos = 0;" << std::endl;
  kernel << loop_code.first;

  for (const auto& code_block : kernel_->kernel_code_blocks) {
    for (const auto& upper : code_block->upper_block) {
      kernel << upper << std::endl;
    }

    kernel << "write_pos = atomic_add(global_write_pos, result_increment);"
           << std::endl;
    for (const auto& materialize : code_block->materialize_result_block) {
      kernel << materialize << std::endl;
    }

    for (const auto& lower : code_block->lower_block) {
      kernel << lower << std::endl;
    }
  }

  kernel << loop_code.second;

  kernel << "}" << std::endl << std::endl;

  return kernel.str();
}

const std::string getCodeAggregationKernel(
    ExecutionStrategy::GeneratedKernelPtr aggr_kernel_,
    const std::string& loop_variable_name, uint64_t global_worksize,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars,
    const std::map<std::string, std::string>& aggr_vars,
    MemoryAccessPattern mem_access) {
  std::stringstream kernel;

  kernel << getCodeAggregationKernelSignature(input_vars, output_vars,
                                              aggr_vars)
         << "{" << std::endl;
  std::pair<std::string, std::string> loop_code =
      getCodeOCLKernelMemoryTraversal(loop_variable_name, mem_access);

  kernel << loop_code.first;

  for (const auto& code_block : aggr_kernel_->kernel_code_blocks) {
    for (const auto& upper : code_block->upper_block) {
      kernel << upper << std::endl;
    }

    for (const auto& lower : code_block->lower_block) {
      kernel << lower << std::endl;
    }
  }

  kernel << loop_code.second;

  kernel << "}" << std::endl << std::endl;

  return kernel.str();
}

const std::string getCodeCreateOCLDefaultKernelInputBuffers(
    const std::string& num_elements_for_loop, cl_device_type dev_type) {
  std::stringstream out;
  out << "char* tempory_flag_array = NULL;" << std::endl;
  out << "TID* tempory_prefix_sum_array = NULL;" << std::endl;
  //  out << "cl_int _err = 0;" << std::endl;

  if (dev_type == CL_DEVICE_TYPE_CPU) {
    out << "tempory_flag_array = (char*) realloc(NULL, "
        << num_elements_for_loop << " * sizeof (char));" << std::endl;
    out << "tempory_prefix_sum_array = (TID*) realloc(NULL, "
        << num_elements_for_loop << " * sizeof (TID));" << std::endl;

#ifdef GENERATE_PROFILING_CODE
    out << "ocl_start_timer();" << std::endl;
#endif

    out << "memset(tempory_flag_array, 0, " << num_elements_for_loop
        << " * sizeof (char));" << std::endl;

#ifdef GENERATE_PROFILING_CODE
    out << "ocl_stop_timer(\"Memset\");" << std::endl;
#endif

    out << "cl_mem cl_output_mem_flags "
        << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE | "
           "CL_MEM_USE_HOST_PTR,"
        << "sizeof (char)*" << num_elements_for_loop << ","
        << "tempory_flag_array, &_err);" << std::endl;
    out << "assert(_err == CL_SUCCESS);" << std::endl;

    out << "cl_mem cl_output_prefix_sum"
        << " = clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE | "
           "CL_MEM_USE_HOST_PTR,"
        << " sizeof (TID) * " << num_elements_for_loop << ","
        << "tempory_prefix_sum_array, &_err);" << std::endl;
    out << "assert(_err == CL_SUCCESS);" << std::endl;
  } else if (dev_type == CL_DEVICE_TYPE_GPU ||
             dev_type == CL_DEVICE_TYPE_ACCELERATOR) {
    out << "cl_mem cl_output_mem_flags "
        << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE,"
        << "sizeof (char)*" << num_elements_for_loop << ","
        << "NULL, &_err);" << std::endl;
    out << "assert(_err == CL_SUCCESS);" << std::endl;
    /* perform memset on flag array memory buffer */
    out << "oclFillBuffer_char("
        << "ocl_getTransferHostToDeviceCommandQueue(context), "
        << "cl_output_mem_flags, 0, 0, " << num_elements_for_loop << ");"
        << std::endl;
    out << "clFinish(ocl_getTransferHostToDeviceCommandQueue(context));"
        << std::endl;
    out << "assert(_err == CL_SUCCESS);" << std::endl;

    out << "cl_mem cl_output_prefix_sum"
        << " = clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE,"
        << " sizeof (TID) * " << num_elements_for_loop << ","
        << "NULL, &_err);" << std::endl;
    out << "assert(_err == CL_SUCCESS);" << std::endl;
  }

  return out.str();
}

const std::string getCodeCreateOCLInputBuffers(
    const std::string& num_elements_for_loop,
    const std::map<std::string, std::string>& variables,
    cl_device_type dev_type, cl_device_id dev_id, bool enable_caching) {
  std::stringstream out;

  cl_bool unified_host_memory = true;
  CL_CHECK(clGetDeviceInfo(dev_id, CL_DEVICE_HOST_UNIFIED_MEMORY,
                           sizeof(cl_bool), &unified_host_memory, NULL));

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  if (dev_type == CL_DEVICE_TYPE_CPU || unified_host_memory) {
    for (const auto& var : variables) {
      if (isPointerType(var.second)) {
        auto num_elements = var.first + "_length";

        out << "cl_mem " << getCLBufferInputVarName(var.first)
            << " = clCreateBuffer(ocl_getContext(context), CL_MEM_READ_ONLY | "
               "CL_MEM_USE_HOST_PTR, "
            << "sizeof (" << removePointer(var.second) << ") * " << var.first
            << "_length , "
            << "(void*) " << var.first << ", &_err);" << std::endl;
        out << "assert(_err == CL_SUCCESS);" << std::endl;
      }
    }
  } else if (dev_type == CL_DEVICE_TYPE_GPU ||
             dev_type == CL_DEVICE_TYPE_ACCELERATOR) {
    if (enable_caching) {
      /* copy data to device, use cached copy call */
      int blocking_write = 1;
      out << "ocl_enter_critical_section_copy_host_to_device();" << std::endl;
      /* first reserve required input memory */
      for (const auto& var : variables) {
        if (isPointerType(var.second)) {
          auto num_elements = var.first + "_length";
          out << "cl_mem " << getCLBufferInputVarName(var.first)
              << " = oclCachedCopyHostToDevice(" << var.first << ", "
              << "sizeof (" << removePointer(var.second) << ") * "
              << num_elements << ", "
              << "ocl_getDeviceID(context), &_err);" << std::endl;
          out << "assert(_err == CL_SUCCESS);" << std::endl;
          out << "assert(" << getCLBufferInputVarName(var.first) << "!=NULL);"
              << std::endl;
          /* increment reference count, so the mem_obj is not deleted when
           * clReleaseMemObject is called later */
          out << "_err=clRetainMemObject(" << getCLBufferInputVarName(var.first)
              << ");" << std::endl;
        }
      }
      out << "ocl_leave_critical_section_copy_host_to_device();" << std::endl;

      /* Wait for the command queue to get serviced before reading back results
       */
      out << "clFinish(ocl_getTransferHostToDeviceCommandQueue(context));"
          << std::endl;
      out << "assert(_err == CL_SUCCESS);" << std::endl;
    } else {
      /* copy data to device (Note: for now, we transfer synchron!) */
      int blocking_write = 1;
      out << "ocl_enter_critical_section_copy_host_to_device();" << std::endl;
      /* first reserve required input memory */
      for (const auto& var : variables) {
        if (isPointerType(var.second)) {
          auto num_elements = var.first + "_length";

          out << "cl_mem " << getCLBufferInputVarName(var.first)
              << " = clCreateBuffer(ocl_getContext(context), CL_MEM_READ_ONLY, "
              << "sizeof (" << removePointer(var.second) << ") * "
              << num_elements << ", "
              << "NULL, &_err);" << std::endl;
          out << "assert(_err == CL_SUCCESS);" << std::endl;

          out << "CL_CHECK(oclCopyHostToDevice(" << var.first << ", "
              << "sizeof (" << removePointer(var.second) << ") * "
              << num_elements << ", " << getCLBufferInputVarName(var.first)
              << ", "
              << "ocl_getContext(context), "
              << "ocl_getTransferHostToDeviceCommandQueue(context), "
              << blocking_write << ", NULL));" << std::endl;
        }
      }
      out << "ocl_leave_critical_section_copy_host_to_device();" << std::endl;

      /* Wait for the command queue to get serviced before reading back results
       */
      out << "clFinish(ocl_getTransferHostToDeviceCommandQueue(context));"
          << std::endl;
      out << "assert(_err == CL_SUCCESS);" << std::endl;
    }
  } else {
    COGADB_FATAL_ERROR("Unknown or Unsupported Device Type!", "");
  }
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Data Transfer Host to Device\");" << std::endl;
#endif

  return out.str();
}

const std::string getCodeCallFilterKernel(
    const std::string& num_elements_for_loop, size_t global_worksize,
    const std::map<std::string, std::string>& variables) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif
  out << "cl_kernel select_hashprobe_kernel=NULL;" << std::endl;
  out << "select_hashprobe_kernel = clCreateKernel(ocl_getProgram(context), "
         "\"select_and_hashprobe_kernel\", &_err);"
      << std::endl;
  out << "assert(_err == CL_SUCCESS);" << std::endl;
  out << "CL_CHECK(clSetKernelArg(select_hashprobe_kernel, 0, sizeof ("
      << num_elements_for_loop << "), &" << num_elements_for_loop << "));"
      << std::endl;
  out << "CL_CHECK(clSetKernelArg(select_hashprobe_kernel, 1, sizeof "
         "(cl_output_mem_flags), &cl_output_mem_flags));"
      << std::endl;
  unsigned int index = 2;
  for (const auto& var : variables) {
    auto var_name = isPointerType(var.second)
                        ? getCLBufferInputVarName(var.first)
                        : var.first;
    out << "CL_CHECK(clSetKernelArg(select_hashprobe_kernel, " << index++
        << ", sizeof (" << var_name << ") , &" << var_name << "));"
        << std::endl;
  }
  out << "{" << std::endl;
  out << "cl_event kernel_completion;" << std::endl;
  out << "size_t global_work_size[1] = {" << global_worksize << "};"
      << std::endl;
#ifdef LOCK_OPENCL_CALLS
  out << "ocl_enter_critical_section_schedule_computation();" << std::endl;
#endif
  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
      << "select_hashprobe_kernel, 1, NULL, global_work_size, NULL, 0, NULL, "
      << "&kernel_completion));" << std::endl;
#ifdef LOCK_OPENCL_CALLS
  out << "ocl_leave_critical_section_schedule_computation();" << std::endl;
#endif
  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
  out << "}" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Filter Kernel\");" << std::endl;
#endif

  return out.str();
}

const std::string getCodeCallProjectionKernel(
    const std::string& num_elements_for_loop, size_t global_worksize,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  out << "hashprobe_aggregate_and_project_kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"hashprobe_aggregate_and_project_kernel\", &_err);"
      << std::endl;
  out << "CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 0, "
         "sizeof ("
      << num_elements_for_loop << "), &" << num_elements_for_loop << "));"
      << std::endl;
  out << "CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 1, "
         "sizeof (cl_output_mem_flags), &cl_output_mem_flags));"
      << std::endl;
  out << "CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, 2, "
         "sizeof (cl_output_prefix_sum), &cl_output_prefix_sum));"
      << std::endl;

  unsigned int index = 3;
  for (const auto& var : input_vars) {
    auto var_name = isPointerType(var.second)
                        ? getCLBufferInputVarName(var.first)
                        : var.first;
    out << "CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, "
        << index++ << ", sizeof (" << var_name << ") , &" << var_name << "));"
        << std::endl;
  }

  for (const auto& var : output_vars) {
    out << "CL_CHECK(clSetKernelArg(hashprobe_aggregate_and_project_kernel, "
        << index++ << ", sizeof (" << getCLBufferResultVarName(var.first)
        << ") , &" << getCLBufferResultVarName(var.first) << "));" << std::endl;
  }

  out << "{" << std::endl;
  out << "cl_event kernel_completion;" << std::endl;
  out << "size_t global_work_size[1] = {" << global_worksize << "};"
      << std::endl;
  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "hashprobe_aggregate_and_project_kernel, 1, NULL, global_work_size, "
         "NULL, 0, NULL, &kernel_completion));"
      << std::endl;
  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
  out << "}" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Projection Kernel\");" << std::endl;
#endif
  return out.str();
}

const std::string getCodeCallSerialSinglePassKernel(
    const std::string& num_elements_for_loop,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  out << "kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"serial_single_pass_kernel\", &_err);"
      << std::endl;
  out << "CL_CHECK(clSetKernelArg(kernel, 0, "
         "sizeof ("
      << num_elements_for_loop << "), &" << num_elements_for_loop << "));"
      << std::endl;
  out << "CL_CHECK(clSetKernelArg(kernel, 1, "
         "sizeof (cl_output_number_of_result_tuples), "
         "&cl_output_number_of_result_tuples));"
      << std::endl;

  unsigned int index = 2;
  for (const auto& var : input_vars) {
    auto var_name = isPointerType(var.second)
                        ? getCLBufferInputVarName(var.first)
                        : var.first;
    out << "CL_CHECK(clSetKernelArg(kernel, " << index++ << ", sizeof ("
        << var_name << ") , &" << var_name << "));" << std::endl;
  }

  for (const auto& var : output_vars) {
    out << "CL_CHECK(clSetKernelArg(kernel, " << index++ << ", sizeof ("
        << getCLBufferResultVarName(var.first) << ") , &"
        << getCLBufferResultVarName(var.first) << "));" << std::endl;
  }

  out << "{" << std::endl;
  out << "cl_event kernel_completion;" << std::endl;
  out << "size_t global_work_size[1] = {" << 1 << "};" << std::endl;
  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "kernel, 1, NULL, global_work_size, "
         "NULL, 0, NULL, &kernel_completion));"
      << std::endl;
  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
  out << "}" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Kernel\");" << std::endl;
#endif
  return out.str();
}

const std::string getCodeCallParallelGlobalAtomicSinglePassKernel(
    const std::string& num_elements_for_loop, uint64_t global_worksize,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  out << "kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"serial_single_pass_kernel\", &_err);"
      << std::endl;
  out << "CL_CHECK(clSetKernelArg(kernel, 0, "
         "sizeof ("
      << num_elements_for_loop << "), &" << num_elements_for_loop << "));"
      << std::endl;

  out << "CL_CHECK(clSetKernelArg(kernel, 1, "
         "sizeof (cl_output_number_of_result_tuples), "
         "&cl_output_number_of_result_tuples));"
      << std::endl;

  unsigned int index = 2;
  for (const auto& var : input_vars) {
    auto var_name = isPointerType(var.second)
                        ? getCLBufferInputVarName(var.first)
                        : var.first;
    out << "CL_CHECK(clSetKernelArg(kernel, " << index++ << ", sizeof ("
        << var_name << ") , &" << var_name << "));" << std::endl;
  }

  for (const auto& var : output_vars) {
    out << "CL_CHECK(clSetKernelArg(kernel, " << index++ << ", sizeof ("
        << getCLBufferResultVarName(var.first) << ") , &"
        << getCLBufferResultVarName(var.first) << "));" << std::endl;
  }

  out << "{" << std::endl;
  out << "cl_event kernel_completion;" << std::endl;
  out << "size_t global_work_size[1] = {" << global_worksize << "};"
      << std::endl;
  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "kernel, 1, NULL, global_work_size, "
         "NULL, 0, NULL, &kernel_completion));"
      << std::endl;
  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
  out << getCodeReleaseKernel("kernel");
  out << "}" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Kernel\");" << std::endl;
#endif
  return out.str();
}

const std::string getCodeCallAggregationKernel(
    const std::string& num_elements_for_loop, uint64_t global_worksize,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  out << "cl_kernel aggregation_kernel = NULL;" << std::endl;
  out << "aggregation_kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"aggregation_kernel\", &_err);"
      << std::endl;
  out << "CL_CHECK(clSetKernelArg(aggregation_kernel, 0, "
      << "sizeof (" << num_elements_for_loop << "), &" << num_elements_for_loop
      << "));" << std::endl;

  unsigned int index = 1;
  for (const auto& var : input_vars) {
    auto var_name = isPointerType(var.second)
                        ? getCLBufferInputVarName(var.first)
                        : var.first;
    out << "CL_CHECK(clSetKernelArg(aggregation_kernel, " << index++
        << ", sizeof(" << var_name << ") , &" << var_name << "));" << std::endl;
  }

  for (const auto& var : output_vars) {
    out << "CL_CHECK(clSetKernelArg(aggregation_kernel, " << index++
        << ", sizeof(" << getCLBufferResultVarName(var.first) << ") , &"
        << getCLBufferResultVarName(var.first) << "));" << std::endl;
  }

  out << "{" << std::endl;
  out << "cl_event kernel_completion;" << std::endl;

  out << "size_t global_work_size = " << global_worksize << ";" << std::endl;
  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "aggregation_kernel, 1, NULL, &global_work_size, "
         "NULL, 0, NULL, &kernel_completion));"
      << std::endl;
  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
  out << "}" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Aggregation Kernel\");" << std::endl;
#endif
  return out.str();
}

const std::string getCodePrefixSum(
    const std::string& current_result_size_var_name,
    const std::string& num_elements_for_loop,
    const std::string& cl_mem_flag_array_name,
    const std::string& cl_mem_prefix_sum_array_name) {
  std::stringstream out;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  out << "/* prefix sum */" << std::endl;
  out << current_result_size_var_name
      << " = ocl_prefix_sum(ocl_getComputeCommandQueue(context)"
         ", ocl_getProgram(context), "
      << cl_mem_flag_array_name << ", " << cl_mem_prefix_sum_array_name << ", "
      << num_elements_for_loop << ");" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Prefix Sum Kernel\");" << std::endl;
#endif
  return out.str();
}

const std::string getCoderReduceAggregation(
    const std::map<std::string, std::string>& aggr_variables,
    const uint64_t global_worksize) {
  std::stringstream out;

  for (const auto& var : aggr_variables) {
    out << var.first << " = ocl_reduce_" << removePointer(var.second)
        << "(ocl_getTransferDeviceToHostCommandQueue(context), "
        << getCLBufferResultVarName(var.first) << ", " << global_worksize
        << ", " << SUM << ");" << std::endl;
  }

  return out.str();
}

const std::string getCodeReduction(std::string type, std::string input_var,
                                   std::string output_var) {
  std::stringstream out;

  out << output_var << " = ocl_reduce_" << removePointer(type)
      << "(ocl_getTransferDeviceToHostCommandQueue(context), "
      << getCLBufferResultVarName(input_var) << ", " << 8192 << ", " << SUM
      << ");" << std::endl;

  return out.str();
}

const std::string getCodeSort_uint64_t(std::string keys, std::string vals,
                                       std::string num_elements) {
  std::stringstream code;
  code << "ocl_sort_by_key_uint64_t("
       << "ocl_getTransferDeviceToHostCommandQueue(context), " << num_elements
       << ", " << keys << ", " << vals << ");" << std::endl;
  return code.str();
}

const std::string getCodeReduceByKeys(
    std::string num_elements, std::string keys, std::string keys_out,
    std::pair<std::string, std::string> values,
    std::pair<std::string, std::string> values_out, bool take_any_value) {
  std::stringstream code;
  code << " = ocl_reduce_by_key_" << values.second << "("
       << "ocl_getTransferDeviceToHostCommandQueue(context), " << num_elements
       << ", " << keys << ", " << values.first << ", " << keys_out << ", "
       << values_out.first << ", " << take_any_value << ");" << std::endl;
  return code.str();
}

const std::string getCodeGather(std::string map, std::string value_type,
                                std::string values_in, std::string values_out,
                                std::string num_elements) {
  std::stringstream code;
  code << "ocl_gather_" << value_type << "("
       << "ocl_getTransferDeviceToHostCommandQueue(context), " << num_elements
       << ", " << map << ", " << values_in << ", " << values_out << ");"
       << std::endl;

  return code.str();
}

std::string getCodePrintBuffer(std::string num_elements, std::string mem,
                               std::string type) {
  std::stringstream code;
  code << "ocl_print_" << type << "("
       << "ocl_getTransferDeviceToHostCommandQueue(context), " << num_elements
       << ", " << mem << ");" << std::endl;
  return code.str();
}

const std::string getCodeDeclareProjectionKernel() {
  std::stringstream out;
  out << "cl_kernel hashprobe_aggregate_and_project_kernel=NULL;" << std::endl;
  return out.str();
}

const std::string getCodeDeclareOCLResultBuffer(
    const std::pair<std::string, std::string>& var) {
  std::stringstream out;

  if (isPointerType(var.second)) {
    out << "cl_mem " << getCLBufferResultVarName(var.first) << " = NULL;"
        << std::endl;
  }

  return out.str();
}

const std::string getCodeDeclareOCLResultBuffers(
    const std::map<std::string, std::string>& variables) {
  std::stringstream out;
  for (const auto& var : variables) {
    out << getCodeDeclareOCLResultBuffer(var);
  }
  return out.str();
}

const std::string getCodeDeclareOCLAggregationResultBuffers(
    const std::map<std::string, std::string>& variables,
    const std::map<std::string, std::string>& aggr_variables,
    const uint64_t global_work_size) {
  std::stringstream out;

  for (const auto& var : variables) {
    if (aggr_variables.find(var.first) != aggr_variables.end()) {
      out << getCodeDeclareOCLResultBuffer(
          std::make_pair(var.first, removePointer(var.second) + "*"));
    } else {
      out << getCodeDeclareOCLResultBuffer(var);
    }

    out << "uint64_t " << var.first << "_length = " << global_work_size << ";"
        << std::endl;
  }

  return out.str();
}

const std::string getCodeInitOCLResultBuffer(
    cl_device_type dev_type, cl_device_id dev_id,
    const std::pair<std::string, std::string>& var, bool init_buffer,
    const std::string& init_value, bool use_host_pointer_if_possible) {
  std::stringstream out;

  cl_bool unified_host_memory = true;
  CL_CHECK(clGetDeviceInfo(dev_id, CL_DEVICE_HOST_UNIFIED_MEMORY,
                           sizeof(cl_bool), &unified_host_memory, NULL));

  if (dev_type == CL_DEVICE_TYPE_CPU || unified_host_memory) {
    if (isPointerType(var.second)) {
      out << getCLBufferResultVarName(var.first)
          << " = clCreateBuffer(ocl_getContext(context), CL_MEM_WRITE_ONLY";

      if (use_host_pointer_if_possible) {
        out << " | CL_MEM_USE_HOST_PTR";
      }

      out << ", sizeof (" << removePointer(var.second) << ") * " << var.first
          << "_length, ";

      if (use_host_pointer_if_possible) {
        out << var.first;
      } else {
        out << "NULL";
      }

      out << ", &_err);" << std::endl;
      out << "assert(_err == CL_SUCCESS);" << std::endl;
    }
  } else if (dev_type == CL_DEVICE_TYPE_GPU ||
             dev_type == CL_DEVICE_TYPE_ACCELERATOR) {
    if (isPointerType(var.second)) {
      out << getCLBufferResultVarName(var.first)
          << " = clCreateBuffer(ocl_getContext(context), CL_MEM_WRITE_ONLY, "
          << "sizeof (" << removePointer(var.second) << ") * " << var.first
          << "_length, NULL , &_err);" << std::endl;
      out << "assert(_err == CL_SUCCESS);" << std::endl;

      //!!!!HACK!!!
      if (var.first.find("ht_") == 0) {
        out << "CL_CHECK(oclCopyHostToDevice(" << var.first << ", "
            << "sizeof (" << removePointer(var.second) << ") * " << var.first
            << "_length, " << getCLBufferResultVarName(var.first) << ", "
            << "ocl_getContext(context), "
            << "ocl_getTransferHostToDeviceCommandQueue(context), " << 1
            << ", NULL"
            << "));" << std::endl;
      }
    }
  } else {
    COGADB_FATAL_ERROR("Unknown or unsupported device type!", "");
  }

  if (init_buffer) {
    out << removePointer(var.second) << " "
        << getCLBufferResultVarName(var.first) << "_init_value = " << init_value
        << ";" << std::endl;

    out << "oclFillBuffer_" << removePointer(var.second) << "("
        << "ocl_getTransferHostToDeviceCommandQueue(context), "
        << getCLBufferResultVarName(var.first) << ", "
        << getCLBufferResultVarName(var.first) << "_init_value, 0, "
        << var.first << "_length);" << std::endl;
  }

  return out.str();
}

const std::string getCodeInitOCLAggregationResultBuffers(
    const std::string& current_result_size_var_name, cl_device_type dev_type,
    cl_device_id dev_id, const std::map<std::string, std::string>& variables,
    const std::map<std::string, std::string>& aggr_variables,
    uint64_t global_worksize) {
  std::stringstream out;
  for (const auto& var : variables) {
    if (aggr_variables.find(var.first) != aggr_variables.end()) {
      out << getCodeInitOCLResultBuffer(
                 dev_type, dev_id,
                 std::make_pair(var.first, removePointer(var.second) + "*"),
                 true, var.first, false)
          << std::endl;
    } else {
      out << getCodeInitOCLResultBuffer(dev_type, dev_id, var) << std::endl;
    }
  }
  return out.str();
}

const std::string getCodeInitOCLResultBuffers(
    const std::string& current_result_size_var_name, cl_device_type dev_type,
    cl_device_id dev_id, const std::map<std::string, std::string>& variables) {
  std::stringstream out;
  for (const auto& var : variables) {
    out << getCodeInitOCLResultBuffer(dev_type, dev_id, var) << std::endl;
  }
  return out.str();
}

const std::string getCodeCopyOCLResultValueToVariable(
    const std::pair<std::string, std::string>& var) {
  std::stringstream out;
  // if (isPointerType(var.second)) {
  out << "_err = "
         "clEnqueueReadBuffer(ocl_getTransferDeviceToHostCommandQueue("
         "context), "
      << getCLBufferResultVarName(var.first) << ", CL_TRUE, 0,"
      << "sizeof (" << var.second << ")"
      << ", (void*)&" << var.first << ", "
      << "0, NULL, NULL);" << std::endl;
  out << "assert(_err == CL_SUCCESS);" << std::endl;
  //}

  return out.str();
}

const std::string getCodeCopyOCLResultBufferToHost(
    const std::pair<std::string, std::string>& var) {
  std::stringstream out;
  if (isPointerType(var.second)) {
    out << "_err = "
           "clEnqueueReadBuffer(ocl_getTransferDeviceToHostCommandQueue("
           "context), "
        << getCLBufferResultVarName(var.first) << ", CL_TRUE, 0,"
        << "sizeof (" << removePointer(var.second) << ") * " << var.first
        << "_length, " << var.first << ", "
        << "0, NULL, NULL);" << std::endl;
    out << "assert(_err == CL_SUCCESS);" << std::endl;
  }

  return out.str();
}

const std::string getCodeCopyOCLResultBuffersToHost(
    const std::string& current_result_size_var_name, cl_device_type dev_type,
    cl_device_id dev_id, const std::map<std::string, std::string>& variables) {
  std::stringstream out;

  cl_bool unified_host_memory = true;
  CL_CHECK(clGetDeviceInfo(dev_id, CL_DEVICE_HOST_UNIFIED_MEMORY,
                           sizeof(cl_bool), &unified_host_memory, NULL));

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  if (dev_type != CL_DEVICE_TYPE_CPU && !unified_host_memory) {
    out << "if (" << current_result_size_var_name << " > 0) {" << std::endl;

    for (const auto& var : variables) {
      out << getCodeCopyOCLResultBufferToHost(var);
    }

    out << "}" << std::endl;

    /* Wait for the command queue to get serviced before reading back results */
    out << "clFinish(ocl_getTransferDeviceToHostCommandQueue(context));"
        << std::endl;
  }
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Data Transfer Device To Host\");" << std::endl;
#endif
  return out.str();
}

const std::string getCodeCopyOCLAggregationResultBuffersToHost(
    const std::string& current_result_size_var_name, cl_device_type dev_type,
    cl_device_id dev_id, const std::map<std::string, std::string>& variables,
    const std::map<std::string, std::string>& aggr_vars) {
  std::stringstream out;

  cl_bool unified_host_memory = true;
  CL_CHECK(clGetDeviceInfo(dev_id, CL_DEVICE_HOST_UNIFIED_MEMORY,
                           sizeof(cl_bool), &unified_host_memory, NULL));

  if (dev_type != CL_DEVICE_TYPE_CPU && !unified_host_memory) {
    for (const auto& var : variables) {
      if (aggr_vars.find(var.first) == aggr_vars.end()) {
        out << getCodeCopyOCLResultBufferToHost(var);
      }
    }

    /* Wait for the command queue to get serviced before reading back results */
    out << "clFinish(ocl_getTransferDeviceToHostCommandQueue(context));"
        << std::endl;
  }
  return out.str();
}

const std::string getCodeCopyOCLGroupedAggregationResultBuffersToHost(
    const std::string& current_result_size_var_name, cl_device_type dev_type,
    cl_device_id dev_id, const std::map<std::string, std::string>& variables,
    const std::string& hash_map_name) {
  std::stringstream out;

  cl_bool unified_host_memory = true;
  CL_CHECK(clGetDeviceInfo(dev_id, CL_DEVICE_HOST_UNIFIED_MEMORY,
                           sizeof(cl_bool), &unified_host_memory, NULL));

  if (dev_type != CL_DEVICE_TYPE_CPU && !unified_host_memory) {
    for (const auto& var : variables) {
      out << getCodeCopyOCLResultBufferToHost(var);
    }

    out << getCodeCopyOCLResultBufferToHost(
        std::make_pair(hash_map_name, "AggregationPayload*"));

    /* Wait for the command queue to get serviced before reading back results */
    out << "clFinish(ocl_getTransferDeviceToHostCommandQueue(context));"
        << std::endl;
  }
  return out.str();
}

const std::string getCodeCleanupDefaultKernelOCLStructures() {
  std::stringstream out;

  return out.str();
}

const std::string getCodeCleanupOCLStructures(
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars) {
  std::stringstream out;

  for (const auto& var : input_vars) {
    if (isPointerType(var.second)) {
      out << "if(" << getCLBufferInputVarName(var.first) << ") {" << std::endl
          << "CL_CHECK(clReleaseMemObject("
          << getCLBufferInputVarName(var.first) << "));" << std::endl
          << getCLBufferInputVarName(var.first) << " = NULL;" << std::endl
          << "}" << std::endl;
    }
  }

  for (const auto& var : output_vars) {
    if (isPointerType(var.second)) {
      out << "if(" << getCLBufferResultVarName(var.first) << ") {" << std::endl
          << "CL_CHECK(clReleaseMemObject("
          << getCLBufferResultVarName(var.first) << "));" << std::endl
          << getCLBufferResultVarName(var.first) << " = NULL;" << std::endl
          << "}" << std::endl;
    }
  }
  return out.str();
}

const std::string getCodeCleanupOCLAggregationStructures(
    const std::map<std::string, std::string>& aggr_vars) {
  std::stringstream out;

  for (const auto& var : aggr_vars) {
    out << "if(" << getCLBufferResultVarName(var.first) << ") {" << std::endl
        << "CL_CHECK(clReleaseMemObject(" << getCLBufferResultVarName(var.first)
        << "));" << std::endl
        << getCLBufferResultVarName(var.first) << " = NULL;" << std::endl
        << "}" << std::endl;
  }

  return out.str();
}

const std::string getCLInputVarName(const AttributeReference& ref) {
  std::stringstream str;
  str << "__global const ";

  if (isAttributeDictionaryCompressed(ref)) {
    str << toCType(UINT32);
  } else {
    str << toCType(ref.getAttributeType());
  }

  str << "* const " << getVarName(ref);

  return str.str();
}

const std::string getCLResultVarName(const AttributeReference& ref) {
  std::stringstream str;
  str << "__global ";

  if (isAttributeDictionaryCompressed(ref)) {
    str << toCType(UINT32);
  } else {
    str << toCType(ref.getAttributeType());
  }
  str << "* " << getResultArrayVarName(ref);
  return str.str();
}

const std::string getCodeMurmur3Hashing(bool version_32bit) {
  std::stringstream ss;

  if (version_32bit) {
    ss << "  index ^= index >> 16;" << std::endl
       << "  index *= 0x85ebca6b;" << std::endl
       << "  index ^= index >> 13;" << std::endl
       << "  index *= 0xc2b2ae35;" << std::endl
       << "  index ^= index >> 16;" << std::endl;
  } else {
    ss << "  index ^= index >> 33;" << std::endl
       << "  index *= 0xff51afd7ed558ccd;" << std::endl
       << "  index ^= index >> 33;" << std::endl
       << "  index *= 0xc4ceb9fe1a85ec53;" << std::endl
       << "  index ^= index >> 33;" << std::endl;
  }

  return ss.str();
}

const std::string getCodeMultiplyShift() {
  return "index *= 123456789123456789ul;";
}

const std::string getCodeMultiplyAddShift() {
  return "index = index * 789 + 321;";
}

const std::string getCodeReleaseKernel(const std::string& kernel_var_name) {
  std::stringstream out;

  out << "if(" << kernel_var_name << ") {" << std::endl
      << "  CL_CHECK(clReleaseKernel(" << kernel_var_name << "));" << std::endl
      << "  " << kernel_var_name << " = NULL;" << std::endl
      << "}" << std::endl;

  return out.str();
}

}  // end namespace CoGaDB
