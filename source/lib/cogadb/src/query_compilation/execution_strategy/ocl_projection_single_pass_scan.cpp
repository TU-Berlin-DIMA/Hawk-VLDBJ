/*
 * author: henning funke
 * date: 22.06.2016
 */

#include <query_compilation/gpu_utilities/util/divup.h>
#include <boost/thread.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/code_generators/ocl_workgroup_utils.hpp>
#include <query_compilation/execution_strategy/ocl_projection_single_pass_scan.hpp>
#include <query_compilation/pipeline_selectivity_estimates.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLProjectionSinglePassScan::OCLProjectionSinglePassScan(
    bool use_host_ptr, MemoryAccessPattern mem_access, cl_device_id dev_id)
    : OCLProjection(use_host_ptr, mem_access, dev_id),
      kernel_(boost::make_shared<GeneratedKernel>()) {
  // kernel configuration
  global_worksize_ = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_projection.gpu.single_pass_scan.global_size");
  local_worksize_ = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_projection.gpu.single_pass_scan.local_size");
  values_per_thread_ = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_projection.gpu.single_pass_scan.values_per_thread");
  std::string wg_implementation =
      VariableManager::instance().getVariableValueString(
          "code_gen.ocl.workgroup_function_implementation");
  if (wg_implementation.compare("atomic") == 0) {
    write_pos_mode_ = ATOMICS_ONLY;
  } else if (wg_implementation.compare("fragmented") == 0) {
    write_pos_mode_ = FRAGMENTED_WRITE;
  } else {
    write_pos_mode_ = LOCAL_RESOLUTION;
  }
}

void OCLProjectionSinglePassScan::addInstruction_impl(InstructionPtr instr) {
  if (instr->getInstructionType() == LOOP_INSTR) {
    first_loop_ = boost::dynamic_pointer_cast<Loop>(instr);
    assert(first_loop_ != nullptr);
  }
  GeneratedCodePtr gen_code = instr->getCode(OCL_TARGET_CODE);
  if (!gen_code) {
    COGADB_FATAL_ERROR("", "");
  }
  if (!addToCode(code_, gen_code)) {
    COGADB_FATAL_ERROR("", "");
  }
  if (instr->getInstructionType() == LOOP_INSTR) {
    return;
  }
  CodeBlockPtr block = kernel_->kernel_code_blocks.back();
  if (instr->getInstructionType() != MATERIALIZATION_INSTR &&
      instr->getInstructionType() != HASH_TABLE_BUILD_INSTR &&
      instr->getInstructionType() != MAP_UDF_INSTR) {
    block->upper_block.insert(block->upper_block.end(),
                              gen_code->upper_code_block_.begin(),
                              gen_code->upper_code_block_.end());

    block->lower_block.insert(block->lower_block.begin(),
                              gen_code->lower_code_block_.begin(),
                              gen_code->lower_code_block_.end());
  } else {
    block->materialize_result_block.insert(
        block->materialize_result_block.end(),
        gen_code->upper_code_block_.begin(), gen_code->upper_code_block_.end());
  }
  for (auto& variable : instr->getInputVariables(OCL_TARGET_CODE)) {
    kernel_input_vars_.insert(variable);
  }
  for (auto& variable : instr->getOutputVariables(OCL_TARGET_CODE)) {
    kernel_output_vars_.insert(variable);
  }
  kernel_init_vars_ << gen_code->init_variables_code_block_.str();
}

const std::map<std::string, std::string>
OCLProjectionSinglePassScan::getInputVariables() const {
  auto merged_input_vars = kernel_input_vars_;
  for (const auto& var : kernel_input_vars_) {
    merged_input_vars.insert(var);
  }
  return merged_input_vars;
}

const std::map<std::string, std::string>
OCLProjectionSinglePassScan::getOutputVariables() const {
  auto merged_output_vars = kernel_output_vars_;

  for (const auto& var : kernel_output_vars_) {
    merged_output_vars.insert(var);
  }
  return merged_output_vars;
}

const std::string OCLProjectionSinglePassScan::getCodeCallComputeKernels(
    const std::string& num_elements_for_loop, size_t global_worksize,
    cl_device_type dev_type_) const {
  std::stringstream out;
  out << "current_result_size=allocated_result_elements;" << std::endl;
  /* in case output is not empty, we initialize the previously declared output
   * buffers */
  out << getCodeInitOCLResultBuffers("current_result_size", dev_type_, dev_id_,
                                     kernel_output_vars_);
  /* generate code that calls the kernel that writes the result */
  out << getCodeCallSinglePassScanKernel(num_elements_for_loop, global_worksize,
                                         local_worksize_, kernel_input_vars_,
                                         kernel_output_vars_);

  out << "uint32_t size_tmp = 0;" << std::endl;
  out << "_err = "
         "clEnqueueReadBuffer(ocl_getTransferDeviceToHostCommandQueue("
         "context), "
      << "cl_output_number_of_result_tuples"
      << ", CL_TRUE, 0,"
      << "sizeof (size_tmp), "
      << "&size_tmp"
      << ", "
      << "0, NULL, NULL);" << std::endl;
  out << "assert(_err == CL_SUCCESS);" << std::endl;
  out << "current_result_size=size_tmp;" << std::endl;
  out << "printf(\"Input Table Size: %lu, Pipeline Result Size: %lu\\n\", "
      << num_elements_for_loop << ", current_result_size);" << std::endl;
  /* copy data from device back to CPU main memory, if required */

  out << getCodeCopyOCLResultBuffersToHost("current_result_size", dev_type_,
                                           dev_id_, kernel_output_vars_);
  return out.str();
}

const std::string getCodeCallSinglePassScanKernel(
    const std::string& num_elements_for_loop, uint64_t global_worksize,
    size_t local_worksize, const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  out << "kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"single_pass_kernel\", &_err);"
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
  out << "size_t local_work_size[1] = {" << local_worksize << "};" << std::endl;
  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "kernel, 1, NULL, global_work_size, "
         "local_work_size, 0, NULL, &kernel_completion));"
      << std::endl;
  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
  out << "}" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Kernel\");" << std::endl;
#endif
  return out.str();
}

const std::pair<std::string, std::string> getCodeOCLKernelMemoryTraversal(
    const std::string& loop_variable_name, size_t global_worksize,
    MemoryAccessPattern mem_access, size_t local_worksize,
    bool divup_local_worksize) {
  std::stringstream begin_loop;
  std::stringstream end_loop;
  begin_loop << "uint32_t result_increment=1;" << std::endl;
  if (mem_access == COALESCED_MEMORY_ACCESS) {
    begin_loop << "uint64_t " << loop_variable_name << " = get_global_id(0);"
               << std::endl;
    begin_loop << "const uint64_t number_of_threads = get_global_size(0);"
               << std::endl;

    if (divup_local_worksize) {
      uint64_t upperLimit =
          divUp(global_worksize, local_worksize) * local_worksize;
      begin_loop << "size_t divup = (num_elements + get_local_size(0) - 1) / "
                    "get_local_size(0);"
                 << std::endl
                 << "divup *= get_local_size(0);" << std::endl;

      begin_loop << "while(" << loop_variable_name << " < divup){" << std::endl;
    } else {
      begin_loop << "while(" << loop_variable_name << " < num_elements){"
                 << std::endl;
    }

  } else if (mem_access == BLOCK_MEMORY_ACCESS) {
    begin_loop << "uint64_t block_size = (num_elements + " << global_worksize
               << " - 1) / " << global_worksize << ";" << std::endl;
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

const std::string getCodeSinglePassScanKernel(
    ExecutionStrategy::GeneratedKernelPtr kernel_,
    const std::string& loop_variable_name,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars,
    MemoryAccessPattern mem_access, size_t local_worksize,
    WritePositionMode write_pos_mode, GeneratedCodePtr code) {
  std::stringstream kernel;
  kernel << code->kernel_header_and_types_code_block_.str();

  WorkgroupCodeBlock workgroup_scan =
      getScanCode("num_thread_out", "local_offset", local_worksize);
  kernel << workgroup_scan.global_init;

  kernel << getCodeKernelSignature("single_pass_kernel",
                                   "uint64_t num_elements, "
                                   "__global uint32_t* global_write_pos",
                                   input_vars, output_vars)
         << "{" << std::endl;
  std::pair<std::string, std::string> loop_code =
      getCodeOCLKernelMemoryTraversal(loop_variable_name, 1, mem_access,
                                      local_worksize, true);

  kernel << "int lid = get_local_id(0);" << std::endl
         << "int localSize = get_local_size(0);" << std::endl
         << "int globalIdx = get_group_id(0);" << std::endl;
  kernel << workgroup_scan.kernel_init;

  kernel << "uint32_t write_pos = 0;" << std::endl;
  kernel << "__local uint32_t global_group_offset[" << (local_worksize / 32)
         << "];" << std::endl;

  kernel << loop_code.first;

  for (const auto& code_block : kernel_->kernel_code_blocks) {
    kernel << "int num_thread_out=0;" << std::endl;
    kernel << "bool selected = false;" << std::endl;

    std::stringstream declarations;
    std::stringstream other;

    // filter out declarations
    for (const auto& upper : code_block->upper_block) {
      if (upper.find("double ") == 0 || upper.find("float ") == 0 ||
          upper.find("TID ") == 0) {
        declarations << upper << std::endl;
      } else {
        other << upper << std::endl;
      }
    }
    kernel << declarations.str();
    kernel << "if(" << loop_variable_name << "< num_elements) {" << std::endl;
    kernel << other.str();
    kernel << "selected=true;" << std::endl;
    kernel << "num_thread_out=1;" << std::endl;
    kernel << "}" << std::endl;

    for (const auto& lower : code_block->lower_block) {
      kernel << lower << std::endl;
    }

    if (write_pos_mode == ATOMICS_ONLY) {
      kernel << "if(num_thread_out > 0) write_pos = "
                "atomic_add(global_write_pos, num_thread_out);"
             << std::endl;
    } else if (write_pos_mode == LOCAL_RESOLUTION) {
      kernel << workgroup_scan.local_init;
      kernel << workgroup_scan.computation;
      kernel << "if((lid+1)%" << getGroupResultOffset() << "==0) {" << std::endl
             << "    global_group_offset[" << getGroupVariableIndex()
             << "] = atomic_add(global_write_pos, group_total);" << std::endl
             << "}" << std::endl
             << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
             << "write_pos = global_group_offset[" << getGroupVariableIndex()
             << "] + local_offset;" << std::endl;
    } else if (write_pos_mode == FRAGMENTED_WRITE) {
      kernel << "write_pos = " << loop_variable_name << ";" << std::endl;
    }

    if (write_pos_mode != FRAGMENTED_WRITE) {
      kernel << "if(selected) {" << std::endl;
      for (const auto& materialize : code_block->materialize_result_block) {
        kernel << materialize << std::endl;
      }
      kernel << "}" << std::endl;
    } else {
      kernel << "if(" << loop_variable_name << "< num_elements) {" << std::endl;
      for (const auto& materialize : code_block->materialize_result_block) {
        kernel << materialize << std::endl;
      }
      kernel << "}" << std::endl;
    }
  }
  kernel << loop_code.second;
  kernel << "}" << std::endl << std::endl;
  return kernel.str();
}

const std::string OCLProjectionSinglePassScan::getKernelCode(
    const std::string& loop_variable_name, size_t global_worksize,
    MemoryAccessPattern mem_access_) {
  std::stringstream kernel;
  kernel << getKernelHeadersAndTypes();
  kernel << getCodeSinglePassScanKernel(
      kernel_, loop_variable_name, kernel_input_vars_, kernel_output_vars_,
      mem_access_, local_worksize_, write_pos_mode_, code_);
  return kernel.str();
}

const std::pair<std::string, std::string> OCLProjectionSinglePassScan::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  GeneratedCodePtr code = code_;
  assert(first_loop_ != NULL);
  std::string num_elements_for_loop = first_loop_->getVarNameNumberOfElements();
  size_t global_worksize = global_worksize_;

  if (!columns_to_decompress.empty()) {
    COGADB_FATAL_ERROR("Cannot work with uncompressed columns!", "");
  }

  bool cache_input_data = VariableManager::instance().getVariableValueBoolean(
      "code_gen.enable_caching");

  /* generate OpenCL kernel */
  std::stringstream kernel;
  kernel << getKernelCode(first_loop_->getLoopVariableName(), global_worksize,
                          mem_access_);
  /* generate host code that prepares OpenCL buffers and launches the kernel */
  std::stringstream out;
  out << getDefaultIncludeHeaders();
  /* all imports and declarations */
  out << code->header_and_types_code_block_.str() << std::endl;
  /* include the generated kernels in the host code as comment for easier
   * debugging */
  out << "/*" << std::endl;
  out << kernel.str() << std::endl;
  out << "*/" << std::endl;
  /* write function signature */
  out << getFunctionSignature() + " {" << std::endl;
  out << code->fetch_input_code_block_.str() << std::endl;
  /* all code for query function definition and input array retrieval */
  out << code->declare_variables_code_block_.str() << std::endl;
  /* generate code that retrieves number of elements from loop table */
  out << first_loop_->getNumberOfElementsExpression() << std::endl;
  // TODO, we need to find a better place for this!
  out << "uint64_t current_result_size = 0;" << std::endl;

  double selectivity_estimate =
      PipelineSelectivityTable::instance().getSelectivity(result_table_name);

  std::cout << "selectivity estimate for " << result_table_name << " = "
            << selectivity_estimate << std::endl;

  out << "uint64_t allocated_result_elements = "
      << "(double)" << first_loop_->getVarNameNumberOfElements() << " * "
      << selectivity_estimate << ";" << std::endl;

  out << "cl_int _err = 0;" << std::endl;
  // out << filter_kernel_init_vars_.str();
  /* create additional data structures required by execution strategy */
  out << this->getCodeCreateCustomStructures(num_elements_for_loop, dev_type_);

  out << kernel_init_vars_.str();

  out << getCodeCreateOCLInputBuffers(num_elements_for_loop,
                                      this->getInputVariables(), dev_type_,
                                      dev_id_, cache_input_data)
      << std::endl;
  /* foward declare OCL result buffers */
  out << getCodeDeclareOCLResultBuffers(this->getOutputVariables());
  /* generate code that calls kernels */
  out << getCodeCallComputeKernels(num_elements_for_loop, global_worksize,
                                   dev_type_);
  /* clean up previously allocated OpenCL data structures */
  out << this->getCodeCleanupCustomStructures();
  out << getCodeCleanupOCLStructures(this->getInputVariables(),
                                     this->getOutputVariables());
  /* generate code that builds the result table using the minimal API */
  out << generateCCodeCreateResultTable(
             param, code->create_result_table_code_block_.str(),
             code->clean_up_code_block_.str(), result_table_name)
      << std::endl;
  out << "}" << std::endl;
  return std::make_pair(out.str(), kernel.str());
}

const std::string OCLProjectionSinglePassScan::getCodeCreateCustomStructures(
    const std::string& num_elements_for_loop, cl_device_type dev_type) {
  std::stringstream out;
  out << "cl_kernel kernel=NULL;" << std::endl;

  out << "cl_mem cl_output_number_of_result_tuples "
      << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE, "
      << "sizeof (uint32_t)"
      << ","
      << "NULL, &_err);" << std::endl;
  out << "assert(_err == CL_SUCCESS);" << std::endl;
  out << "uint32_t fill_buffer_init_value=0;" << std::endl;
  out << "CL_CHECK(clEnqueueFillBuffer(ocl_"
         "getTransferHostToDeviceCommandQueue(context), "
      << "cl_output_number_of_result_tuples, &fill_buffer_init_value, "
         "sizeof(fill_buffer_init_value), 0, "
      << "sizeof(fill_buffer_init_value), "
      << "0, NULL, NULL));" << std::endl;
  out << "clFinish(ocl_getTransferHostToDeviceCommandQueue(context));"
      << std::endl;
  return out.str();
}

const std::string OCLProjectionSinglePassScan::getCodeCleanupCustomStructures()
    const {
  std::stringstream out;
  out << "if(kernel) "
         "CL_CHECK(clReleaseKernel(kernel));"
      << std::endl;
  out << "if(cl_output_number_of_result_tuples) "
         "CL_CHECK(clReleaseMemObject(cl_output_number_of_result_tuples));"
      << std::endl;
  return out.str();
}

}  // namespace ExecutionStrategy

}  // namespace CoGaDB
