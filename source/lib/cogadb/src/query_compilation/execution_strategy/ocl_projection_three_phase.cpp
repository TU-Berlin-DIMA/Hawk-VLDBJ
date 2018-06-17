#include <query_compilation/execution_strategy/ocl_projection_three_phase.hpp>

#include <boost/thread.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLProjectionThreePhase::OCLProjectionThreePhase(bool use_host_ptr,
                                                 MemoryAccessPattern mem_access,
                                                 cl_device_id dev_id)
    : OCLProjection(use_host_ptr, mem_access, dev_id),
      filter_kernel_(boost::make_shared<GeneratedKernel>()),
      projection_kernel_(boost::make_shared<GeneratedKernel>()) {}

void OCLProjectionThreePhase::addInstruction_impl(InstructionPtr instr) {
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

  if (instr->getInstructionType() == FILTER_INSTR ||
      instr->getInstructionType() == HASH_TABLE_PROBE_INSTR ||
      instr->getInstructionType() == CROSS_JOIN_INSTR) {
    CodeBlockPtr block = filter_kernel_->kernel_code_blocks.back();
    block->upper_block.insert(block->upper_block.end(),
                              gen_code->upper_code_block_.begin(),
                              gen_code->upper_code_block_.end());

    block->lower_block.insert(block->lower_block.begin(),
                              gen_code->lower_code_block_.begin(),
                              gen_code->lower_code_block_.end());

    for (auto& variable : instr->getInputVariables(OCL_TARGET_CODE)) {
      filter_kernel_input_vars_.insert(variable);
    }

    for (auto& variable : instr->getOutputVariables(OCL_TARGET_CODE)) {
      filter_kernel_output_vars_.insert(variable);
    }

    filter_kernel_init_vars_ << gen_code->init_variables_code_block_.str();
  }

  if (instr->getInstructionType() == HASH_TABLE_PROBE_INSTR ||
      instr->getInstructionType() == HASH_TABLE_BUILD_INSTR ||
      instr->getInstructionType() == MAP_UDF_INSTR ||
      instr->getInstructionType() == HASH_AGGREGATE_INSTR ||
      instr->getInstructionType() == CROSS_JOIN_INSTR ||
      instr->getInstructionType() == PRODUCE_TUPLE_INSTR ||
      instr->getInstructionType() == MATERIALIZATION_INSTR) {
    CodeBlockPtr block = projection_kernel_->kernel_code_blocks.back();
    block->upper_block.insert(block->upper_block.end(),
                              gen_code->upper_code_block_.begin(),
                              gen_code->upper_code_block_.end());

    block->lower_block.insert(block->lower_block.begin(),
                              gen_code->lower_code_block_.begin(),
                              gen_code->lower_code_block_.end());

    for (auto& variable : instr->getInputVariables(OCL_TARGET_CODE)) {
      projection_kernel_input_vars_.insert(variable);
    }

    for (auto& variable : instr->getOutputVariables(OCL_TARGET_CODE)) {
      projection_kernel_output_vars_.insert(variable);
    }

    projection_kernel_init_vars_ << gen_code->init_variables_code_block_.str();
  }

  if (instr->getInstructionType() == ALGEBRA_INSTR) {
    CodeBlockPtr block = projection_kernel_->kernel_code_blocks.back();

    block->upper_block.push_back(gen_code->declare_variables_code_block_.str());

    block->upper_block.insert(block->upper_block.end(),
                              gen_code->upper_code_block_.begin(),
                              gen_code->upper_code_block_.end());

    block->lower_block.insert(block->lower_block.begin(),
                              gen_code->lower_code_block_.begin(),
                              gen_code->lower_code_block_.end());

    for (auto& variable : instr->getInputVariables(OCL_TARGET_CODE)) {
      projection_kernel_input_vars_.insert(variable);
    }

    for (auto& variable : instr->getOutputVariables(OCL_TARGET_CODE)) {
      projection_kernel_output_vars_.insert(variable);
    }

    projection_kernel_init_vars_ << gen_code->init_variables_code_block_.str();
  }
}

const std::map<std::string, std::string>
OCLProjectionThreePhase::getInputVariables() const {
  auto merged_input_vars = filter_kernel_input_vars_;

  for (const auto& var : projection_kernel_input_vars_) {
    merged_input_vars.insert(var);
  }
  return merged_input_vars;
}

const std::map<std::string, std::string>
OCLProjectionThreePhase::getOutputVariables() const {
  auto merged_output_vars = filter_kernel_output_vars_;

  for (const auto& var : projection_kernel_output_vars_) {
    merged_output_vars.insert(var);
  }
  return merged_output_vars;
}

const std::string OCLProjectionThreePhase::getCodeCallComputeKernels(
    const std::string& num_elements_for_loop, size_t global_worksize,
    cl_device_type dev_type_) const {
  std::stringstream out;
  /* generate code that calls OpenCL kernel that performs filtering
   * and produces a flag array marking matching tuples
   */
  out << "/* first phase: pass over data and compute flag array */"
      << std::endl;
  out << getCodeCallFilterKernel(num_elements_for_loop, global_worksize,
                                 filter_kernel_input_vars_);
  /* compute write positions */
  out << "/* second phase: compute write positions from flag array by using a "
         "prefix sum */"
      << std::endl;
  out << getCodePrefixSum("current_result_size", num_elements_for_loop,
                          "cl_output_mem_flags", "cl_output_prefix_sum");

  out << "/* third phase: pass over data and write result */" << std::endl;
  /* forward declare projection kernel */
  out << getCodeDeclareProjectionKernel();
  /* if the result is empty, we do not allocate result buffers and do not call
   * the
   * projection kernel, but create an empty result table right away
   */
  out << "if(current_result_size>0){" << std::endl;
  out << "allocated_result_elements=current_result_size;" << std::endl;
  out << projection_kernel_init_vars_.str();
  /* in case output is not empty, we initialize the previously declared output
   * buffers */
  out << getCodeInitOCLResultBuffers("current_result_size", dev_type_, dev_id_,
                                     projection_kernel_output_vars_);
  /* generate code that calls projection kernel that writes the result */
  out << getCodeCallProjectionKernel(num_elements_for_loop, global_worksize,
                                     projection_kernel_input_vars_,
                                     projection_kernel_output_vars_);
  /* copy data from device back to CPU main memory, if required */
  out << getCodeCopyOCLResultBuffersToHost("current_result_size", dev_type_,
                                           dev_id_,
                                           projection_kernel_output_vars_);
  out << "}" << std::endl;
  return out.str();
}
const std::string OCLProjectionThreePhase::getKernelCode(
    const std::string& loop_variable_name, size_t global_worksize,
    MemoryAccessPattern mem_access_) {
  std::stringstream kernel;
  kernel << getCodeFilterKernel(filter_kernel_, loop_variable_name,
                                global_worksize, filter_kernel_input_vars_,
                                filter_kernel_output_vars_, mem_access_);
  kernel << getCodeProjectionKernel(
      projection_kernel_, loop_variable_name, global_worksize,
      projection_kernel_input_vars_, projection_kernel_output_vars_,
      mem_access_);

  return kernel.str();
}

const std::string OCLProjectionThreePhase::getCodeCreateCustomStructures(
    const std::string& num_elements_for_loop, cl_device_type dev_type) {
  std::stringstream out;
  out << filter_kernel_init_vars_.str();
  out << "char* tempory_flag_array = NULL;" << std::endl;
  out << "TID* tempory_prefix_sum_array = NULL;" << std::endl;

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

    out << "cl_mem cl_output_prefix_sum"
        << " = clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE,"
        << " sizeof (TID) * " << num_elements_for_loop << ","
        << "NULL, &_err);" << std::endl;
    out << "assert(_err == CL_SUCCESS);" << std::endl;
  }

  return out.str();
}

const std::string OCLProjectionThreePhase::getCodeCleanupCustomStructures()
    const {
  std::stringstream out;
  out << "if(tempory_flag_array) free(tempory_flag_array);" << std::endl;
  out << "if(tempory_prefix_sum_array) free(tempory_prefix_sum_array);"
      << std::endl;
  out << "if(cl_output_mem_flags) "
         "CL_CHECK(clReleaseMemObject(cl_output_mem_flags));"
      << std::endl;
  out << "if(cl_output_prefix_sum) "
         "CL_CHECK(clReleaseMemObject(cl_output_prefix_sum));"
      << std::endl;
  out << "if(select_hashprobe_kernel) "
         "CL_CHECK(clReleaseKernel(select_hashprobe_kernel));"
      << std::endl;
  out << "if(hashprobe_aggregate_and_project_kernel) "
         "CL_CHECK(clReleaseKernel(hashprobe_aggregate_and_project_kernel));"
      << std::endl;

  return out.str();
}

}  // namespace ExecutionStrategy

}  // namespace CoGaDB
