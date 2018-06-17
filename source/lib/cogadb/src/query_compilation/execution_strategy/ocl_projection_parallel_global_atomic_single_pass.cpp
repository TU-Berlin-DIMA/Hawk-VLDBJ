#include <core/variable_manager.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/execution_strategy/ocl_projection_parallel_global_atomic_single_pass.hpp>

#include <boost/thread.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLProjectionParallelGlobalAtomicSinglePass::
    OCLProjectionParallelGlobalAtomicSinglePass(bool use_host_ptr,
                                                MemoryAccessPattern mem_access,
                                                cl_device_id dev_id)
    : OCLProjection(use_host_ptr, mem_access, dev_id),
      kernel_(boost::make_shared<GeneratedKernel>()) {}

void OCLProjectionParallelGlobalAtomicSinglePass::addInstruction_impl(
    InstructionPtr instr) {
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
      instr->getInstructionType() != HASH_TABLE_BUILD_INSTR) {
    block->upper_block.insert(block->upper_block.end(),
                              gen_code->upper_code_block_.begin(),
                              gen_code->upper_code_block_.end());

    block->lower_block.insert(block->lower_block.begin(),
                              gen_code->lower_code_block_.begin(),
                              gen_code->lower_code_block_.end());

  } else {
    /* add materialization code to special intermdiate block,
     * so we can add code before and after it to update the write_pos
     * (normally, we add it after, but if we use atomics, we need to do it
     * before)
     */
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
OCLProjectionParallelGlobalAtomicSinglePass::getInputVariables() const {
  auto merged_input_vars = kernel_input_vars_;

  for (const auto& var : kernel_input_vars_) {
    merged_input_vars.insert(var);
  }
  return merged_input_vars;
}

const std::map<std::string, std::string>
OCLProjectionParallelGlobalAtomicSinglePass::getOutputVariables() const {
  auto merged_output_vars = kernel_output_vars_;

  for (const auto& var : kernel_output_vars_) {
    merged_output_vars.insert(var);
  }
  return merged_output_vars;
}

const std::string
OCLProjectionParallelGlobalAtomicSinglePass::getCodeCallComputeKernels(
    const std::string& num_elements_for_loop, size_t global_worksize,
    cl_device_type dev_type_) const {
  std::stringstream out;
  out << "current_result_size = allocated_result_elements";

  if (VariableManager::instance().getVariableValueBoolean(
          "code_gen.opt.enable_predication")) {
    out << " + 1";
  }

  out << ";" << std::endl;
  /* in case output is not empty, we initialize the previously declared output
   * buffers */
  out << getCodeInitOCLResultBuffers("current_result_size", dev_type_, dev_id_,
                                     kernel_output_vars_);
  /* generate code that calls the kernel that writes the result */
  out << getCodeCallParallelGlobalAtomicSinglePassKernel(
      num_elements_for_loop, global_worksize, kernel_input_vars_,
      kernel_output_vars_);

  out << "uint32_t size_tmp;" << std::endl;
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
  out << "clFinish(ocl_getTransferDeviceToHostCommandQueue(context));"
      << std::endl;
  out << "current_result_size=size_tmp;" << std::endl;

  /* copy data from device back to CPU main memory, if required */
  out << getCodeCopyOCLResultBuffersToHost("current_result_size", dev_type_,
                                           dev_id_, kernel_output_vars_);

  return out.str();
}
const std::string OCLProjectionParallelGlobalAtomicSinglePass::getKernelCode(
    const std::string& loop_variable_name, size_t global_worksize,
    MemoryAccessPattern mem_access_) {
  std::stringstream kernel;
  kernel << getCodeParallelGlobalAtomicSinglePassKernel(
      kernel_, loop_variable_name, kernel_input_vars_, kernel_output_vars_,
      mem_access_);

  return kernel.str();
}

const std::string
OCLProjectionParallelGlobalAtomicSinglePass::getCodeCreateCustomStructures(
    const std::string& num_elements_for_loop, cl_device_type dev_type) {
  std::stringstream out;
  out << kernel_init_vars_.str();

  out << "cl_kernel kernel=NULL;" << std::endl;

  out << "cl_mem cl_output_number_of_result_tuples "
      << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE, "
      << "sizeof (uint32_t)"
      << ","
      << "NULL, &_err);" << std::endl;
  out << "assert(_err == CL_SUCCESS);" << std::endl;
  out << "oclFillBuffer_uint32_t("
      << "ocl_getTransferHostToDeviceCommandQueue(context), "
      << "cl_output_number_of_result_tuples, 0, 0, 1);" << std::endl;
  return out.str();
}

const std::string
OCLProjectionParallelGlobalAtomicSinglePass::getCodeCleanupCustomStructures()
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
