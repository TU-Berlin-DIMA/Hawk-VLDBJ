#include <core/variable_manager.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/execution_strategy/ocl_projection_serial_single_pass.hpp>

#include <boost/thread.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLProjectionSerialSinglePass::OCLProjectionSerialSinglePass(
    bool use_host_ptr, MemoryAccessPattern mem_access, cl_device_id dev_id)
    : OCLProjection(use_host_ptr, mem_access, dev_id),
      kernel_(boost::make_shared<GeneratedKernel>()) {}

void OCLProjectionSerialSinglePass::addInstruction_impl(InstructionPtr instr) {
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
  block->upper_block.insert(block->upper_block.end(),
                            gen_code->upper_code_block_.begin(),
                            gen_code->upper_code_block_.end());

  block->lower_block.insert(block->lower_block.begin(),
                            gen_code->lower_code_block_.begin(),
                            gen_code->lower_code_block_.end());

  for (auto& variable : instr->getInputVariables(OCL_TARGET_CODE)) {
    kernel_input_vars_.insert(variable);
  }

  for (auto& variable : instr->getOutputVariables(OCL_TARGET_CODE)) {
    kernel_output_vars_.insert(variable);
  }

  kernel_init_vars_ << gen_code->init_variables_code_block_.str();
}

const std::map<std::string, std::string>
OCLProjectionSerialSinglePass::getInputVariables() const {
  auto merged_input_vars = kernel_input_vars_;

  for (const auto& var : kernel_input_vars_) {
    merged_input_vars.insert(var);
  }
  return merged_input_vars;
}

const std::map<std::string, std::string>
OCLProjectionSerialSinglePass::getOutputVariables() const {
  auto merged_output_vars = kernel_output_vars_;

  for (const auto& var : kernel_output_vars_) {
    merged_output_vars.insert(var);
  }
  return merged_output_vars;
}

const std::string OCLProjectionSerialSinglePass::getCodeCallComputeKernels(
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
  out << getCodeCallSerialSinglePassKernel(
      num_elements_for_loop, kernel_input_vars_, kernel_output_vars_);

  out << "_err = "
         "clEnqueueReadBuffer(ocl_getTransferDeviceToHostCommandQueue("
         "context), "
      << "cl_output_number_of_result_tuples"
      << ", CL_TRUE, 0,"
      << "sizeof (uint64_t), "
      << "&current_result_size"
      << ", "
      << "0, NULL, NULL);" << std::endl;
  out << "assert(_err == CL_SUCCESS);" << std::endl;

  /* copy data from device back to CPU main memory, if required */
  out << getCodeCopyOCLResultBuffersToHost("current_result_size", dev_type_,
                                           dev_id_, kernel_output_vars_);

  return out.str();
}
const std::string OCLProjectionSerialSinglePass::getKernelCode(
    const std::string& loop_variable_name, size_t global_worksize,
    MemoryAccessPattern mem_access_) {
  std::stringstream kernel;
  kernel << getCodeSerialSinglePassKernel(kernel_, loop_variable_name,
                                          kernel_input_vars_,
                                          kernel_output_vars_, mem_access_);

  return kernel.str();
}

const std::string OCLProjectionSerialSinglePass::getCodeCreateCustomStructures(
    const std::string& num_elements_for_loop, cl_device_type dev_type) {
  std::stringstream out;
  out << kernel_init_vars_.str();

  out << "cl_kernel kernel=NULL;" << std::endl;

  out << "cl_mem cl_output_number_of_result_tuples "
      << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE, "
      << "sizeof (uint64_t)"
      << ","
      << "NULL, &_err);" << std::endl;
  out << "assert(_err == CL_SUCCESS);" << std::endl;

  return out.str();
}

const std::string
OCLProjectionSerialSinglePass::getCodeCleanupCustomStructures() const {
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
