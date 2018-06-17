#include <query_compilation/execution_strategy/ocl_aggregation.h>

#include <core/variable_manager.hpp>

#include <boost/thread.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLAggregation::OCLAggregation(bool use_host_ptr,
                               MemoryAccessPattern mem_access,
                               cl_device_id dev_id)
    : OCL(use_host_ptr, mem_access, dev_id),
      aggregation_kernel_(boost::make_shared<GeneratedKernel>()) {}

void OCLAggregation::addInstruction_impl(InstructionPtr instr) {
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

  if (instr->getInstructionType() == HASH_TABLE_PROBE_INSTR ||
      instr->getInstructionType() == MAP_UDF_INSTR ||
      instr->getInstructionType() == PRODUCE_TUPLE_INSTR ||
      instr->getInstructionType() == MATERIALIZATION_INSTR ||
      instr->getInstructionType() == AGGREGATE_INSTR ||
      instr->getInstructionType() == ALGEBRA_INSTR ||
      instr->getInstructionType() == FILTER_INSTR) {
    CodeBlockPtr block = aggregation_kernel_->kernel_code_blocks.back();
    block->upper_block.insert(block->upper_block.end(),
                              gen_code->upper_code_block_.begin(),
                              gen_code->upper_code_block_.end());

    block->lower_block.insert(block->lower_block.begin(),
                              gen_code->lower_code_block_.begin(),
                              gen_code->lower_code_block_.end());

    for (auto& variable : instr->getInputVariables(OCL_TARGET_CODE)) {
      aggregation_kernel_input_vars_.insert(variable);
    }

    for (auto& variable : instr->getOutputVariables(OCL_TARGET_CODE)) {
      aggregation_kernel_output_vars_.insert(variable);

      if (instr->getInstructionType() == AGGREGATE_INSTR) {
        aggregation_kernel_aggregate_vars_.insert(variable);
      }
    }
  }
}

const std::pair<std::string, std::string> OCLAggregation::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  GeneratedCodePtr code = code_;
  assert(first_loop_ != NULL);

  std::string num_elements_for_loop = first_loop_->getVarNameNumberOfElements();

  bool cache_input_data = VariableManager::instance().getVariableValueBoolean(
      "code_gen.enable_caching");
  uint64_t global_worksize;

  if (dev_type_ == CL_DEVICE_TYPE_CPU) {
    global_worksize = boost::thread::hardware_concurrency();
  } else if (dev_type_ == CL_DEVICE_TYPE_GPU ||
             dev_type_ == CL_DEVICE_TYPE_ACCELERATOR) {
    cl_uint max_compute_units;
    clGetDeviceInfo(dev_id_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                    &max_compute_units, nullptr);

    const auto global_mutliplier = 10000u;
    global_worksize = max_compute_units * global_mutliplier;
  } else {
    COGADB_FATAL_ERROR(
        "Cannot determine global work size for unknown device type!", "");
  }

  if (!columns_to_decompress.empty()) {
    COGADB_FATAL_ERROR("Cannot work with uncompressed columns!", "");
  }

  std::stringstream kernel;
  kernel << getKernelHeadersAndTypes();
  kernel << code->kernel_header_and_types_code_block_.str();
  kernel << getCodeAggregationKernel(
      aggregation_kernel_, first_loop_->getLoopVariableName(), global_worksize,
      aggregation_kernel_input_vars_, aggregation_kernel_output_vars_,
      aggregation_kernel_aggregate_vars_, mem_access_);

  /* generate host code that prepares OpenCL buffers and launches the kernel */
  std::stringstream out;
  /* include the generated kernels in the host code as comment for easier
   * debugging, but do so only if we do not already print the generated code */
  if (VariableManager::instance().getVariableValueBoolean(
          "show_generated_kernel_code") == false &&
      VariableManager::instance().getVariableValueBoolean(
          "show_generated_code") == false) {
    out << "/*" << std::endl;
    out << kernel.str() << std::endl;
    out << "*/" << std::endl;
  }
  out << getDefaultIncludeHeaders();
  /* all imports and declarations */
  out << code->header_and_types_code_block_.str() << std::endl;
  /* write function signature */
  out << getFunctionSignature() + " {" << std::endl;
  out << code->fetch_input_code_block_.str() << std::endl;
  /* all code for query function definition and input array retrieval */
  out << code->declare_variables_code_block_.str() << std::endl;
  // TODO, we need to find a better place for this!
  out << "uint64_t current_result_size = 0;" << std::endl;
  out << "uint64_t allocated_result_elements = 10000;" << std::endl;
  out << "cl_int _err = 0;" << std::endl;
  out << code->init_variables_code_block_.str();
  /* generate code that retrieves number of elements from loop table */
  out << first_loop_->getNumberOfElementsExpression() << std::endl;

  out << getCodeCreateOCLInputBuffers(num_elements_for_loop,
                                      aggregation_kernel_input_vars_, dev_type_,
                                      dev_id_, cache_input_data)
      << std::endl;

  out << getCodeDeclareOCLAggregationResultBuffers(
      aggregation_kernel_output_vars_, aggregation_kernel_aggregate_vars_,
      global_worksize);

  out << getCodeInitOCLAggregationResultBuffers(
             "current_result_size", dev_type_, dev_id_,
             aggregation_kernel_output_vars_,
             aggregation_kernel_aggregate_vars_, global_worksize)
      << std::endl;

  /* generate code that calls OpenCL kernel that performs the aggregations */
  out << getCodeCallAggregationKernel(num_elements_for_loop, global_worksize,
                                      aggregation_kernel_input_vars_,
                                      aggregation_kernel_output_vars_);

  /* copy data from device back to CPU main memory, if required */
  out << getCodeCopyOCLAggregationResultBuffersToHost(
      "current_result_size", dev_type_, dev_id_,
      aggregation_kernel_output_vars_, aggregation_kernel_aggregate_vars_);

  out << getCoderReduceAggregation(aggregation_kernel_aggregate_vars_,
                                   global_worksize);

  /* if we do materialize into a hash table during aggregation,
     write copy result from hash table to output arrays */
  if (pipe_end == MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
    out << code->after_for_loop_code_block_.str();
  }
  /* clean up previously allocated OpenCL data structures */
  out << getCodeCleanupOCLStructures(aggregation_kernel_input_vars_,
                                     aggregation_kernel_output_vars_);

  out << getCodeCleanupOCLAggregationStructures(
      aggregation_kernel_aggregate_vars_);

  /* generate code that builds the result table using the minimal API */
  out << generateCCodeCreateResultTable(
             param, code->create_result_table_code_block_.str(),
             code->clean_up_code_block_.str(), result_table_name)
      << std::endl;

  out << "}" << std::endl;

  return std::make_pair(out.str(), kernel.str());
}

}  // namespace ExecutionStrategy

}  // namespace CoGaDB
