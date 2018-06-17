#include <boost/thread.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/execution_strategy/ocl_projection.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLProjection::OCLProjection(bool use_host_ptr, MemoryAccessPattern mem_access,
                             cl_device_id dev_id)
    : OCL(use_host_ptr, mem_access, dev_id), first_loop_() {}

uint64_t OCLProjection::getGlobalSize(cl_device_type dev_type,
                                      size_t num_elements) {
  uint64_t global_worksize, local_size;
  if (dev_type == CL_DEVICE_TYPE_CPU || dev_type == CL_DEVICE_TYPE_GPU ||
      dev_type == CL_DEVICE_TYPE_ACCELERATOR) {
    local_size = 1;

    auto max_compute_units = boost::compute::device(dev_id_).compute_units();
    auto multiplier = VariableManager::instance().getVariableValueInteger(
        "code_gen.projection.global_size_multiplier");

    global_worksize =
        max_compute_units * local_size * static_cast<unsigned int>(multiplier);
  } else {
    COGADB_FATAL_ERROR(
        "Cannot determine global work size for unknown device type!", "");
  }

  if (global_worksize > num_elements) {
    global_worksize = num_elements;
  }

  return global_worksize;
}

const std::pair<std::string, std::string> OCLProjection::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  GeneratedCodePtr code = code_;
  assert(first_loop_ != NULL);
  std::string num_elements_for_loop = first_loop_->getVarNameNumberOfElements();
  uint64_t global_worksize =
      getGlobalSize(dev_type_, first_loop_->getNumberOfElements());

  // std::cout << "default_hash_table: " <<
  // VariableManager::instance().getVariableValueString("default_hash_table") <<
  // std::endl;
  std::string ht_implementation =
      VariableManager::instance().getVariableValueString("default_hash_table");

  if (pipe_end != MATERIALIZE_FROM_ARRAY_TO_ARRAY) {
    if ((ht_implementation != "ocl_linear_probing") &&
        (ht_implementation != "ocl_cuckoo2hashes") &&
        (ht_implementation != "ocl_seeded_linear_probing")) {
      global_worksize = 1;
    }
  }

  if (!columns_to_decompress.empty()) {
    COGADB_FATAL_ERROR("Cannot work with uncompressed columns!", "");
  }

  bool cache_input_data = VariableManager::instance().getVariableValueBoolean(
      "code_gen.enable_caching");

  /* generate OpenCL kernel */
  std::stringstream kernel;
  kernel << getKernelHeadersAndTypes();
  kernel << code->kernel_header_and_types_code_block_.str();
  kernel << getKernelCode(first_loop_->getLoopVariableName(), global_worksize,
                          mem_access_);
  /* generate host code that prepares OpenCL buffers and launches the kernel */
  std::stringstream out;
  out << getDefaultIncludeHeaders();
  /* all imports and declarations */
  out << code->header_and_types_code_block_.str() << std::endl;
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
  /* write function signature */
  out << getFunctionSignature() + " {" << std::endl;
  out << code->fetch_input_code_block_.str() << std::endl;
  /* all code for query function definition and input array retrieval */
  out << code->declare_variables_code_block_.str() << std::endl;
  /* generate code that retrieves number of elements from loop table */
  out << first_loop_->getNumberOfElementsExpression() << std::endl;
  // TODO, we need to find a better place for this!
  out << "uint64_t current_result_size = 0;" << std::endl;

  out << "uint64_t allocated_result_elements = "
      << first_loop_->getVarNameNumberOfElements() << ";" << std::endl;

  out << "cl_int _err = 0;" << std::endl;
  /* create additional data structures required by execution strategy */
  out << this->getCodeCreateCustomStructures(num_elements_for_loop, dev_type_);

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

}  // namespace ExecutionStrategy

}  // namespace CoGaDB
