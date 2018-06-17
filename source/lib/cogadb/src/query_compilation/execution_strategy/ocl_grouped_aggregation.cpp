#include <query_compilation/execution_strategy/ocl_grouped_aggregation.h>

#include <core/variable_manager.hpp>
#include <query_compilation/primitives/aggregationPrimitives.hpp>

#include <boost/thread.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

const std::string OCLGroupedAggregation::AggregationHashTableVarName =
    "aggregation_hash_map";

const std::string OCLGroupedAggregation::InvalidKey = "0xFFFFFFFFFFFFFFFF";

OCLGroupedAggregation::OCLGroupedAggregation(bool use_host_ptr,
                                             MemoryAccessPattern mem_access,
                                             cl_device_id dev_id)
    : OCL(use_host_ptr, mem_access, dev_id),
      aggregation_kernel_(boost::make_shared<GeneratedKernel>()),
      sequential(false),
      hack_enable_manual_ht_size_(
          VariableManager::instance().getVariableValueBoolean(
              "code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_"
              "size")) {}

void OCLGroupedAggregation::addInstruction_impl(InstructionPtr instr) {
  if (instr->getInstructionType() == LOOP_INSTR) {
    first_loop_ = boost::dynamic_pointer_cast<Loop>(instr);
    assert(first_loop_ != nullptr);
  }

  GeneratedCodePtr gen_code = instr->getCode(OCL_TARGET_CODE);

  if (!gen_code) {
    COGADB_FATAL_ERROR("", "");
  }

  if (instr->getInstructionType() == HASH_AGGREGATE_INSTR) {
    auto hash_aggr = boost::static_pointer_cast<HashGroupAggregate>(instr);
    grouping_attrs_ = hash_aggr->getGroupingAttributes();
    aggr_specs_ = hash_aggr->getAggregateSpecifications();
    projection_param_ = hash_aggr->getProjectionParam();

    for (auto& variable : instr->getInputVariables(OCL_TARGET_CODE)) {
      aggregation_kernel_input_vars_.insert(variable);
    }
  } else if (!addToCode(code_, gen_code)) {
    COGADB_FATAL_ERROR("", "");
  }

  if (instr->getInstructionType() == GENERIC_GROUPING_KEY_INSTR) {
    COGADB_FATAL_ERROR("Generic grouping is not supported in OpenCL!", "");
  }

  if (instr->getInstructionType() == HASH_TABLE_PROBE_INSTR ||
      instr->getInstructionType() == MAP_UDF_INSTR ||
      instr->getInstructionType() == BITPACKED_GROUPING_KEY_INSTR ||
      instr->getInstructionType() == PRODUCE_TUPLE_INSTR ||
      instr->getInstructionType() == MATERIALIZATION_INSTR ||
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
    }
  }
}

const std::pair<std::string, std::string> OCLGroupedAggregation::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  GeneratedCodePtr code = code_;
  assert(first_loop_ != NULL);

  std::string num_elements_for_loop = first_loop_->getVarNameNumberOfElements();

  bool cache_input_data = VariableManager::instance().getVariableValueBoolean(
      "code_gen.enable_caching");

  uint64_t global_worksize, local_size;

  if (dev_type_ == CL_DEVICE_TYPE_CPU || dev_type_ == CL_DEVICE_TYPE_GPU ||
      dev_type_ == CL_DEVICE_TYPE_ACCELERATOR) {
    // As long as we are using a semaphore, we can not use a local size bigger
    // than one, otherwise the workgroup deadlocks and the whole kernel
    // deadlocks
    local_size = 1;

    auto max_compute_units = boost::compute::device(dev_id_).compute_units();

    auto multiplier = VariableManager::instance().getVariableValueInteger(
        "code_gen.opt.ocl_grouped_aggregation.global_size_multiplier");

    global_worksize =
        max_compute_units * local_size * static_cast<unsigned int>(multiplier);
  } else {
    COGADB_FATAL_ERROR(
        "Cannot determine global work size for unknown device type!", "");
  }

  if (!columns_to_decompress.empty()) {
    COGADB_FATAL_ERROR("Cannot work with uncompressed columns!", "");
  }

  // Overwrite device specific settings if the sequential strategy is selected
  if (VariableManager::instance().getVariableValueString(
          "code_gen.opt.ocl_grouped_aggregation_strategy") == "sequential") {
    sequential = true;
    global_worksize = 1;
    local_size = 1;
  }

  std::stringstream kernel;
  kernel << getKernelHeadersAndTypes();
  kernel << code->kernel_header_and_types_code_block_.str();
  kernel << getKernelCode(global_worksize);

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

  out << getAggregationPayloadStruct() << std::endl;

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

  out << getCreateAggregationHashMapBuffer() << std::endl;

  out << getCodeCreateOCLInputBuffers(num_elements_for_loop,
                                      aggregation_kernel_input_vars_, dev_type_,
                                      dev_id_, cache_input_data)
      << std::endl;

  out << getCodeDeclareOCLResultBuffers(aggregation_kernel_output_vars_);

  out << getCodeInitOCLResultBuffers("current_result_size", dev_type_, dev_id_,
                                     aggregation_kernel_output_vars_)
      << std::endl;

  /* generate code that calls OpenCL kernel that performs the aggregations */
  out << getCallKernels(global_worksize, local_size);

  /* copy data from device back to CPU main memory, if required */
  out << getCodeCopyOCLGroupedAggregationResultBuffersToHost(
      "current_result_size", dev_type_, dev_id_,
      aggregation_kernel_output_vars_, AggregationHashTableVarName);

  /* if we do materialize into a hash table during aggregation,
     write copy result from hash table to output arrays */
  if (pipe_end == MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
    out << code->after_for_loop_code_block_.str();
  }

  out << getCreateResult();

  /* clean up previously allocated OpenCL data structures */
  out << getCodeCleanupOCLStructures(aggregation_kernel_input_vars_,
                                     aggregation_kernel_output_vars_);

  out << "if(" << getCLBufferResultVarName(AggregationHashTableVarName) << ") {"
      << std::endl
      << "CL_CHECK(clReleaseMemObject("
      << getCLBufferResultVarName(AggregationHashTableVarName) << "));"
      << std::endl
      << getCLBufferResultVarName(AggregationHashTableVarName) << " = NULL;"
      << std::endl
      << "}" << std::endl;

  out << "free(" << AggregationHashTableVarName << ");" << std::endl;

  /* generate code that builds the result table using the minimal API */
  out << generateCCodeCreateResultTable(
             param, code->create_result_table_code_block_.str(),
             code->clean_up_code_block_.str(), result_table_name)
      << std::endl;

  out << "}" << std::endl;

  return std::make_pair(out.str(), kernel.str());
}

std::string OCLGroupedAggregation::getKernelCode(uint64_t global_worksize) {
  std::stringstream ss;

  ss << getAggregationPayloadStruct();

  if (!sequential) {
    ss << "void lock_payload(__global AggregationPayload* payload) {"
       << std::endl
       << "  while (atomic_xchg(&payload->semaphore, 0) == 0);" << std::endl
       << "}" << std::endl
       << "void unlock_payload(__global AggregationPayload* payload) {"
       << std::endl
       << "  atomic_xchg(&payload->semaphore, 1);" << std::endl
       << "}" << std::endl;
  }

  ss << "uint64_t getPayloadIndex(uint64_t key, "
        "__global AggregationPayload* hash_map, const uint64_t hash_map_mask) {"
     << std::endl
     << "  uint64_t index = 0; " << std::endl
     << getCodePayloadIndexFunctionImpl() << std::endl
     << "}" << std::endl;

  ss << getCodeKernelSignature(
      "grouped_aggregation_init_kernel",
      "__global AggregationPayload* hash_map, uint64_t num_elements");
  ss << " {" << std::endl;

  auto loop_code = getCodeInitKernelIndexCalculation(global_worksize);

  ss << loop_code.first;

  for (const auto& aggr_spec : aggr_specs_) {
    ss << aggr_spec->getCodeInitializeAggregationPayloadFields(
        "hash_map[table_index].");
  }

  ss << "hash_map[table_index]."
     << "key = " << InvalidKey << ";" << std::endl;
  if (!sequential) {
    ss << "hash_map[table_index].semaphore = 1;" << std::endl;
  }

  ss << loop_code.second;

  ss << "}" << std::endl;

  ss << getCodeKernelSignature("grouped_aggregation_kernel",
                               "__global AggregationPayload* hash_map, const "
                               "uint64_t hash_map_mask,"
                               " uint64_t num_elements",
                               aggregation_kernel_input_vars_,
                               aggregation_kernel_output_vars_)
     << " {" << std::endl;

  loop_code = getCodeOCLKernelMemoryTraversal(
      first_loop_->getLoopVariableName(), mem_access_);

  ss << loop_code.first;

  for (const auto& code_block : aggregation_kernel_->kernel_code_blocks) {
    for (const auto& upper : code_block->upper_block) {
      ss << upper << std::endl;
    }
  }

  ss << "  uint64_t payload_index = getPayloadIndex(group_key, hash_map, "
     << "hash_map_mask);" << std::endl
     << getAggregationCode(grouping_attrs_, aggr_specs_,
                           "hash_map[payload_index].")
     << std::endl;
  if (!sequential) {
    ss << "unlock_payload(&hash_map[payload_index]);" << std::endl;
  }
  ss << std::endl;

  for (const auto& code_block : aggregation_kernel_->kernel_code_blocks) {
    for (const auto& lower : code_block->lower_block) {
      ss << lower << std::endl;
    }
  }

  ss << loop_code.second;

  ss << "}" << std::endl << std::endl;

  return ss.str();
}

std::pair<std::string, std::string>
OCLGroupedAggregation::getCodeInitKernelIndexCalculation(
    uint64_t global_worksize) {
  if (dev_type_ == CL_DEVICE_TYPE_GPU) {
    return std::make_pair("uint64_t table_index = get_global_id(0);", "");
  } else {
    return getCodeOCLKernelMemoryTraversal("table_index", mem_access_);
  }
}

std::string OCLGroupedAggregation::getCodeCheckKeyInHashTable() {
  const std::string hash_map_access = "hash_map[index]";
  std::stringstream ss;

  ss << "  index ";
  if (hack_enable_manual_ht_size_) {
    ss << "%= hash_map_mask + 1;";
  } else {
    ss << "&= hash_map_mask;";
  }

  if (!sequential) {
    ss << "  lock_payload(&" << hash_map_access << ");" << std::endl;
  }
  ss << "  if (" << hash_map_access << ".key == key) {" << std::endl
     << "    return index;" << std::endl
     << "  } else if (" << hash_map_access << ".key == " << InvalidKey << ") {"
     << std::endl
     << "    " << hash_map_access << ".key = key;" << std::endl
     << "    return index;" << std::endl
     << "  }" << std::endl;
  if (!sequential) {
    ss << "  unlock_payload(&" << hash_map_access << ");" << std::endl;
  }

  return ss.str();
}

std::string OCLGroupedAggregation::getAggregationCode(
    const GroupingAttributes& grouping_columns,
    const AggregateSpecifications& aggregation_specs,
    const std::string access_ht_entry_expression) {
  return getAggregationCodeGeneric(grouping_columns, aggregation_specs,
                                   access_ht_entry_expression);
}

std::string OCLGroupedAggregation::getAggregationPayloadStruct() {
  std::stringstream ss;

  ss << "struct AggregationPayload {" << std::endl;

  ss << getAggregationPayloadCodeForGroupingAttributes(grouping_attrs_);

  std::set<AggregateSpecification::AggregationPayloadField> struct_fields;

  for (const auto& aggr_spec : aggr_specs_) {
    auto& payload_fields = aggr_spec->getAggregationPayloadFields();
    struct_fields.insert(payload_fields.begin(), payload_fields.end());
  }

  for (const auto& struct_field : struct_fields) {
    ss << struct_field << ";" << std::endl;
  }

  ss << "uint64_t key;" << std::endl;
  if (!sequential) {
    ss << "uint32_t semaphore;" << std::endl;
  }

  ss << "};" << std::endl;
  ss << "typedef struct AggregationPayload AggregationPayload;" << std::endl;

  return ss.str();
}

std::string OCLGroupedAggregation::getCreateAggregationHashMapBuffer() {
  std::stringstream ss;

  const std::string hash_map_length = AggregationHashTableVarName + "_length";

  auto hashtable_size_multiplier =
      VariableManager::instance().getVariableValueFloat(
          "code_gen.opt.ocl_grouped_aggregation.hashtable_size_multiplier");

  ss << "uint64_t " << hash_map_length << " = 1;" << std::endl
     << "while(" << hash_map_length << " < "
     << first_loop_->getTable()->getNumberofRows() << " * "
     << hashtable_size_multiplier << ") {" << std::endl
     << "  " << hash_map_length << " <<= 1;" << std::endl
     << "}" << std::endl;

  if (hack_enable_manual_ht_size_) {
    auto manual_size = VariableManager::instance().getVariableValueInteger(
        "code_gen.opt.ocl_grouped_aggregation.hack.ht_size");
    ss << hash_map_length << " = " << manual_size << ";" << std::endl;
  }

  ss << "uint64_t " << AggregationHashTableVarName
     << "_mask = " << hash_map_length << " - 1;" << std::endl;

  ss << "AggregationPayload* " << AggregationHashTableVarName << " = malloc("
     << hash_map_length << " * sizeof(AggregationPayload));" << std::endl;

  const auto var =
      std::make_pair(AggregationHashTableVarName, "AggregationPayload*");

  ss << getCodeDeclareOCLResultBuffer(var);

  ss << getCodeInitOCLResultBuffer(dev_type_, dev_id_, var, false, "", true);

  return ss.str();
}

std::string OCLGroupedAggregation::getCallKernels(uint64_t global_worksize,
                                                  uint64_t local_worksize) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  const std::string hash_map_length = AggregationHashTableVarName + "_length";
  const std::string hash_map_mask = AggregationHashTableVarName + "_mask";

  out << "cl_kernel grouped_aggregation_init_kernel = NULL;" << std::endl;
  out << "grouped_aggregation_init_kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"grouped_aggregation_init_kernel\", &_err);"
      << std::endl;

  out << "cl_event kernel_completion;" << std::endl;
  out << "size_t global_work_size = ";

  if (dev_type_ == CL_DEVICE_TYPE_GPU) {
    out << hash_map_length;
  } else {
    out << global_worksize;
  }

  out << ";" << std::endl;

  out << "CL_CHECK(clSetKernelArg(grouped_aggregation_init_kernel, 0, "
      << "sizeof (cl_mem), &"
      << getCLBufferResultVarName(AggregationHashTableVarName) << "));"
      << std::endl;

  out << "CL_CHECK(clSetKernelArg(grouped_aggregation_init_kernel, 1, "
      << "sizeof (" << hash_map_length << "), &" << hash_map_length << "));"
      << std::endl;

  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "grouped_aggregation_init_kernel, 1, NULL, &global_work_size, "
         "NULL, 0, NULL, &kernel_completion));"
      << std::endl;

  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
  out << getCodeReleaseKernel("grouped_aggregation_init_kernel");

  out << "cl_kernel grouped_aggregation_kernel = NULL;" << std::endl;
  out << "grouped_aggregation_kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"grouped_aggregation_kernel\", &_err);"
      << std::endl;

  out << "CL_CHECK(clSetKernelArg(grouped_aggregation_kernel, 0, "
      << "sizeof (cl_mem), &"
      << getCLBufferResultVarName(AggregationHashTableVarName) << "));"
      << std::endl;

  out << "CL_CHECK(clSetKernelArg(grouped_aggregation_kernel, 1, "
      << "sizeof (" << hash_map_mask << "), &" << hash_map_mask << "));"
      << std::endl;

  out << "CL_CHECK(clSetKernelArg(grouped_aggregation_kernel, 2, "
      << "sizeof (" << first_loop_->getVarNameNumberOfElements() << "), &"
      << first_loop_->getVarNameNumberOfElements() << "));" << std::endl;

  unsigned int index = 3;
  for (const auto& var : aggregation_kernel_input_vars_) {
    auto var_name = isPointerType(var.second)
                        ? getCLBufferInputVarName(var.first)
                        : var.first;
    out << "CL_CHECK(clSetKernelArg(grouped_aggregation_kernel, " << index++
        << ", sizeof(" << var_name << ") , &" << var_name << "));" << std::endl;
  }

  for (const auto& var : aggregation_kernel_output_vars_) {
    out << "CL_CHECK(clSetKernelArg(grouped_aggregation_kernel, " << index++
        << ", sizeof(" << getCLBufferResultVarName(var.first) << ") , &"
        << getCLBufferResultVarName(var.first) << "));" << std::endl;
  }

  out << "{" << std::endl;

  out << "global_work_size = " << global_worksize << ";" << std::endl;
  out << "size_t local_work_size = " << local_worksize << ";" << std::endl;

  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "grouped_aggregation_kernel, 1, NULL, &global_work_size, "
         "&local_work_size, 0, NULL, &kernel_completion));"
      << std::endl;
  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
  out << getCodeReleaseKernel("grouped_aggregation_kernel");
  out << "}" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Grouped Aggregation Kernel\");" << std::endl;
#endif
  return out.str();
}

std::string OCLGroupedAggregation::getCreateResult() {
  const std::string hash_map_access = AggregationHashTableVarName + "[i].";

  std::stringstream ss;

  ss << getCodeDeclareResultMemory(projection_param_, true);
  ss << getCodeMallocResultMemory(projection_param_, true);

  ss << "for(uint64_t i = 0; i < " << AggregationHashTableVarName << "_length;"
     << "++i) {" << std::endl;
  ss << "  if (" << hash_map_access << "key != " << InvalidKey << ") {"
     << std::endl;

  for (const auto& aggr_spec : aggr_specs_) {
    ss << aggr_spec->getCodeFetchResultsFromHashTableEntry(hash_map_access);
  }

  ss << getCodeProjectGroupingColumnsFromHashTable(grouping_attrs_,
                                                   hash_map_access);

  ss << "    ++current_result_size;" << std::endl;

  ss << "  }" << std::endl;

  ss << "if (current_result_size >= allocated_result_elements) {" << std::endl;
  ss << "   allocated_result_elements *= 1.4;" << std::endl;
  ss << getCodeReallocResultMemory(projection_param_, true);
  ss << "}" << std::endl;

  ss << "}" << std::endl;

  return ss.str();
}

std::string OCLGroupedAggregation::getCodePayloadIndexFunctionImpl() {
  const auto impl_name = VariableManager::instance().getVariableValueString(
      "code_gen.opt.ocl_grouped_aggregation_hashtable");

  if (impl_name == "linear_probing") {
    return getCodePayloadIndexFunctionLinearProbingImpl();
  } else if (impl_name == "quadratic_probing") {
    return getCodePayloadIndexFunctionQuadraticProbingImpl();
  } else if (impl_name == "cuckoo_hashing") {
    return getCodePayloadIndexFunctionCuckooHashingImpl();
  } else {
    COGADB_FATAL_ERROR(
        "Unknown opencl grouped aggregation implementation "
        "selected! Selected: "
            << impl_name,
        "");
  }
}

std::string OCLGroupedAggregation::getCodeMurmur3Hashing() {
  std::stringstream ss;

  ss << "  index ^= index >> 33;" << std::endl
     << "  index *= 0xff51afd7ed558ccd;" << std::endl
     << "  index ^= index >> 33;" << std::endl
     << "  index *= 0xc4ceb9fe1a85ec53;" << std::endl
     << "  index ^= index >> 33;" << std::endl;

  return ss.str();
}

std::string
OCLGroupedAggregation::getCodePayloadIndexFunctionLinearProbingImpl() {
  std::stringstream ss;

  std::string hash_func_code = "";
  const auto hash_func_name =
      VariableManager::instance().getVariableValueString(
          "code_gen.opt.ocl_grouped_aggregation.hash_function");

  if (hash_func_name == "multiply_shift") {
    hash_func_code = "index * 123456789123456789ul;";
  } else if (hash_func_name == "murmur3") {
    hash_func_code = getCodeMurmur3Hashing();
  }

  ss << "  index = key;" << std::endl
     << hash_func_code << "  while (1) {" << std::endl
     << getCodeCheckKeyInHashTable() << "    index += 1; " << std::endl
     << "  }" << std::endl;

  return ss.str();
}

std::string
OCLGroupedAggregation::getCodePayloadIndexFunctionQuadraticProbingImpl() {
  std::stringstream ss;

  ss << "  index = key;" << std::endl
     << getCodeMurmur3Hashing() << "  uint64_t displacement = 0;" << std::endl
     << "  while (1) {" << std::endl
     << getCodeCheckKeyInHashTable() << "    index += displacement >> 1 + "
     << "(displacement >> 1) * (displacement >> 1);" << std::endl
     << "    displacement += 1; " << std::endl
     << "  }" << std::endl;

  return ss.str();
}

std::string
OCLGroupedAggregation::getCodePayloadIndexFunctionCuckooHashingImpl() {
  std::stringstream ss;

  std::string hash_func_code = "", hash_func_code2 = "";
  const auto hash_func_name =
      VariableManager::instance().getVariableValueString(
          "code_gen.opt.ocl_grouped_aggregation.hash_function");

  if (hash_func_name == "multiply_shift") {
    hash_func_code = "index * 123456789123456789ul;";
    hash_func_code2 = getCodeMurmur3Hashing();
  } else if (hash_func_name == "murmur3") {
    hash_func_code = getCodeMurmur3Hashing();
    hash_func_code2 = "index * 123456789123456789ul;";
  }

  ss << "  index = key;" << std::endl
     << hash_func_code << getCodeCheckKeyInHashTable() << std::endl;

  // use multiply shift hashing
  ss << "  index = key;" << std::endl
     << "  index * 123456789123456789ul;" << std::endl
     << getCodeCheckKeyInHashTable() << std::endl;

  // use multiply add shift hashing
  ss << "  index = key;" << std::endl
     << "  index * 789 + 321;" << std::endl
     << getCodeCheckKeyInHashTable() << std::endl;

  // fallback to linear probing
  ss << getCodePayloadIndexFunctionLinearProbingImpl();

  return ss.str();
}

}  // namespace ExecutionStrategy

}  // namespace CoGaDB
