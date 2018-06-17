#include <query_compilation/execution_strategy/ocl_grouped_aggregation_atomic_workgroup_ht.h>

#include <core/variable_manager.hpp>
#include <query_compilation/primitives/aggregationPrimitives.hpp>

#include <boost/thread.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

const std::string getGroupingAttributePayloadVarName(
    const AttributeReference& attr) {
  if (isComputed(attr)) {
    return getVarName(attr);
  } else {
    return getGroupTIDVarName(attr);
  }
}

OCLGroupedAggregationAtomicWorkGroupHT::OCLGroupedAggregationAtomicWorkGroupHT(
    bool use_host_ptr, MemoryAccessPattern mem_access, cl_device_id dev_id)
    : OCLGroupedAggregationAtomic(use_host_ptr, mem_access, dev_id) {}

void OCLGroupedAggregationAtomicWorkGroupHT::addInstruction_impl(
    InstructionPtr instr) {
  if (instr->getInstructionType() == LOOP_INSTR) {
    first_loop_ = boost::dynamic_pointer_cast<Loop>(instr);
    assert(first_loop_ != nullptr);
  }

  enableIntelAggregationHACK(atomics_32bit_);

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

  enableIntelAggregationHACK(false);
}

const std::pair<std::string, std::string>
OCLGroupedAggregationAtomicWorkGroupHT::getCode(
    const ProjectionParam& param, const ScanParam&, PipelineEndType pipe_end,
    const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  GeneratedCodePtr code = code_;
  assert(first_loop_ != NULL);
  enableIntelAggregationHACK(atomics_32bit_);

  std::string num_elements_for_loop = first_loop_->getVarNameNumberOfElements();

  bool cache_input_data = VariableManager::instance().getVariableValueBoolean(
      "code_gen.enable_caching");

  uint64_t global_worksize, local_worksize;

  if (dev_type_ == CL_DEVICE_TYPE_CPU || dev_type_ == CL_DEVICE_TYPE_GPU ||
      dev_type_ == CL_DEVICE_TYPE_ACCELERATOR) {
    auto max_compute_units = boost::compute::device(dev_id_).compute_units();

    auto multiplier = VariableManager::instance().getVariableValueInteger(
        "code_gen.opt.ocl_grouped_aggregation.global_size_multiplier");

    global_worksize = max_compute_units * static_cast<unsigned int>(multiplier);
  } else {
    COGADB_FATAL_ERROR(
        "Cannot determine global work size for unknown device type!", "");
  }

  local_worksize = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_grouped_aggregation.atomic.workgroup.local_size");

  if (!columns_to_decompress.empty()) {
    COGADB_FATAL_ERROR("Cannot work with uncompressed columns!", "");
  }

  global_worksize = adjustGlobalWorkSize(global_worksize);

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

  out << getCreateAggregationHashMapBuffer(global_worksize) << std::endl;

  out << getCodeCreateOCLInputBuffers(num_elements_for_loop,
                                      aggregation_kernel_input_vars_, dev_type_,
                                      dev_id_, cache_input_data)
      << std::endl;

  out << getCodeDeclareOCLResultBuffers(aggregation_kernel_output_vars_);

  out << getCodeInitOCLResultBuffers("current_result_size", dev_type_, dev_id_,
                                     aggregation_kernel_output_vars_)
      << std::endl;

  /* generate code that calls OpenCL kernel that performs the aggregations */
  out << getCallKernels(global_worksize, local_worksize);

  /* copy data from device back to CPU main memory, if required */
  out << getCodeCopyOCLGroupedAggregationResultBuffersToHost(
      "current_result_size", dev_type_, dev_id_,
      aggregation_kernel_output_vars_, AggregationHashTableVarName);

  /* if we do materialize into a hash table during aggregation,
     write copy result from hash table to output arrays */
  if (pipe_end == MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
    out << code->after_for_loop_code_block_.str();
  }

  out << getCreateResult(global_worksize);

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

  enableIntelAggregationHACK(false);

  return std::make_pair(out.str(), kernel.str());
}

std::string OCLGroupedAggregationAtomicWorkGroupHT::getKernelCode(
    uint64_t global_worksize) {
  std::stringstream ss;

  ss << getAggregationPayloadStruct();

  if (atomics_32bit_) {
    get32BitAtomics(ss);
  } else {
    get64BitAtomics(ss);
  }

  std::string index_type = atomics_32bit_ ? "uint32_t" : "uint64_t";

  ss << index_type << " getPayloadIndex(" << index_type
     << " key, "
        "__global AggregationPayload* hash_map, const uint64_t hash_map_mask, "
        "const uint64_t hash_table_offset) {"
     << std::endl
     << "  " << index_type << " index = 0; " << std::endl
     << "  " << index_type << " real_index = 0; " << std::endl
     << "  " << index_type << " old = 0;" << std::endl
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
     << "key = " << (atomics_32bit_ ? InvalidKey32Bit : InvalidKey) << ";"
     << std::endl;
  ss << loop_code.second;

  ss << "}" << std::endl;

  ss << getCodeKernelSignature("grouped_aggregation_kernel",
                               "__global AggregationPayload* hash_map, const "
                               "uint64_t hash_map_mask,"
                               " uint64_t num_elements",
                               aggregation_kernel_input_vars_,
                               aggregation_kernel_output_vars_)
     << " {" << std::endl;

  ss << "uint64_t hash_map_offset = get_group_id(0) * (hash_map_mask + 1);"
     << std::endl;

  loop_code = getCodeOCLKernelMemoryTraversal(
      first_loop_->getLoopVariableName(), mem_access_);

  ss << loop_code.first;

  for (const auto& code_block : aggregation_kernel_->kernel_code_blocks) {
    for (const auto& upper : code_block->upper_block) {
      ss << upper << std::endl;
    }
  }

  ss << "  uint64_t payload_index = getPayloadIndex(group_key, hash_map, "
     << "hash_map_mask, hash_map_offset);" << std::endl
     << getAggregationCode(grouping_attrs_, aggr_specs_,
                           "hash_map[payload_index].")
     << std::endl;

  for (const auto& code_block : aggregation_kernel_->kernel_code_blocks) {
    for (const auto& lower : code_block->lower_block) {
      ss << lower << std::endl;
    }
  }

  ss << loop_code.second;

  ss << "if (get_group_id(0) != 0) {" << std::endl;
  ss << "  barrier(CLK_GLOBAL_MEM_FENCE);" << std::endl;
  ss << "  for (unsigned int i = hash_map_offset + get_local_id(0); "
     << "i <= hash_map_offset + hash_map_mask; i += get_local_size(0)) {"
     << std::endl
     << "    if (hash_map[i].key != "
     << (atomics_32bit_ ? InvalidKey32Bit : InvalidKey) << ") {" << std::endl
     << "      uint64_t payload_index = getPayloadIndex(hash_map[i].key, "
     << "hash_map, hash_map_mask, 0);" << std::endl
     << getCopyHashTableEntryCode(grouping_attrs_, aggr_specs_,
                                  "hash_map[payload_index].", "hash_map[i].")
     << "  }" << std::endl
     << "}" << std::endl;

  ss << "}" << std::endl;

  ss << "}" << std::endl << std::endl;

  return ss.str();
}

std::string
OCLGroupedAggregationAtomicWorkGroupHT::getCodeCheckKeyInHashTable() {
  const std::string hash_map_access = "hash_map[real_index]";
  std::stringstream ss;
  const std::string atomic_cmpxchg =
      atomics_32bit_ ? "atomic_cmpxchg" : "atom_cmpxchg";

  ss << "  index ";
  if (hack_enable_manual_ht_size_) {
    ss << "%= hash_map_mask + 1;";
  } else {
    ss << "&= hash_map_mask;";
  }

  ss << std::endl;

  ss << "  real_index = index + hash_table_offset;" << std::endl
     << "  old = " << hash_map_access << ".key;" << std::endl
     << "  if (old == key) {" << std::endl
     << "    return real_index;" << std::endl
     << "  } else if (old == "
     << (atomics_32bit_ ? InvalidKey32Bit : InvalidKey) << ") {" << std::endl
     << "    old = " << atomic_cmpxchg << "(&" << hash_map_access
     << ".key, old, key);" << std::endl
     << "    if (old == " << (atomics_32bit_ ? InvalidKey32Bit : InvalidKey)
     << " || old == key) {" << std::endl
     << "      return real_index;" << std::endl
     << "    }" << std::endl
     << "  }" << std::endl;

  return ss.str();
}

uint64_t OCLGroupedAggregationAtomicWorkGroupHT::adjustGlobalWorkSize(
    uint64_t global_worksize) {
  auto max_global_mem_usage =
      boost::compute::device(dev_id_).max_memory_alloc_size() / 2.0;

  auto hashtable_size_multiplier =
      VariableManager::instance().getVariableValueFloat(
          "code_gen.opt.ocl_grouped_aggregation.hashtable_size_multiplier");

  if (estimateHashTableSize(global_worksize, hashtable_size_multiplier) >
      max_global_mem_usage) {
    COGADB_ERROR(
        "GlobalWorkSize to big, decreasing so that the hashtable fits "
        "into the device memory.",
        "");

    while (estimateHashTableSize(global_worksize, hashtable_size_multiplier) >
           max_global_mem_usage) {
      global_worksize /= 2;
    }
  }

  return global_worksize;
}

double OCLGroupedAggregationAtomicWorkGroupHT::estimateHashTableSize(
    uint64_t global_worksize, float hashtable_size_multiplier) {
  auto estimated_hashtable_size =
      first_loop_->getNumberOfElements() * hashtable_size_multiplier;

  if (hack_enable_manual_ht_size_) {
    estimated_hashtable_size =
        VariableManager::instance().getVariableValueInteger(
            "code_gen.opt.ocl_grouped_aggregation.hack.ht_size");
  }

  estimated_hashtable_size *= global_worksize;

  auto element_size = (atomics_32bit_ ? 4 : 8);

  estimated_hashtable_size *=
      (aggr_specs_.size() * element_size + element_size +
       sizeof(TID) * grouping_attrs_.size());

  return estimated_hashtable_size;
}

std::string OCLGroupedAggregationAtomicWorkGroupHT::getCopyHashTableEntryCode(
    const GroupingAttributes& grouping_columns,
    const AggregateSpecifications& aggregation_specs,
    const std::string access_src_ht_entry_expression,
    const std::string access_dst_ht_entry_expression) {
  std::stringstream hash_aggregate;
  for (size_t i = 0; i < aggregation_specs.size(); ++i) {
    hash_aggregate << aggregation_specs[i]->getCodeCopyHashTableEntry(
        access_src_ht_entry_expression, access_dst_ht_entry_expression);
    for (size_t i = 0; i < grouping_columns.size(); ++i) {
      AttributeReference attr_ref = grouping_columns[i];
      hash_aggregate << access_src_ht_entry_expression
                     << getGroupingAttributePayloadVarName(attr_ref) << " = "
                     << access_dst_ht_entry_expression
                     << getGroupingAttributePayloadVarName(attr_ref) << ";"
                     << std::endl;
    }
  }
  return hash_aggregate.str();
}

std::string
OCLGroupedAggregationAtomicWorkGroupHT::getCreateAggregationHashMapBuffer(
    uint64_t global_worksize) {
  std::stringstream ss;

  auto hashtable_size_multiplier =
      VariableManager::instance().getVariableValueFloat(
          "code_gen.opt.ocl_grouped_aggregation.hashtable_size_multiplier");

  ss << "uint64_t " << AggregationHashTableLengthVarName << " = 1;" << std::endl
     << "while(" << AggregationHashTableLengthVarName << " < "
     << first_loop_->getTable()->getNumberofRows() << " * "
     << hashtable_size_multiplier << ") {" << std::endl
     << "  " << AggregationHashTableLengthVarName << " <<= 1;" << std::endl
     << "}" << std::endl;

  if (hack_enable_manual_ht_size_) {
    auto manual_size = VariableManager::instance().getVariableValueInteger(
        "code_gen.opt.ocl_grouped_aggregation.hack.ht_size");
    ss << AggregationHashTableLengthVarName << " = " << manual_size << ";"
       << std::endl;
  }

  ss << "uint64_t " << AggregationHashTableVarName
     << "_mask = " << AggregationHashTableLengthVarName << " - 1;" << std::endl;

  ss << AggregationHashTableLengthVarName << " *= " << global_worksize << ";"
     << std::endl;

  ss << "AggregationPayload* " << AggregationHashTableVarName << " = malloc("
     << AggregationHashTableLengthVarName << " * sizeof(AggregationPayload));"
     << std::endl;

  const auto var =
      std::make_pair(AggregationHashTableVarName, "AggregationPayload*");

  ss << getCodeDeclareOCLResultBuffer(var);

  ss << getCodeInitOCLResultBuffer(dev_type_, dev_id_, var);

  return ss.str();
}

std::string OCLGroupedAggregationAtomicWorkGroupHT::getAggKernelGlobalLocalSize(
    uint64_t global_worksize, uint64_t local_worksize) {
  std::stringstream out;

  out << "local_work_size = " << local_worksize << ";" << std::endl;

  out << "size_t local_work_size_tmp = 0;" << std::endl;
  out << "clGetKernelWorkGroupInfo(grouped_aggregation_kernel, "
      << "ocl_getDeviceID(context), CL_KERNEL_WORK_GROUP_SIZE, "
      << "sizeof(size_t), &local_work_size_tmp, NULL);" << std::endl;
  out << "if (local_work_size_tmp < local_work_size) {" << std::endl
      << "printf(\"OCLGroupedAggregationAtomicWorkGroupHT changing workgroup "
      << "size to: %lu\\n\", local_work_size_tmp);"
      << "  local_work_size = local_work_size_tmp;" << std::endl
      << "}" << std::endl;

  out << "global_work_size = " << global_worksize << " * local_work_size;"
      << std::endl;
  out << "global_work_size_ptr = &global_work_size;" << std::endl;
  out << "local_work_size_ptr = &local_work_size;" << std::endl;

  return out.str();
}

std::string OCLGroupedAggregationAtomicWorkGroupHT::getInitKernelHashMapSize(
    uint64_t global_worksize) {
  return AggregationHashTableLengthVarName;
}

}  // namespace ExecutionStrategy

}  // namespace CoGaDB
