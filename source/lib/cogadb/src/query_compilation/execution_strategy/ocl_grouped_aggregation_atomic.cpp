#include <query_compilation/execution_strategy/ocl_grouped_aggregation_atomic.h>

#include <core/variable_manager.hpp>
#include <query_compilation/primitives/aggregationPrimitives.hpp>

#include <boost/thread.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

const std::string OCLGroupedAggregationAtomic::AggregationHashTableVarName =
    "aggregation_hash_map";

const std::string
    OCLGroupedAggregationAtomic::AggregationHashTableLengthVarName =
        "aggregation_hash_map_length";

const std::string OCLGroupedAggregationAtomic::InvalidKey =
    "0xFFFFFFFFFFFFFFFF";
const std::string OCLGroupedAggregationAtomic::InvalidKey32Bit = "0xFFFFFFFF";

OCLGroupedAggregationAtomic::OCLGroupedAggregationAtomic(
    bool use_host_ptr, MemoryAccessPattern mem_access, cl_device_id dev_id)
    : OCL(use_host_ptr, mem_access, dev_id),
      aggregation_kernel_(boost::make_shared<GeneratedKernel>()),
      // We want 32bit for all devices
      atomics_32bit_(VariableManager::instance().getVariableValueBoolean(
          "code_gen.cl_properties.use_32bit_atomics")),
      hack_enable_manual_ht_size_(
          VariableManager::instance().getVariableValueBoolean(
              "code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_"
              "size")) {}

void OCLGroupedAggregationAtomic::addInstruction_impl(InstructionPtr instr) {
  enableIntelAggregationHACK(atomics_32bit_);

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

  enableIntelAggregationHACK(false);
}

const std::pair<std::string, std::string> OCLGroupedAggregationAtomic::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  GeneratedCodePtr code = code_;
  assert(first_loop_ != NULL);

  enableIntelAggregationHACK(atomics_32bit_);

  std::string num_elements_for_loop = first_loop_->getVarNameNumberOfElements();

  bool cache_input_data = VariableManager::instance().getVariableValueBoolean(
      "code_gen.enable_caching");

  uint64_t global_worksize;

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

  if (!columns_to_decompress.empty()) {
    COGADB_FATAL_ERROR("Cannot work with uncompressed columns!", "");
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
  out << getCallKernels(global_worksize);

  /* copy data from device back to CPU main memory, if required */
  out << getCodeCopyOCLGroupedAggregationResultBuffersToHost(
      "current_result_size", dev_type_, dev_id_,
      aggregation_kernel_output_vars_, AggregationHashTableVarName);

  /* if we do materialize into a hash table during aggregation,
     write copy result from hash table to output arrays */
  if (pipe_end == MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
    out << code->after_for_loop_code_block_.str();
  }

  out << getCreateResult(1);

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

void OCLGroupedAggregationAtomic::get64BitAtomics(std::stringstream& stream) {
  stream << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable"
         << std::endl;

  stream << "#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable"
         << std::endl;

  stream << "#undef C_MIN_uint64_t" << std::endl
         << "#undef C_MIN_double" << std::endl
         << "#undef C_MAX_uint64_t" << std::endl
         << "#undef C_MAX_double" << std::endl
         << "#undef C_SUM_uint64_t" << std::endl
         << "#undef C_SUM_double" << std::endl;

  stream << "#define C_MIN_uint64_t(a, b) (atom_min(&a, b))" << std::endl
         << "#define C_MAX_uint64_t(a, b) (atom_max(&a, b))" << std::endl
         << "#define C_SUM_uint64_t(a, b) (atom_add(&a, b))" << std::endl;

  stream << "void atomic_min_double(volatile __global double* p, double val) {"
         << std::endl
         << "  union {" << std::endl
         << "    double d;" << std::endl
         << "    uint64_t i;" << std::endl
         << "  } prev;" << std::endl
         << "  union {" << std::endl
         << "    double d;" << std::endl
         << "    uint64_t i;" << std::endl
         << "  } new;" << std::endl
         << "  do {" << std::endl
         << "    prev.i = new.i = *(volatile __global uint64_t*)p;" << std::endl
         << "    new.d = new.d < val ? new.d : val;" << std::endl
         << "  } while (atom_cmpxchg((volatile __global uint64_t*)p, prev.i,"
            " new.i) != prev.i);"
         << std::endl
         << "}" << std::endl
         << "#define C_MIN_double(a, b) atomic_min_double(&a, b)" << std::endl;

  stream << "void atomic_max_double(volatile __global double* p, double val) {"
         << std::endl
         << "  union {" << std::endl
         << "    double d;" << std::endl
         << "    uint64_t i;" << std::endl
         << "  } prev;" << std::endl
         << "  union {" << std::endl
         << "    double d;" << std::endl
         << "    uint64_t i;" << std::endl
         << "  } new;" << std::endl
         << "  do {" << std::endl
         << "    prev.i = new.i = *(volatile __global uint64_t*)p;" << std::endl
         << "    new.d = new.d > val ? new.d : val;" << std::endl
         << "  } while (atom_cmpxchg((volatile __global uint64_t*)p, prev.i,"
            " new.i) != prev.i);"
         << std::endl
         << "}" << std::endl
         << "#define C_MAX_double(a, b) atomic_max_double(&a, b)" << std::endl;

  stream << "void atomic_sum_double(volatile __global double* p, double val) {"
         << std::endl
         << "  union {" << std::endl
         << "    double d;" << std::endl
         << "    uint64_t i;" << std::endl
         << "  } prev;" << std::endl
         << "  union {" << std::endl
         << "    double d;" << std::endl
         << "    uint64_t i;" << std::endl
         << "  } new;" << std::endl
         << "  do {" << std::endl
         << "    prev.i = new.i = *(volatile __global uint64_t*)p;" << std::endl
         << "    new.d += val;" << std::endl
         << "  } while (atom_cmpxchg((volatile __global uint64_t*)p, prev.i,"
            " new.i) != prev.i);"
         << std::endl
         << "}" << std::endl
         << "#define C_SUM_double(a, b) atomic_sum_double(&a, b)" << std::endl;
}

void OCLGroupedAggregationAtomic::get32BitAtomics(std::stringstream& stream) {
  stream << "#undef C_MIN_uint32_t" << std::endl
         << "#undef C_MIN_float" << std::endl
         << "#undef C_MAX_uint32_t" << std::endl
         << "#undef C_MAX_float" << std::endl
         << "#undef C_SUM_uint32_t" << std::endl
         << "#undef C_SUM_float" << std::endl;

  stream << "#define C_MIN_uint32_t(a, b) (atomic_min(&a, b))" << std::endl
         << "#define C_MAX_uint32_t(a, b) (atomic_max(&a, b))" << std::endl
         << "#define C_SUM_uint32_t(a, b) (atomic_add(&a, b))" << std::endl;

  stream << "void atomic_min_float(volatile __global float* p, float val) {"
         << std::endl
         << "  union {" << std::endl
         << "    float d;" << std::endl
         << "    uint32_t i;" << std::endl
         << "  } prev;" << std::endl
         << "  union {" << std::endl
         << "    float d;" << std::endl
         << "    uint32_t i;" << std::endl
         << "  } new;" << std::endl
         << "  do {" << std::endl
         << "    prev.i = new.i = *(volatile __global uint32_t*)p;" << std::endl
         << "    new.d = new.d < val ? new.d : val;" << std::endl
         << "  } while (atomic_cmpxchg((volatile __global uint32_t*)p, prev.i,"
            " new.i) != prev.i);"
         << std::endl
         << "}" << std::endl
         << "#define C_MIN_float(a, b) atomic_min_float(&a, b)" << std::endl;

  stream << "void atomic_max_float(volatile __global float* p, float val) {"
         << std::endl
         << "  union {" << std::endl
         << "    float d;" << std::endl
         << "    uint32_t i;" << std::endl
         << "  } prev;" << std::endl
         << "  union {" << std::endl
         << "    float d;" << std::endl
         << "    uint32_t i;" << std::endl
         << "  } new;" << std::endl
         << "  do {" << std::endl
         << "    prev.i = new.i = *(volatile __global uint32_t*)p;" << std::endl
         << "    new.d = new.d > val ? new.d : val;" << std::endl
         << "  } while (atomic_cmpxchg((volatile __global uint32_t*)p, prev.i,"
            " new.i) != prev.i);"
         << std::endl
         << "}" << std::endl
         << "#define C_MAX_float(a, b) atomic_max_float(&a, b)" << std::endl;

  stream << "void atomic_sum_float(volatile __global float* p, float val) {"
         << std::endl
         << "  union {" << std::endl
         << "    float d;" << std::endl
         << "    uint32_t i;" << std::endl
         << "  } prev;" << std::endl
         << "  union {" << std::endl
         << "    float d;" << std::endl
         << "    uint32_t i;" << std::endl
         << "  } new;" << std::endl
         << "  do {" << std::endl
         << "    prev.i = new.i = *(volatile __global uint32_t*)p;" << std::endl
         << "    new.d += val;" << std::endl
         << "  } while (atomic_cmpxchg((volatile __global uint32_t*)p, prev.i,"
            " new.i) != prev.i);"
         << std::endl
         << "}" << std::endl
         << "#define C_SUM_float(a, b) atomic_sum_float(&a, b)" << std::endl;
}

std::string OCLGroupedAggregationAtomic::getKernelCode(
    uint64_t global_worksize) {
  std::stringstream ss;

  ss << getAggregationPayloadStruct();

  if (atomics_32bit_) {
    get32BitAtomics(ss);
  } else {
    get64BitAtomics(ss);
  }

  std::string index_type = atomics_32bit_ ? "uint32_t" : "uint64_t";

  ss << index_type << " getPayloadIndex(" << index_type << " key, "
     << "__global AggregationPayload* hash_map, const uint64_t hash_map_mask) {"
     << std::endl
     << "  " << index_type << " index = 0; " << std::endl
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
OCLGroupedAggregationAtomic::getCodeInitKernelIndexCalculation(
    uint64_t global_worksize) {
  if (dev_type_ == CL_DEVICE_TYPE_GPU) {
    return std::make_pair("uint64_t table_index = get_global_id(0);", "");
  } else {
    return getCodeOCLKernelMemoryTraversal("table_index", mem_access_);
  }
}

std::string OCLGroupedAggregationAtomic::getCodeCheckKeyInHashTable() {
  const std::string hash_map_access = "hash_map[index]";
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

  ss << "  old = " << hash_map_access << ".key;" << std::endl
     << "  if (old == key) {" << std::endl
     << "    return index;" << std::endl
     << "  } else if (old == "
     << (atomics_32bit_ ? InvalidKey32Bit : InvalidKey) << ") {" << std::endl
     << "    old = " << atomic_cmpxchg << "(&" << hash_map_access
     << ".key, old, key);" << std::endl
     << "    if (old == " << (atomics_32bit_ ? InvalidKey32Bit : InvalidKey)
     << " || old == key) {" << std::endl
     << "      return index;" << std::endl
     << "    }" << std::endl
     << "  }" << std::endl;

  return ss.str();
}

std::string OCLGroupedAggregationAtomic::getAggregationCode(
    const GroupingAttributes& grouping_columns,
    const AggregateSpecifications& aggregation_specs,
    const std::string access_ht_entry_expression) {
  return getAggregationCodeGeneric(grouping_columns, aggregation_specs,
                                   access_ht_entry_expression);
}

std::string OCLGroupedAggregationAtomic::getAggregationPayloadStruct() {
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

  if (atomics_32bit_) {
    ss << "uint32_t key;" << std::endl;
  } else {
    ss << "uint64_t key;" << std::endl;
  }

  ss << "};" << std::endl;
  ss << "typedef struct AggregationPayload AggregationPayload;" << std::endl;

  return ss.str();
}

std::string OCLGroupedAggregationAtomic::getCreateAggregationHashMapBuffer() {
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

  ss << "AggregationPayload* " << AggregationHashTableVarName << " = malloc("
     << AggregationHashTableLengthVarName << " * sizeof(AggregationPayload));"
     << std::endl;

  const auto var =
      std::make_pair(AggregationHashTableVarName, "AggregationPayload*");

  ss << getCodeDeclareOCLResultBuffer(var);

  ss << getCodeInitOCLResultBuffer(dev_type_, dev_id_, var, false, "", true);

  return ss.str();
}

std::string OCLGroupedAggregationAtomic::getCallKernels(
    uint64_t global_worksize, uint64_t local_worksize) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  const std::string hash_map_mask = AggregationHashTableVarName + "_mask";

  out << "cl_kernel grouped_aggregation_init_kernel = NULL;" << std::endl;
  out << "grouped_aggregation_init_kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"grouped_aggregation_init_kernel\", &_err);"
      << std::endl;

  out << "cl_event kernel_completion;" << std::endl;
  out << "size_t global_work_size, *global_work_size_ptr = NULL, "
         "local_work_size, *local_work_size_ptr = NULL;"
      << std::endl;

  out << getInitKernelGlobalLocalSize(global_worksize);

  out << "CL_CHECK(clSetKernelArg(grouped_aggregation_init_kernel, 0, "
      << "sizeof (cl_mem), &"
      << getCLBufferResultVarName(AggregationHashTableVarName) << "));"
      << std::endl;

  out << "uint64_t init_kernel_hash_map_size = "
      << getInitKernelHashMapSize(global_worksize) << ";" << std::endl;
  out << "CL_CHECK(clSetKernelArg(grouped_aggregation_init_kernel, 1, "
      << "sizeof (init_kernel_hash_map_size), &init_kernel_hash_map_size));"
      << std::endl;

  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "grouped_aggregation_init_kernel, 1, NULL, global_work_size_ptr, "
         "local_work_size_ptr, 0, NULL, &kernel_completion));"
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

  out << getAggKernelGlobalLocalSize(global_worksize, local_worksize);

  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "grouped_aggregation_kernel, 1, NULL, global_work_size_ptr, "
         "local_work_size_ptr, 0, NULL, &kernel_completion));"
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

std::string OCLGroupedAggregationAtomic::getInitKernelGlobalLocalSize(
    uint64_t global_worksize) {
  std::stringstream out;

  out << "global_work_size = ";

  if (dev_type_ == CL_DEVICE_TYPE_GPU) {
    out << AggregationHashTableLengthVarName;
  } else {
    out << global_worksize;
  }

  out << ";" << std::endl;
  out << "global_work_size_ptr = &global_work_size;" << std::endl;
  out << "local_work_size_ptr = NULL;" << std::endl;

  return out.str();
}

std::string OCLGroupedAggregationAtomic::getAggKernelGlobalLocalSize(
    uint64_t global_worksize, uint64_t) {
  std::stringstream out;

  out << "global_work_size = " << global_worksize << ";" << std::endl;
  out << "global_work_size_ptr = &global_work_size;" << std::endl;
  out << "local_work_size_ptr = NULL;" << std::endl;

  return out.str();
}

std::string OCLGroupedAggregationAtomic::getInitKernelHashMapSize(uint64_t) {
  return AggregationHashTableLengthVarName;
}

std::string OCLGroupedAggregationAtomic::getCreateResult(
    uint64_t work_group_count) {
  const std::string hash_map_access = AggregationHashTableVarName + "[i].";

  std::stringstream ss;

  ss << getCodeDeclareResultMemory(projection_param_, true);
  ss << getCodeMallocResultMemory(projection_param_, true);

  ss << "for(uint64_t i = 0; i < " << AggregationHashTableLengthVarName << " / "
     << work_group_count << "; ++i) {" << std::endl;
  ss << "  if (" << hash_map_access
     << "key != " << (atomics_32bit_ ? InvalidKey32Bit : InvalidKey) << ") {"
     << std::endl;

  for (const auto& aggr_spec : aggr_specs_) {
    ss << aggr_spec->getCodeFetchResultsFromHashTableEntry(hash_map_access)
       << std::endl;
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

std::string OCLGroupedAggregationAtomic::getCodePayloadIndexFunctionImpl() {
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

std::string OCLGroupedAggregationAtomic::getCodeMurmur3Hashing() {
  return CoGaDB::getCodeMurmur3Hashing(atomics_32bit_);
}

std::string
OCLGroupedAggregationAtomic::getCodePayloadIndexFunctionLinearProbingImpl() {
  std::stringstream ss;

  std::string hash_func_code = "";
  const auto hash_func_name =
      VariableManager::instance().getVariableValueString(
          "code_gen.opt.ocl_grouped_aggregation.hash_function");

  if (hash_func_name == "multiply_shift") {
    hash_func_code = getCodeMultiplyShift();
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
OCLGroupedAggregationAtomic::getCodePayloadIndexFunctionQuadraticProbingImpl() {
  std::stringstream ss;

  ss << "  index = key;" << std::endl
     << getCodeMurmur3Hashing() << "  uint64_t displacement = 0;" << std::endl
     << "  while (1) {" << std::endl
     << getCodeCheckKeyInHashTable() << "    index += (displacement >> 1) + "
     << "(displacement >> 1) * (displacement >> 1);" << std::endl
     << "    displacement += 1; " << std::endl
     << "  }" << std::endl;

  return ss.str();
}

std::string
OCLGroupedAggregationAtomic::getCodePayloadIndexFunctionCuckooHashingImpl() {
  std::stringstream ss;

  std::string hash_func_code = "", hash_func_code2 = "";
  const auto hash_func_name =
      VariableManager::instance().getVariableValueString(
          "code_gen.opt.ocl_grouped_aggregation.hash_function");

  if (hash_func_name == "multiply_shift") {
    hash_func_code = getCodeMultiplyShift();
    hash_func_code2 = getCodeMurmur3Hashing();
  } else if (hash_func_name == "murmur3") {
    hash_func_code = getCodeMurmur3Hashing();
    hash_func_code2 = getCodeMultiplyShift();
  }

  ss << "  index = key;" << std::endl
     << hash_func_code << getCodeCheckKeyInHashTable() << std::endl;

  // use multiply shift hashing
  ss << "  index = key;" << std::endl
     << hash_func_code2 << std::endl
     << getCodeCheckKeyInHashTable() << std::endl;

  // use multiply add shift hashing
  ss << "  index = key;" << std::endl
     << "  " << getCodeMultiplyAddShift() << std::endl
     << getCodeCheckKeyInHashTable() << std::endl;

  // fallback to linear probing
  ss << getCodePayloadIndexFunctionLinearProbingImpl();

  return ss.str();
}

}  // namespace ExecutionStrategy

}  // namespace CoGaDB
