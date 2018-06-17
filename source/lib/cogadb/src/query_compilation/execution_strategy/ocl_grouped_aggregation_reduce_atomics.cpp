/*
 * author: henning funke
 * date: 08.07.2016
 */

#include <query_compilation/execution_strategy/ocl_grouped_aggregation_reduce_atomics.h>
#include <boost/thread.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/algebraic_aggregate_specification.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/ocl_workgroup_utils.hpp>
#include <query_compilation/primitives/aggregationPrimitives.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

const std::string
    OCLGroupedAggregationLocalReduceAtomics::AggregationHashTableVarName =
        "aggregation_hash_map";

const std::string
    OCLGroupedAggregationLocalReduceAtomics::AggregationHashTableLengthVarName =
        "aggregation_hash_map_length";

const std::string OCLGroupedAggregationLocalReduceAtomics::InvalidKey =
    "0xFFFFFFFFFFFFFFFF";
const std::string OCLGroupedAggregationLocalReduceAtomics::InvalidKey32Bit =
    "0xFFFFFFFF";

OCLGroupedAggregationLocalReduceAtomics::
    OCLGroupedAggregationLocalReduceAtomics(bool use_host_ptr,
                                            MemoryAccessPattern mem_access,
                                            cl_device_id dev_id)
    : OCL(use_host_ptr, mem_access, dev_id),
      aggregation_kernel_(boost::make_shared<GeneratedKernel>()) {
  hack_enable_manual_ht_size_ =
      VariableManager::instance().getVariableValueBoolean(
          "code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_size");
}

void OCLGroupedAggregationLocalReduceAtomics::addInstruction_impl(
    InstructionPtr instr) {
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

  if (instr->getInstructionType() == ALGEBRA_INSTR) {
    CodeBlockPtr block = aggregation_kernel_->kernel_code_blocks.back();

    block->upper_block.push_back(gen_code->declare_variables_code_block_.str());

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
      std::string var = variable.first;
      std::string type = variable.second;
      std::string findStar = "*";
      std::string findComputed = "computed_var_";
      std::cout << "OVAR: " << var << std::endl;
      if (var.find(findComputed) ==
          std::string::npos) {  //&& type.find(findStar) != std::string::npos) {
        aggregation_kernel_input_vars_.insert(variable);
      }
      aggregation_kernel_output_vars_.insert(variable);
    }
  }
}

const std::pair<std::string, std::string>
OCLGroupedAggregationLocalReduceAtomics::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  GeneratedCodePtr code = code_;
  assert(first_loop_ != NULL);

  // kernel configuration
  global_worksize_ = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_grouped_aggregation.gpu.reduce_atomics.global_size");
  local_worksize_ = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_grouped_aggregation.gpu.reduce_atomics.local_size");
  values_per_thread_ = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_grouped_aggregation.gpu.reduce_atomics.values_per_"
      "thread");
  use_buffer_ = VariableManager::instance().getVariableValueBoolean(
      "code_gen.opt.ocl_grouped_aggregation.gpu.reduce_atomics.enable_buffer");
  // todo
  //  use_buffer_ = VariableManager::instance().getVariableValueBoolean(
  //        "code_gen.opt.ocl_grouped_aggregation.gpu.reduce_atomics.buffer_write_strategy");
  bool cache_input_data = VariableManager::instance().getVariableValueBoolean(
      "code_gen.enable_caching");

  std::cout << "configuration: " << global_worksize_ << ", " << local_worksize_
            << ", " << values_per_thread_ << ", " << use_buffer_ << std::endl;

  std::string num_elements_for_loop = first_loop_->getVarNameNumberOfElements();

  if (!columns_to_decompress.empty()) {
    COGADB_FATAL_ERROR("Cannot work with uncompressed columns!", "");
  }

  std::stringstream kernel;
  kernel << getKernelHeadersAndTypes();
  kernel << getKernelCode(global_worksize_);

  /* generate host code that prepares OpenCL buffers and launches the kernel */
  std::stringstream out;
  out << "/*" << std::endl;
  out << kernel.str() << std::endl;
  out << "*/" << std::endl;
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
  out << getCallKernels(global_worksize_, local_worksize_);

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

/*
const std::string getAggregationCodeGeneric(
    const GroupingAttributes& grouping_columns,
    const AggregateSpecifications& aggregation_specs,
    const std::string access_ht_entry_expression){
  std::stringstream hash_aggregate;
  for (size_t i = 0; i < aggregation_specs.size(); ++i) {
    hash_aggregate << aggregation_specs[i]->getCodeHashGroupBy(
        access_ht_entry_expression);
    for (size_t i = 0; i < grouping_columns.size(); ++i) {
      AttributeReference attr_ref = grouping_columns[i];
      if(!isComputed(attr_ref)){
        hash_aggregate << access_ht_entry_expression
                       << getGroupTIDVarName(attr_ref) << " = "
                       << getTupleIDVarName(attr_ref) << ";" << std::endl;
      }else{
        hash_aggregate << access_ht_entry_expression << getVarName(attr_ref)
                       << " = " << getVarName(attr_ref) << ";" << std::endl;
      }
    }
  }
  return hash_aggregate.str();
}
*/

std::string getAggregationCode(const GroupingAttributes& grouping_columns,
                               const AggregateSpecifications& aggregation_specs,
                               const std::string access_ht_entry_expression,
                               std::string suffix) {
  // std::stringstream hash_aggregate;
  // for (size_t i = 0; i < aggregation_specs.size(); ++i) {
  //  hash_aggregate << aggregation_specs[i]->getCodeHashGroupBy(
  //      access_ht_entry_expression);
  //
  //  for (size_t i = 0; i < grouping_columns.size(); ++i) {
  //    AttributeReference attr_ref = grouping_columns[i];
  //    hash_aggregate << access_ht_entry_expression
  //                   << getGroupTIDVarName(attr_ref) << " = "
  //                   << getGroupTIDVarName(attr_ref) << "_reg;" << std::endl;
  //  }
  //}
  std::stringstream hash_aggregate;
  for (size_t i = 0; i < aggregation_specs.size(); ++i) {
    hash_aggregate << aggregation_specs[i]->getCodeHashGroupBy(
        access_ht_entry_expression);
    for (size_t i = 0; i < grouping_columns.size(); ++i) {
      AttributeReference attr_ref = grouping_columns[i];
      if (!isComputed(attr_ref)) {
        hash_aggregate << access_ht_entry_expression
                       << getGroupTIDVarName(attr_ref) << " = "
                       << getGroupTIDVarName(attr_ref) << "_reg;" << std::endl;
      } else {
        hash_aggregate << access_ht_entry_expression << getVarName(attr_ref)
                       << " = " << getVarName(attr_ref) << "_reg;" << std::endl;
      }
    }
  }

  /*
      if(isComputed(attr_ref)){
        arr = getVarName(attr_ref);
      }else{
        arr = getGroupTIDVarName(attr_ref);
      }
  */

  // replace tuple input with group aggregate as sum of local and register
  std::string code = hash_aggregate.str();
  for (size_t i = 0; i < aggregation_specs.size(); ++i) {
    boost::shared_ptr<AlgebraicAggregateSpecification> agg_spec =
        boost::dynamic_pointer_cast<AlgebraicAggregateSpecification>(
            aggregation_specs[i]);
    for (auto& var :
         aggregation_specs[i]->getOutputVariables(OCL_TARGET_CODE)) {
      if (var.first.find("COUNT") == 0) {
        std::string replace = " 1);";

        std::string insert;
        if (isInclusiveImplementation()) {
          insert = "(" + var.first + suffix + "[lid]));";
        } else {
          insert = "(" + var.first + "_reg+" + var.first + suffix + "[lid]));";
        }

        int pos = code.find(replace);
        code.replace(pos, replace.length(), insert);
      } else {
        std::string replace = getElementAccessExpression(agg_spec->scan_attr_);

        std::string insert;
        if (isInclusiveImplementation()) {
          insert = "(" + var.first + suffix + "[lid])";
        } else {
          insert = "(" + var.first + "_reg+" + var.first + suffix + "[lid])";
        }

        int pos = code.find(replace);
        code.replace(pos, replace.length(), insert);
      }
    }
  }
  std::stringstream result;
  result << code << std::endl;
  return result.str();
}

const std::pair<std::string, std::string> getCodeOCLKernelMemoryTraversal(
    const std::string& loop_variable_name, size_t global_worksize,
    size_t local_worksize, MemoryAccessPattern mem_access,
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

/*
std::string getCustomSortCode(std::string key_var, std::string value_var) {

  std::stringstream sort;
  sort <<
  "// Loop on sorted sequence length" << std::endl <<
  "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl <<
  "int i=lid;int wg=localSize;" << std::endl <<
  "for (int length=1;length<wg;length<<=1)" << std::endl <<
  "{" << std::endl <<
  "  bool direction = ((i & (length<<1)) != 0); // direction of sort: 0=asc,
1=desc" << std::endl <<
  "  // Loop on comparison distance (between keys)" << std::endl <<
  "  for (int inc=length;inc>0;inc>>=1)" << std::endl <<
  "  {" << std::endl <<
  "    int j = i ^ inc; // sibling to compare" << std::endl <<
  "    int iVal = src_lid_loc[i];" << std::endl <<
  "    TID iKey = key" << suffix << "[i];" << std::endl;

  //sort << "if(j >= localSize || j < 0) printf(\"something is wrong C, j:
%i\\n\", j);" << std::endl;

  sort <<
  "    int jVal = src_lid_loc[j];" << std::endl <<
  "    TID jKey = key" << suffix << "[j];" << std::endl <<
  "    bool smaller = (jKey < iKey) || ( jKey == iKey && j < i );" << std::endl
<<
  "    bool swap = smaller ^ (j < i) ^ direction;" << std::endl <<
  "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl <<
  "    key" << suffix << "[i] = (swap)?jKey:iKey;" << std::endl <<
  "    src_lid_loc[i] = (swap)?jVal:iVal;" << std::endl <<
  "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl <<
  "  }" << std::endl <<
  "}" << std::endl;

}
*/

std::string getGroupingArray(AttributeReference attr_ref) {
  std::string arr;
  if (isComputed(attr_ref)) {
    arr = getVarName(attr_ref);
  } else {
    arr = getGroupTIDVarName(attr_ref);
  }
  return arr;
}

std::string OCLGroupedAggregationLocalReduceAtomics::getKernelCode(
    uint64_t global_worksize) {
  std::stringstream ss;

  ss << code_->kernel_header_and_types_code_block_.str();

  ss << getAggregationPayloadStruct();

  ss << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable"
     << std::endl;

  ss << "#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable"
     << std::endl;

  ss << "#undef C_MIN_uint64_t" << std::endl
     << "#undef C_MIN_double" << std::endl
     << "#undef C_MAX_uint64_t" << std::endl
     << "#undef C_MAX_double" << std::endl
     << "#undef C_SUM_uint64_t" << std::endl
     << "#undef C_SUM_double" << std::endl;

  ss << "#define C_MIN_uint64_t(a, b) (atomic_min(&a, b))" << std::endl
     << "#define C_MAX_uint64_t(a, b) (atomic_max(&a, b))" << std::endl
     << "#define C_SUM_uint64_t(a, b) (atomic_add(&a, b))" << std::endl;

  ss << "void atomic_min_double(volatile __global double* p, double val) {"
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

  ss << "void atomic_max_double(volatile __global double* p, double val) {"
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

  ss << "void atomic_sum_double(volatile __global double* p, double val) {"
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

  ss << "uint64_t getPayloadIndex(uint64_t key, "
        "__global AggregationPayload* hash_map, const uint64_t hash_map_mask) {"
     << std::endl
     << "  uint64_t index = 0; " << std::endl
     << "  uint64_t old = 0;" << std::endl
     << getCodePayloadIndexFunctionImpl() << std::endl
     << "}" << std::endl;

  ss << "#define WARP_SHIFT 4" << std::endl
     << "#define GRP_SHIFT 8" << std::endl
     << "#define BANK_OFFSET(n)     ((n) >> WARP_SHIFT + (n) >> GRP_SHIFT)"
     << std::endl;

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
  ss << loop_code.second;

  ss << "}" << std::endl;

  ss << getCodeKernelSignature("grouped_aggregation_kernel",
                               "__global AggregationPayload* hash_map, const "
                               "uint64_t hash_map_mask,"
                               " uint64_t num_elements",
                               aggregation_kernel_input_vars_,
                               aggregation_kernel_output_vars_)
     << " {" << std::endl;

  ss << "int lid = get_local_id(0);" << std::endl
     << "int localSize = get_local_size(0);" << std::endl
     << "int globalIdx = get_group_id(0);" << std::endl;

  // iterate over aggregate specifications to collect output variables
  std::map<std::string, std::string> aggVars;
  for (size_t i = 0; i < aggr_specs_.size(); ++i) {
    AggregateSpecificationPtr general_specs = aggr_specs_[i];
    std::map<std::string, std::string> specsVars =
        general_specs->getOutputVariables(OCL_TARGET_CODE);
    for (auto& var : specsVars) {
      aggVars.insert(var);
    }
  }

  std::string suffix = "_loc";
  if (use_buffer_) suffix = "_buf";
  std::string idvar = first_loop_->getLoopVariableName();

  std::stringstream local_declaration;
  for (auto& var : aggVars) {
    local_declaration << "__local " << var.second << " " << var.first << "_loc["
                      << local_worksize_ << "];" << std::endl;
  }
  for (const auto& attr_ref : grouping_attrs_) {
    if (isComputed(attr_ref)) {
      local_declaration << "__local " << getResultType(attr_ref, false) << " "
                        << getVarName(attr_ref) << "_loc[" << local_worksize_
                        << "];" << std::endl;
    } else {
      local_declaration << "__local TID " << getGroupTIDVarName(attr_ref)
                        << "_loc[" << local_worksize_ << "];" << std::endl;
    }
  }
  local_declaration << "__local TID key_loc[" << local_worksize_ << "];"
                    << std::endl;
  local_declaration << "__local int src_lid_loc[" << local_worksize_ << "];"
                    << std::endl;
  local_declaration << "__local char start_flags_loc[" << local_worksize_ + 1
                    << "];" << std::endl;
  local_declaration << "__local char flags_loc[" << local_worksize_ << "];"
                    << std::endl;
  local_declaration << "start_flags_loc[" << local_worksize_
                    << "]=1; //last thread" << std::endl;
  local_declaration << "int reduce_aggregate_count=0;" << std::endl;
  // buffer
  if (use_buffer_) {
    for (auto& var : aggVars) {
      local_declaration << "__local " << var.second << " " << var.first
                        << "_buf[" << local_worksize_ << "];" << std::endl;
    }

    for (const auto& attr_ref : grouping_attrs_) {
      if (isComputed(attr_ref)) {
        local_declaration << "__local " << getResultType(attr_ref, false) << " "
                          << getVarName(attr_ref) << "_buf[" << local_worksize_
                          << "];" << std::endl;
      } else {
        local_declaration << "__local TID " << getGroupTIDVarName(attr_ref)
                          << "_buf[" << local_worksize_ << "];" << std::endl;
      }
    }

    local_declaration << "__local TID key_buf[" << local_worksize_ << "];"
                      << std::endl;
    local_declaration << "__local int buffer_fill;" << std::endl;
    local_declaration << "__local int total;" << std::endl;
    local_declaration << "__local int group_offset[32];" << std::endl;
    local_declaration << "if(lid == 0) buffer_fill = 0;" << std::endl;
    local_declaration << "key_buf[lid] = 0xffffffffffffffff;" << std::endl;
    local_declaration << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
    local_declaration << "int loop_count=0;" << std::endl;
    local_declaration << "group_offset[lid/32] = 0;" << std::endl;
  }

  std::stringstream local_init;
  local_init << "key_loc[lid] = 0xffffffffffffffff;" << std::endl;
  local_init << "start_flags_loc[lid] = 0;" << std::endl;
  for (auto& var : aggVars) {
    local_init << var.first << "_loc[lid] = (" << var.second << ")0;"
               << std::endl;
  }

  std::stringstream load_local;
  for (size_t i = 0; i < aggr_specs_.size(); ++i) {
    AggregateSpecificationPtr general_specs = aggr_specs_[i];
    boost::shared_ptr<AlgebraicAggregateSpecification> specs =
        boost::dynamic_pointer_cast<AlgebraicAggregateSpecification>(
            general_specs);
    std::string agg_field = getAggregationPayloadFieldVarName(
        specs->result_attr_, specs->agg_func_);
    std::string input_array_element =
        getElementAccessExpression(specs->scan_attr_);
    for (auto& var : general_specs->getOutputVariables(OCL_TARGET_CODE)) {
      if (var.first.find("COUNT") == 0)
        load_local << var.first << "_loc[lid] = 1;" << std::endl;
      else
        load_local << var.first << "_loc[lid] = " << input_array_element << ";"
                   << std::endl;
    }
  }
  for (const auto& attr_ref : grouping_attrs_) {
    if (isComputed(attr_ref)) {
      load_local << getGroupingArray(attr_ref)
                 << "_loc[lid] = " << getGroupingArray(attr_ref) << ";"
                 << std::endl;
    } else {
      load_local << getGroupTIDVarName(attr_ref)
                 << "_loc[lid] = " << getTupleIDVarName(attr_ref) << ";"
                 << std::endl;
    }
  }
  load_local << "key_loc[lid] = group_key;" << std::endl;

  std::stringstream scan_buffer;
  scan_buffer << "int flag = (key_loc[lid] != 0xffffffffffffffff);"
              << std::endl;
  scan_buffer << "total = 0;" << std::endl;
  scan_buffer << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  WorkgroupCodeBlock scan_codeBlock =
      getScanCode("flag", "buffer_scan", local_worksize_, "scan_buf");

  /*
    scan_buffer << scan_codeBlock.computation
                << "if ((lid+1)%" << getGroupResultOffset() << "==0) {" <<
    std::endl
                << "  group_offset[" << getGroupVariableIndex() << "] =
    atomic_add(&total,  group_total);" << std::endl
                << "}" << std::endl
                << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  */

  scan_buffer << "if(flag) scan_buf[lid] = atomic_add(&total,  1);"
              << std::endl;
  scan_buffer << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // scatter to local memory buffer
  std::stringstream fill_buffer;

  fill_buffer << "int write_pos = buffer_fill + group_offset["
              << getGroupVariableIndex() << "] + scan_buf[lid];" << std::endl;

  fill_buffer << "if(flag) {" << std::endl;

  for (auto& var : aggVars) {
    fill_buffer << var.first << "_buf[write_pos] = " << var.first
                << "_loc[lid];" << std::endl;
  }
  for (const auto& attr_ref : grouping_attrs_) {
    fill_buffer << getGroupingArray(attr_ref)
                << "_buf[write_pos] = " << getGroupingArray(attr_ref)
                << "_loc[lid];" << std::endl;
  }
  fill_buffer << "key_buf[write_pos] = key_loc[lid];" << std::endl;
  fill_buffer << "}" << std::endl;
  fill_buffer << "if(lid == localSize-1) buffer_fill += total;" << std::endl;
  fill_buffer << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  std::string varn =
      aggr_specs_[0]->getOutputVariables(OCL_TARGET_CODE).begin()->first;
  std::stringstream print_buffer_code;
  print_buffer_code
      << "if(globalIdx == 10 && total > 0) {" << std::endl
      << "  if(lid == 0) printf(\"Print for attribute " << varn << "\\n\");"
      << std::endl
      << "  printf(\"buffer: lid %i, group key %lu, val %f\\n \" , lid, key"
      << suffix << "[lid]," << varn << suffix << "[lid]);" << std::endl
      << "}" << std::endl;

  std::stringstream sort;
  sort
      << "src_lid_loc[lid] = lid;" << std::endl
      << "// <code from "
         "http://www.bealto.com/gpu-sorting_parallel-merge-local.html>"
      << std::endl
      << "int wg = get_local_size(0); // workgroup size = block size, power of "
         "2"
      << std::endl
      << "barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date"
      << std::endl
      << "// Now we will merge sub-sequences of length 1,2,...,WG/2"
      << std::endl
      << "for (int length=1;length<wg;length<<=1)" << std::endl
      << "{" << std::endl
      << "  int iVal = src_lid_loc[lid];" << std::endl
      << "  uint64_t iKey = key" << suffix << "[lid];" << std::endl
      << "  int ii = lid & (length-1);  // index in our sequence in 0..length-1"
      << std::endl
      << "  int sibling = (lid - ii) ^ length; // beginning of the sibling "
         "sequence"
      << std::endl
      << "  int pos = 0;" << std::endl
      << "  for (int inc=length;inc>0;inc>>=1) // increment for dichotomic "
         "search"
      << std::endl
      << "  {" << std::endl
      << "    int j = sibling+pos+inc-1;" << std::endl
      << "    uint64_t jKey = key" << suffix << "[j];" << std::endl
      << "    bool smaller = (jKey < iKey) || ( jKey == iKey && j < lid );"
      << std::endl
      << "    pos += (smaller)?inc:0;" << std::endl
      << "    pos = min(pos,length);" << std::endl
      << "  }" << std::endl
      << "  int bits = 2*length-1; // mask for destination" << std::endl
      << "  int dest = ((ii + pos) & bits) | (lid & ~bits); // destination "
         "index in merged sequence"
      << std::endl
      << "  barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
      << "  key" << suffix << "[dest] = iKey;" << std::endl
      << "  src_lid_loc[dest] = iVal;" << std::endl
      << "  barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
      << "}" << std::endl
      << "// <end code>" << std::endl;

  // load attributes in sorted order by group key
  std::stringstream reorder_and_set;
  reorder_and_set << "// load payload to register" << std::endl;
  reorder_and_set << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  for (auto& var : aggVars) {
    reorder_and_set << var.second << " " << var.first << "_reg = " << var.first
                    << suffix << "[src_lid_loc[lid]];" << std::endl;
  }
  reorder_and_set << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  reorder_and_set << "// store reordered payload in local" << std::endl;
  for (auto& var : aggVars) {
    reorder_and_set << var.first << suffix << "[lid] = " << var.first << "_reg;"
                    << std::endl;
  }
  reorder_and_set << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  for (const auto& attr_ref : grouping_attrs_) {
    if (isComputed(attr_ref)) {
      reorder_and_set << getResultType(attr_ref, false) << " "
                      << getVarName(attr_ref)
                      << "_reg = " << getVarName(attr_ref) << suffix
                      << "[src_lid_loc[lid]];" << std::endl;
    } else {
      reorder_and_set << "TID " << getGroupTIDVarName(attr_ref)
                      << "_reg = " << getGroupTIDVarName(attr_ref) << suffix
                      << "[src_lid_loc[lid]];" << std::endl;
    }
  }
  reorder_and_set << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  reorder_and_set << "start_flags_loc[lid] = (lid == 0) ? 1 : (key" << suffix
                  << "[lid] != key" << suffix << "[lid-1]);" << std::endl
                  << "if(lid%32==0) start_flags_loc[lid] = 1;" << std::endl
                  << "flags_loc[lid] = start_flags_loc[lid];" << std::endl;
  reorder_and_set << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // segmented reduction
  std::string reduce_segments_code = getSegmentedScan(aggVars, suffix);

  std::string varname =
      aggr_specs_[0]->getOutputVariables(OCL_TARGET_CODE).begin()->first;
  std::stringstream print_reduction_code;
  if (use_buffer_)
    print_reduction_code << "if(globalIdx == 10) {" << std::endl;
  else
    print_reduction_code << "if(" << idvar << " >= 1000*localSize && " << idvar
                         << " < 1001*localSize) {" << std::endl;

  print_reduction_code << "  if(lid == 0) printf(\"Print for attribute "
                       << varname << "\\n\");" << std::endl
                       << "  printf(\"lid %i, group key %lu, start flag %i, "
                          "val %f, seg scan %f.\\n \" , lid, key"
                       << suffix << "[lid], start_flags_loc[lid]," << varname
                       << "_reg," << varname << suffix << "[lid]);" << std::endl
                       << "}" << std::endl;

  // <<<<<<<<<<<<<<<<<<<<<< code construction
  ss << local_declaration.str();

  ss << scan_codeBlock.kernel_init;

  ss << scan_codeBlock.local_init;

  loop_code = getCodeOCLKernelMemoryTraversal(
      first_loop_->getLoopVariableName(), global_worksize_, local_worksize_,
      mem_access_, true);

  ss << loop_code.first;

  ss << local_init.str();

  ss << "if(" << idvar << " < num_elements) {" << std::endl;
  for (const auto& code_block : aggregation_kernel_->kernel_code_blocks) {
    for (const auto& upper : code_block->upper_block) {
      ss << upper << std::endl;
    }
  }
  ss << load_local.str();
  ss << "}" << std::endl;

  for (const auto& code_block : aggregation_kernel_->kernel_code_blocks) {
    for (const auto& lower : code_block->lower_block) {
      ss << lower << std::endl;
    }
  }

  if (use_buffer_) {
    ss << scan_buffer.str();
    ss << "if(buffer_fill + total >= localSize ||" << idvar
       << " > (num_elements - localSize)) {" << std::endl;
  }

  ss << "reduce_aggregate_count++;" << std::endl;

  // testprint
  // if(buffer)
  //  ss << "if(globalIdx == 10) {" << std::endl;
  // else
  //  ss << "if(" << idvar << ">= 1000*localSize &&" << idvar << "<
  //  1001*localSize) {" << std::endl;
  // ss   << "  if(lid == 0) printf(\"Print for attribute " << varname <<
  // "\\n\");" << std::endl
  //     << "  printf(\"before sort: locid %i, group key %lu, val %f.\\n \" ,
  //     lid, key" << suffix << "[lid],"
  //     << varname << suffix << "[lid]);" << std::endl;
  // ss   << "}" << std::endl;

  ss << sort.str();

  ss << reorder_and_set.str();

  // testprint
  // if(buffer)
  //  ss << "if(globalIdx == 10) {" << std::endl;
  // else
  //  ss << "if(" << idvar << ">= 1000*localSize &&" << idvar << "<
  //  1001*localSize) {" << std::endl;
  //
  // ss   << "  if(lid == 0) printf(\"Print for attribute " << varname <<
  // "\\n\");" << std::endl
  //     << "  printf(\"after sort: locid %i, group key %lu, start flag %i, val
  //     %f.\\n \" , lid, key" << suffix << "[lid], start_flags_loc[lid],"
  //     << varname << suffix << "[lid]);" << std::endl;
  // ss   << "}" << std::endl;

  ss << reduce_segments_code;

  // ss << print_reduction_code.str();

  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  ss << "if(start_flags_loc[lid+1]==1 && key" << suffix
     << "[lid] != 0xffffffffffffffff) {" << std::endl
     << "  uint64_t payload_index = getPayloadIndex(key" << suffix
     << "[lid], hash_map, hash_map_mask);" << std::endl
     << getAggregationCode(grouping_attrs_, aggr_specs_,
                           "hash_map[payload_index].", suffix)
     << std::endl;

  ss << "}" << std::endl;

  // testprint
  /*
  std::string grouptid = getTupleIDVarName(grouping_attrs_[0]);
  ss   << "if(" << grouptid << ">= 1000*localSize &&" << grouptid << "<
  1001*localSize) {" << std::endl
       << "  if(lid == 0) printf(\"Print for attribute " << varname << "\\n\");"
  << std::endl
       << "  printf(\"locid %i, group key %lu, start flag %i, val %f, seg scan
  %f.\\n \" , lid, key_loc[lid], start_flags_loc[lid],"
       << varname << "_reg," << varname << "_reg+" << varname << "_loc[lid]);"
  << std::endl;
  ss   << "}" << std::endl;
*/
  if (use_buffer_) {
    ss << "  if(lid==localSize-1) buffer_fill = 0;" << std::endl;
    ss << "  key_buf[lid] = 0xffffffffffffffff;" << std::endl;
    ss << "  barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
    ss << "}" << std::endl;
    ss << fill_buffer.str();
    // ss << print_buffer_code.str();
    ss << "loop_count++;" << std::endl;
  }

  ss << loop_code.second;

  // ss << "  if(lid==localSize-1) printf(\"reduce_aggregate_count %i\",
  // reduce_aggregate_count);" << std::endl;

  ss << "}" << std::endl << std::endl;

  return ss.str();
}

std::pair<std::string, std::string>
OCLGroupedAggregationLocalReduceAtomics::getCodeInitKernelIndexCalculation(
    uint64_t global_worksize) {
  // if (dev_type_ == CL_DEVICE_TYPE_GPU) {
  return std::make_pair("uint64_t table_index = get_global_id(0);", "");
  //} else {
  //  return getCodeOCLKernelMemoryTraversal("table_index", global_worksize,
  //                                         mem_access_, false);
  //}
}

std::string
OCLGroupedAggregationLocalReduceAtomics::getCodeCheckKeyInHashTable() {
  const std::string hash_map_access = "hash_map[index]";
  std::stringstream ss;

  ss << "  index &= hash_map_mask;" << std::endl
     << "  old = " << hash_map_access << ".key;" << std::endl
     << "  if (old == key) {" << std::endl
     << "    return index;" << std::endl
     << "  } else if (old == " << InvalidKey << ") {" << std::endl
     << "    old = atom_cmpxchg(&" << hash_map_access << ".key, old, key);"
     << std::endl
     << "    if (old == " << InvalidKey << " || old == key) {" << std::endl
     << "      return index;" << std::endl
     << "    }" << std::endl
     << "  }" << std::endl;

  return ss.str();
}

std::string
OCLGroupedAggregationLocalReduceAtomics::getAggregationPayloadStruct() {
  std::stringstream ss;

  ss << "struct AggregationPayload {" << std::endl;

  // for (const auto& attr_ref : grouping_attrs_) {
  //  ss << getAggregationGroupTIDPayloadFieldCode(attr_ref) << std::endl;
  //}
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

  ss << "};" << std::endl;
  ss << "typedef struct AggregationPayload AggregationPayload;" << std::endl;

  return ss.str();
}

std::string
OCLGroupedAggregationLocalReduceAtomics::getCreateAggregationHashMapBuffer() {
  std::stringstream ss;

  auto hashtable_size_multiplier =
      VariableManager::instance().getVariableValueFloat(
          "code_gen.opt.ocl_grouped_aggregation.hashtable_size_multiplier");

  ss << "uint64_t " << AggregationHashTableLengthVarName << " = 1;" << std::endl
     << "while(" << AggregationHashTableLengthVarName << " < "
     << first_loop_->getVarNameNumberOfElements() << " * "
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

std::string OCLGroupedAggregationLocalReduceAtomics::getCallKernels(
    uint64_t global_worksize, size_t local_worksize) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  const std::string hash_map_size = AggregationHashTableLengthVarName;
  const std::string hash_map_mask = AggregationHashTableVarName + "_mask";

  out << "cl_kernel grouped_aggregation_init_kernel = NULL;" << std::endl;
  out << "grouped_aggregation_init_kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"grouped_aggregation_init_kernel\", &_err);"
      << std::endl;

  out << "cl_event kernel_completion;" << std::endl;
  out << "size_t global_work_size = ";

  if (dev_type_ == CL_DEVICE_TYPE_GPU) {
    out << hash_map_size;
  } else {
    out << global_worksize;
  }

  out << ";" << std::endl;

  out << "CL_CHECK(clSetKernelArg(grouped_aggregation_init_kernel, 0, "
      << "sizeof (cl_mem), &"
      << getCLBufferResultVarName(AggregationHashTableVarName) << "));"
      << std::endl;

  out << "CL_CHECK(clSetKernelArg(grouped_aggregation_init_kernel, 1, "
      << "sizeof (" << hash_map_size << "), &" << hash_map_size << "));"
      << std::endl;

  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "grouped_aggregation_init_kernel, 1, NULL, &global_work_size, "
         "NULL, 0, NULL, &kernel_completion));"
      << std::endl;

  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;

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

  // out << "{" << std::endl;

  out << "global_work_size = " << global_worksize << ";" << std::endl;
  out << "size_t local_work_size = " << local_worksize << ";" << std::endl;

  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "grouped_aggregation_kernel, 1, NULL, &global_work_size, "
         "&local_work_size, 0, NULL, &kernel_completion));"
      << std::endl;
  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
// out << "}" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Grouped Aggregation Kernel\");" << std::endl;
#endif
  return out.str();
}

std::string OCLGroupedAggregationLocalReduceAtomics::getCreateResult() {
  const std::string hash_map_access = AggregationHashTableVarName + "[i].";

  std::stringstream ss;

  ss << getCodeDeclareResultMemory(projection_param_, true);
  ss << getCodeMallocResultMemory(projection_param_, true);

  ss << "for(uint64_t i = 0; i < " << AggregationHashTableLengthVarName << ";"
     << "++i) {" << std::endl;
  ss << "  if (" << hash_map_access << "key != " << InvalidKey << ") {"
     << std::endl;

  for (const auto& aggr_spec : aggr_specs_) {
    ss << aggr_spec->getCodeFetchResultsFromHashTableEntry(hash_map_access);
  }

  // for (const auto& grouping_attr : grouping_attrs_) {
  //  ss << "    " << getResultArrayVarName(grouping_attr)
  //     << "[current_result_size] = "
  //     << getCompressedElementAccessExpression(
  //            grouping_attr,
  //            hash_map_access + getGroupTIDVarName(grouping_attr))
  //     << ";" << std::endl;
  //}
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

std::string
OCLGroupedAggregationLocalReduceAtomics::getCodePayloadIndexFunctionImpl() {
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

std::string OCLGroupedAggregationLocalReduceAtomics::getCodeMurmur3Hashing() {
  std::stringstream ss;

  ss << "  index ^= index >> 33;" << std::endl
     << "  index *= 0xff51afd7ed558ccd;" << std::endl
     << "  index ^= index >> 33;" << std::endl
     << "  index *= 0xc4ceb9fe1a85ec53;" << std::endl
     << "  index ^= index >> 33;" << std::endl;

  return ss.str();
}

std::string OCLGroupedAggregationLocalReduceAtomics::
    getCodePayloadIndexFunctionLinearProbingImpl() {
  std::stringstream ss;

  ss << "  index = key;" << std::endl
     << getCodeMurmur3Hashing() << "  while (1) {" << std::endl
     << getCodeCheckKeyInHashTable() << "    index += 1; " << std::endl
     << "  }" << std::endl;

  return ss.str();
}

std::string OCLGroupedAggregationLocalReduceAtomics::
    getCodePayloadIndexFunctionQuadraticProbingImpl() {
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

std::string OCLGroupedAggregationLocalReduceAtomics::
    getCodePayloadIndexFunctionCuckooHashingImpl() {
  std::stringstream ss;

  ss << "  index = key;" << std::endl
     << getCodeMurmur3Hashing() << getCodeCheckKeyInHashTable() << std::endl;

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
