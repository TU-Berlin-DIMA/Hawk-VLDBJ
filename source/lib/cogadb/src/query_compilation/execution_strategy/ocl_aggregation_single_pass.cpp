/*
 * author: henning funke
 * date: 26.06.2016
 */

#include <query_compilation/execution_strategy/ocl_aggregation_single_pass.h>
#include <core/variable_manager.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/code_generators/ocl_workgroup_utils.hpp>

#include <boost/thread.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLAggregationSinglePassReduce::OCLAggregationSinglePassReduce(
    bool use_host_ptr, MemoryAccessPattern mem_access, cl_device_id dev_id)
    : OCL(use_host_ptr, mem_access, dev_id),
      aggregation_kernel_(boost::make_shared<GeneratedKernel>()) {}

void OCLAggregationSinglePassReduce::addInstruction_impl(InstructionPtr instr) {
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

  if (instr->getInstructionType() == AGGREGATE_INSTR) {
    boost::shared_ptr<Aggregation> aggregateInstruction =
        boost::dynamic_pointer_cast<Aggregation>(instr);
    alg_agg_spec_ =
        boost::dynamic_pointer_cast<AlgebraicAggregateSpecification>(
            aggregateInstruction->getAggregateSpecifications());

    for (auto& variable : instr->getInputVariables(OCL_TARGET_CODE)) {
      aggregation_kernel_input_vars_.insert(variable);
    }

    for (auto& variable : instr->getOutputVariables(OCL_TARGET_CODE)) {
      std::string first = variable.first;
      std::string second = variable.second;

      if (alg_agg_spec_->getAggregationFunction() == AVERAGE) {
        // SUM is somewhere renamed to AVERAGE
        if (first.find("SUM") < first.length())
          first.replace(first.find("SUM"), 3, "AVERAGE");
      }
      aggregation_kernel_aggregate_vars_.insert(
          std::pair<std::string, std::string>(first, second));
      aggregation_kernel_output_vars_.insert(
          std::pair<std::string, std::string>(first, second));
    }
  }

  if (instr->getInstructionType() == HASH_TABLE_PROBE_INSTR ||
      instr->getInstructionType() == MAP_UDF_INSTR ||
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

      if (instr->getInstructionType() == AGGREGATE_INSTR) {
        aggregation_kernel_aggregate_vars_.insert(variable);
      }
    }
  }
}

std::string removePointer(std::string type) {
  auto find = type.find_first_of("*");

  if (find == std::string::npos) {
    return type;
  } else {
    if (find != type.find_last_of("*")) {
      COGADB_FATAL_ERROR("We don't support '" << type << "**'", "");
    }

    return type.erase(find, 1);
  }
}

const std::string getCodeAggregationKernelSignature(
    const std::map<std::string, std::string>& input_vars,
    std::map<std::string, std::string> output_vars,
    const std::map<std::string, std::string>& aggr_vars) {
  for (const auto& var : aggr_vars) {
    auto find = output_vars.find(var.first);
    if (find != output_vars.end()) {
      find->second = removePointer(find->second) + "*";
    }
  }

  std::stringstream str;
  return getCodeKernelSignature("aggregation_kernel", "uint64_t num_elements",
                                input_vars, output_vars);
}

std::string getAtomicDoubleFunctions() {
  std::stringstream ss;

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

  return ss.str();
}

const std::string getCodeCallReduceAggregationKernel(
    const std::string& num_elements_for_loop, uint64_t global_worksize,
    size_t local_worksize, const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars,
    cl_device_type dev_type) {
  std::stringstream out;
#ifdef GENERATE_PROFILING_CODE
  out << "ocl_start_timer();" << std::endl;
#endif

  out << "cl_kernel aggregation_kernel = NULL;" << std::endl;
  out << "aggregation_kernel = "
         "clCreateKernel(ocl_getProgram(context), "
         "\"aggregation_kernel\", &_err);"
      << std::endl;
  out << "CL_CHECK( _err );" << std::endl;

  out << "CL_CHECK(clSetKernelArg(aggregation_kernel, 0, "
      << "sizeof (" << num_elements_for_loop << "), &" << num_elements_for_loop
      << "));" << std::endl;

  unsigned int index = 1;
  for (const auto& var : input_vars) {
    auto var_name = isPointerType(var.second)
                        ? getCLBufferInputVarName(var.first)
                        : var.first;
    out << "CL_CHECK(clSetKernelArg(aggregation_kernel, " << index++
        << ", sizeof(" << var_name << ") , &" << var_name << "));" << std::endl;
  }

  for (const auto& var : output_vars) {
    out << "CL_CHECK(clSetKernelArg(aggregation_kernel, " << index++
        << ", sizeof(" << getCLBufferResultVarName(var.first) << ") , &"
        << getCLBufferResultVarName(var.first) << "));" << std::endl;
  }

  out << "cl_event kernel_completion;" << std::endl;
  out << "size_t global_work_size = " << global_worksize << ";" << std::endl;
  out << "size_t local_work_size = " << local_worksize << ";" << std::endl;

  out << "CL_CHECK(clEnqueueNDRangeKernel(ocl_getComputeCommandQueue(context), "
         "aggregation_kernel, 1, NULL, &global_work_size, "
         "&local_work_size, 0, NULL, &kernel_completion));"
      << std::endl;
  out << "CL_CHECK(clWaitForEvents(1, &kernel_completion));" << std::endl;
  out << "CL_CHECK(clReleaseEvent(kernel_completion));" << std::endl;
// out << "}" << std::endl;

#ifdef GENERATE_PROFILING_CODE
  out << "ocl_stop_timer(\"Reduce Aggregation Kernel\");" << std::endl;
#endif
  return out.str();
}

const std::string getCodeReduceAggregationKernel(
    ExecutionStrategy::GeneratedKernelPtr aggr_kernel_,
    const std::string& loop_variable_name, uint64_t global_worksize,
    int local_worksize, bool use_atomics,
    const std::map<std::string, std::string>& input_vars,
    const std::map<std::string, std::string>& output_vars,
    const std::map<std::string, std::string>& agg_vars,
    MemoryAccessPattern mem_access,
    boost::shared_ptr<AlgebraicAggregateSpecification> alg_agg_spec,
    GeneratedCodePtr code) {
  std::stringstream declare_code;
  for (const auto& var : agg_vars) {
    declare_code << var.second << " " << var.first << "_reg;" << std::endl;
    declare_code << "__local " << var.second << " " << var.first << "_loc["
                 << local_worksize << "];" << std::endl;
  }

  std::stringstream init_code;
  for (const auto& var : agg_vars) {
    init_code << var.first << "_reg = (" << var.second << ")0;" << std::endl;
  }

  std::string aggregate_code =
      alg_agg_spec->getCodeAggregationComputation(C_TARGET_CODE);
  for (const auto& var : agg_vars) {
    int f = aggregate_code.find(var.first) + var.first.length();
    aggregate_code.insert(f, "_reg");
  }

  std::stringstream load_loc_code;
  for (const auto& var : agg_vars) {
    load_loc_code << var.first << "_loc[lid]=" << var.first << "_reg;"
                  << std::endl;
  }

  std::stringstream kernel;
  kernel << getKernelHeadersAndTypes();
  kernel << code->kernel_header_and_types_code_block_.str();
  kernel << getAtomicDoubleFunctions();
  kernel << "#define WARP_SHIFT 4" << std::endl;
  kernel << "#define GRP_SHIFT 8" << std::endl;
  kernel << "#define BANK_OFFSET(n)     ((n) >> WARP_SHIFT + (n) >> GRP_SHIFT)"
         << std::endl;
  kernel << getCodeAggregationKernelSignature(input_vars, output_vars, agg_vars)
         << "{" << std::endl;
  kernel << "int lid = get_local_id(0);" << std::endl
         << "int localSize = get_local_size(0);" << std::endl
         << "int globalIdx = get_group_id(0);" << std::endl;
  kernel << declare_code.str();
  kernel << init_code.str();

  std::pair<std::string, std::string> loop_code =
      getCodeOCLKernelMemoryTraversal(loop_variable_name, mem_access);

  kernel << loop_code.first;

  for (const auto& code_block : aggr_kernel_->kernel_code_blocks) {
    for (const auto& upper : code_block->upper_block) {
      kernel << upper << std::endl;
    }

    kernel << aggregate_code << std::endl;
    // todo: if(alg_agg_spec->getAggregationFunction() == MIN) ..

    for (const auto& lower : code_block->lower_block) {
      kernel << lower << std::endl;
    }
  }

  kernel << loop_code.second;

  // write to shared
  kernel << load_loc_code.str() << std::endl;

  if (use_atomics) {
    for (const auto& var : agg_vars) {
      kernel << "C_SUM_" << var.second << "(" << var.first << "[0], "
             << var.first << "_loc[lid] + " << var.first << "_reg);"
             << std::endl;
    }
  } else {
    std::stringstream reduce_operation;
    for (const auto& var : agg_vars) {
      reduce_operation << var.first << "_loc[bi] += " << var.first
                       << "_loc[ai];" << std::endl;
    }
    WorkgroupCodeBlock reduce_code = getLowLevelReductionVar1(
        reduce_operation.str(), UPSWEEP, local_worksize);
    kernel << reduce_code.local_init;
    kernel << reduce_code.computation;
    kernel << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
    kernel << "if(lid == localSize-1) {" << std::endl;
    for (const auto& var : agg_vars) {
      kernel << "C_SUM_" << var.second << "(" << var.first << "[0], "
             << var.first << "_loc[lid] + " << var.first << "_reg);"
             << std::endl;
    }
    kernel << "}";
  }

  kernel << "}" << std::endl << std::endl;

  return kernel.str();
}

const std::pair<std::string, std::string>
OCLAggregationSinglePassReduce::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  // kernel configuration
  global_worksize_ = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_aggregation.gpu.single_pass_reduce.global_size");
  local_worksize_ = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_aggregation.gpu.single_pass_reduce.local_size");
  values_per_thread_ = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_aggregation.gpu.single_pass_reduce.values_per_thread");
  use_atomics_ = VariableManager::instance().getVariableValueBoolean(
      "code_gen.opt.ocl_aggregation.gpu.single_pass_reduce.use_atomics");

  GeneratedCodePtr code = code_;
  assert(first_loop_ != NULL);
  std::string num_elements_for_loop = first_loop_->getVarNameNumberOfElements();

  bool cache_input_data = VariableManager::instance().getVariableValueBoolean(
      "code_gen.enable_caching");

  std::stringstream kernel;
  kernel << getKernelHeadersAndTypes();
  kernel << getCodeReduceAggregationKernel(
      aggregation_kernel_, first_loop_->getLoopVariableName(), global_worksize_,
      local_worksize_, use_atomics_, aggregation_kernel_input_vars_,
      aggregation_kernel_output_vars_, aggregation_kernel_aggregate_vars_,
      mem_access_, alg_agg_spec_, code_);

  /* generate host code that prepares OpenCL buffers and launches the kernel */
  std::stringstream out;
  out << "/*" << std::endl;
  out << kernel.str() << std::endl;
  out << "*/" << std::endl;
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
  out << "uint64_t allocated_result_elements = 1;" << std::endl;
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
      global_worksize_);

  out << getCodeInitOCLAggregationResultBuffers(
             "current_result_size", dev_type_, dev_id_,
             aggregation_kernel_output_vars_,
             aggregation_kernel_aggregate_vars_, 1)
      << std::endl;

  /* generate code that calls OpenCL kernel that performs the aggregations */
  out << getCodeCallReduceAggregationKernel(
      num_elements_for_loop, global_worksize_, local_worksize_,
      aggregation_kernel_input_vars_, aggregation_kernel_output_vars_,
      dev_type_);

  // isPointerType condition in Util function very difficult to work around for
  // aggregation
  // because of the assumption that results are only passed via global reduction
  for (auto const& var : aggregation_kernel_aggregate_vars_) {
    out << getCodeCopyOCLResultValueToVariable(var);
  }

  out << getCodeCopyOCLResultBuffersToHost("current_result_size", dev_type_,
                                           dev_id_,
                                           aggregation_kernel_output_vars_);

  out << std::endl;
  // out << getCoderReduceAggregation(aggregation_kernel_aggregate_vars_,
  //                                 global_worksize_/local_worksize_);

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
