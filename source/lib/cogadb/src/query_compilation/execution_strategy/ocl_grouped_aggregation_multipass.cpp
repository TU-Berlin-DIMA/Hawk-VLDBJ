/*
 * author: henning funke
 * date: 08.07.2016
 */

#include <query_compilation/execution_strategy/multipass_grouping.h>
#include <boost/thread.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/execution_strategy/ocl_grouped_aggregation_multipass.hpp>
#include <query_compilation/primitives/aggregationPrimitives.hpp>
#include <query_compilation/primitives/loopPrimitives.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLGroupedAggregationMultipass::OCLGroupedAggregationMultipass(
    bool use_host_ptr, MemoryAccessPattern mem_access, cl_device_id dev_id)
    : OCLProjectionThreePhase(use_host_ptr, mem_access, dev_id) {}

const std::pair<std::string, std::string>
OCLGroupedAggregationMultipass::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  std::pair<std::string, std::string> code = OCLProjectionThreePhase::getCode(
      param, scanned_attributes, pipe_end, result_table_name,
      columns_to_decompress);

  std::string host_code = patchGroupAggregateHostCode(
      code.first, aggregationAttributes, computedAttributes);

  std::string kernel_code = patchGroupAggregateKernelCode(code.second);

  std::stringstream out;
  out << host_code;
  code.first = out.str();
  code.second = kernel_code;
  return code;
}

void OCLGroupedAggregationMultipass::addInstruction_impl(InstructionPtr instr) {
  if (instr->getInstructionType() == HASH_AGGREGATE_INSTR) {
    auto hash_aggr = boost::static_pointer_cast<HashGroupAggregate>(instr);
    ProjectionParam originalParam = hash_aggr->getGroupingAttributes();
    ProjectionParam param;
    param.insert(param.end(), originalParam.begin(), originalParam.end());
    groupingAttributes.insert(groupingAttributes.end(), originalParam.begin(),
                              originalParam.end());
    AggregateSpecifications agg_specs = hash_aggr->getAggregateSpecifications();

    for (auto spec : agg_specs) {
      auto atts = spec->getScannedAttributes();
      for (auto a : atts) {
        AttributeReferencePtr doubleAttribute =
            boost::make_shared<AttributeReference>(*a);

        param.push_back(*a);
        aggregationAttributes.push_back(*a);
        std::cout << "INPUT VARNAME: " << getResultArrayVarName(*a)
                  << std::endl;
      }
      auto outputs = spec->getComputedAttributes();
      for (auto o : outputs) {
        // param.push_back(*o);
        computedAttributes.push_back(*o);
        std::cout << "OUTPUT VARNAME: " << getResultArrayVarName(*o)
                  << std::endl;
      }
    }
    InstructionPtr ptr = boost::make_shared<Materialization>(param);
    OCLProjectionThreePhase::addInstruction_impl(ptr);
  } else if (instr->getInstructionType() == BITPACKED_GROUPING_KEY_INSTR) {
    GeneratedCodePtr gen_code = instr->getCode(OCL_TARGET_CODE);
    CodeBlockPtr block = projection_kernel_->kernel_code_blocks.back();
    block->upper_block.insert(block->upper_block.end(),
                              gen_code->upper_code_block_.begin(),
                              gen_code->upper_code_block_.end());

    block->lower_block.insert(block->lower_block.begin(),
                              gen_code->lower_code_block_.begin(),
                              gen_code->lower_code_block_.end());

    projection_kernel_output_vars_.insert(
        std::make_pair<std::string, std::string>("grouping_keys", "uint64_t*"));
    projection_kernel_output_vars_.insert(
        std::make_pair<std::string, std::string>("source_indices",
                                                 "uint64_t*"));
    block->upper_block.push_back(getCodeProjectGroupAggregateData());
  } else {
    OCLProjectionThreePhase::addInstruction_impl(instr);
  }
}

const std::string
OCLGroupedAggregationMultipass::getCodeCleanupCustomStructures() const {
  std::stringstream out;
  out << OCLProjectionThreePhase::getCodeCleanupCustomStructures();
  out << getCodeCleanupGroupAggregateStructures(groupingAttributes,
                                                aggregationAttributes);
  return out.str();
}

const std::string OCLGroupedAggregationMultipass::getCodeCallComputeKernels(
    const std::string& num_elements_for_loop, size_t global_worksize,
    cl_device_type dev_type_) const {
  std::stringstream out;

  global_worksize = 16384;

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

  out << getCodeCreateGroupAggregateStructures(
      groupingAttributes, aggregationAttributes, "current_result_size");

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
  out << "uint64_t grouping_keys_length = current_result_size;" << std::endl;
  out << "uint64_t source_indices_length = current_result_size;" << std::endl;
  out << getCodeInitOCLResultBuffers("current_result_size", dev_type_, dev_id_,
                                     projection_kernel_output_vars_);
  /* generate code that calls projection kernel that writes the result */
  out << getCodeCallProjectionKernel(num_elements_for_loop, global_worksize,
                                     projection_kernel_input_vars_,
                                     projection_kernel_output_vars_);

  out << getCodeMultipassGroupAggregate(
      groupingAttributes, aggregationAttributes, "current_result_size");

  out << getCodeCreateGroupAggregateResult(groupingAttributes,
                                           aggregationAttributes);

  out << "}" << std::endl;
  return out.str();
}
}
}
