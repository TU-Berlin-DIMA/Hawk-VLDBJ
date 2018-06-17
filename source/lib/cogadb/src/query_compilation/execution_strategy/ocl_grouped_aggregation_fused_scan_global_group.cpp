/*
 * author: henning funke
 * date: 29.08.2016
 */

#include <query_compilation/execution_strategy/ocl_grouped_aggregation_fused_scan_global_group.hpp>
//#include <core/variable_manager.hpp>
#include <query_compilation/execution_strategy/multipass_grouping.h>
#include <boost/thread.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/primitives/aggregationPrimitives.hpp>
#include <query_compilation/primitives/loopPrimitives.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLGroupedAggregationFusedScanGlobalGroup::
    OCLGroupedAggregationFusedScanGlobalGroup(bool use_host_ptr,
                                              MemoryAccessPattern mem_access,
                                              cl_device_id dev_id)
    : OCLProjectionSinglePassScan(use_host_ptr, mem_access, dev_id) {}

const std::pair<std::string, std::string>
OCLGroupedAggregationFusedScanGlobalGroup::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  std::pair<std::string, std::string> code =
      OCLProjectionSinglePassScan::getCode(param, scanned_attributes, pipe_end,
                                           result_table_name,
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

void OCLGroupedAggregationFusedScanGlobalGroup::addInstruction_impl(
    InstructionPtr instr) {
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
      }
      auto outputs = spec->getComputedAttributes();
      for (auto o : outputs) {
        computedAttributes.push_back(*o);
      }
    }
    InstructionPtr ptr = boost::make_shared<Materialization>(param);
    OCLGroupedAggregationFusedScanGlobalGroup::addInstruction_impl(ptr);
  } else if (instr->getInstructionType() == BITPACKED_GROUPING_KEY_INSTR) {
    GeneratedCodePtr gen_code = instr->getCode(OCL_TARGET_CODE);
    CodeBlockPtr block = kernel_->kernel_code_blocks.back();
    kernel_output_vars_.insert(
        std::make_pair<std::string, std::string>("grouping_keys", "uint64_t*"));
    kernel_output_vars_.insert(std::make_pair<std::string, std::string>(
        "source_indices", "uint64_t*"));
    block->materialize_result_block.insert(
        block->materialize_result_block.end(),
        gen_code->upper_code_block_.begin(), gen_code->upper_code_block_.end());
    block->materialize_result_block.push_back(
        getCodeProjectGroupAggregateData());
  } else {
    OCLProjectionSinglePassScan::addInstruction_impl(instr);
  }
}

const std::string
OCLGroupedAggregationFusedScanGlobalGroup::getCodeCleanupCustomStructures()
    const {
  std::stringstream out;
  out << OCLProjectionSinglePassScan::getCodeCleanupCustomStructures();
  out << getCodeCleanupGroupAggregateStructures(groupingAttributes,
                                                aggregationAttributes);
  return out.str();
}

const std::string
OCLGroupedAggregationFusedScanGlobalGroup::getCodeCallComputeKernels(
    const std::string& num_elements_for_loop, size_t global_worksize,
    cl_device_type dev_type_) const {
  std::stringstream out;
  out << "current_result_size=allocated_result_elements;" << std::endl;
  /* in case output is not empty, we initialize the previously declared output
   * buffers */
  out << "uint64_t grouping_keys_length = allocated_result_elements;"
      << std::endl;
  out << "uint64_t source_indices_length = allocated_result_elements;"
      << std::endl;
  out << getCodeInitOCLResultBuffers("allocated_result_elements", dev_type_,
                                     dev_id_, kernel_output_vars_);
  /* generate code that calls the kernel that writes the result */
  out << getCodeCallSinglePassScanKernel(num_elements_for_loop, global_worksize,
                                         local_worksize_, kernel_input_vars_,
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
  out << "current_result_size=size_tmp;" << std::endl;
  out << "allocated_result_elements=current_result_size;" << std::endl;
  out << "printf(\"current_result_size: %lu\\n\", current_result_size);"
      << std::endl;
  out << getCodeCreateGroupAggregateStructures(
      groupingAttributes, aggregationAttributes, "current_result_size");
  /* in case output is not empty, we initialize the previously declared output
   * buffers */
  out << "if(current_result_size>0){" << std::endl;
  out << getCodeMultipassGroupAggregate(
      groupingAttributes, aggregationAttributes, "current_result_size");
  out << getCodeCreateGroupAggregateResult(groupingAttributes,
                                           aggregationAttributes);
  out << "}" << std::endl;
  return out.str();
}

}  // end namespace ExecutionStrategy

}  // end namespace CoGaDB
