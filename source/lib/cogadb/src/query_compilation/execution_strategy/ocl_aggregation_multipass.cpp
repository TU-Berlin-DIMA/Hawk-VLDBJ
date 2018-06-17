/*
 * author: henning funke
 * date: 09.08.2016
 */

#include <boost/thread.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/execution_strategy/ocl_aggregation_multipass.hpp>
#include <query_compilation/primitives/aggregationPrimitives.hpp>
#include <query_compilation/primitives/loopPrimitives.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCLAggregationMultipass::OCLAggregationMultipass(bool use_host_ptr,
                                                 MemoryAccessPattern mem_access,
                                                 cl_device_id dev_id)
    : OCLProjectionThreePhase(use_host_ptr, mem_access, dev_id) {}

const std::pair<std::string, std::string> OCLAggregationMultipass::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  std::pair<std::string, std::string> code = OCLProjectionThreePhase::getCode(
      param, scanned_attributes, pipe_end, result_table_name,
      columns_to_decompress);

  std::string host_code = code.first;

  //    createResultArray_double("LINEORDER.LO_REVENUE.1",
  //    result_array_LINEORDER_LO_REVENUE_1, current_result_size)

  for (uint i = 0; i < computedAttributes.size(); i++) {
    auto aggr = aggregationAttributes[i];
    auto comp = computedAttributes[i];
    std::string type = "double";
    if (aggr.getAttributeReferenceType() == INPUT_ATTRIBUTE) {
      type = "float";
    }
    std::string output_var = " reduced_" + getVarName(aggr, false);
    std::string findStr = "createResultArray_double(\"" +
                          comp.getResultAttributeName() + "\", " +
                          getResultArrayVarName(comp);
    std::string replaceStr = "createResultArray_" + type + "(\"" +
                             aggr.getResultAttributeName() + "\", " +
                             output_var;

    std::cout << "find: " << findStr << std::endl;
    std::cout << "replace: " << replaceStr << std::endl;

    while (host_code.find(findStr) < std::string::npos) {
      int pos = host_code.find(findStr);
      host_code.replace(pos, findStr.length(), replaceStr);
    }
  }

  std::stringstream out;
  out << host_code;
  /* copy data from device back to CPU main memory, if required */
  code.first = out.str();
  return code;
}

void OCLAggregationMultipass::addInstruction_impl(InstructionPtr instr) {
  if (instr->getInstructionType() == AGGREGATE_INSTR) {
    auto aggregate = boost::static_pointer_cast<Aggregation>(instr);
    ProjectionParam param;
    AggregateSpecificationPtr agg_spec =
        aggregate->getAggregateSpecifications();
    auto atts = agg_spec->getScannedAttributes();
    for (auto a : atts) {
      param.push_back(*a);
      aggregationAttributes.push_back(*a);
      std::cout << "INPUT VARNAME: " << getResultArrayVarName(*a) << std::endl;
    }
    auto comp_atts = agg_spec->getComputedAttributes();
    for (auto a : comp_atts) {
      computedAttributes.push_back(*a);
      std::cout << "OUTPUT VARNAME: " << getResultArrayVarName(*a) << std::endl;
    }
    InstructionPtr ptr = boost::make_shared<Materialization>(param);
    OCLProjectionThreePhase::addInstruction_impl(ptr);

  } else {
    OCLProjectionThreePhase::addInstruction_impl(instr);
  }
}

const std::string
OCLAggregationMultipass::getCodeCreateCustomStructuresAfterKernel(
    std::string num_elements) const {
  std::stringstream out;
  /*
   for(auto a : aggregationAttributes) {
     out << "cl_mem cl_mem_" << getVarName(a, false) << "_reduced"
         << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE,"
         << "sizeof (" << getResultType(a, false) << ")*" << num_elements << ","
         << "NULL, &_err);" << std::endl
         << "assert(_err == CL_SUCCESS);" << std::endl;
   }*/
  return out.str();
}

std::string OCLAggregationMultipass::getCodeMultipassAggregate(
    std::string input_size) const {
  std::string output_size;
  std::stringstream code;

  code << "ocl_start_timer();" << std::endl;

  std::map<std::string, std::string> aggregation_vars;

  // reduce each column:
  for (auto a : aggregationAttributes) {
    std::string varname = getResultArrayVarName(a);
    std::string type = getResultType(a, false);
    std::string output_var = " reduced_" + getVarName(a, false);

    code << getCodeReduction(type, varname, output_var + "[0]");
  }
  code << "current_result_size=1;" << std::endl;

  code << "ocl_stop_timer(\"multipass aggregation\");" << std::endl;

  return code.str();
}

const std::string OCLAggregationMultipass::getCodeCreateResult() const {
  std::stringstream ss;
  /*
    for(auto a : aggregationAttributes) {
       std::string type = getResultType(a, false);
       std::string varname = "cl_mem_" + getVarName(a,false) + "_reduced";
       std::string host_varname = getResultArrayVarName(a);

       ss << "_err = "
             "clEnqueueReadBuffer(ocl_getTransferDeviceToHostCommandQueue("
             "context), "
          << varname << ", CL_TRUE, 0,"
          << "sizeof (" << type << ") * 1"
          << ", " << host_varname << ", "
          << "0, NULL, NULL);" << std::endl
          << "assert(_err == CL_SUCCESS);" << std::endl;
    }
    ss << "current_result_size=1;" << std::endl;
  */
  return ss.str();
}

const std::string OCLAggregationMultipass::getCodeCallComputeKernels(
    const std::string& num_elements_for_loop, size_t global_worksize,
    cl_device_type dev_type_) const {
  std::stringstream out;

  global_worksize = 16384;

  for (auto a : aggregationAttributes) {
    std::string varname = getResultArrayVarName(a);
    std::string type = getResultType(a, false);
    std::string output_var = " reduced_" + getVarName(a, false);

    out << type << "* " << output_var << " = malloc(sizeof(" << type << "));"
        << std::endl;
  }

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
  out << getCodeInitOCLResultBuffers("current_result_size", dev_type_, dev_id_,
                                     projection_kernel_output_vars_);
  /* generate code that calls projection kernel that writes the result */
  out << getCodeCallProjectionKernel(num_elements_for_loop, global_worksize,
                                     projection_kernel_input_vars_,
                                     projection_kernel_output_vars_);

  out << getCodeMultipassAggregate("current_result_size");

  out << getCodeCreateResult();

  out << "}" << std::endl;
  return out.str();
}
}
}
