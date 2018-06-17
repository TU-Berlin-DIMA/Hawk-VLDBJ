/*
 * author: henning funke
 * date: 29.08.2016
 */

#include <query_compilation/execution_strategy/multipass_grouping.h>
#include <core/attribute_reference.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>

namespace CoGaDB {

std::string getCodeProjectGroupAggregateData() {
  std::stringstream write_groupid;
  write_groupid << "grouping_keys[write_pos] = group_key;" << std::endl;
  write_groupid << "source_indices[write_pos] = write_pos;" << std::endl;
  return write_groupid.str();
}

std::string patchGroupAggregateKernelCode(std::string kernel_code) {
  std::stringstream kernel_lines;
  kernel_lines << kernel_code;
  std::string line;
  std::stringstream patched;
  while (std::getline(kernel_lines, line)) {
    std::string findA = "_COUNT[write_pos] =";
    int posA = line.find(findA);
    if (line.find(findA) < std::string::npos) {
      int posA = line.find(findA);
      std::string newl;
      newl = line.substr(0, posA) + findA + "1;";
      line = newl;
    }
    patched << line << std::endl;
  }
  return patched.str();
}

std::string patchGroupAggregateHostCode(
    std::string host_code,
    std::vector<AttributeReference> aggregationAttributes,
    std::vector<AttributeReference> computedAttributes) {
  // replace double with float for non-computed attributes
  std::string findStr;
  std::string replaceStr;
  for (uint i = 0; i < aggregationAttributes.size(); i++) {
    auto a = aggregationAttributes[i];
    auto c = computedAttributes[i];
    if (a.getAttributeType() == FLOAT) {
      findStr = "createResultArray_double(\"" + c.getResultAttributeName();
      replaceStr = "createResultArray_float(\"" + c.getResultAttributeName();
      while (host_code.find(findStr) < std::string::npos) {
        int pos = host_code.find(findStr);
        host_code.replace(pos, findStr.length(), replaceStr);
      }
    }
  }

  // replace generated name with specified name
  for (uint i = 0; i < computedAttributes.size(); i++) {
    findStr = getResultArrayVarName(aggregationAttributes[i]);
    replaceStr = getResultArrayVarName(computedAttributes[i]);
    if (findStr.compare(replaceStr) != 0) {
      while (host_code.find(findStr) < std::string::npos) {
        int pos = host_code.find(findStr);
        host_code.replace(pos, findStr.length(), replaceStr);
      }
    }
  }

  // replace float with int32_t for count
  findStr = "createResultArray_float(\"COUNT_";
  replaceStr = "createResultArray_int32_t(\"COUNT_";
  while (host_code.find(findStr) < std::string::npos) {
    int pos = host_code.find(findStr);
    host_code.replace(pos, findStr.length(), replaceStr);
  }

  return host_code;
}

std::string getCodeCreateGroupAggregateStructures(
    std::vector<AttributeReference> groupingAttributes,
    std::vector<AttributeReference> aggregationAttributes,
    std::string num_elements) {
  std::stringstream out;

  out << "printf(\"number of grouping inputs: %lu\\n\", " << num_elements
      << ");" << std::endl;

  out << "cl_mem cl_mem_grouping_keys_reduced "
      << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE,"
      << "sizeof (uint64_t)*" << num_elements << ","
      << "NULL, &_err);" << std::endl
      << "assert(_err == CL_SUCCESS);" << std::endl;

  out << "cl_mem cl_mem_source_indices_reduced "
      << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE,"
      << "sizeof (uint64_t)*" << num_elements << ","
      << "NULL, &_err);" << std::endl
      << "assert(_err == CL_SUCCESS);" << std::endl;

  for (auto a : aggregationAttributes) {
    out << "cl_mem cl_mem_" << getVarName(a, false) << "_sorted_reduced"
        << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE,"
        << "sizeof (" << getResultType(a, false) << ")*" << num_elements << ","
        << "NULL, &_err);" << std::endl
        << "assert(_err == CL_SUCCESS);" << std::endl;
    out << "cl_mem cl_mem_" << getVarName(a, false) << "_sorted"
        << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE,"
        << "sizeof (" << getResultType(a, false) << ")*" << num_elements << ","
        << "NULL, &_err);" << std::endl
        << "assert(_err == CL_SUCCESS);" << std::endl;
  }
  for (auto a : groupingAttributes) {
    out << "cl_mem cl_mem_" << getVarName(a, false) << "_gathered"
        << "= clCreateBuffer(ocl_getContext(context), CL_MEM_READ_WRITE,"
        << "sizeof (" << getResultType(a, false) << ")*" << num_elements << ","
        << "NULL, &_err);" << std::endl
        << "assert(_err == CL_SUCCESS);" << std::endl;
  }
  return out.str();
}

std::string getCodeCleanupGroupAggregateStructures(
    std::vector<AttributeReference> groupingAttributes,
    std::vector<AttributeReference> aggregationAttributes) {
  std::stringstream out;
  out << "CL_CHECK(clReleaseMemObject(cl_mem_grouping_keys_reduced));"
      << std::endl;
  out << "CL_CHECK(clReleaseMemObject(cl_mem_source_indices_reduced));"
      << std::endl;
  for (auto a : aggregationAttributes) {
    out << "CL_CHECK(clReleaseMemObject(cl_mem_" << getVarName(a, false)
        << "_sorted));" << std::endl;
    out << "CL_CHECK(clReleaseMemObject(cl_mem_" << getVarName(a, false)
        << "_sorted_reduced));" << std::endl;
  }
  for (auto a : groupingAttributes) {
    out << "CL_CHECK(clReleaseMemObject(cl_mem_" << getVarName(a, false)
        << "_gathered));" << std::endl;
  }
  return out.str();
}

std::string getCodeMultipassGroupAggregate(
    std::vector<AttributeReference> groupingAttributes,
    std::vector<AttributeReference> aggregationAttributes,
    std::string input_size) {
  std::string output_size;
  std::stringstream code;
  code << "ocl_start_timer();" << std::endl;
  code << getCodeSort_uint64_t("cl_result_mem_grouping_keys",
                               "cl_result_mem_source_indices", input_size);
  for (auto a : aggregationAttributes) {
    std::string varname = "cl_mem_" + getVarName(a, false);
    std::string type = getResultType(a, false);
    std::pair<std::string, std::string> aggregation_attribute(
        "cl_result_mem_" + getResultArrayVarName(a), type);
    std::pair<std::string, std::string> attribute_sorted(varname + "_sorted",
                                                         type);
    std::pair<std::string, std::string> attribute_reduced(
        varname + "_sorted_reduced", type);
    code << getCodeGather("cl_result_mem_source_indices", type,
                          aggregation_attribute.first, attribute_sorted.first,
                          input_size);
    code << "uint64_t num_groups ";
    code << getCodeReduceByKeys(
        "current_result_size", "cl_result_mem_grouping_keys",
        "cl_mem_grouping_keys_reduced", attribute_sorted, attribute_reduced);
  }
  std::pair<std::string, std::string> tid_vals_in(
      "cl_result_mem_source_indices", "uint64_t");
  std::pair<std::string, std::string> tid_vals_out(
      "cl_mem_source_indices_reduced", "uint64_t");

  // reduce by keys: keep any index for grouping attribute values
  code << "num_groups ";
  code << getCodeReduceByKeys(
      "current_result_size", "cl_result_mem_grouping_keys",
      "cl_mem_grouping_keys_reduced", tid_vals_in, tid_vals_out, true);

  // post processing: get grouping attribute values
  for (auto a : groupingAttributes) {
    std::string varname = "cl_mem_" + getVarName(a, false);
    std::string type = getResultType(a, false);
    std::pair<std::string, std::string> grouping_attribute(
        "cl_result_mem_" + getResultArrayVarName(a), type);
    std::pair<std::string, std::string> attribute_gathered(
        varname + "_gathered", type);
    code << getCodeGather("cl_mem_source_indices_reduced", type,
                          grouping_attribute.first, attribute_gathered.first,
                          "num_groups");
  }
  code << "ocl_stop_timer(\"multipass grouped aggregation\");" << std::endl;
  return code.str();
}

std::string getCodeCreateGroupAggregateResult(
    std::vector<AttributeReference> groupingAttributes,
    std::vector<AttributeReference> aggregationAttributes) {
  std::stringstream ss;
  for (auto a : aggregationAttributes) {
    std::string type = getResultType(a, false);
    std::string varname = "cl_mem_" + getVarName(a, false) + "_sorted_reduced";
    std::string host_varname = getResultArrayVarName(a);

    ss << "_err = "
          "clEnqueueReadBuffer(ocl_getTransferDeviceToHostCommandQueue("
          "context), "
       << varname << ", CL_TRUE, 0,"
       << "sizeof (" << type << ") * "
       << "num_groups"
       << ", " << host_varname << ", "
       << "0, NULL, NULL);" << std::endl
       << "assert(_err == CL_SUCCESS);" << std::endl;
  }

  for (auto a : groupingAttributes) {
    std::string type = getResultType(a, false);
    std::string varname = "cl_mem_" + getVarName(a, false) + "_gathered";
    std::string host_varname = getResultArrayVarName(a);

    ss << "_err = "
          "clEnqueueReadBuffer(ocl_getTransferDeviceToHostCommandQueue("
          "context), "
       << varname << ", CL_TRUE, 0,"
       << "sizeof (" << type << ") * "
       << "num_groups"
       << ", " << host_varname << ", "
       << "0, NULL, NULL);" << std::endl
       << "assert(_err == CL_SUCCESS);" << std::endl;
  }
  ss << "current_result_size=num_groups;" << std::endl;

  return ss.str();
}

}  // end namespace CoGaDB
