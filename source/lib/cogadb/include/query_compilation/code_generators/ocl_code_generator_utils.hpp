

#ifndef OCL_CODE_GENERATOR_UTILS_HPP
#define OCL_CODE_GENERATOR_UTILS_HPP

#define GENERATE_PROFILING_CODE

#include <iomanip>
#include <list>
#include <set>
#include <sstream>
#include <string>

#include <core/attribute_reference.hpp>
#include <query_compilation/code_generator.hpp>
#include <util/opencl_runtime.hpp>

namespace CoGaDB {

  namespace ExecutionStrategy {
    class GeneratedKernel;
    typedef boost::shared_ptr<GeneratedKernel> GeneratedKernelPtr;
  }

  bool isPointerType(const std::string& type);

  std::string removePointer(std::string type);

  const std::string getCodeKernelSignature(
      const std::string& name, const std::string& extra_vars,
      const std::map<std::string, std::string>& input_vars =
          std::map<std::string, std::string>(),
      const std::map<std::string, std::string>& output_vars =
          std::map<std::string, std::string>());

  const std::string getCodeFilterKernelSignature(
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars);

  const std::string getCodeProjectionKernelSignature(
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars);

  const std::string getCLBufferInputVarName(const AttributeReference& attr);
  const std::string getCLBufferInputVarName(const std::string& varname);

  const std::string getCLBufferResultVarName(const AttributeReference& attr);
  const std::string getCLBufferResultVarName(const std::string& varname);

  const std::string getKernelHeadersAndTypes();

  const std::pair<std::string, std::string> getCodeOCLKernelMemoryTraversal(
      const std::string& loop_variable_name, MemoryAccessPattern mem_access);

  const std::string getCodeFilterKernel(
      ExecutionStrategy::GeneratedKernelPtr filter_kernel_,
      const std::string& loop_variable_name, size_t global_worksize,
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars,
      MemoryAccessPattern mem_access_);

  const std::string getCodeProjectionKernel(
      ExecutionStrategy::GeneratedKernelPtr projection_kernel,
      const std::string& loop_variable_name, size_t global_worksize,
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars,
      MemoryAccessPattern mem_access);

  const std::string getCodeSerialSinglePassKernel(
      ExecutionStrategy::GeneratedKernelPtr kernel_,
      const std::string& loop_variable_name,
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars,
      MemoryAccessPattern mem_access);

  const std::string getCodeParallelGlobalAtomicSinglePassKernel(
      ExecutionStrategy::GeneratedKernelPtr kernel_,
      const std::string& loop_variable_name,
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars,
      MemoryAccessPattern mem_access);

  const std::string getCodeAggregationKernel(
      ExecutionStrategy::GeneratedKernelPtr aggr_kernel_,
      const std::string& loop_variable_name, uint64_t global_worksize,
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars,
      const std::map<std::string, std::string>& aggr_vars,
      MemoryAccessPattern mem_access_);

  const std::string getCodeCreateOCLDefaultKernelInputBuffers(
      const std::string& num_elements_for_loop, cl_device_type dev_type);

  const std::string getCodeCreateOCLInputBuffers(
      const std::string& num_elements_for_loop,
      const std::map<std::string, std::string>& variables,
      cl_device_type dev_type, cl_device_id dev_id, bool enable_caching);

  const std::string getCodeCopyOCLResultValueToVariable(
      const std::pair<std::string, std::string>& var);

  const std::string getCodeCallFilterKernel(
      const std::string& num_elements_for_loop, size_t global_worksize,
      const std::map<std::string, std::string>& variables);

  const std::string getCodeCallProjectionKernel(
      const std::string& num_elements_for_loop, size_t global_worksize,
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars);

  const std::string getCodeCallSerialSinglePassKernel(
      const std::string& num_elements_for_loop,
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars);

  const std::string getCodeCallParallelGlobalAtomicSinglePassKernel(
      const std::string& num_elements_for_loop, uint64_t global_worksize,
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars);

  const std::string getCodeCallAggregationKernel(
      const std::string& num_elements_for_loop, uint64_t global_worksize,
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars);

  const std::string getCodePrefixSum(
      const std::string& current_result_size_var_name,
      const std::string& num_elements_for_loop,
      const std::string& cl_mem_flag_array_name,
      const std::string& cl_mem_prefix_sum_array_name);

  const std::string getCoderReduceAggregation(
      const std::map<std::string, std::string>& aggr_variables,
      const uint64_t global_worksize);

  const std::string getCodeReduction(std::string type, std::string input_var,
                                     std::string output_var);

  const std::string getCodeSort_uint64_t(std::string keys, std::string vals,
                                         std::string num_elements);

  const std::string getCodeReduceByKeys(
      std::string num_elements, std::string keys, std::string keys_out,
      std::pair<std::string, std::string> values,
      std::pair<std::string, std::string> values_out,
      bool take_any_value = false);

  const std::string getCodeGather(std::string map, std::string value_type,
                                  std::string values_in, std::string values_out,
                                  std::string num_elements);

  std::string getCodePrintBuffer(std::string num_elements, std::string mem,
                                 std::string type);

  const std::string getCodeDeclareProjectionKernel();

  const std::string getCodeDeclareOCLResultBuffer(
      const std::pair<std::string, std::string>& var);

  const std::string getCodeDeclareOCLResultBuffers(
      const std::map<std::string, std::string>& variables);

  const std::string getCodeDeclareOCLAggregationResultBuffers(
      const std::map<std::string, std::string>& variables,
      const std::map<std::string, std::string>& aggr_variables,
      const uint64_t global_work_size);

  const std::string getCodeInitOCLAggregationResultBuffers(
      const std::string& current_result_size_var_name, cl_device_type dev_type,
      cl_device_id dev_id, const std::map<std::string, std::string>& variables,
      const std::map<std::string, std::string>& aggr_variables,
      uint64_t global_worksize);

  const std::string getCodeInitOCLResultBuffer(
      cl_device_type dev_type, cl_device_id dev_id,
      const std::pair<std::string, std::string>& var, bool init_buffer = false,
      const std::string& init_value = "",
      bool use_host_pointer_if_possible = true);

  const std::string getCodeInitOCLResultBuffers(
      const std::string& current_result_size_var_name, cl_device_type dev_type,
      cl_device_id dev_id, const std::map<std::string, std::string>& variables);

  const std::string getCodeCopyOCLResultBuffersToHost(
      const std::string& current_result_size_var_name, cl_device_type dev_type,
      cl_device_id dev_id, const std::map<std::string, std::string>& variables);

  const std::string getCodeCopyOCLAggregationResultBuffersToHost(
      const std::string& current_result_size_var_name, cl_device_type dev_type,
      cl_device_id dev_id, const std::map<std::string, std::string>& variables,
      const std::map<std::string, std::string>& aggr_vars);

  const std::string getCodeCopyOCLGroupedAggregationResultBuffersToHost(
      const std::string& current_result_size_var_name, cl_device_type dev_type,
      cl_device_id dev_id, const std::map<std::string, std::string>& variables,
      const std::string& hash_map_name);

  const std::string getCodeCleanupDefaultKernelOCLStructures();

  const std::string getCodeCleanupOCLStructures(
      const std::map<std::string, std::string>& input_vars,
      const std::map<std::string, std::string>& output_vars);

  const std::string getCodeCleanupOCLAggregationStructures(
      const std::map<std::string, std::string>& aggr_vars);

  const std::string getCLInputVarName(const AttributeReference& ref);

  const std::string getCLResultVarName(const AttributeReference& ref);

  const std::string getCodeMurmur3Hashing(bool version_32bit = false);

  const std::string getCodeMultiplyShift();

  const std::string getCodeMultiplyAddShift();

  const std::string getCodeReleaseKernel(const std::string& kernel_var_name);
}

#endif  // OCL_CODE_GENERATOR_UTILS_HPP
