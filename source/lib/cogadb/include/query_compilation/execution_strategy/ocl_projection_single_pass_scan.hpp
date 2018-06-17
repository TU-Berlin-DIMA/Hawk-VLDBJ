/*
 * author: henning funke
 * date: 22.06.2016
 */

#ifndef OCL_PROJECTION_SINGLE_PASS_SCAN_HPP
#define OCL_PROJECTION_SINGLE_PASS_SCAN_HPP

#include <core/global_definitions.hpp>
#include <query_compilation/execution_strategy/ocl.hpp>
#include <query_compilation/execution_strategy/ocl_projection.hpp>

namespace CoGaDB {
  namespace ExecutionStrategy {

    enum WritePositionMode { LOCAL_RESOLUTION, ATOMICS_ONLY, FRAGMENTED_WRITE };

    const std::string getCodeSinglePassScanKernel(
        ExecutionStrategy::GeneratedKernelPtr kernel_,
        const std::string& loop_variable_name,
        const std::map<std::string, std::string>& input_vars,
        const std::map<std::string, std::string>& output_vars,
        MemoryAccessPattern mem_access, size_t local_worksize,
        WritePositionMode write_pos_mode, GeneratedCodePtr code);

    const std::string getCodeCallSinglePassScanKernel(
        const std::string& num_elements_for_loop, uint64_t global_worksize,
        size_t local_worksize,
        const std::map<std::string, std::string>& input_vars,
        const std::map<std::string, std::string>& output_vars);

    class OCLProjectionSinglePassScan : public OCLProjection {
     public:
      OCLProjectionSinglePassScan(bool use_host_ptr,
                                  MemoryAccessPattern mem_access,
                                  cl_device_id dev_id);

      void addInstruction_impl(InstructionPtr instr);
      const std::map<std::string, std::string> getInputVariables() const;
      const std::map<std::string, std::string> getOutputVariables() const;
      const std::string getCodeCallComputeKernels(
          const std::string& num_elements_for_loop, size_t global_worksize,
          cl_device_type dev_type_) const;
      const std::string getKernelCode(const std::string& loop_variable_name,
                                      size_t global_worksize,
                                      MemoryAccessPattern mem_access);
      const std::string getCodeCreateCustomStructures(
          const std::string& num_elements_for_loop, cl_device_type dev_type_);
      const std::string getCodeCleanupCustomStructures() const;

      const std::pair<std::string, std::string> getCode(
          const ProjectionParam& param, const ScanParam& scanned_attributes,
          PipelineEndType pipe_end, const std::string& result_table_name,
          const std::map<std::string, AttributeReferencePtr>&
              columns_to_decompress);

     protected:
      GeneratedKernelPtr kernel_;
      std::map<std::string, std::string> kernel_input_vars_;
      std::map<std::string, std::string> kernel_output_vars_;
      std::stringstream kernel_init_vars_;

      size_t global_worksize_;
      size_t local_worksize_;
      int values_per_thread_;
      WritePositionMode write_pos_mode_;
    };
  }
}

#endif
