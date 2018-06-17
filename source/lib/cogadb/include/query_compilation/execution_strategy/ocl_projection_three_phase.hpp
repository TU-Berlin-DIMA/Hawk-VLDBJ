#ifndef OCL_PROJECTION_THREE_PHASE_EXECUTION_STRATEGY_HPP
#define OCL_PROJECTION_THREE_PHASE_EXECUTION_STRATEGY_HPP

#include <core/global_definitions.hpp>

#include <query_compilation/execution_strategy/ocl.hpp>
#include <query_compilation/execution_strategy/ocl_projection.hpp>

namespace CoGaDB {
  namespace ExecutionStrategy {
    class OCLProjectionThreePhase : public OCLProjection {
     public:
      OCLProjectionThreePhase(bool use_host_ptr, MemoryAccessPattern mem_access,
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

     protected:
      GeneratedKernelPtr filter_kernel_;
      GeneratedKernelPtr projection_kernel_;
      std::map<std::string, std::string> projection_kernel_input_vars_;
      std::map<std::string, std::string> projection_kernel_output_vars_;
      std::stringstream projection_kernel_init_vars_;
      std::map<std::string, std::string> filter_kernel_input_vars_;
      std::map<std::string, std::string> filter_kernel_output_vars_;
      std::stringstream filter_kernel_init_vars_;
    };
  }
}

#endif  // OCL_PROJECTION_THREE_PHASE_EXECUTION_STRATEGY_HPP
