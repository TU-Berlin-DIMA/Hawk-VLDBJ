#ifndef OCL_PROJECTION_SERIAL_SINGLE_PASS_HPP
#define OCL_PROJECTION_SERIAL_SINGLE_PASS_HPP

#include <core/global_definitions.hpp>

#include <query_compilation/execution_strategy/ocl.hpp>
#include <query_compilation/execution_strategy/ocl_projection.hpp>

namespace CoGaDB {
  namespace ExecutionStrategy {
    class OCLProjectionSerialSinglePass : public OCLProjection {
     public:
      OCLProjectionSerialSinglePass(bool use_host_ptr,
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

     private:
      GeneratedKernelPtr kernel_;
      std::map<std::string, std::string> kernel_input_vars_;
      std::map<std::string, std::string> kernel_output_vars_;
      std::stringstream kernel_init_vars_;
    };
  }
}

#endif  // OCL_PROJECTION_SERIAL_SINGLE_PASS_HPP
