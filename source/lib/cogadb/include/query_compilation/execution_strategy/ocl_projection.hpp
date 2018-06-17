#ifndef OCL_PROJECTION_HPP
#define OCL_PROJECTION_HPP

#include <core/global_definitions.hpp>

#include <query_compilation/execution_strategy/ocl.hpp>

namespace CoGaDB {
  namespace ExecutionStrategy {
    /* \brief this class is the base class for all
     * execution strategies that target projection
     * pipelines
     */
    class OCLProjection : public OCL {
     public:
      OCLProjection(bool use_host_ptr, MemoryAccessPattern mem_access,
                    cl_device_id dev_id);

      virtual void addInstruction_impl(InstructionPtr instr) = 0;

      const std::pair<std::string, std::string> getCode(
          const ProjectionParam& param, const ScanParam& scanned_attributes,
          PipelineEndType pipe_end, const std::string& result_table_name,
          const std::map<std::string, AttributeReferencePtr>&
              columns_to_decompress);

     protected:
      LoopPtr first_loop_;
      virtual const std::map<std::string, std::string> getInputVariables()
          const = 0;
      virtual const std::map<std::string, std::string> getOutputVariables()
          const = 0;
      virtual const std::string getCodeCallComputeKernels(
          const std::string& num_elements_for_loop, size_t global_worksize,
          cl_device_type dev_type_) const = 0;
      virtual const std::string getKernelCode(
          const std::string& loop_variable_name, size_t global_worksize,
          MemoryAccessPattern mem_access) = 0;
      virtual const std::string getCodeCreateCustomStructures(
          const std::string& num_elements_for_loop,
          cl_device_type dev_type_) = 0;
      virtual const std::string getCodeCleanupCustomStructures() const = 0;

     private:
      uint64_t getGlobalSize(cl_device_type dev_type, size_t num_elements);
    };
  }
}

#endif  // OCL_PROJECTION_HPP
