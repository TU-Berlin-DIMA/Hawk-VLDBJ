#ifndef OCL_AGGREGATION_H
#define OCL_AGGREGATION_H

#include <core/global_definitions.hpp>

#include <query_compilation/execution_strategy/ocl.hpp>

namespace CoGaDB {

  namespace ExecutionStrategy {
    /* \brief this class is the base class for all
     * execution strategies that target aggregation
     * pipelines
     */
    class OCLAggregation : public OCL {
     public:
      OCLAggregation(bool use_host_ptr, MemoryAccessPattern mem_access,
                     cl_device_id dev_id);

      void addInstruction_impl(InstructionPtr instr);

      const std::pair<std::string, std::string> getCode(
          const ProjectionParam& param, const ScanParam& scanned_attributes,
          PipelineEndType pipe_end, const std::string& result_table_name,
          const std::map<std::string, AttributeReferencePtr>&
              columns_to_decompress);

     private:
      LoopPtr first_loop_;
      GeneratedKernelPtr aggregation_kernel_;
      std::map<std::string, std::string> aggregation_kernel_input_vars_;
      std::map<std::string, std::string> aggregation_kernel_output_vars_;
      std::map<std::string, std::string> aggregation_kernel_aggregate_vars_;
    };

  }  // namespace ExecutionStrategy

}  // namespace CoGaDB

#endif  // OCL_AGGREGATION_H
