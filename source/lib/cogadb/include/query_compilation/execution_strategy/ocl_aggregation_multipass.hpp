/*
 * author: henning funke
 * date: 09.08.2016
 */

#ifndef OCL_AGGREGATION_MULTIPASS_H
#define OCL_AGGREGATION_MULTIPASS_H

#include <core/global_definitions.hpp>
#include <query_compilation/execution_strategy/ocl.hpp>
#include <query_compilation/execution_strategy/ocl_projection_three_phase.hpp>

namespace CoGaDB {

  namespace ExecutionStrategy {
    class OCLAggregationMultipass : public OCLProjectionThreePhase {
     public:
      OCLAggregationMultipass(bool use_host_ptr, MemoryAccessPattern mem_access,
                              cl_device_id dev_id);

      const std::pair<std::string, std::string> getCode(
          const ProjectionParam& param, const ScanParam& scanned_attributes,
          PipelineEndType pipe_end, const std::string& result_table_name,
          const std::map<std::string, AttributeReferencePtr>&
              columns_to_decompress);

      void addInstruction_impl(InstructionPtr instr);

      const std::string getCodeCallComputeKernels(
          const std::string& num_elements_for_loop, size_t global_worksize,
          cl_device_type dev_type_) const;

      const std::string getCodeCreateCustomStructuresAfterKernel(
          std::string num_elements) const;

      std::string getCodeMultipassAggregate(std::string input_size) const;

      const std::string getCodeCreateResult() const;

      std::vector<AttributeReference> aggregationAttributes;
      std::vector<AttributeReference> computedAttributes;
    };
  }
}

#endif
