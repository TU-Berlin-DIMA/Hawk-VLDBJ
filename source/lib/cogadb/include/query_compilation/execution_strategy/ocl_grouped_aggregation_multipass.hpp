/*
 * author: henning funke
 * date: 27.07.2016
 */

#ifndef OCL_GROUPED_AGGREGATION_MULTIPASS_H
#define OCL_GROUPED_AGGREGATION_MULTIPASS_H

#include <core/global_definitions.hpp>
#include <query_compilation/execution_strategy/ocl.hpp>
#include <query_compilation/execution_strategy/ocl_projection_three_phase.hpp>

namespace CoGaDB {

  namespace ExecutionStrategy {
    class OCLGroupedAggregationMultipass : public OCLProjectionThreePhase {
     public:
      OCLGroupedAggregationMultipass(bool use_host_ptr,
                                     MemoryAccessPattern mem_access,
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

      const std::string getCodeCleanupCustomStructures() const;

      std::vector<AttributeReference> groupingAttributes;
      std::vector<AttributeReference> aggregationAttributes;
      // for output variable name replacement
      std::vector<AttributeReference> computedAttributes;

      std::map<std::string, std::string> grouping_result_vars_;
    };
  }
}

#endif
