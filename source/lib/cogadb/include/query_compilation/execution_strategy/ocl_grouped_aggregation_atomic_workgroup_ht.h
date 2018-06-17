#ifndef OCL_GROUPED_AGGREGATION_ATOMIC_WORKGROUP_HT_H
#define OCL_GROUPED_AGGREGATION_ATOMIC_WORKGROUP_HT_H

#include <core/global_definitions.hpp>

#include <query_compilation/execution_strategy/ocl_grouped_aggregation_atomic.h>

namespace CoGaDB {

  namespace ExecutionStrategy {

    class OCLGroupedAggregationAtomicWorkGroupHT
        : public OCLGroupedAggregationAtomic {
     public:
      OCLGroupedAggregationAtomicWorkGroupHT(bool use_host_ptr,
                                             MemoryAccessPattern mem_access,
                                             cl_device_id dev_id);

      void addInstruction_impl(InstructionPtr instr) override;

      const std::pair<std::string, std::string> getCode(
          const ProjectionParam& param, const ScanParam& scanned_attributes,
          PipelineEndType pipe_end, const std::string& result_table_name,
          const std::map<std::string, AttributeReferencePtr>&
              columns_to_decompress) override;

     private:
      std::string getKernelCode(uint64_t global_worksize) override;
      std::string getCopyHashTableEntryCode(
          const GroupingAttributes& grouping_columns,
          const AggregateSpecifications& aggregation_specs,
          const std::string access_src_ht_entry_expression,
          const std::string access_dst_ht_entry_expression);

      std::string getCreateAggregationHashMapBuffer(uint64_t global_worksize);

      std::string getAggKernelGlobalLocalSize(uint64_t global_worksize,
                                              uint64_t local_worksize) override;
      std::string getInitKernelHashMapSize(uint64_t global_worksize) override;

      std::string getCodeCheckKeyInHashTable() override;

      uint64_t adjustGlobalWorkSize(uint64_t global_worksize);
      double estimateHashTableSize(uint64_t global_worksize,
                                   float hashtable_size_multiplier);
    };

  }  // namespace ExecutionStrategy

}  // namespace CoGaDB

#endif  // OCL_GROUPED_AGGREGATION_ATOMIC_WORKGROUP_HT_H
