#ifndef OCL_GROUPED_AGGREGATION_ATOMIC_H
#define OCL_GROUPED_AGGREGATION_ATOMIC_H

#include <core/global_definitions.hpp>

#include <query_compilation/execution_strategy/ocl.hpp>

namespace CoGaDB {

  namespace ExecutionStrategy {
    class OCLGroupedAggregationAtomic : public OCL {
     public:
      OCLGroupedAggregationAtomic(bool use_host_ptr,
                                  MemoryAccessPattern mem_access,
                                  cl_device_id dev_id);

      void addInstruction_impl(InstructionPtr instr) override;

      const std::pair<std::string, std::string> getCode(
          const ProjectionParam& param, const ScanParam& scanned_attributes,
          PipelineEndType pipe_end, const std::string& result_table_name,
          const std::map<std::string, AttributeReferencePtr>&
              columns_to_decompress) override;

     protected:
      virtual std::string getKernelCode(uint64_t global_worksize);
      void get64BitAtomics(std::stringstream& stream);
      void get32BitAtomics(std::stringstream& stream);
      std::string getAggregationCode(
          const GroupingAttributes& grouping_columns,
          const AggregateSpecifications& aggregation_specs,
          const std::string access_ht_entry_expression);
      std::string getAggregationPayloadStruct();

      std::string getCreateAggregationHashMapBuffer();

      std::string getCallKernels(uint64_t global_worksize,
                                 uint64_t local_worksize = 0);
      virtual std::string getInitKernelGlobalLocalSize(
          uint64_t global_worksize);
      virtual std::string getAggKernelGlobalLocalSize(uint64_t global_worksize,
                                                      uint64_t local_worksize);
      virtual std::string getInitKernelHashMapSize(uint64_t global_worksize);

      std::string getCreateResult(uint64_t work_group_count);
      virtual std::string getCodeCheckKeyInHashTable();
      std::string getCodePayloadIndexFunctionImpl();
      std::string getCodePayloadIndexFunctionLinearProbingImpl();
      std::string getCodePayloadIndexFunctionQuadraticProbingImpl();
      std::string getCodePayloadIndexFunctionCuckooHashingImpl();
      std::pair<std::string, std::string> getCodeInitKernelIndexCalculation(
          uint64_t global_worksize);

      std::string getCodeMurmur3Hashing();

      LoopPtr first_loop_;
      GeneratedKernelPtr aggregation_kernel_;
      std::map<std::string, std::string> aggregation_kernel_input_vars_;
      std::map<std::string, std::string> aggregation_kernel_output_vars_;

      GroupingAttributes grouping_attrs_;
      AggregateSpecifications aggr_specs_;
      ProjectionParam projection_param_;
      bool atomics_32bit_;
      bool hack_enable_manual_ht_size_;

      const static std::string AggregationHashTableVarName;
      const static std::string AggregationHashTableLengthVarName;
      const static std::string InvalidKey;
      const static std::string InvalidKey32Bit;
    };

  }  // namespace ExecutionStrategy

}  // namespace CoGaDB

#endif  // OCL_GROUPED_AGGREGATION_H
