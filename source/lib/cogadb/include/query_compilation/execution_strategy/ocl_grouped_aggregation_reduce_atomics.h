
/*
 * author: henning funke
 * date: 08.07.2016
 */

#ifndef OCL_GROUPED_AGGREGATION_REDUCE_ATOMICS_H
#define OCL_GROUPED_AGGREGATION_REDUCE_ATOMICS_H

#include <core/global_definitions.hpp>

#include <query_compilation/execution_strategy/ocl.hpp>

namespace CoGaDB {

  namespace ExecutionStrategy {
    class OCLGroupedAggregationLocalReduceAtomics : public OCL {
     public:
      OCLGroupedAggregationLocalReduceAtomics(bool use_host_ptr,
                                              MemoryAccessPattern mem_access,
                                              cl_device_id dev_id);

      void addInstruction_impl(InstructionPtr instr);

      const std::pair<std::string, std::string> getCode(
          const ProjectionParam& param, const ScanParam& scanned_attributes,
          PipelineEndType pipe_end, const std::string& result_table_name,
          const std::map<std::string, AttributeReferencePtr>&
              columns_to_decompress);

     private:
      std::string getKernelCode(uint64_t global_worksize);
      // std::string getAggregationCode(
      //    const GroupingAttributes& grouping_columns,
      //    const AggregateSpecifications& aggregation_specs,
      //    const std::string access_ht_entry_expression);
      std::string getAggregationPayloadStruct();

      std::string getCreateAggregationHashMapBuffer();

      std::string getCallKernels(uint64_t global_worksize,
                                 size_t local_worksize);

      std::string getCreateResult();
      std::string getCodeCheckKeyInHashTable();
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

      size_t global_worksize_;
      size_t local_worksize_;
      int values_per_thread_;
      bool use_buffer_;

      const static std::string AggregationHashTableVarName;
      const static std::string AggregationHashTableLengthVarName;
      const static std::string InvalidKey;
      const static std::string InvalidKey32Bit;

      bool hack_enable_manual_ht_size_;
    };

  }  // namespace ExecutionStrategy

}  // namespace CoGaDB

#endif  // OCL_GROUPED_AGGREGATION_REDUCE_ATOMICS_H
