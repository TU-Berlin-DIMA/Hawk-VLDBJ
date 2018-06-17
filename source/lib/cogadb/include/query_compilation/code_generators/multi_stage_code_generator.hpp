#ifndef MULTI_STAGE_CODE_GENERATOR_HPP
#define MULTI_STAGE_CODE_GENERATOR_HPP

#include <dlfcn.h>
#include <boost/enable_shared_from_this.hpp>
#include <boost/filesystem.hpp>
#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <list>
#include <ostream>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/code_generators/c_code_generator.hpp>
#include <query_compilation/execution_strategy/pipeline.hpp>
#include <query_compilation/pipeline.hpp>
#include <sstream>

#include <query_compilation/aggregate_specification.hpp>
#include <query_compilation/predicate_expression.hpp>
#include <query_compilation/primitives/UDFPrimitives.hpp>
#include <query_compilation/primitives/aggregationPrimitives.hpp>
#include <query_compilation/primitives/algebraPrimitives.hpp>
#include <query_compilation/primitives/filterPrimitives.hpp>
#include <query_compilation/primitives/hashPrimitives.hpp>
#include <query_compilation/primitives/instruction.hpp>
#include <query_compilation/primitives/joinPrimitives.hpp>
#include <query_compilation/primitives/loopPrimitives.hpp>

namespace CoGaDB {

  enum SIMDType { SSE42, AVX2 };

  class MultiStageCodeGenerator
      : public CCodeGenerator,
        public boost::enable_shared_from_this<MultiStageCodeGenerator> {
   public:
    MultiStageCodeGenerator(const ProjectionParam& param, const TablePtr table,
                            uint32_t version = 1);

    MultiStageCodeGenerator(const ProjectionParam& param);

    void init();

    virtual bool consumeSelection_impl(const PredicateExpressionPtr pred_expr);

    virtual bool consumeBuildHashTable_impl(const AttributeReference& attr);

    virtual bool consumeProbeHashTable_impl(
        const AttributeReference& hash_table_attr,
        const AttributeReference& probe_attr);

    virtual bool consumeHashGroupAggregate_impl(
        const GroupByAggregateParam& param);

    virtual bool consumeAggregate_impl(const AggregateSpecifications& param);

    virtual bool consumeCrossJoin_impl(const AttributeReference& attr);

    virtual bool consumeNestedLoopJoin_impl(
        const PredicateExpressionPtr pred_expr);

    virtual const std::pair<bool, std::vector<AttributeReferencePtr>>
    consumeMapUDF_impl(const Map_UDF_ParamPtr param);

    virtual const std::pair<bool, AttributeReference>
    consumeAlgebraComputation_impl(const AttributeReference& left_attr,
                                   const AttributeReference& right_attr,
                                   const ColumnAlgebraOperation& alg_op);

    virtual const std::pair<bool, AttributeReference>
    consumeAlgebraComputation_impl(const AttributeReference& left_attr,
                                   const boost::any constant,
                                   const ColumnAlgebraOperation& alg_op);

    virtual const std::pair<bool, AttributeReference>
    consumeAlgebraComputation_impl(const boost::any constant,
                                   const AttributeReference& right_attr,
                                   const ColumnAlgebraOperation& alg_op);

    virtual void printCode(std::ostream& out);

    virtual const PipelinePtr compile();
    CodeGenerationTarget getTarget() const;
    const PipelinePtr compileOneVariant(
        const ExecutionStrategy::PipelinePtr exec_strat,
        cl_device_type& device_type, cl_device_id& device_id);

   private:
    const PipelinePtr skipCompilationOfEmptyPipeline();
    void setCodeGeneratorTarget();
    void createExecutionStrategyAndDeviceType(ExecutionStrategy::PipelinePtr&,
                                              cl_device_type& device_type,
                                              cl_device_id& device_id);
    void generateCode(std::string& host_source_code,
                      std::string& kernel_source_code,
                      ExecutionStrategy::PipelinePtr);
    std::list<InstructionPtr> programPrimitives;
    uint32_t programPos;
    CodeGenerationTarget target;
    //        void optimizePrimitivePipeline();
    //        void optimizePrimitivePipeline(std::list<InstructionPtr>&
    //        program);
    void generateCode_BitpackedGroupingKeyComputation(
        const GroupingAttributes& grouping_attrs);
    void generateCode_GenericGroupingKeyComputation(
        const GroupingAttributes& grouping_attrs);
    virtual bool createForLoop_impl(const TablePtr table, uint32_t version);

    virtual const std::string getCodeWriteResult() const;
    // const std::string getSIMDExpression(const Predicate& pred,SIMDType
    // SIMDtype ) const;
    /*      virtual bool createHashTable(const AttributeReference& attr);

          virtual const std::string getCodeAllocateResultTable() const;
          virtual const std::string getCodeWriteResultFromHashTable() const;
          virtual const std::string createResultTable() const;


          const std::string getCPPExpression(const Predicate& pred) const;
          const std::string getAggregationCode(const GroupingAttributes&
       grouping_columns, const AggregateSpecifications& aggregation_specs,
                                               const std::string
       access_ht_entry_expression) const;

          const AttributeReference getAttributeReference(const std::string&
       column_name) const;
          bool produceTuples(const ScanParam& param);

          std::stringstream mHeaderAndTypesBlock;
          std::stringstream mFetchInputCodeBlock;
          std::stringstream mGeneratedCode;
          std::list<std::string> mUpperCodeBlock;
          std::list<std::string> mLowerCodeBlock;
          std::stringstream mAfterForLoopBlock;
          std::stringstream mCreateResulttableCodeBlock;
          std::stringstream mCleanupCode;

          const static std::string MaxCStringLengthVarName;
    */

    bool grouped_aggregation = false;
  };
  typedef boost::shared_ptr<MultiStageCodeGenerator> MultiStageCodeGeneratorPtr;
  void optimizePrimitivePipeline(std::list<InstructionPtr>& program);

}  // end namespace CoGaDB

#endif /* MULTI_STAGE_GENERATOR_HPP */
