#ifndef C_CODE_GENERATOR_HPP
#define C_CODE_GENERATOR_HPP

#include <dlfcn.h>
#include <boost/filesystem.hpp>
#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <list>
#include <ostream>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/pipeline.hpp>
#include <sstream>

#include <query_compilation/aggregate_specification.hpp>
#include <query_compilation/hash_table_generator.hpp>
#include <query_compilation/predicate_expression.hpp>

namespace CoGaDB {

  class HashTableGenerator;
  typedef boost::shared_ptr<HashTableGenerator> HashTableGeneratorPtr;

  const std::string generateCode_BitpackedGroupingKeyComputation(
      const GroupingAttributes& grouping_attrs);
  const std::string generateCode_GenericGroupingKeyComputation(
      const GroupingAttributes& grouping_attrs);

  class CCodeGenerator : public CodeGenerator {
   public:
    CCodeGenerator(const ProjectionParam& param, const TablePtr table,
                   uint32_t version = 1);

    CCodeGenerator(const ProjectionParam& param);

    void init();

    virtual bool consumeSelection_impl(const PredicateExpressionPtr pred_expr);

    virtual bool consumeBuildHashTable_impl(const AttributeReference& attr);

    virtual bool consumeProbeHashTable_impl(
        const AttributeReference& hash_table_attr,
        const AttributeReference& probe_attr);

    virtual bool consumeHashGroupAggregate_impl(
        const GroupByAggregateParam& param);

    virtual bool consumeAggregate_impl(const AggregateSpecifications& params);

    virtual bool consumeCrossJoin_impl(const AttributeReference& attr);

    virtual bool consumeNestedLoopJoin_impl(
        const PredicateExpressionPtr pred_expr);

    virtual const std::pair<bool, std::vector<AttributeReferencePtr> >
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

   protected:
    virtual bool createForLoop_impl(const TablePtr table, uint32_t version);
    virtual bool createHashTable(const AttributeReference& attr);

    virtual const std::string getCodeAllocateResultTable() const;
    virtual const std::string getCodeWriteResult() const;
    virtual const std::string getCodeWriteResultFromHashTable() const;
    virtual const std::string createResultTable() const;

    const std::string getAggregationCode(
        const GroupingAttributes& grouping_columns,
        const AggregateSpecifications& aggregation_specs,
        const std::string access_ht_entry_expression) const;

    const AttributeReference getAttributeReference(
        const std::string& column_name) const;
    bool produceTuples(const ScanParam& param);

    void compile(const std::string& source,
                 SharedCLibPipelineQueryPtr& query_ptr,
                 boost::shared_ptr<llvm::ExecutionEngine>& engine,
                 boost::shared_ptr<llvm::LLVMContext>& context);

    std::stringstream mHeaderAndTypesBlock;
    std::stringstream mFetchInputCodeBlock;
    std::stringstream mGeneratedCode;
    std::list<std::string> mUpperCodeBlock;
    std::list<std::string> mLowerCodeBlock;
    std::stringstream mAfterForLoopBlock;
    std::stringstream mCreateResulttableCodeBlock;
    std::stringstream mCleanupCode;
    HashTableGeneratorPtr ht_gen;
    std::map<std::string, AttributeReferencePtr> mColumnsToDecompress;
    ProjectionParam aggr_result_params_;

    const static std::string MaxCStringLengthVarName;
  };

}  // end namespace CoGaDB

#endif /* C_CODE_GENERATOR_HPP */
