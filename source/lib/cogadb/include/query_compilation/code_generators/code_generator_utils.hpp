#ifndef CODE_GENERATOR_UTILS_HPP
#define CODE_GENERATOR_UTILS_HPP

#include <core/table.hpp>
#include <query_compilation/aggregate_specification.hpp>
#include <query_compilation/pipeline.hpp>
#include <query_compilation/predicate_expression.hpp>

#include <query_compilation/code_generator.hpp>

namespace CoGaDB {

  const std::string toCPPType(const AttributeType& attr);

  const std::string toCType(const AttributeType& attr);

  const std::string toType(const AttributeType& attr);

  const std::string getResultType(const AttributeReference& attr,
                                  bool ignore_compressed);

  const std::string getCTypeFunctionPostFix(const AttributeType& attr);

  const std::string getVarName(const AttributeReference& attr_ref,
                               const bool ignore_compressed = false);

  const std::string getSSEVarName(const uint32_t pred_num = 1);

  const std::string getTupleIDVarName(const TablePtr table, uint32_t version);

  const std::string getTupleIDVarName(const AttributeReference& attr_ref);

  const std::string getInputArrayVarName(const AttributeReference& attr_ref);

  const std::string getResultArrayVarName(const AttributeReference& attr_ref);

  const std::string getHashTableVarName(const AttributeReference& attr_ref);

  const std::string getTableVarName(const AttributeReference& attr_ref);

  const std::string getTableVarName(const TablePtr table, uint32_t version);

  const std::string getExpression(LogicalOperation log_op);

  const std::string getExpression(ValueComparator x);

  const std::string getSSEExpression(LogicalOperation log_op);

  const std::string getSSEExpression(ValueComparator x);
  const std::string getSSEExpressionFloat(ValueComparator x);
  const std::string getSSEExpressionDouble(ValueComparator x);

  const std::string getConstant(const boost::any& constant);

  const std::string getSSEConstantFloat(const boost::any& constant);

  const std::string getConstant(const std::string& constant);

  const std::string getElementAccessExpression(
      const AttributeReference& attr_ref, std::string tuple_id = "");

  const std::string getElementAccessExpressionSIMD(
      const AttributeReference& attr_ref, std::string vector_type,
      std::string tuple_id = "");

  const std::string getInputColumnVarName(const AttributeReference& attr_ref);

  const std::string getGroupTIDVarName(const AttributeReference& attr_ref);

  const std::string getCompressedElementAccessExpression(
      const AttributeReference& attr_ref, std::string tuple_id = "");

  const std::string getCompressedElementAccessExpressionSIMD(
      const AttributeReference& attr_ref, std::string vector_type,
      std::string tuple_id = "");

  const std::string getAggregationPayloadFieldVarName(
      const AttributeReference& attr_ref, const AggregationFunction& agg_func);

  const std::string getAggregationPayloadFieldVarName(
      const AttributeReference& attr_ref, const AggregationParam& param);

  const std::string getAggregationPayloadFieldCode(
      const AttributeReference& attr_ref, AggregationFunction agg_func);

  //! REMOVE ME!!!
  void enableIntelAggregationHACK(bool enable);

  const std::string getAggregationResultCType(
      const AttributeReference& attr_ref, AggregationFunction agg_func);

  const std::string getComputeGroupIDExpression(
      const GroupingAttributes& grouping_attrs);

  const std::string getAggregationGroupTIDPayloadFieldCode(
      const AttributeReference& attr_ref);

  bool isBitpackedGroupbyOptimizationApplicable(
      const GroupingAttributes& grouping_attrs);

  const std::string getAggregationCodeGeneric(
      const GroupingAttributes& grouping_columns,
      const AggregateSpecifications& aggregation_specs,
      const std::string access_ht_entry_expression);

  const std::string getAggregationPayloadCodeForGroupingAttributes(
      const GroupingAttributes& grouping_attrs);

  const std::string getCodeProjectGroupingColumnsFromHashTable(
      const GroupingAttributes& grouping_attrs,
      const std::string& hash_map_access);

  const std::string getCodeDeclareResultMemory(const AttributeReference& ref,
                                               bool uses_c_string = true);

  const std::string getCodeDeclareResultMemory(const ProjectionParam& param,
                                               bool uses_c_string = true);

  const std::string getCodeMalloc(const std::string& variable,
                                  const std::string& type,
                                  const std::string& count, const bool realloc);

  const std::string getCodeMallocResultMemory(const AttributeReference& ref,
                                              bool uses_c_string = true);

  const std::string getCodeMallocResultMemory(const ProjectionParam& param,
                                              bool uses_c_string = true);

  const std::string getCodeReallocResultMemory(const ProjectionParam& param,
                                               bool uses_c_string = true);

  const std::string getVariableFromAttributeName(
      const std::string qualified_attribute_name);

  bool targetingCLang();

  std::string getMinFunction();

  std::string getMaxFunction();

  void convertToCodeGenerator(const std::string& value,
                              CodeGeneratorType& code_gen);

  const std::string generateCCodeWriteResult(ProjectionParam param);

  const std::string generateCCodeWriteResultFromHashTable(
      ProjectionParam param);

  const std::string generateCCodeCreateResultTable(
      const ProjectionParam& param,
      const std::string& createResulttableCodeBlock,
      const std::string& cleanupCode, const std::string& result_table_name);

  const std::string generateCCodeAllocateResultTable(
      const ProjectionParam& param);

  bool isAttributeDictionaryCompressed(const AttributeReference& attr);

  std::string getStringCompareExpression(const AttributeReference& left,
                                         const AttributeReference& right,
                                         ValueComparator comp);

  std::string getStringCompareExpression(const AttributeReference& attr,
                                         const std::string& constant,
                                         ValueComparator comp);

  //! WARNING: this call fetches the column from disk, as we need to access the
  // dictionary!
  bool getDictionaryIDForPredicate(const ColumnPtr col,
                                   const ValueComparator comp,
                                   const std::string& comparison_val,
                                   uint32_t& result_id,
                                   ValueComparator& rewritten_value_comparator);

  std::string getArrayFromColumnCode(const AttributeReference& attr,
                                     bool ignore_compressed = false);

}  // namespace CoGaDB

#endif  // CODE_GENERATOR_UTILS_HPP
