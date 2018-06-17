/*
 * File:   aggregate_specification.hpp
 * Author: sebastian
 *
 * Created on 24. August 2015, 11:06
 */

#ifndef AGGREGATE_SPECIFICATION_HPP
#define AGGREGATE_SPECIFICATION_HPP

#include <core/global_definitions.hpp>

#include <core/attribute_reference.hpp>
#include <core/operator_parameter_types.hpp>
#include <query_compilation/primitives/instruction.hpp>
#include <query_compilation/user_defined_code.hpp>
#include <util/code_generation.hpp>

namespace CoGaDB {

  struct AggregateSpecification;
  typedef boost::shared_ptr<AggregateSpecification> AggregateSpecificationPtr;

  enum AggregationFunctionType { DISTRIBUTIVE, ALGEBRAIC, HOLISTIC };

  struct AggregateSpecification {
    typedef std::string AggregationPayloadField;

    virtual const std::vector<AttributeReferencePtr> getScannedAttributes()
        const = 0;
    virtual const std::vector<AttributeReferencePtr> getComputedAttributes()
        const = 0;

    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) = 0;

    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) = 0;

    virtual void replaceTablePointerInAttributeReferences(
        const ScanParam& scanned_attributes) = 0;

    virtual const std::string getCodeHashGroupBy(
        const std::string& access_ht_entry_expression) = 0;
    virtual const std::string getCodeCopyHashTableEntry(
        const std::string& access_dst_ht_entry_expression,
        const std::string& access_src_ht_entry_expression) {
      COGADB_NOT_IMPLEMENTED;
    }

    virtual const std::vector<AggregationPayloadField>
    getAggregationPayloadFields() = 0;
    virtual const std::string getCodeInitializeAggregationPayloadFields(
        const std::string& access_ht_entry_expression) = 0;
    virtual const std::string getCodeFetchResultsFromHashTableEntry(
        const std::string& access_ht_entry_expression) = 0;

    virtual const std::string getCodeAggregationDeclareVariables(
        CodeGenerationTarget target) = 0;
    virtual const std::string getCodeAggregationInitializeVariables(
        CodeGenerationTarget target) = 0;
    virtual const std::string getCodeAggregationComputation(
        CodeGenerationTarget target) = 0;
    virtual const std::string getCodeAggregationWriteResult(
        CodeGenerationTarget target) = 0;

    virtual const std::string toString() const = 0;

    virtual const AggregationFunctionType getAggregationFunctionType()
        const = 0;

    virtual bool supportsPredication() const;
    void setPredicationMode(const PredicationMode& pred_mode);
    PredicationMode getPredicationMode() const;

    virtual ~AggregateSpecification();

   protected:
    AggregateSpecification();

   private:
    AggregateSpecification(const AggregateSpecification&);
    AggregateSpecification& operator=(const AggregateSpecification&);
    PredicationMode predication_mode_;
  };

  const AggregateSpecificationPtr createAggregateSpecification(
      const AttributeReference& attr, const AggregationFunction& agg_func);

  const AggregateSpecificationPtr createAggregateSpecification(
      const AttributeReference& attr, const AggregationFunction& agg_func,
      const std::string& result_name);

  const AggregateSpecificationPtr createAggregateSpecificationMinBy(
      const AttributeReference& min_attr,
      const AttributeReference& project_attr);

  const AggregateSpecificationPtr createAggregateSpecificationUDF(
      AggregationFunctionType agg_func_type,
      const std::vector<StructFieldPtr>& fields_for_aggregation,
      UDF_CodePtr aggregate_udf_code, UDF_CodePtr final_aggregate_udf_code);

  typedef std::vector<AggregateSpecificationPtr> AggregateSpecifications;

  typedef std::vector<AttributeReference> GroupingAttributes;

  struct GroupByAggregateParam;
  typedef boost::shared_ptr<GroupByAggregateParam> GroupByAggregateParamPtr;

  struct GroupByAggregateParam {
    GroupByAggregateParam(const ProcessorSpecification& proc_spec,
                          const GroupingAttributes& grouping_attrs,
                          const AggregateSpecifications& aggregation_specs);
    ProcessorSpecification proc_spec;
    GroupingAttributes grouping_attrs;
    AggregateSpecifications aggregation_specs;
  };

  typedef std::pair<AttributeReferencePtr, SortOrder> SortStep;
  typedef std::vector<SortStep> SortSpecification;

  const std::string toString(const GroupingAttributes& grouping_attrs,
                             const AggregateSpecifications& aggregation_specs);

}  // end namespace CoGaDB

#endif /* AGGREGATE_SPECIFICATION_HPP */
