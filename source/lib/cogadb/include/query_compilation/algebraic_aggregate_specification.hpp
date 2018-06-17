/*
 * File:   algebraic_aggregate_specification.hpp
 * Author: sebastian
 *
 * Created on 30. Dezember 2015, 11:31
 */

#ifndef ALGEBRAIC_AGGREGATE_SPECIFICATION_HPP
#define ALGEBRAIC_AGGREGATE_SPECIFICATION_HPP

#include <query_compilation/aggregate_specification.hpp>

namespace CoGaDB {

  struct AlgebraicAggregateSpecification : public AggregateSpecification {
    AlgebraicAggregateSpecification(const AttributeReference& _scan_attr,
                                    const AttributeReference& _result_attr,
                                    const AggregationFunction& _agg_func);

    const std::vector<AttributeReferencePtr> getScannedAttributes() const;
    const std::vector<AttributeReferencePtr> getComputedAttributes() const;

    std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target);

    std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target);

    void replaceTablePointerInAttributeReferences(
        const ScanParam& scanned_attributes);

    const std::string getCodeHashGroupBy(
        const std::string& access_ht_entry_expression);

    virtual const std::string getCodeCopyHashTableEntry(
        const std::string& access_dst_ht_entry_expression,
        const std::string& access_src_ht_entry_expression);

    const std::vector<AggregationPayloadField> getAggregationPayloadFields();
    const std::string getCodeInitializeAggregationPayloadFields(
        const std::string& access_ht_entry_expression);
    const std::string getCodeFetchResultsFromHashTableEntry(
        const std::string& access_ht_entry_expression);

    const std::string getCodeAggregationDeclareVariables(
        CodeGenerationTarget target);
    const std::string getCodeAggregationInitializeVariables(
        CodeGenerationTarget target);
    const std::string getCodeAggregationComputation(
        CodeGenerationTarget target);
    const std::string getCodeAggregationWriteResult(
        CodeGenerationTarget target);

    const std::string toString() const;
    const AggregationFunctionType getAggregationFunctionType() const;

    AggregationFunction getAggregationFunction() const;

    bool supportsPredication() const;

    AttributeReference scan_attr_;
    AttributeReference result_attr_;
    AggregationFunction agg_func_;
  };

}  // end namespace CoGaDB

#endif /* ALGEBRAIC_AGGREGATE_SPECIFICATION_HPP */
