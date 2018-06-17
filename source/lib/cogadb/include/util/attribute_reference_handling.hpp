

#ifndef ATTRIBUTE_REFERENCE_HANDLING_HPP
#define ATTRIBUTE_REFERENCE_HANDLING_HPP

#include <core/attribute_reference.hpp>

namespace CoGaDB {

  class PredicateExpression;
  typedef boost::shared_ptr<PredicateExpression> PredicateExpressionPtr;

  struct AggregateSpecification;
  typedef boost::shared_ptr<AggregateSpecification> AggregateSpecificationPtr;
  typedef std::vector<AggregateSpecificationPtr> AggregateSpecifications;

  void replaceAttributeTablePointersWithScannedAttributeTablePointers(
      const ScanParam& scanned_attributes, AttributeReference& attr);

  void replaceAttributeTablePointersWithScannedAttributeTablePointers(
      const ScanParam& scanned_attributes,
      std::vector<AttributeReference>& attribute_references);

  void replaceAttributeTablePointersWithScannedAttributeTablePointers(
      const ScanParam& scanned_attributes,
      std::vector<AttributeReferencePtr>& attribute_references);

  void replaceAttributeTablePointersWithScannedAttributeTablePointers(
      const ScanParam& scanned_attributes, PredicateExpressionPtr pred_expr);

  void replaceAttributeTablePointersWithScannedAttributeTablePointers(
      const ScanParam& scanned_attributes, AggregateSpecifications& agg_specs);

}  // end namespace CoGaDB

#endif  // ATTRIBUTE_REFERENCE_HANDLING_HPP
