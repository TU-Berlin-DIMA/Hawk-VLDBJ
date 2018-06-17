/*
 * File:   predicate_specification.hpp
 * Author: sebastian
 *
 * Created on 24. August 2015, 10:48
 */

#ifndef PREDICATE_SPECIFICATION_HPP
#define PREDICATE_SPECIFICATION_HPP

#include <query_compilation/predicate_expression.hpp>

#include "primitives/instruction.hpp"

namespace CoGaDB {

  class PredicateSpecification;
  typedef boost::shared_ptr<PredicateSpecification> PredicateSpecificationPtr;

  class PredicateSpecification : public PredicateExpression {
   public:
    PredicateSpecification(PredicateSpecificationType pred_t,
                           const AttributeReferencePtr left_attr,
                           const AttributeReferencePtr right_attr,
                           ValueComparator comp);

    PredicateSpecification(PredicateSpecificationType pred_t,
                           const AttributeReferencePtr left_attr,
                           const boost::any& right_constant,
                           ValueComparator comp);

    PredicateSpecification(PredicateSpecificationType pred_t,
                           const AttributeReferencePtr left_attr,
                           const std::string& regular_expression,
                           ValueComparator comp);

    const std::string getCPPExpression() const;
    const std::pair<std::string, std::string> getSSEExpression(
        uint32_t& pred_num) const;
    // const std::vector<boost::any> getConstants() const;
    const std::vector<AttributeReferencePtr> getScannedAttributes() const;
    const std::vector<AttributeReferencePtr> getColumnsToDecompress() const;

    PredicateSpecificationType getPredicateSpecificationType() const;
    const AttributeReferencePtr getLeftAttribute() const;
    const AttributeReferencePtr getRightAttribute() const;
    const boost::any& getConstant() const;
    const std::string& getRegularExpression() const;
    ValueComparator getValueComparator() const;
    void print() const;
    const std::string toString() const;
    /*! \brief inverts the order of Value Value Predicates
                e.g., Attr1>Attr2 -> Attr2<Attr1
                this is especially useful for join path optimization*/
    void invertOrder();
    //        bool operator< (const Predicate& p) const;
    void replaceTablePointerInAttributeReferences(
        const ScanParam& scanned_attributes);

   private:
    PredicateSpecificationType pred_t_;
    AttributeReferencePtr left_attr_;
    AttributeReferencePtr right_attr_;
    boost::any constant_;
    std::string reg_ex_;
    ValueComparator comp_;
  };

  class PredicateCombination : public PredicateExpression {
   public:
    PredicateCombination(
        const std::vector<PredicateExpressionPtr>& predicate_expressions,
        LogicalOperation log_op_);

    const std::string getCPPExpression() const;
    const std::pair<std::string, std::string> getSSEExpression(
        uint32_t& pred_num) const;
    // const std::vector<boost::any> getConstants() const;
    const std::vector<AttributeReferencePtr> getScannedAttributes() const;
    const std::vector<PredicateExpressionPtr> getPredicateExpression() const;
    const std::vector<AttributeReferencePtr> getColumnsToDecompress() const;
    const std::string toString() const;
    void replaceTablePointerInAttributeReferences(
        const ScanParam& scanned_attributes);

   private:
    std::vector<PredicateExpressionPtr> predicate_expressions_;
    LogicalOperation log_op_;
  };

}  // end namespace CoGaDB

#endif /* PREDICATE_SPECIFICATION_HPP */
