
#include <boost/tr1/memory.hpp>
#include <query_compilation/predicate_expression.hpp>
#include <query_compilation/predicate_specification.hpp>

namespace CoGaDB {

PredicateExpression::~PredicateExpression() {}

const PredicateExpressionPtr createColumnColumnComparisonPredicateExpression(
    const AttributeReferencePtr& left, const AttributeReferencePtr& right,
    ValueComparator comp) {
  PredicateExpressionPtr ptr(
      new PredicateSpecification(ValueValuePredicateSpec, left, right, comp));
  return ptr;
}

const PredicateExpressionPtr createColumnConstantComparisonPredicateExpression(
    const AttributeReferencePtr left_attr, const boost::any& right_constant,
    ValueComparator comp) {
  PredicateExpressionPtr ptr(new PredicateSpecification(
      ValueConstantPredicateSpec, left_attr, right_constant, comp));
  return ptr;
}

const PredicateExpressionPtr createRegularExpressionPredicateExpression(
    const AttributeReferencePtr left_attr,
    const std::string& regular_expression, ValueComparator comp) {
  PredicateExpressionPtr ptr(
      new PredicateSpecification(ValueRegularExpressionPredicateSpec, left_attr,
                                 regular_expression, comp));
  return ptr;
}

const PredicateExpressionPtr createPredicateExpression(
    const std::vector<PredicateExpressionPtr>& pred_expression,
    const LogicalOperation& log_op) {
  PredicateExpressionPtr ptr(new PredicateCombination(pred_expression, log_op));
  return ptr;
}

const std::vector<PredicateExpressionPtr> getPredicates(
    PredicateExpressionPtr pred_expr) {
  std::vector<PredicateExpressionPtr> result;
  if (pred_expr) {
    result.push_back(pred_expr);
    boost::shared_ptr<PredicateCombination> combinator;
    combinator = boost::dynamic_pointer_cast<PredicateCombination>(pred_expr);
    if (combinator) {
      std::vector<PredicateExpressionPtr> pred_exprs =
          combinator->getPredicateExpression();
      for (size_t i = 0; i < pred_exprs.size(); ++i) {
        std::vector<PredicateExpressionPtr> preds =
            getPredicates(pred_exprs[i]);
        result.insert(result.end(), preds.begin(), preds.end());
      }
    }
  }
  return result;
}

}  // end namespace CoGaDB
