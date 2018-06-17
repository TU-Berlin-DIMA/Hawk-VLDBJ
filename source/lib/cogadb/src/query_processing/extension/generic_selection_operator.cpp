
#include <query_compilation/code_generator.hpp>
#include <query_compilation/predicate_expression.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/generic_selection_operator.hpp>

namespace CoGaDB {
namespace query_processing {
namespace logical_operator {

Logical_GenericSelection::Logical_GenericSelection(
    PredicateExpressionPtr pred_expr)
    : TypedNode_Impl<TablePtr, map_init_function_dummy>(false,
                                                        hype::ANY_DEVICE),
      pred_expr_(pred_expr),
      could_not_be_pushed_down_further_(false) {}

unsigned int Logical_GenericSelection::getOutputResultSize() const {
  if (this->getLeft()) {
    return this->getLeft()->getOutputResultSize() *
           this->getCalculatedSelectivity();
  } else {
    return 10;
  }
}

double Logical_GenericSelection::getCalculatedSelectivity() const {
  return 0.1;
}

std::string Logical_GenericSelection::getOperationName() const {
  return "GENERIC_SELECTION";
}

std::string Logical_GenericSelection::toString(bool verbose) const {
  std::string result = this->getOperationName();
  if (verbose) {
    result += pred_expr_->toString();
  }
  return result;
}

const PredicateExpressionPtr
Logical_GenericSelection::getPredicateExpression() {
  return pred_expr_;
}

bool Logical_GenericSelection::couldNotBePushedDownFurther() {
  return could_not_be_pushed_down_further_;
}

void Logical_GenericSelection::couldNotBePushedDownFurther(bool val) {
  could_not_be_pushed_down_further_ = val;
}

const std::list<std::string>
Logical_GenericSelection::getNamesOfReferencedColumns() const {
  std::list<std::string> result;
  std::vector<AttributeReferencePtr> attrs = pred_expr_->getScannedAttributes();
  for (size_t i = 0; i < attrs.size(); ++i) {
    result.push_back(CoGaDB::toString(*attrs[i]));
  }
  return result;
}

void Logical_GenericSelection::produce_impl(CodeGeneratorPtr code_gen,
                                            QueryContextPtr context) {
  /* add the attributes accessed by this operator to the list in
   * the query context */
  auto referenced_colums = getNamesOfReferencedColumns();
  for (auto referenced_column : referenced_colums) {
    context->addAccessedColumn(referenced_column);
  }
  left_->produce(code_gen, context);
}

void Logical_GenericSelection::consume_impl(CodeGeneratorPtr code_gen,
                                            QueryContextPtr context) {
  if (!code_gen->consumeSelection(pred_expr_)) {
    COGADB_FATAL_ERROR("Code Generation Failed for generic selection!", "");
  }

  if (parent_) {
    parent_->consume(code_gen, context);
  }
}

}  // end namespace logical_operator
}  // end namespace query_processing
}  // end namespace CoGaDB
