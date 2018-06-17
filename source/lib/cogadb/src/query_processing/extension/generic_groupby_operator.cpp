
#include <query_processing/extension/generic_groupby_operator.hpp>
#include <util/hardware_detector.hpp>

#include <core/variable_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <statistics/column_statistics.hpp>
#include <util/code_generation.hpp>

namespace CoGaDB {

namespace query_processing {

namespace logical_operator {

Logical_GenericGroupby::Logical_GenericGroupby(
    const GroupByAggregateParam& groupby_param)
    : TypedNode_Impl<TablePtr,
                     physical_operator::map_init_function_groupby_operator>(
          false, hype::ANY_DEVICE),
      groupby_param_(groupby_param) {}

unsigned int Logical_GenericGroupby::getOutputResultSize() const { return 10; }

double Logical_GenericGroupby::getCalculatedSelectivity() const { return 0.3; }

std::string Logical_GenericGroupby::getOperationName() const {
  //                    return "GROUPBY";
  return "GENERIC_GROUPBY";
}
std::string Logical_GenericGroupby::toString(bool verbose) const {
  std::string result = getOperationName();
  if (verbose) {
    result += " (";
    result += CoGaDB::toString(this->groupby_param_.grouping_attrs,
                               this->groupby_param_.aggregation_specs);
    result += ")";
  }
  return result;
}

const std::list<std::string>
Logical_GenericGroupby::getNamesOfReferencedColumns() const {
  std::list<std::string> result;
  for (auto attr : groupby_param_.grouping_attrs) {
    result.push_back(attr.getUnversionedAttributeName());
  }
  for (auto aggr_spec : groupby_param_.aggregation_specs) {
    std::vector<AttributeReferencePtr> scanned_attributes =
        aggr_spec->getScannedAttributes();
    for (auto scanned_attr : scanned_attributes) {
      result.push_back(scanned_attr->getUnversionedAttributeName());
    }
  }
  return result;
}

const GroupingAttributes Logical_GenericGroupby::getGroupingAttributes() {
  return groupby_param_.grouping_attrs;
}

const AggregateSpecifications
Logical_GenericGroupby::getAggregateSpecifications() {
  return groupby_param_.aggregation_specs;
}

void Logical_GenericGroupby::produce_impl(CodeGeneratorPtr code_gen,
                                          QueryContextPtr context) {
  /* The build of the aggregation hash table is a pipeline breaker.
   * Thus, we generate a new code generator and a new query context. */
  CodeGeneratorPtr code_gen_hash_aggregate;
  QueryContextPtr context_hash_aggregate;

  /* if this operator is the root, then we use the code generator
   and context passed to us */
  code_gen_hash_aggregate = code_gen;
  context_hash_aggregate = context;

  /* add the attributes accessed by this operator to the list in
   * the query context */
  std::list<std::string> referenced_colums = getNamesOfReferencedColumns();
  std::list<std::string>::const_iterator cit;
  for (cit = referenced_colums.begin(); cit != referenced_colums.end(); ++cit) {
    context_hash_aggregate->addAccessedColumn(*cit);
    context_hash_aggregate->addColumnToProjectionList(*cit);
  }
  left_->produce(code_gen_hash_aggregate, context_hash_aggregate);
}

void Logical_GenericGroupby::consume_impl(CodeGeneratorPtr code_gen,
                                          QueryContextPtr context) {
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");

  //                GroupByAggregateParam groupby_param(proc_spec,
  //                grouping_columns, agg_specs);

  if (groupby_param_.grouping_attrs.empty()) {
    /* aggregation without groupby */
    if (!code_gen->consumeAggregate(groupby_param_.aggregation_specs)) {
      COGADB_FATAL_ERROR("Failed to consume Aggregate Operation!", "");
    }
  } else {
    /* aggregation with groupby */
    if (!code_gen->consumeHashGroupAggregate(groupby_param_)) {
      COGADB_FATAL_ERROR("Failed to consume Hash Aggregate Operation!", "");
    }
  }
  if (debug_code_generator) code_gen->print();
  /* This operator is a pipeline breaker, so we do not call consume
   * of the parent operator! */
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CoGaDB
