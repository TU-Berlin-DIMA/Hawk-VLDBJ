#include <query_processing/extension/map_udf_operator.hpp>
#include <util/hardware_detector.hpp>

#include <core/variable_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <statistics/column_statistics.hpp>
#include <util/code_generation.hpp>

namespace CoGaDB {

namespace query_processing {

namespace logical_operator {

Logical_MapUDF::Logical_MapUDF(const Map_UDF_ParamPtr param)
    : TypedNode_Impl<TablePtr, map_init_function_dummy>(false,
                                                        hype::ANY_DEVICE),
      param_(param) {}

unsigned int Logical_MapUDF::getOutputResultSize() const { return 10; }

double Logical_MapUDF::getCalculatedSelectivity() const { return 0.3; }

std::string Logical_MapUDF::getOperationName() const {
  //                    return "GROUPBY";
  return "MAP_UDF";
}

std::string Logical_MapUDF::toString(bool verbose) const {
  std::string result = getOperationName();
  if (verbose) {
    result += " (";
    //                    for (size_t i = 0; i <
    //                    groupby_param_.grouping_attrs.size(); ++i) {
    //                        result +=
    //                        CoGaDB::toString(groupby_param_.grouping_attrs[i]);
    //                        if (i + 1 < groupby_param_.grouping_attrs.size())
    //                        {
    //                            result += ",";
    //                        }
    //                    }
    //                    result += "; ";
    //                    for (size_t i = 0; i <
    //                    groupby_param_.aggregation_specs.size(); ++i) {
    //                        result +=
    //                        groupby_param_.aggregation_specs[i]->toString();
    //                        if (i + 1 < groupby_param_.grouping_attrs.size())
    //                        {
    //                            result += ",";
    //                        }
    //                    }
    result += ")";
  }
  return result;
}

const std::list<std::string> Logical_MapUDF::getNamesOfReferencedColumns()
    const {
  std::list<std::string> result;
  //                    std::list<std::string>
  //                    result(grouping_columns_.begin(),grouping_columns_.end());
  //                    std::list<ColumnAggregation>::const_iterator
  //                    agg_func_cit;
  //                    for(agg_func_cit=aggregation_functions_.begin();agg_func_cit!=aggregation_functions_.end();++agg_func_cit){
  //                        result.push_back(agg_func_cit->first);
  ////                        std::cout << "Aggregated Column: " <<
  /// agg_func_cit->first << std::endl;
  //                    }

  ScanParam scanned_attributes = param_->getScannedAttributes();
  for (size_t i = 0; i < scanned_attributes.size(); ++i) {
    result.push_back(scanned_attributes[i].getUnversionedAttributeName());
  }

  return result;
}

void Logical_MapUDF::produce_impl(CodeGeneratorPtr code_gen,
                                  QueryContextPtr context) {
  /* add the attributes accessed by this operator to the list in
   * the query context */
  std::list<std::string> referenced_colums = getNamesOfReferencedColumns();
  std::list<std::string>::const_iterator cit;
  for (cit = referenced_colums.begin(); cit != referenced_colums.end(); ++cit) {
    context->addAccessedColumn(*cit);
  }
  left_->produce(code_gen, context);
}

void Logical_MapUDF::consume_impl(CodeGeneratorPtr code_gen,
                                  QueryContextPtr context) {
  std::pair<bool, std::vector<AttributeReferencePtr> > map_udf_ret;
  map_udf_ret = code_gen->consumeMapUDF(param_);

  assert(map_udf_ret.first == true);
  auto projected_attributes = context->getProjectionList();
  for (size_t i = 0; i < map_udf_ret.second.size(); ++i) {
    for (auto projected_attr : projected_attributes) {
      assert(map_udf_ret.second[i] != NULL);
      std::cout << "Compare Attribute "
                << CoGaDB::toString(*map_udf_ret.second[i]) << " with "
                << CoGaDB::toString(*projected_attr) << std::endl;
      if (*map_udf_ret.second[i] == *projected_attr) {
        code_gen->addAttributeProjection(*map_udf_ret.second[i]);
      }
    }
  }
  /* project all columns */
  if (projected_attributes.empty()) {
    for (size_t i = 0; i < map_udf_ret.second.size(); ++i) {
      code_gen->addAttributeProjection(*map_udf_ret.second[i]);
    }
  }

  if (parent_) {
    parent_->consume(code_gen, context);
  }
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CoGaDB
