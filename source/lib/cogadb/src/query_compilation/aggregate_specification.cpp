
#include <boost/make_shared.hpp>
#include <core/global_definitions.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/aggregate_specification.hpp>
#include <query_compilation/algebraic_aggregate_specification.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/cpp_code_generator.hpp>
#include <util/attribute_reference_handling.hpp>
#include <util/getname.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <limits>
#include <util/code_generation.hpp>
#include "types.h"

//#ifndef COGADB_NOT_IMPLEMENTED
// #define COGADB_NOT_IMPLEMENTED COGADB_FATAL_ERROR("Not Implemented!", "")
//#endif

namespace CoGaDB {

//    struct AlgebraicAggregateSpecification : public AggregateSpecification {
//        AlgebraicAggregateSpecification(const AttributeReference& _scan_attr,
//                const AttributeReference& _result_attr,
//                const AggregationFunction& _agg_func);
//
//        const std::vector<AttributeReferencePtr> getScannedAttributes() const;
//        const std::vector<AttributeReferencePtr> getComputedAttributes()
//        const;
//
//        const std::string getCodeHashGroupBy(const std::string&
//        access_ht_entry_expression);
//        const std::vector<AggregationPayloadField>
//        getAggregationPayloadFields();
//        const std::string getCodeInitializeAggregationPayloadFields();
//        const std::string getCodeFetchResultsFromHashTableEntry(const
//        std::string& access_ht_entry_expression);
//
//        const std::string getCodeAggregationDeclareVariables();
//        const std::string getCodeAggregationInitializeVariables();
//        const std::string getCodeAggregationComputation();
//        const std::string getCodeAggregationWriteResult();
//
//        const std::string toString() const;
//        const AggregationFunctionType getAggregationFunctionType() const;
//
//        AttributeReference scan_attr;
//        AttributeReference result_attr;
//        AggregationFunction agg_func;
//    };

struct MinByAggregateSpecification : public AggregateSpecification {
  MinByAggregateSpecification(
      const std::vector<AttributeReferencePtr>& _scanned_attrs,
      const std::vector<AttributeReferencePtr>& _result_attrs,
      const std::map<AttributeReferencePtr, std::string>& init_values);

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
  const std::vector<AggregationPayloadField> getAggregationPayloadFields();
  const std::string getCodeInitializeAggregationPayloadFields(
      const std::string& access_ht_entry_expression);
  const std::string getCodeFetchResultsFromHashTableEntry(
      const std::string& access_ht_entry_expression);

  const std::string getCodeAggregationDeclareVariables(
      CodeGenerationTarget target);
  const std::string getCodeAggregationInitializeVariables(
      CodeGenerationTarget target);
  const std::string getCodeAggregationComputation(CodeGenerationTarget target);
  const std::string getCodeAggregationWriteResult(CodeGenerationTarget target);

  const std::string toString() const;

  const AggregationFunctionType getAggregationFunctionType() const;

  std::vector<AttributeReferencePtr> scanned_attrs;
  std::vector<AttributeReferencePtr> result_attrs;
  std::map<AttributeReferencePtr, std::string> init_values;
};

struct UDFAggregateSpecification : public AggregateSpecification {
  UDFAggregateSpecification(
      AggregationFunctionType agg_func_type,
      const std::vector<StructFieldPtr>& fields_for_aggregation,
      UDF_CodePtr aggregate_udf_code, UDF_CodePtr final_aggregate_udf_code);

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
  const std::vector<AggregationPayloadField> getAggregationPayloadFields();
  const std::string getCodeInitializeAggregationPayloadFields(
      const std::string& access_ht_entry_expression);
  const std::string getCodeFetchResultsFromHashTableEntry(
      const std::string& access_ht_entry_expression);

  const std::string getCodeAggregationDeclareVariables(
      CodeGenerationTarget target);
  const std::string getCodeAggregationInitializeVariables(
      CodeGenerationTarget target);
  const std::string getCodeAggregationComputation(CodeGenerationTarget target);
  const std::string getCodeAggregationWriteResult(CodeGenerationTarget target);

  const std::string toString() const;

  virtual const AggregationFunctionType getAggregationFunctionType() const;

  AggregationFunctionType agg_func_type;
  std::vector<StructFieldPtr> fields_for_aggregation;
  UDF_CodePtr aggregate_udf_code;
  UDF_CodePtr final_aggregate_udf_code;
};

AggregateSpecification::AggregateSpecification()
    : predication_mode_(BRANCHED_EXECUTION) {}

bool AggregateSpecification::supportsPredication() const {
  /* predication support is off by default,
   * needs to be explictely overridden by
   * overriding this function in a base class
   */
  return false;
}

void AggregateSpecification::setPredicationMode(
    const PredicationMode& pred_mode) {
  if (BRANCHED_EXECUTION == pred_mode) {
    if (!this->supportsPredication()) {
      COGADB_FATAL_ERROR(
          "Predication not supported by aggregation: " << this->toString(), "");
    }
  }
  predication_mode_ = pred_mode;
}

PredicationMode AggregateSpecification::getPredicationMode() const {
  return predication_mode_;
}

AggregateSpecification::~AggregateSpecification() {}

GroupByAggregateParam::GroupByAggregateParam(
    const ProcessorSpecification& _proc_spec,
    const GroupingAttributes& _grouping_attrs,
    const AggregateSpecifications& _aggregation_specs)
    : proc_spec(_proc_spec),
      grouping_attrs(_grouping_attrs),
      aggregation_specs(_aggregation_specs) {}

const std::string toString(const GroupingAttributes& grouping_attrs,
                           const AggregateSpecifications& aggregation_specs) {
  std::string result;
  for (size_t i = 0; i < grouping_attrs.size(); ++i) {
    result += toString(grouping_attrs[i]);
    //    if(isPersistent(grouping_attrs[i].getTable() )){
    //      result += " (Persistent)";
    //    }else{
    //      std::stringstream str;
    //      str << " (Intermediate Result TablePtr: " <<
    //      grouping_attrs[i].getTable() << ")";
    //      result += str.str();
    //    }
    if (i + 1 < grouping_attrs.size()) {
      result += ", ";
    }
  }
  result += "; ";
  for (size_t i = 0; i < aggregation_specs.size(); ++i) {
    result += aggregation_specs[i]->toString();
    if (i + 1 < aggregation_specs.size()) {
      result += ", ";
    }
  }
  return result;
}

const AggregateSpecificationPtr createAggregateSpecification(
    const AttributeReference& attr_ref, const AggregationFunction& agg_func,
    const std::string& result_name) {
  AttributeReference computed_attr = createComputedAttribute(
      AttributeReference(attr_ref.getVersionedAttributeName(), DOUBLE,
                         result_name, 1),
      agg_func, result_name);

  return AggregateSpecificationPtr(
      new AlgebraicAggregateSpecification(attr_ref, computed_attr, agg_func));
}

const AggregateSpecificationPtr createAggregateSpecification(
    const AttributeReference& attr_ref, const AggregationFunction& agg_func) {
  AttributeReference computed_attr = createComputedAttribute(
      AttributeReference(attr_ref.getVersionedAttributeName(), DOUBLE,
                         attr_ref.getVersionedAttributeName(), 1),
      agg_func);

  return AggregateSpecificationPtr(
      new AlgebraicAggregateSpecification(attr_ref, computed_attr, agg_func));
}

const AggregateSpecificationPtr createAggregateSpecificationMinBy(
    const AttributeReference& min_attr,
    const AttributeReference& project_attr) {
  std::vector<AttributeReferencePtr> scanned_attrs;
  std::vector<AttributeReferencePtr> result_attrs;

  //        scanned_attrs.push_back(AggregateSpecificationPtr(new
  //        AttributeReference(attr)));
  //        result_attrs.push_back(createAggregateSpecification(attr,
  //        UDF_AGGREGATION, "MIN_VALUE"));

  AttributeReference computed_attr = createComputedAttribute(
      AttributeReference(min_attr.getVersionedAttributeName(), DOUBLE,
                         "MIN_VALUE", 1),
      UDF_AGGREGATION, "MIN_VALUE");

  AttributeReference computed_attr2 = createComputedAttribute(
      AttributeReference(project_attr.getVersionedAttributeName(),
                         project_attr.getAttributeType(), "CID", 1),
      UDF_AGGREGATION, "CID");

  assert(project_attr.getAttributeType() == INT);
  assert(computed_attr2.getAttributeType() == INT);

  AttributeReferencePtr computed_min_attr =
      boost::make_shared<AttributeReference>(computed_attr);
  AttributeReferencePtr computed_project_attr =
      boost::make_shared<AttributeReference>(computed_attr2);

  scanned_attrs.push_back(boost::make_shared<AttributeReference>(min_attr));
  scanned_attrs.push_back(boost::make_shared<AttributeReference>(project_attr));
  result_attrs.push_back(computed_min_attr);
  result_attrs.push_back(computed_project_attr);

  std::map<AttributeReferencePtr, std::string> init_values;
  std::stringstream max_value;
  max_value << boost::lexical_cast<std::string>(
      std::numeric_limits<double>::max());
  init_values[computed_min_attr] = max_value.str();

  AggregateSpecificationPtr ptr(new MinByAggregateSpecification(
      scanned_attrs, result_attrs, init_values));

  return ptr;
}

const AggregateSpecificationPtr createAggregateSpecificationUDF(
    AggregationFunctionType agg_func_type,
    const std::vector<StructFieldPtr>& fields_for_aggregation,
    UDF_CodePtr aggregate_udf_code, UDF_CodePtr final_aggregate_udf_code) {
  return boost::make_shared<UDFAggregateSpecification>(
      agg_func_type, fields_for_aggregation, aggregate_udf_code,
      final_aggregate_udf_code);
}

MinByAggregateSpecification::MinByAggregateSpecification(
    const std::vector<AttributeReferencePtr>& _scanned_attrs,
    const std::vector<AttributeReferencePtr>& _result_attrs,
    const std::map<AttributeReferencePtr, std::string>& _init_values)
    : scanned_attrs(_scanned_attrs),
      result_attrs(_result_attrs),
      init_values(_init_values) {}

const std::vector<AttributeReferencePtr>
MinByAggregateSpecification::getScannedAttributes() const {
  return scanned_attrs;
}

const std::vector<AttributeReferencePtr>
MinByAggregateSpecification::getComputedAttributes() const {
  return result_attrs;
}

std::map<std::string, std::string>
MinByAggregateSpecification::getInputVariables(CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

std::map<std::string, std::string>
MinByAggregateSpecification::getOutputVariables(CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

void MinByAggregateSpecification::replaceTablePointerInAttributeReferences(
    const ScanParam& scanned_attributes) {
  CoGaDB::replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, this->scanned_attrs);
}

const std::string MinByAggregateSpecification::getCodeHashGroupBy(
    const std::string& access_ht_entry_expression) {
  std::stringstream compute_expr;

  //        compute_expr << access_ht_entry_expression <<
  //        getAggregationPayloadFieldVarName(*result_attrs[0],
  //                UDF_AGGREGATION) << " = std::min("
  //                << access_ht_entry_expression <<
  //                getAggregationPayloadFieldVarName(*result_attrs[0],
  //                UDF_AGGREGATION) << ","
  //                << getElementAccessExpression(*scanned_attrs[0]) << ");";
  //        compute_expr << "std::cout << \"Tuple: \" << " <<
  //        getElementAccessExpression(*scanned_attrs[0])
  //                     << "<< \", \" << " <<
  //                     getElementAccessExpression(*scanned_attrs[1])
  //                     << "<< \", \" << " <<
  //                     getTupleIDVarName(*scanned_attrs[1]) << " <<
  //                     std::endl;" << std::endl;

  //        compute_expr << "std::cout << \"if(\" << "
  //                << access_ht_entry_expression <<
  //                getAggregationPayloadFieldVarName(*result_attrs[0],
  //                UDF_AGGREGATION) << " << \" > \" << "
  //                << getElementAccessExpression(*scanned_attrs[0]) << " <<
  //                \"){\" << std::endl;" << std::endl;

  compute_expr << "if(" << access_ht_entry_expression
               << getAggregationPayloadFieldVarName(*result_attrs[0],
                                                    UDF_AGGREGATION)
               << " > " << getElementAccessExpression(*scanned_attrs[0])
               << "){";

  compute_expr << access_ht_entry_expression
               << getAggregationPayloadFieldVarName(*result_attrs[0],
                                                    UDF_AGGREGATION)
               << " = " << getElementAccessExpression(*scanned_attrs[0]) << ";";

  compute_expr << access_ht_entry_expression
               << getAggregationPayloadFieldVarName(*result_attrs[1],
                                                    UDF_AGGREGATION)
               << " = " << getElementAccessExpression(*scanned_attrs[1]) << ";";
  //        compute_expr << "std::cout << \"Update Aggregation: \" << " <<
  //        getElementAccessExpression(*scanned_attrs[1])  << "<< \", \" << " <<
  //        getTupleIDVarName(*scanned_attrs[1]) << " << std::endl;" <<
  //        std::endl;
  compute_expr << "}";

  return compute_expr.str();
}

const std::vector<AggregateSpecification::AggregationPayloadField>
MinByAggregateSpecification::getAggregationPayloadFields() {
  std::vector<AggregateSpecification::AggregationPayloadField> result;
  for (size_t i = 0; i < result_attrs.size(); ++i) {
    result.push_back(
        getAggregationPayloadFieldCode(*result_attrs[i], UDF_AGGREGATION));
  }

  return result;
}

const std::string
MinByAggregateSpecification::getCodeInitializeAggregationPayloadFields(
    const std::string& access_ht_entry_expression) {
  std::stringstream initialization_code;

  for (size_t i = 0; i < result_attrs.size(); ++i) {
    std::string c_type = toCType(result_attrs[i]->getAttributeType());
    initialization_code << access_ht_entry_expression
                        << getAggregationPayloadFieldVarName(*result_attrs[i],
                                                             UDF_AGGREGATION)
                        << " = ";

    auto cit = init_values.find(result_attrs[i]);
    if (cit == init_values.end()) {
      initialization_code << "(" << c_type << ") 0;" << std::endl;
    } else {
      initialization_code << cit->second << ";" << std::endl;
    }
  }

  return initialization_code.str();
}

const std::string
MinByAggregateSpecification::getCodeFetchResultsFromHashTableEntry(
    const std::string& access_ht_entry_expression) {
  std::stringstream write_result_expr;
  for (size_t i = 0; i < result_attrs.size(); ++i) {
    write_result_expr << getResultArrayVarName(*result_attrs[i])
                      << "[current_result_size] = "
                      << access_ht_entry_expression
                      << getAggregationPayloadFieldVarName(*result_attrs[i],
                                                           UDF_AGGREGATION)
                      << ";";
  }
  return write_result_expr.str();
}

const std::string
MinByAggregateSpecification::getCodeAggregationDeclareVariables(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

const std::string
MinByAggregateSpecification::getCodeAggregationInitializeVariables(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

const std::string MinByAggregateSpecification::getCodeAggregationComputation(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

const std::string MinByAggregateSpecification::getCodeAggregationWriteResult(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

const std::string MinByAggregateSpecification::toString() const {
  std::stringstream result;
  result << "REDUCE_UDF(";
  for (size_t i = 0; i < scanned_attrs.size(); ++i) {
    result << createFullyQualifiedColumnIdentifier(*scanned_attrs[i]);
    if (i + 1 < scanned_attrs.size()) result << ", ";
  }
  result << " => ";
  for (size_t i = 0; i < result_attrs.size(); ++i) {
    result << createFullyQualifiedColumnIdentifier(*result_attrs[i]);
    if (i + 1 < result_attrs.size()) result << ", ";
  }
  result << ");";

  return result.str();
}

const AggregationFunctionType
MinByAggregateSpecification::getAggregationFunctionType() const {
  return ALGEBRAIC;
}

UDFAggregateSpecification::UDFAggregateSpecification(
    AggregationFunctionType _agg_func_type,
    const std::vector<StructFieldPtr>& _fields_for_aggregation,
    UDF_CodePtr _aggregate_udf_code, UDF_CodePtr _final_aggregate_udf_code)
    : agg_func_type(_agg_func_type),
      fields_for_aggregation(_fields_for_aggregation),
      aggregate_udf_code(_aggregate_udf_code),
      final_aggregate_udf_code(_final_aggregate_udf_code) {}

const std::vector<AttributeReferencePtr>
UDFAggregateSpecification::getScannedAttributes() const {
  std::vector<AttributeReferencePtr> result;
  result.insert(result.begin(), aggregate_udf_code->scanned_attributes.begin(),
                aggregate_udf_code->scanned_attributes.end());
  result.insert(result.begin(),
                final_aggregate_udf_code->scanned_attributes.begin(),
                final_aggregate_udf_code->scanned_attributes.end());
  return result;
}

const std::vector<AttributeReferencePtr>
UDFAggregateSpecification::getComputedAttributes() const {
  std::vector<AttributeReferencePtr> result;
  result.insert(result.begin(), aggregate_udf_code->result_attributes.begin(),
                aggregate_udf_code->result_attributes.end());
  result.insert(result.begin(),
                final_aggregate_udf_code->result_attributes.begin(),
                final_aggregate_udf_code->result_attributes.end());
  return result;
}

std::map<std::string, std::string> UDFAggregateSpecification::getInputVariables(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

std::map<std::string, std::string>
UDFAggregateSpecification::getOutputVariables(CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

void UDFAggregateSpecification::replaceTablePointerInAttributeReferences(
    const ScanParam& scanned_attributes) {
  CoGaDB::replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, aggregate_udf_code->scanned_attributes);

  CoGaDB::replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, final_aggregate_udf_code->scanned_attributes);
}

const std::string UDFAggregateSpecification::getCodeHashGroupBy(
    const std::string& access_ht_entry_expression) {
  std::string code = aggregate_udf_code->getCode();
  boost::replace_all(code, "<HASH_ENTRY>.", access_ht_entry_expression);

  return code;
}

const std::vector<AggregateSpecification::AggregationPayloadField>
UDFAggregateSpecification::getAggregationPayloadFields() {
  std::vector<AggregateSpecification::AggregationPayloadField> result;
  for (size_t i = 0; i < this->fields_for_aggregation.size(); ++i) {
    std::stringstream payload_field_code;
    payload_field_code << toCPPType(this->fields_for_aggregation[i]->field_type)
                       << " " << this->fields_for_aggregation[i]->field_name
                       << ";" << std::endl;
    result.push_back(payload_field_code.str());
  }

  return result;
}

const std::string
UDFAggregateSpecification::getCodeInitializeAggregationPayloadFields(
    const std::string& access_ht_entry_expression) {
  std::stringstream initialization_code;

  for (size_t i = 0; i < fields_for_aggregation.size(); ++i) {
    initialization_code << access_ht_entry_expression
                        << fields_for_aggregation[i]->field_name << " = "
                        << fields_for_aggregation[i]->field_init_val << ";"
                        << std::endl;
  }

  return initialization_code.str();
}

const std::string
UDFAggregateSpecification::getCodeFetchResultsFromHashTableEntry(
    const std::string& access_ht_entry_expression) {
  std::string code = final_aggregate_udf_code->getCode();
  boost::replace_all(code, "<HASH_ENTRY>.", access_ht_entry_expression);

  return code;
}

const std::string UDFAggregateSpecification::getCodeAggregationDeclareVariables(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

const std::string
UDFAggregateSpecification::getCodeAggregationInitializeVariables(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

const std::string UDFAggregateSpecification::getCodeAggregationComputation(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

const std::string UDFAggregateSpecification::getCodeAggregationWriteResult(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

const std::string UDFAggregateSpecification::toString() const {
  std::stringstream result;
  result << "REDUCE_UDF(";
  std::vector<AttributeReferencePtr> scanned_attrs = getScannedAttributes();
  std::vector<AttributeReferencePtr> result_attrs = getComputedAttributes();
  for (size_t i = 0; i < scanned_attrs.size(); ++i) {
    result << createFullyQualifiedColumnIdentifier(*scanned_attrs[i]);
    if (i + 1 < scanned_attrs.size()) result << ", ";
  }
  result << " => ";
  for (size_t i = 0; i < result_attrs.size(); ++i) {
    result << CoGaDB::toString(*result_attrs[i]);
    if (i + 1 < result_attrs.size()) result << ", ";
  }
  result << ");";

  return result.str();
}

const AggregationFunctionType
UDFAggregateSpecification::getAggregationFunctionType() const {
  return agg_func_type;
}

}  // end namespace CoGaDB
