

#include <boost/make_shared.hpp>
#include <core/global_definitions.hpp>
#include <core/operator_parameter_types.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/code_generators/c_code_generator.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/cpp_code_generator.hpp>
#include <query_compilation/code_generators/cuda_c_code_generator.hpp>
#include <query_compilation/code_generators/multi_stage_code_generator.hpp>
#include <sstream>
#include <util/getname.hpp>

#include <query_compilation/query_context.hpp>
#include <query_compilation/user_defined_code.hpp>

#include <core/data_dictionary.hpp>
#include <persistence/storage_manager.hpp>
#include <util/attribute_reference_handling.hpp>

namespace CoGaDB {

std::string getComputedAttributeVarName(const AttributeReference& left_attr,
                                        const AttributeReference& right_attr,
                                        const ColumnAlgebraOperation& alg_op) {
  std::stringstream ss;
  ss << left_attr.getVersionedAttributeName() << "_" << util::getName(alg_op)
     << "_" << right_attr.getVersionedAttributeName();
  return ss.str();
}

std::string getComputedAttributeVarName(const AttributeReference& attr,
                                        const AggregationFunction agg_func) {
  std::stringstream ss;
  ss << attr.getVersionedAttributeName() << "_" << util::getName(agg_func);
  return ss.str();
}

AttributeReference createComputedAttribute(
    const AttributeReference& left_attr, const AttributeReference& right_attr,
    const ColumnAlgebraOperation& alg_op) {
  std::stringstream new_name;
  new_name << getComputedAttributeVarName(left_attr, right_attr, alg_op);
  return AttributeReference(new_name.str(), DOUBLE, new_name.str(), 1);
}

AttributeReference createComputedAttribute(
    const AttributeReference& left_attr, const boost::any& constant,
    const ColumnAlgebraOperation& alg_op) {
  std::stringstream new_name;
  new_name << left_attr.getVersionedAttributeName() << util::getName(alg_op)
           << getConstant(constant);
  return AttributeReference(new_name.str(), DOUBLE, new_name.str(), 1);
}

AttributeReference createComputedAttribute(
    const boost::any& constant, const AttributeReference& right_attr,
    const ColumnAlgebraOperation& alg_op) {
  std::stringstream new_name;
  new_name << getConstant(constant) << util::getName(alg_op)
           << right_attr.getVersionedAttributeName();
  return AttributeReference(new_name.str(), DOUBLE, new_name.str(), 1);
}

AttributeReference createComputedAttribute(
    const AttributeReference& attr, const AggregationFunction& agg_func) {
  std::stringstream new_name;
  new_name << getComputedAttributeVarName(attr, agg_func);
  AttributeType type = attr.getAttributeType();
  if (type == FLOAT) {
    type = DOUBLE;
  }
  return AttributeReference(new_name.str(), type, new_name.str(), 1);
}

AttributeReference createComputedAttribute(const AttributeReference& attr,
                                           const AggregationFunction& agg_func,
                                           const std::string& result_name) {
  std::stringstream new_name;
  new_name << getComputedAttributeVarName(attr, agg_func);
  AttributeType type = attr.getAttributeType();
  if (type == FLOAT) {
    type = DOUBLE;
  }
  return AttributeReference(new_name.str(), type, result_name, 1);
}

CodeGenerator::CodeGenerator(const CodeGeneratorType& _generator_type,
                             const ProjectionParam& _param,
                             const TablePtr table, const uint32_t version)
    : param(_param),
      scanned_attributes(),
      pipe_end(MATERIALIZE_FROM_ARRAY_TO_ARRAY),
      input_table(table),
      input_table_version(version),
      tuples_produced(false),
      is_empty_pipeline(true),
      groupby_param(),
      generator_type(_generator_type) {
  assert(input_table != NULL);
  for (size_t i = 0; i < param.size(); ++i) {
    addToScannedAttributes(param[i]);
  }
}

CodeGenerator::CodeGenerator(const CodeGeneratorType& _generator_type,
                             const ProjectionParam& _param)
    : param(_param),
      scanned_attributes(),
      pipe_end(MATERIALIZE_FROM_ARRAY_TO_ARRAY),
      input_table(),
      input_table_version(0),
      tuples_produced(false),
      is_empty_pipeline(true),
      groupby_param(),
      generator_type(_generator_type) {
  for (size_t i = 0; i < param.size(); ++i) {
    addToScannedAttributes(param[i]);
  }
}

CodeGenerator::~CodeGenerator() {}

bool CodeGenerator::addAttributeProjection(const AttributeReference& attr) {
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");
  if (debug_code_generator)
    std::cout << "[DEBUG]: addAttributeProjection: add "
              << attr.getVersionedTableName() << "."
              << attr.getVersionedAttributeName() << std::endl;
  //    std::cout << "=====================================================" <<
  //    std::endl;
  //    CoGaDB::printStackTrace(std::cout);
  //    std::cout << "=====================================================" <<
  //    std::endl;
  if (attr.getAttributeReferenceType() == INPUT_ATTRIBUTE)
    addToScannedAttributes(attr);
  bool found = false;
  for (size_t i = 0; i < param.size(); ++i) {
    if (param[i].getVersionedAttributeName() ==
            attr.getVersionedAttributeName() &&
        param[i].getAttributeReferenceType() == INPUT_ATTRIBUTE) {
      found = true;
    }
  }

  if (found) {
    return false;
  } else {
    param.push_back(attr);
    return true;
  }
}

AttributeReferencePtr CodeGenerator::getProjectionAttributeByName(
    const std::string& name) const {
  for (size_t i = 0; i < param.size(); ++i) {
    if (param[i].getUnversionedAttributeName() == name) {
      return AttributeReferencePtr(new AttributeReference(param[i]));
    }
  }
  return AttributeReferencePtr();
}

AttributeReferencePtr CodeGenerator::getScannedAttributeByName(
    const std::string& name) const {
  std::string qualified_column_name =
      convertToFullyQualifiedNameIfRequired(name);
  for (size_t i = 0; i < scanned_attributes.size(); ++i) {
    AttributeReferencePtr current_attr(
        new AttributeReference(scanned_attributes[i]));
    std::string current_column =
        createFullyQualifiedColumnIdentifier(current_attr);
    if (current_column == qualified_column_name) {
      return current_attr;
    }
  }
  return AttributeReferencePtr();
}

bool CodeGenerator::isEmpty() const { return is_empty_pipeline; }

void CodeGenerator::print() const {
  std::ostream& out = std::cout;
  out << "Code Generator: " << std::endl;
  out << "\tScanned Attributes: " << std::endl;
  for (size_t i = 0; i < scanned_attributes.size(); ++i) {
    out << "\t\t" << toString(scanned_attributes[i]) << std::endl;
  }
  out << "\tProjected Attributes: " << std::endl;
  for (size_t i = 0; i < param.size(); ++i) {
    out << "\t\t" << toString(param[i]) << std::endl;
  }
  out << "\tIs Empty Pipeline: " << is_empty_pipeline << std::endl;
  if (input_table) {
    out << "\t" << input_table->getName() << std::endl;
  }
  //        out << "\tPipeline End Type: " << getName(pipe_end) << std::endl;
}

bool CodeGenerator::dropProjectionAttributes() {
  param.clear();

  return true;
}

bool CodeGenerator::createForLoop() {
  return createForLoop_impl(this->input_table, this->input_table_version);
}

bool CodeGenerator::createForLoop(const TablePtr table, uint32_t version) {
  if (!table) return false;
  if (!this->input_table) {
    this->input_table = table;
    this->input_table_version = version;
  }
  return createForLoop_impl(table, version);
}

bool CodeGenerator::addToScannedAttributes(const AttributeReference& attr) {
  bool found = false;
  /* ignore all attributes that are computed and add the input columns only */
  if (attr.getAttributeReferenceType() != INPUT_ATTRIBUTE) {
    return false;
  }
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");
  for (size_t i = 0; i < scanned_attributes.size(); ++i) {
    if (scanned_attributes[i].getVersionedAttributeName() ==
        attr.getVersionedAttributeName()) {
      found = true;
      if (debug_code_generator) {
        std::cout << "[DEBUG]: found attribute in scanned attributes: "
                  << attr.getVersionedTableName() << "."
                  << attr.getVersionedAttributeName() << std::endl;
      }
    }
  }
  if (found) {
    if (debug_code_generator) {
      std::cout << "[DEBUG]: addToScannedAttributes: I will NOT add "
                << attr.getVersionedTableName() << "."
                << attr.getVersionedAttributeName() << std::endl;
    }
    return false;
  } else {
    if (debug_code_generator) {
      std::cout << "[DEBUG]: addToScannedAttributes: add "
                << attr.getVersionedTableName() << "."
                << attr.getVersionedAttributeName() << std::endl;
      if ("DATES.D_YEAR.1" == attr.getUnversionedAttributeName()) {
        std::cout << "Bingo!" << std::endl;
      }
    }
    scanned_attributes.push_back(attr);
    return true;
  }
}

bool CodeGenerator::consumeSelection(const PredicateExpressionPtr pred_expr) {
  if (!pred_expr) {
    return false;
  }
  is_empty_pipeline = false;
  std::vector<AttributeReferencePtr> scanned =
      pred_expr->getScannedAttributes();
  for (size_t i = 0; i < scanned.size(); ++i) {
    if (scanned[i]) this->addToScannedAttributes(*scanned[i]);
  }
  pred_expr->replaceTablePointerInAttributeReferences(scanned_attributes);
  return consumeSelection_impl(pred_expr);
}

bool CodeGenerator::consumeBuildHashTable(const AttributeReference& attr) {
  this->pipe_end = MATERIALIZE_FROM_ARRAY_TO_JOIN_HASH_TABLE_AND_ARRAY;
  is_empty_pipeline = false;
  this->addToScannedAttributes(attr);
  AttributeReference copy_attr(attr);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_attr);
  return consumeBuildHashTable_impl(copy_attr);
}

bool CodeGenerator::consumeProbeHashTable(
    const AttributeReference& hash_table_attr,
    const AttributeReference& probe_attr) {
  //        std::cout << toString(hash_table_attr) << std::endl;
  assert(hash_table_attr.getHashTable() != NULL);
  assert(hash_table_attr.getTable()->getHashTablebyName(
             createFullyQualifiedColumnIdentifier(
                 boost::make_shared<AttributeReference>(hash_table_attr))) !=
         NULL);
  is_empty_pipeline = false;
  this->addToScannedAttributes(hash_table_attr);
  this->addToScannedAttributes(probe_attr);
  AttributeReference copy_hash_table_attr(hash_table_attr);
  AttributeReference copy_probe_attr(probe_attr);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_hash_table_attr);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_probe_attr);
  return consumeProbeHashTable_impl(copy_hash_table_attr, copy_probe_attr);
}

bool CodeGenerator::consumeCrossJoin(const AttributeReference& attr) {
  is_empty_pipeline = false;
  this->addToScannedAttributes(attr);
  AttributeReference copy_attr(attr);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_attr);
  return consumeCrossJoin_impl(copy_attr);
}

bool CodeGenerator::consumeNestedLoopJoin(
    const PredicateExpressionPtr pred_expr) {
  is_empty_pipeline = false;
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  return false;
}

const std::pair<bool, std::vector<AttributeReferencePtr> >
CodeGenerator::consumeMapUDF(const Map_UDF_ParamPtr param) {
  is_empty_pipeline = false;

  ScanParam scanned = param->getScannedAttributes();
  for (size_t i = 0; i < scanned.size(); ++i) {
    this->addToScannedAttributes(scanned[i]);
  }
  param->map_udf->replaceTablePointerInAttributeReferences(scanned_attributes);

  return consumeMapUDF_impl(param);
}

bool CodeGenerator::consumeHashGroupAggregate(
    const GroupByAggregateParam& groupby_param) {
  is_empty_pipeline = false;
  /* check whether we have not inserted a pipeline breaking operator before
     if we did, return with error, otherwise, mark that we inserted a
     pipeline breaker */
  if (pipe_end != MATERIALIZE_FROM_ARRAY_TO_ARRAY) {
    COGADB_FATAL_ERROR(
        "Cannot insert two pipeline breaking operations into the same "
        "pipeline!",
        "");
    return false;
  }
  pipe_end = MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY;

  GroupByAggregateParam copy_groupby_param(groupby_param);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_groupby_param.grouping_attrs);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_groupby_param.aggregation_specs);
  //@Bastian: I will probably need this code later, and I will keep it as a
  // comment to preserve it!
  //  std::stringstream str;
  //  str << "GROUPBY Attributes: ";
  //  for(auto i : copy_groupby_param.grouping_attrs){
  //      str << CoGaDB::toString(i);
  //      if(isPersistent(i.getTable() )){
  //        str << " (Persistent)";
  //      }else{
  //        str << " (Intermediate Result TablePtr: " << i.getTable() << ")";
  //      }
  //      str << ", ";
  //  }
  //  str << std::endl;
  //  str << "Aggregation Attributes: ";
  //  for(auto aggr : copy_groupby_param.aggregation_specs){
  //      for (auto i : aggr->getScannedAttributes()){
  //      str << CoGaDB::toString(*i);
  //      if(isPersistent(i->getTable() )){
  //        str << " (Persistent)";
  //      }else{
  //        str << " (Intermediate Result TablePtr: " << i->getTable() << ")";
  //      }
  //      str << ", ";
  //      }
  //  }
  //  std::cout << str.str();

  this->groupby_param =
      GroupByAggregateParamPtr(new GroupByAggregateParam(copy_groupby_param));

  param.clear();

  const AggregateSpecifications& aggr_specs =
      copy_groupby_param.aggregation_specs;
  const GroupingAttributes& grouping_attrs = copy_groupby_param.grouping_attrs;
  bool ret = true;

  for (size_t i = 0; i < grouping_attrs.size(); ++i) {
    AttributeReference attr_ref = grouping_attrs[i];
    ret = this->addAttributeProjection(attr_ref);
    this->addToScannedAttributes(attr_ref);
    if (!ret) COGADB_FATAL_ERROR("Failed to add Projection Attribute!", "");
  }
  for (size_t i = 0; i < aggr_specs.size(); ++i) {
    std::vector<AttributeReferencePtr> scanned_attrs =
        aggr_specs[i]->getScannedAttributes();
    std::vector<AttributeReferencePtr> computed_attrs =
        aggr_specs[i]->getComputedAttributes();
    for (size_t k = 0; k < scanned_attrs.size(); ++k) {
      assert(scanned_attrs[k] != NULL);
      this->addToScannedAttributes(*scanned_attrs[k]);
    }
    for (size_t k = 0; k < computed_attrs.size(); ++k) {
      assert(computed_attrs[k] != NULL);
      ret = this->addAttributeProjection(*computed_attrs[k]);
      if (!ret) COGADB_FATAL_ERROR("Failed to add Projection Attribute!", "");
    }
  }
  return consumeHashGroupAggregate_impl(copy_groupby_param);
}

bool CodeGenerator::consumeAggregate(
    const AggregateSpecifications& aggr_specs) {
  is_empty_pipeline = false;
  /* check whether we have not inserted a pipeline breaking operator before
     if we did, return with error, otherwise, mark that we inserted a
     pipeline breaker */
  if (pipe_end != MATERIALIZE_FROM_ARRAY_TO_ARRAY) {
    COGADB_FATAL_ERROR(
        "Cannot insert two pipeline breaking operations into the same "
        "pipeline!",
        "");
    return false;
  }
  pipe_end = MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY;

  AggregateSpecifications copy_aggr_specs(aggr_specs);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_aggr_specs);

  ProcessorSpecification proc_spec(hype::PD0);
  GroupingAttributes grouping_attrs;
  this->groupby_param = GroupByAggregateParamPtr(
      new GroupByAggregateParam(proc_spec, grouping_attrs, copy_aggr_specs));

  param.clear();

  bool ret = true;
  for (size_t i = 0; i < copy_aggr_specs.size(); ++i) {
    std::vector<AttributeReferencePtr> scanned_attrs =
        copy_aggr_specs[i]->getScannedAttributes();
    std::vector<AttributeReferencePtr> computed_attrs =
        copy_aggr_specs[i]->getComputedAttributes();
    for (size_t k = 0; k < scanned_attrs.size(); ++k) {
      assert(scanned_attrs[k] != NULL);
      this->addToScannedAttributes(*scanned_attrs[k]);
    }
    for (size_t k = 0; k < computed_attrs.size(); ++k) {
      assert(computed_attrs[k] != NULL);
      ret = this->addAttributeProjection(*computed_attrs[k]);
      if (!ret) COGADB_FATAL_ERROR("Failed to add Projection Attribute!", "");
    }
  }

  return consumeAggregate_impl(copy_aggr_specs);
}

const std::pair<bool, AttributeReference>
CodeGenerator::consumeAlgebraComputation(const AttributeReference& left_attr,
                                         const AttributeReference& right_attr,
                                         const ColumnAlgebraOperation& alg_op) {
  is_empty_pipeline = false;

  AttributeReference copy_left_attr(left_attr);
  AttributeReference copy_right_attr(right_attr);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_left_attr);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_right_attr);
  this->addToScannedAttributes(copy_left_attr);
  this->addToScannedAttributes(copy_right_attr);

  return consumeAlgebraComputation_impl(copy_left_attr, copy_right_attr,
                                        alg_op);
}

const std::pair<bool, AttributeReference>
CodeGenerator::consumeAlgebraComputation(const AttributeReference& left_attr,
                                         const boost::any constant,
                                         const ColumnAlgebraOperation& alg_op) {
  is_empty_pipeline = false;
  AttributeReference copy_left_attr(left_attr);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_left_attr);
  this->addToScannedAttributes(copy_left_attr);
  return consumeAlgebraComputation_impl(copy_left_attr, constant, alg_op);
}

const std::pair<bool, AttributeReference>
CodeGenerator::consumeAlgebraComputation(const boost::any constant,
                                         const AttributeReference& right_attr,
                                         const ColumnAlgebraOperation& alg_op) {
  is_empty_pipeline = false;
  AttributeReference copy_right_attr(right_attr);
  replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, copy_right_attr);
  this->addToScannedAttributes(copy_right_attr);
  return consumeAlgebraComputation_impl(constant, copy_right_attr, alg_op);
}

CodeGeneratorType CodeGenerator::getCodeGeneratorType() const {
  return generator_type;
}

bool CodeGenerator::canOmitCompilation() const {
  /* did we add operators into the pipeline? */
  if (this->isEmpty()) {
    /* When a pipeline has the same input and output and does not do
     * anything with the input, we can omit the compilation step. */
    assert(this->input_table != NULL);
    std::cout << "Empty Pipe:" << std::endl;
    return true;
  } else {
    return false;
  }
}

const CodeGeneratorPtr createCodeGenerator(
    const CodeGeneratorType code_gen, const ProjectionParam& param,
    const TablePtr input_table, const boost::any& generic_code_gen_param,
    uint32_t version) {
  CodeGeneratorPtr resultCodeGenerator;

  if (code_gen == CPP_CODE_GENERATOR) {
    resultCodeGenerator =
        boost::make_shared<CPPCodeGenerator>(param, input_table, version);
  } else if (code_gen == C_CODE_GENERATOR) {
    resultCodeGenerator =
        boost::make_shared<CCodeGenerator>(param, input_table, version);
  } else if (code_gen == CUDA_C_CODE_GENERATOR) {
    resultCodeGenerator =
        boost::make_shared<CUDA_C_CodeGenerator>(param, input_table, version);
  } else if (code_gen == MULTI_STAGE_CODE_GENERATOR) {
    resultCodeGenerator = boost::make_shared<MultiStageCodeGenerator>(
        param, input_table, version);
  } else {
    COGADB_FATAL_ERROR("Unknown Code Generator!", "");
    return CodeGeneratorPtr();
  }

  resultCodeGenerator->createForLoop();
  return resultCodeGenerator;
}

const CodeGeneratorPtr createCodeGenerator(
    const CodeGeneratorType code_gen, const ProjectionParam& param,
    const boost::any& generic_code_gen_param) {
  if (code_gen == CPP_CODE_GENERATOR) {
    return boost::make_shared<CPPCodeGenerator>(param);
  } else if (code_gen == C_CODE_GENERATOR) {
    return boost::make_shared<CCodeGenerator>(param);
  } else if (code_gen == MULTI_STAGE_CODE_GENERATOR) {
    return boost::make_shared<MultiStageCodeGenerator>(param);
  } else if (code_gen == CUDA_C_CODE_GENERATOR) {
    return boost::make_shared<CUDA_C_CodeGenerator>(param);
  } else {
    COGADB_FATAL_ERROR("Unknown Code Generator!", "");
    return CodeGeneratorPtr();
  }
}

bool isEquivalent(const ProjectionParam& projected_attributes,
                  const TableSchema& input_schema) {
  if (projected_attributes.size() != input_schema.size()) return false;
  for (size_t i = 0; i < projected_attributes.size(); ++i) {
    bool found = false;
    TableSchema::const_iterator cit;
    for (cit = input_schema.begin(); cit != input_schema.end(); ++cit) {
      if (projected_attributes[i].getUnversionedAttributeName() == cit->second
          /* do we not perform renaming? If yes, we cannot omit compilation!  */
          //                  &&
          //                  projected_attributes[i].getUnversionedAttributeName()==projected_attributes[i].getResultAttributeName()
          ) {
        found = true;
      }
    }
    std::cout << "Not found: "
              << projected_attributes[i].getUnversionedAttributeName()
              << std::endl;
    if (!found) {
      return false;
    }
  }
  return true;
}

void storeResultTableAttributes(CodeGeneratorPtr code_gen,
                                QueryContextPtr context, TablePtr result) {
  /* we later require the attribute references to the computed result,
   so we store it in the query context */
  std::vector<ColumnProperties> col_props = result->getPropertiesOfColumns();
  auto projection_list = context->getProjectionList();

  std::vector<AttributeReferencePtr> result_attributes;
  for (size_t i = 0; i < col_props.size(); ++i) {
    AttributeReferencePtr attr;
    if (!isFullyQualifiedColumnIdentifier(col_props[i].name)) {
      size_t num_occurences =
          DataDictionary::instance().countNumberOfOccurencesOfColumnName(
              col_props[i].name);
      if (num_occurences > 1) {
        attr =
            boost::make_shared<AttributeReference>(result, col_props[i].name);
      }
    }

    if (!attr) {
      //            std::string qualified_column_name =
      //            convertToFullyQualifiedNameIfRequired(col_props[i].name);
      //            AttributeReferencePtr
      attr = getAttributeFromColumnIdentifier(col_props[i].name);
    }

    if (attr) {
      //            assert(attr!=NULL);
      attr = createInputAttributeForNewTable(*attr, result);
    } else {
      /* is computed attribute, look in symbol table wether we find it there */
      attr = getAttributeReference(col_props[i].name, code_gen, context);
      if (!attr) {
        attr = boost::make_shared<AttributeReference>(
            col_props[i].name, col_props[i].attribute_type, col_props[i].name,
            1);
        assert(attr != NULL);
      }
    }
    assert(attr != NULL);
    context->addReferencedAttributeFromOtherPipeline(attr);
    code_gen->addToScannedAttributes(*attr);
    result_attributes.push_back(attr);
  }
  /* add attribute references of result table to pipelines
   * output schema, if we find them in the list of projected columns */
  for (auto proj_attr : projection_list) {
    for (auto result_attr : result_attributes) {
      if (toString(*proj_attr) == toString(*result_attr)) {
        code_gen->addAttributeProjection(*result_attr);
      }
    }
  }
  /* project all result columns in case projection list is empty*/
  if (projection_list.empty()) {
    for (auto result_attr : result_attributes) {
      code_gen->addAttributeProjection(*result_attr);
    }
  }
}

void retrieveScannedAndProjectedAttributesFromScannedTable(
    CodeGeneratorPtr code_gen, QueryContextPtr context, TablePtr scanned_table,
    uint32_t version) {
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");

  std::map<std::string, std::string> rename_map = context->getRenameMap();
  std::map<std::string, std::string>::const_iterator rename_cit;

  std::vector<AttributeReferencePtr> proj_cols = context->getProjectionList();
  if (!proj_cols.empty()) {
    for (size_t i = 0; i < proj_cols.size(); ++i) {
      std::string new_name = createFullyQualifiedColumnIdentifier(proj_cols[i]);
      if (debug_code_generator) {
        std::cout << "SCAN_OPERATOR: Read Projected Col: " << new_name << " ("
                  << proj_cols[i] << ")" << std::endl;
      }
      if (proj_cols[i]->getVersion() == version &&
          scanned_table->hasColumn(*proj_cols[i])) {
        AttributeReference attr(scanned_table,
                                proj_cols[i]->getUnversionedAttributeName(),
                                new_name, version);
        code_gen->addAttributeProjection(attr);
        if (debug_code_generator) {
          std::cout << "[SCAN]: Add projection attribute: " << toString(attr)
                    << "->" << attr.getResultAttributeName() << " to pipeline "
                    << (void*)context.get() << std::endl;
        }
      } else {
        if (debug_code_generator) {
          std::cout << "[SCAN]: Could not resolve "
                    << "projection attribute '"
                    << createFullyQualifiedColumnIdentifier(proj_cols[i]) << "'"
                    << " in table '" << scanned_table->getName() << "'"
                    << std::endl;
        }
        context->addUnresolvedProjectionAttribute(
            createFullyQualifiedColumnIdentifier(proj_cols[i]));
      }
    }
  } else {
    /* the projection list is empty, which means that
     * we have no projection as parent in the pipeline.
     * Thus we have to add all attributes in the input
     * table to the projection list.
     */
    std::vector<ColumnProperties> col_props =
        scanned_table->getPropertiesOfColumns();
    for (size_t i = 0; i < col_props.size(); ++i) {
      //                                    std::stringstream new_name;
      std::string new_name = col_props[i].name + std::string(".") +
                             boost::lexical_cast<std::string>(version);
      rename_cit = rename_map.find(col_props[i].name);
      if (rename_cit != rename_map.end()) {
        new_name = rename_cit->second;
        if (debug_code_generator) {
          std::cout << "[SCAN]: Rename attribute: " << col_props[i].name << "->"
                    << new_name << std::endl;
        }
      }
      AttributeReference attr(scanned_table, col_props[i].name, new_name,
                              version);
      code_gen->addAttributeProjection(attr);
    }
  }
  /* add all accessed columns we have found so far to the pipeline,
   * if they originate from here */
  std::vector<std::string> accessed_columns = context->getAccessedColumns();
  for (size_t i = 0; i < accessed_columns.size(); ++i) {
    std::string current_column = accessed_columns[i];
    AttributeReferencePtr attr =
        getAttributeFromColumnIdentifier(current_column);
    /* check whether this is a valid attribute reference to a column either
     * stored in a persistent database table or
     * is originating from a persistent database table.In case this pointer is
     * zero,
     * we know that it is a computed attribute. As we do not need to scan
     * computed attributes, we can safely omit this
     * part if the pointer is zero.
     */
    if (attr) {
      if (attr->getVersion() == version && scanned_table->hasColumn(*attr)) {
        std::string new_name = current_column;
        rename_cit = rename_map.find(current_column);
        if (rename_cit != rename_map.end()) {
          new_name = rename_cit->second;
          if (debug_code_generator) {
            std::cout << "[SCAN]: Rename attribute: " << current_column << "->"
                      << new_name << std::endl;
          }
        }
        assert(attr != NULL);
        attr = createInputAttributeForNewTable(*attr, scanned_table);
        assert(attr != NULL);

        if (debug_code_generator) {
          std::cout << "[SCAN]: Add accessed attribute: "
                    << CoGaDB::toString(*attr) << " to pipeline "
                    << (void*)context.get() << std::endl;
        }
        code_gen->addToScannedAttributes(*attr);
      }
    }
  }
}

}  // end namespace CoGaDB
