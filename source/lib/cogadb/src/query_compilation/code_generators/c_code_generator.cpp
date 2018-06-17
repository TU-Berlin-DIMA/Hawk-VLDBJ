
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include <query_compilation/code_generators/c_code_generator.hpp>

#include <iomanip>
#include <list>
#include <set>
#include <sstream>

#include <core/selection_expression.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <util/getname.hpp>
#include <util/iostream.hpp>

#include <dlfcn.h>
#include <stdlib.h>
#include <core/data_dictionary.hpp>
#include <util/functions.hpp>
#include <util/time_measurement.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/make_shared.hpp>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>
#include <ctime>

#include <google/dense_hash_map>  // streaming operators etc.

#include <core/variable_manager.hpp>
#include <util/code_generation.hpp>
#include <util/shared_library.hpp>

#include <query_compilation/pipeline_info.hpp>

#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>

namespace CoGaDB {

CCodeGenerator::CCodeGenerator(const ProjectionParam& _param,
                               const TablePtr table, uint32_t version)
    : CodeGenerator(C_CODE_GENERATOR, _param, table, version),
      mHeaderAndTypesBlock(),
      mFetchInputCodeBlock(),
      mGeneratedCode(),
      mUpperCodeBlock(),
      mLowerCodeBlock(),
      mAfterForLoopBlock(),
      mCreateResulttableCodeBlock(),
      mCleanupCode() {
  init();
}

CCodeGenerator::CCodeGenerator(const ProjectionParam& _param)
    : CodeGenerator(C_CODE_GENERATOR, _param),
      mHeaderAndTypesBlock(),
      mFetchInputCodeBlock(),
      mGeneratedCode(),
      mUpperCodeBlock(),
      mLowerCodeBlock(),
      mAfterForLoopBlock(),
      mCreateResulttableCodeBlock(),
      mCleanupCode() {
  init();
}

void CCodeGenerator::init() {
  /* write function signature */
  mFetchInputCodeBlock << "const C_Table* compiled_query(C_Table** c_tables) {"
                       << std::endl;
  mFetchInputCodeBlock << "size_t current_result_size = 0;" << std::endl;
  mFetchInputCodeBlock << "size_t allocated_result_elements = 10000;"
                       << std::endl;
}

bool CCodeGenerator::produceTuples(const ScanParam& param) {
  if (tuples_produced) {
    return true;
  }

  tuples_produced = true;

  std::set<std::string> scanned_tables;

  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");

  size_t param_id = 0;
  ScanParam::const_iterator cit;
  for (cit = param.begin(); cit != param.end(); ++cit, ++param_id) {
    if (cit->getTable() == NULL) {
      COGADB_FATAL_ERROR("Found no valid TablePtr in Scan Attribute: "
                             << cit->getVersionedAttributeName(),
                         "");
    }

    if (scanned_tables.find(cit->getTable()->getName()) ==
        scanned_tables.end()) {
      // we assume that the same ScanParam passed to this function
      // is also passed to the compiled query function
      mFetchInputCodeBlock << "C_Table* " << getTableVarName(*cit)
                           << " = c_tables[" << param_id << "];" << std::endl;
      mFetchInputCodeBlock << "assert(" << getTableVarName(*cit) << " != NULL);"
                           << std::endl;
      scanned_tables.insert(cit->getTable()->getName());

      if (debug_code_generator) {
        std::cout << "[DEBUG:] Produce Tuples for Table "
                  << getTableVarName(*cit) << std::endl;
      }
    }
    mFetchInputCodeBlock << "uint64_t " << getInputColumnVarName(*cit)
                         << "_length = getNumberOfRows("
                         << getTableVarName(*cit) << ");" << std::endl;
    mFetchInputCodeBlock << "C_Column* " << getInputColumnVarName(*cit)
                         << " = getColumnById(" << getTableVarName(*cit) << ","
                         << getConstant(
                                cit->getTable()->getColumnIdbyColumnName(
                                    CoGaDB::toString(*cit)))
                         << ");" << std::endl;
    mCleanupCode << "releaseColumn(" << getInputColumnVarName(*cit) << ");"
                 << std::endl;

    mFetchInputCodeBlock << "if (!" << getInputColumnVarName(*cit)
                         << ") { printf(\"Column '"
                         << getInputColumnVarName(*cit)
                         << "' not found!\\n\"); return NULL; }" << std::endl;

    mFetchInputCodeBlock << getArrayFromColumnCode(*cit);
  }
  /* generate code to decompress the columns we cannot work on directly
   * (e.g., ">" or "<" predicates on string columns) */
  for (std::map<std::string, AttributeReferencePtr>::const_iterator itr(
           mColumnsToDecompress.begin());
       itr != mColumnsToDecompress.end(); ++itr) {
    mFetchInputCodeBlock << getArrayFromColumnCode(*itr->second, true);
  }
  return true;
}

bool CCodeGenerator::createForLoop_impl(const TablePtr table,
                                        uint32_t version) {
  std::string loop_table_tuple_id = getTupleIDVarName(table, version);
  std::stringstream for_loop;
  for_loop << "size_t num_elements_" << getTupleIDVarName(table, version)
           << " = getNumberOfRows(" << getTableVarName(table, version) << ");"
           << std::endl;

  for_loop << "size_t " << loop_table_tuple_id << ";";

  for_loop << "for (" << loop_table_tuple_id << " = 0; " << loop_table_tuple_id
           << " < num_elements_" << getTupleIDVarName(table, version) << "; ++"
           << loop_table_tuple_id << ") {";

  mUpperCodeBlock.push_back(for_loop.str());
  mLowerCodeBlock.push_front("}");

  return true;
}

const std::string CCodeGenerator::getCodeAllocateResultTable() const {
  return generateCCodeAllocateResultTable(param);
}

const std::string CCodeGenerator::getCodeWriteResult() const {
  return generateCCodeWriteResult(param);
}

const std::string CCodeGenerator::getCodeWriteResultFromHashTable() const {
  return generateCCodeWriteResultFromHashTable(param);
}

const std::string CCodeGenerator::createResultTable() const {
  return generateCCodeCreateResultTable(
      param, mCreateResulttableCodeBlock.str(), mCleanupCode.str(),
      input_table->getName());
}

const AttributeReference CCodeGenerator::getAttributeReference(
    const std::string& column_name) const {
  COGADB_FATAL_ERROR("Called Obsolete Function!", "");
}

bool CCodeGenerator::consumeSelection_impl(
    const PredicateExpressionPtr pred_expr) {
  std::stringstream ss;
  ss << "if (" << pred_expr->getCPPExpression() << ") {";
  mUpperCodeBlock.push_back(ss.str());
  mLowerCodeBlock.push_front("}");

  std::vector<AttributeReferencePtr> columns_to_decompress =
      pred_expr->getColumnsToDecompress();
  for (std::vector<AttributeReferencePtr>::iterator itr(
           columns_to_decompress.begin());
       itr != columns_to_decompress.end(); ++itr) {
    mColumnsToDecompress[(*itr)->getVersionedAttributeName()] = *itr;
  }

  return true;
}

bool CCodeGenerator::createHashTable(const AttributeReference& attr) {
  mGeneratedCode << "hashtable_t* " << getHashTableVarName(attr)
                 << " = hash_new(10000);" << std::endl;
  return true;
}

bool CCodeGenerator::consumeBuildHashTable_impl(
    const AttributeReference& attr) {
  /* add code for building hash table inside for loop */
  std::stringstream hash_table;

  /* add code for hash table creation before for loop */
  this->ht_gen = CoGaDB::createHashTableGenerator(attr);
  mGeneratedCode << ht_gen->generateCodeDeclareHashTable();
  mGeneratedCode << ht_gen->generateCodeInitHashTable(
      true, std::to_string(attr.getTable()->getNumberofRows()));
  // mGeneratedCode << "hashtable_t* " << getHashTableVarName(attr) <<
  // "=hash_new (10000);" << std::endl;
  /* insert values into hash table */

  hash_table << ht_gen->generateCodeInsertIntoHashTable(1);
  /*
  hash_table << "tuple_t t_" <<
  getVariableFromAttributeName(attr.getVersionedAttributeName())
          << " = {" << getInputArrayVarName(attr)
          << "[" << getTupleIDVarName(attr) << "], "
          << "current_result_size};" << std::endl;
  hash_table << "hash_put(" << getHashTableVarName(attr);
  hash_table << ", t_" <<
  getVariableFromAttributeName(attr.getVersionedAttributeName()) << ");" <<
  std::endl;
*/
  mUpperCodeBlock.push_back(hash_table.str());
  mLowerCodeBlock.push_front(" ");

  /* add result creation code*/
  mCreateResulttableCodeBlock
      << "C_HashTable* " << getHashTableVarName(attr) << "_system = "
      << "createSystemHashTable(" << getHashTableVarName(attr)
      << ", (HashTableCleanupFunctionPtr)&"
      << ht_gen->generateIdentifierCleanupHandler() << ", \" "
      << ht_gen->identifier_ << "\");" << std::endl;
  mCreateResulttableCodeBlock
      << "if (!addHashTable(result_table, \"" << attr.getResultAttributeName()
      << "\", " << getHashTableVarName(attr) << "_system)) {"
      << "printf(\"Error adding hash table for attribute '"
      << attr.getResultAttributeName() << "' to result!\\n\");" << std::endl
      << "return NULL;" << std::endl
      << "}" << std::endl;

  mCleanupCode << "releaseHashTable(" << getHashTableVarName(attr)
               << "_system);" << std::endl;

  return true;
}

bool CCodeGenerator::consumeProbeHashTable_impl(
    const AttributeReference& hash_table_attr,
    const AttributeReference& probe_attr) {
  /* code for hash table probes */
  std::stringstream hash_probe;
  /* closing brackets and pointer chasing */
  std::stringstream hash_probe_lower;

  this->ht_gen = CoGaDB::createHashTableGenerator(hash_table_attr);

  mGeneratedCode << "TID " << getTupleIDVarName(hash_table_attr) << "=0;"
                 << std::endl;
  mGeneratedCode << "C_HashTable* generic_hashtable_"
                 << getHashTableVarName(hash_table_attr) << "=getHashTable("
                 << getTableVarName(hash_table_attr) << ", "
                 << "\""
                 << createFullyQualifiedColumnIdentifier(
                        boost::make_shared<AttributeReference>(hash_table_attr))
                 << "\");" << std::endl;
  mGeneratedCode << ht_gen->generateCodeDeclareHashTable();
  mGeneratedCode << ht_gen->generateCodeInitHashTable(
                        false,
                        std::to_string(
                            hash_table_attr.getTable()->getNumberofRows()))
                 << std::endl;

  ProbeHashTableCode probeCode =
      this->ht_gen->generateCodeProbeHashTable(probe_attr, 1);
  hash_probe << probeCode.first;
  hash_probe_lower << probeCode.second;

  mUpperCodeBlock.push_back(hash_probe.str());
  mLowerCodeBlock.push_front(hash_probe_lower.str());

  mCleanupCode << "releaseHashTable(generic_hashtable_"
               << getHashTableVarName(hash_table_attr) << ");" << std::endl;

  return true;
}

const std::string CCodeGenerator::getAggregationCode(
    const GroupingAttributes& grouping_columns,
    const AggregateSpecifications& aggregation_specs,
    const std::string access_ht_entry_expression) const {
  return getAggregationCodeGeneric(grouping_columns, aggregation_specs,
                                   access_ht_entry_expression);
  //  std::stringstream hash_aggregate;
  //  for (size_t i = 0; i < aggregation_specs.size(); ++i) {
  //    hash_aggregate << aggregation_specs[i]->getCodeHashGroupBy(
  //        access_ht_entry_expression);
  //    for (size_t i = 0; i < grouping_columns.size(); ++i) {
  //      AttributeReference attr_ref = grouping_columns[i];
  //      hash_aggregate << access_ht_entry_expression
  //                     << getGroupTIDVarName(attr_ref) << " = "
  //                     << getTupleIDVarName(attr_ref) << ";" << std::endl;
  //    }
  //  }
  //  return hash_aggregate.str();
}

bool CCodeGenerator::consumeAggregate_impl(
    const AggregateSpecifications& params) {
  for (const auto& param : params) {
    mGeneratedCode << param->getCodeAggregationDeclareVariables(C_TARGET_CODE)
                   << std::endl;
    mGeneratedCode << param->getCodeAggregationInitializeVariables(
                          C_TARGET_CODE)
                   << std::endl;

    for (const auto& p : param->getComputedAttributes()) {
      aggr_result_params_.push_back(*p);
    }

    std::string compute_expr =
        param->getCodeAggregationComputation(C_TARGET_CODE);
    mUpperCodeBlock.push_back(compute_expr);
    mLowerCodeBlock.push_front(" ");

    mAfterForLoopBlock << param->getCodeAggregationWriteResult(C_TARGET_CODE)
                       << std::endl;
  }
  mAfterForLoopBlock << "++current_result_size;" << std::endl;

  return true;
}

bool CCodeGenerator::consumeCrossJoin_impl(const AttributeReference& attr) {
  std::stringstream loop;
  loop << "for (size_t " << getTupleIDVarName(attr) << " = 0; "
       << getTupleIDVarName(attr) << "< " << attr.getTable()->getNumberofRows()
       << "ULL; ++" << getTupleIDVarName(attr) << ") {";

  mUpperCodeBlock.push_back(loop.str());
  mLowerCodeBlock.push_front("}");

  return true;
}

bool CCodeGenerator::consumeNestedLoopJoin_impl(
    const PredicateExpressionPtr pred_expr) {
  COGADB_NOT_IMPLEMENTED;
}

const std::string generateCode_BitpackedGroupingKeyComputation(
    const GroupingAttributes& grouping_attrs) {
  std::stringstream hash_aggregate;
  hash_aggregate << "TID group_key = "
                 << getComputeGroupIDExpression(grouping_attrs) << ";"
                 << std::endl;
  return hash_aggregate.str();
}

const std::string generateCode_GenericGroupingKeyComputation(
    const GroupingAttributes& grouping_attrs) {
  std::stringstream compute_group_id;
  compute_group_id << "TID group_key = getGroupKey(" << grouping_attrs.size()
                   << ", ";

  for (size_t col_id = 0; col_id < grouping_attrs.size(); ++col_id) {
    compute_group_id << grouping_attrs[col_id].getAttributeType() << ", "
                     << getCompressedElementAccessExpression(
                            grouping_attrs[col_id]);

    if (col_id + 1 < grouping_attrs.size()) {
      compute_group_id << ", ";
    }
  }

  compute_group_id << ");" << std::endl;
  return compute_group_id.str();
}

bool CCodeGenerator::consumeHashGroupAggregate_impl(
    const GroupByAggregateParam& groupby_param) {
  const AggregateSpecifications& aggr_specs = groupby_param.aggregation_specs;
  const GroupingAttributes& grouping_attrs = groupby_param.grouping_attrs;

  /* generate struct that serves as payload based on aggregation functions */
  mHeaderAndTypesBlock << "struct AggregationPayload {" << std::endl;

  for (size_t i = 0; i < grouping_attrs.size(); ++i) {
    AttributeReference attr_ref = grouping_attrs[i];
    mHeaderAndTypesBlock << getAggregationGroupTIDPayloadFieldCode(attr_ref)
                         << std::endl;
    if (isComputed(attr_ref)) {
      COGADB_FATAL_ERROR(
          "Grouping by a computed attribute is currently not supported in the "
          "C Code Generator. "
          "Please use the multi stage code generateor instead."
          "Too switch to the multi stage code generator, please set the "
          "following variable "
          "in the command line interface: "
              << std::endl
              << "set default_code_generator=multi_staged" << std::endl,
          "");
    }
  }

  std::set<AggregateSpecification::AggregationPayloadField> struct_fields;

  for (size_t i = 0; i < aggr_specs.size(); ++i) {
    std::vector<AggregateSpecification::AggregationPayloadField>
        payload_fields = aggr_specs[i]->getAggregationPayloadFields();
    for (size_t k = 0; k < payload_fields.size(); ++k) {
      struct_fields.insert(payload_fields.begin(), payload_fields.end());
    }
  }

  std::set<AggregateSpecification::AggregationPayloadField>::const_iterator cit;
  for (cit = struct_fields.begin(); cit != struct_fields.end(); ++cit) {
    mHeaderAndTypesBlock << *cit << ";" << std::endl;
  }

  mHeaderAndTypesBlock << "};" << std::endl;
  mHeaderAndTypesBlock
      << "typedef struct AggregationPayload AggregationPayload;" << std::endl;

  /* create hash table */
  mGeneratedCode << "C_AggregationHashTable* aggregation_hash_table = "
                    "createAggregationHashTable(sizeof(AggregationPayload));"
                 << std::endl;
  mCleanupCode << "freeAggregationHashTable(aggregation_hash_table);"
               << std::endl;

  /* determine the grouping key according to grouping columns*/
  std::stringstream hash_aggregate;

  if (isBitpackedGroupbyOptimizationApplicable(grouping_attrs)) {
    hash_aggregate << generateCode_BitpackedGroupingKeyComputation(
        grouping_attrs);
  } else {
    hash_aggregate << generateCode_GenericGroupingKeyComputation(
        grouping_attrs);
    std::cout << "[INFO]: Cannot use bitpacked grouping, fallback to generic "
                 "implementation..."
              << std::endl;
  }

  /* do the usual hash table probe, aggregate using the custom payload for
   * each aggregation function. This performs only one lookup per tuple. */
  hash_aggregate << "AggregationPayload* aggregation_payload = "
                    "(AggregationPayload*)getAggregationHashTablePayload("
                    "aggregation_hash_table, group_key);"
                 << std::endl;
  hash_aggregate << "if(aggregation_payload){" << std::endl;
  hash_aggregate << getAggregationCode(grouping_attrs, aggr_specs,
                                       "aggregation_payload->")
                 << std::endl;
  hash_aggregate << "} else {" << std::endl;
  hash_aggregate << "AggregationPayload payload;" << std::endl;
  /* init payload fields */
  for (size_t i = 0; i < aggr_specs.size(); ++i) {
    hash_aggregate << aggr_specs[i]->getCodeInitializeAggregationPayloadFields(
        "payload.");
  }
  /* insert new key and payload in hash table */
  hash_aggregate << "aggregation_payload = "
                    "(AggregationPayload*)insertAggregationHashTable("
                    "aggregation_hash_table, group_key, &payload);"
                 << std::endl;
  hash_aggregate << getAggregationCode(grouping_attrs, aggr_specs,
                                       "aggregation_payload->")
                 << std::endl;
  hash_aggregate << "}" << std::endl;

  mUpperCodeBlock.push_back(hash_aggregate.str());
  mLowerCodeBlock.push_front(" ");

  /* write result from hash table to output arrays */
  mAfterForLoopBlock << "if "
                        "(getAggregationHashTableSize(aggregation_hash_table) "
                        ">= allocated_result_elements) {"
                     << std::endl;
  mAfterForLoopBlock << "    allocated_result_elements = "
                        "getAggregationHashTableSize(aggregation_hash_table);"
                     << std::endl;
  mAfterForLoopBlock << "    " << getCodeReallocResultMemory(param)
                     << std::endl;
  mAfterForLoopBlock << "}" << std::endl;

  mAfterForLoopBlock << "C_AggregationHashTableIterator* aggr_itr = "
                        "createAggregationHashTableIterator(aggregation_hash_"
                        "table);"
                     << std::endl;
  mAfterForLoopBlock << "for (; hasNextAggregationHashTableIterator(aggr_itr); "
                        "nextAggregationHashTableIterator(aggr_itr)) {"
                     << std::endl;

  for (size_t i = 0; i < aggr_specs.size(); ++i) {
    mAfterForLoopBlock << aggr_specs[i]->getCodeFetchResultsFromHashTableEntry(
        "((AggregationPayload*)getAggregationHashTableIteratorPayload(aggr_itr)"
        ")->");
  }

  for (size_t i = 0; i < grouping_attrs.size(); ++i) {
    mAfterForLoopBlock << "    " << getResultArrayVarName(grouping_attrs[i])
                       << "[current_result_size] = "
                       << getCompressedElementAccessExpression(
                              grouping_attrs[i],
                              "((AggregationPayload*)"
                              "getAggregationHashTableIteratorPayload(aggr_itr)"
                              ")->" +
                                  getGroupTIDVarName(grouping_attrs[i]))
                       << ";" << std::endl;
  }

  mAfterForLoopBlock << "    ++current_result_size;" << std::endl;
  mAfterForLoopBlock << "}" << std::endl;
  mAfterForLoopBlock << "freeAggregationHashTableIterator(aggr_itr);"
                     << std::endl;

  return true;
}

const std::pair<bool, std::vector<AttributeReferencePtr>>
CCodeGenerator::consumeMapUDF_impl(const Map_UDF_ParamPtr param) {
  ProjectionParam project_param;
  Map_UDF_Result result =
      param->map_udf->generateCode(scanned_attributes, project_param);

  if (!result.return_code) {
    COGADB_FATAL_ERROR("Failed to generate code for Map UDF type: "
                           << int(param->map_udf->getMap_UDF_Type()),
                       "");
  }
  mGeneratedCode << result.declared_variables;
  mUpperCodeBlock.push_back(result.generated_code);
  mLowerCodeBlock.push_front(" ");

  return std::pair<bool, std::vector<AttributeReferencePtr>>(
      true, result.computed_attributes);
}

const std::pair<bool, AttributeReference>
CCodeGenerator::consumeAlgebraComputation_impl(
    const AttributeReference& left_attr, const AttributeReference& right_attr,
    const ColumnAlgebraOperation& alg_op) {
  AttributeReference computed_attr =
      createComputedAttribute(left_attr, right_attr, alg_op);

  // mGeneratedCode << getCodeDeclareResultMemory(computed_attr);
  mGeneratedCode << "double " << getElementAccessExpression(computed_attr)
                 << " = 0.0;" << std::endl;

  std::stringstream compute_expr;
  compute_expr << getElementAccessExpression(computed_attr) << " = "
               << getElementAccessExpression(left_attr) << " "
               << toCPPOperator(alg_op) << " "
               << getElementAccessExpression(right_attr) << ";";

  mUpperCodeBlock.push_back(compute_expr.str());
  mLowerCodeBlock.push_front(" ");
  return std::make_pair(true, computed_attr);
}

const std::pair<bool, AttributeReference>
CCodeGenerator::consumeAlgebraComputation_impl(
    const AttributeReference& left_attr, const boost::any constant,
    const ColumnAlgebraOperation& alg_op) {
  AttributeReference computed_attr =
      createComputedAttribute(left_attr, constant, alg_op);

  // mGeneratedCode << getCodeDeclareResultMemory(computed_attr);
  mGeneratedCode << "double " << getElementAccessExpression(computed_attr)
                 << " = 0.0;" << std::endl;

  std::stringstream compute_expr;
  compute_expr << getElementAccessExpression(computed_attr) << " = "
               << getElementAccessExpression(left_attr) << " "
               << toCPPOperator(alg_op) << " " << getConstant(constant) << ";";

  mUpperCodeBlock.push_back(compute_expr.str());
  mLowerCodeBlock.push_front(" ");
  return std::make_pair(true, computed_attr);
}

const std::pair<bool, AttributeReference>
CCodeGenerator::consumeAlgebraComputation_impl(
    const boost::any constant, const AttributeReference& right_attr,
    const ColumnAlgebraOperation& alg_op) {
  AttributeReference computed_attr =
      createComputedAttribute(constant, right_attr, alg_op);

  // mGeneratedCode << getCodeDeclareResultMemory(computed_attr);
  mGeneratedCode << "double " << getElementAccessExpression(computed_attr)
                 << " = 0.0;" << std::endl;

  std::stringstream compute_expr;
  compute_expr << getElementAccessExpression(computed_attr) << " = "
               << getConstant(constant) << " " << toCPPOperator(alg_op) << " "
               << getElementAccessExpression(right_attr) << ";";

  mUpperCodeBlock.push_back(compute_expr.str());
  mLowerCodeBlock.push_front(" ");
  return std::make_pair(true, computed_attr);
}

void CCodeGenerator::printCode(std::ostream& out) {
  bool ret = produceTuples(scanned_attributes);
  assert(ret);

  /* all imports and declarations */
  out << mHeaderAndTypesBlock.str() << std::endl;
  out << mFetchInputCodeBlock.str() << std::endl;
  /* all code for query function definition and input array retrieval */
  out << mGeneratedCode.str() << std::endl;
  /* reserve memory for each attribute in projection param */
  auto params = param;

  // we don't want do declare parameters twice, so remove the already declared
  // parameters
  for (auto& p : aggr_result_params_) {
    auto find = std::find(params.begin(), params.end(), p);

    if (find != params.end()) {
      params.erase(find);
    }
  }

  out << getCodeDeclareResultMemory(params) << std::endl;
  out << generateCCodeAllocateResultTable(params) << std::endl;

  /* add for loop and it's contents */
  std::list<std::string>::const_iterator cit;
  for (cit = mUpperCodeBlock.begin(); cit != mUpperCodeBlock.end(); ++cit) {
    out << *cit << std::endl;
  }

  /* if we do not materialize into a hash table during aggregation,
     write result regularely */
  if (pipe_end != MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
    out << getCodeWriteResult() << std::endl;
  }

  /* generate closing brackets, pointer chasing, and cleanup operations */
  for (cit = mLowerCodeBlock.begin(); cit != mLowerCodeBlock.end(); ++cit) {
    out << *cit << std::endl;
  }

  /* if we do materialize into a hash table during aggregation,
     write copy result from hash table to output arrays */
  if (pipe_end == MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
    out << mAfterForLoopBlock.str();
  }

  /* generate code that builds the reslt table using the minimal API */
  out << createResultTable() << std::endl;
  out << "}" << std::endl;
}

void CCodeGenerator::compile(const std::string& source,
                             SharedCLibPipelineQueryPtr& query_ptr,
                             boost::shared_ptr<llvm::ExecutionEngine>& engine,
                             boost::shared_ptr<llvm::LLVMContext>& context) {
  static const char* arg_buffer[] = {"",
                                     "-x",
                                     "c",
                                     "string-input",
                                     "-Wno-parentheses-equality",
#ifdef SSE41_FOUND
                                     "-msse4.1",
#endif
#ifdef SSE42_FOUND
                                     "-msse4.2",
#endif
#ifdef AVX_FOUND
                                     "-mavx",
#endif
#ifdef AVX2_FOUND
                                     "-mavx2",
#endif
                                     "-fpic",
                                     "-includeminimal_api_c.h"};
  static const int arg_count = sizeof(arg_buffer) / sizeof(arg_buffer[0]);

  query_ptr = nullptr;

  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeNativeTarget();

  // skip program name
  clang::ArrayRef<const char*> args(arg_buffer, arg_count);

  std::unique_ptr<clang::CompilerInvocation> invocation(
      clang::createInvocationFromCommandLine(args));
  if (invocation.get() == nullptr) {
    COGADB_ERROR("Failed to create compiler invocation!", "");
    return;
  }

#if LLVM_VERSION >= 39
  clang::CompilerInvocation::setLangDefaults(
      *invocation->getLangOpts(), clang::IK_C,
      llvm::Triple(invocation->getTargetOpts().Triple),
      invocation->getPreprocessorOpts(), clang::LangStandard::lang_c11);
#else
  clang::CompilerInvocation::setLangDefaults(
      *invocation->getLangOpts(), clang::IK_C, clang::LangStandard::lang_c11);
#endif

  // make sure we free memory (by default it does not)
  invocation->getFrontendOpts().DisableFree = false;
  invocation->getCodeGenOpts().DisableFree = false;

  // create the compiler
  clang::CompilerInstance compiler;
  compiler.setInvocation(invocation.release());
  compiler.createDiagnostics();

  clang::PreprocessorOptions& po =
      compiler.getInvocation().getPreprocessorOpts();
  // replace the string-input argument with the actual source code
  po.addRemappedFile("string-input",
                     llvm::MemoryBuffer::getMemBufferCopy(source).release());

  context = boost::make_shared<llvm::LLVMContext>();
  clang::EmitLLVMOnlyAction action(context.get());
  if (!compiler.ExecuteAction(action)) {
    COGADB_ERROR("Failed to execute action!", "");
    return;
  }

  std::string errStr;
  engine = boost::shared_ptr<llvm::ExecutionEngine>(
      llvm::EngineBuilder(action.takeModule())
          .setErrorStr(&errStr)
          .setEngineKind(llvm::EngineKind::JIT)
          .setMCJITMemoryManager(std::unique_ptr<llvm::SectionMemoryManager>(
              new llvm::SectionMemoryManager()))
          .setVerifyModules(true)
          .create());

  if (!engine) {
    COGADB_ERROR("Could not create ExecutionEngine: " + errStr, "");
    return;
  }

  engine->finalizeObject();
  query_ptr = reinterpret_cast<SharedCLibPipelineQueryPtr>(
      engine->getFunctionAddress("compiled_query"));
}

const PipelinePtr CCodeGenerator::compile() {
  int ret = 0;
  bool show_generated_code =
      VariableManager::instance().getVariableValueBoolean(
          "show_generated_code");
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");
  bool keep_last_generated_query_code =
      VariableManager::instance().getVariableValueBoolean(
          "keep_last_generated_query_code");

  ScanParam& param = scanned_attributes;
  assert(input_table != NULL);

  PipelineInfoPtr pipe_info = boost::make_shared<PipelineInfo>();
  pipe_info->setSourceTable(input_table);
  pipe_info->setPipelineType(pipe_end);
  pipe_info->setGroupByAggregateParam(groupby_param);

  if (canOmitCompilation()) {
    if (debug_code_generator) {
      std::cout << "[Falcon]: Omit compilation of empty pipeline..."
                << std::endl;
    }

    return PipelinePtr(
        new DummyPipeline(input_table, scanned_attributes, pipe_info));
  }

  std::stringstream source_code;
  printCode(source_code);

  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  std::stringstream ss;
  ss << "gen_query_" << uuid << ".c";
  std::string filename = ss.str();

  std::ofstream generated_file(filename.c_str(),
                               std::ios::trunc | std::ios::out);
  generated_file << source_code.str();
  generated_file.close();

  if (keep_last_generated_query_code || show_generated_code ||
      debug_code_generator) {
    std::string format_command = std::string("astyle -q ") + filename;
    ret = system(format_command.c_str());
  }

  if (show_generated_code || debug_code_generator) {
    std::cout << "============================================================="
                 "=========="
              << std::endl;
    std::cout << "=== Generated Code: ===" << std::endl;

    /* try a syntax highlighted output first */
    /* command highlight available? */
    int ret = system("which highlight > /dev/null");
    if (ret == 0) {
      std::string highlight_command =
          std::string("highlight -O ansi ") + filename;
      ret = system(highlight_command.c_str());
    } else {
      /* command 'highlight' not available, use good old cat */
      std::string cat_command = std::string("cat ") + filename;
      ret = system(cat_command.c_str());
    }
  }

  if (keep_last_generated_query_code) {
    std::string copy_last_query_command =
        std::string("cp '") + filename +
        std::string("' last_generated_query.c");
    ret = system(copy_last_query_command.c_str());
  }

  Timestamp begin_compile = getTimestamp();
  std::string path_to_precompiled_header = "minimal_api_c.h.pch";
  std::string path_to_minimal_api_header =
      std::string(PATH_TO_COGADB_SOURCE_CODE) +
      "/lib/cogadb/include/query_compilation/minimal_api_c.h";

  bool rebuild_precompiled_header = false;

  if (!boost::filesystem::exists(path_to_precompiled_header)) {
    rebuild_precompiled_header = true;
  } else {
    std::time_t last_access_pch =
        boost::filesystem::last_write_time(path_to_precompiled_header);
    std::time_t last_access_header =
        boost::filesystem::last_write_time(path_to_minimal_api_header);

    /* pre-compiled header outdated? */
    if (last_access_header > last_access_pch) {
      std::cout << "Pre-compiled header '" << path_to_precompiled_header
                << "' is outdated!" << std::endl;
      rebuild_precompiled_header = true;
    }
  }
  if (rebuild_precompiled_header) {
    std::cout
        << "Precompiled Header not found! Building Precompiled Header now..."
        << std::endl;
    std::stringstream precompile_header;

    precompile_header
        << QUERY_COMPILATION_CC << " -std=c11 -x c-header -fno-trigraphs -fpic "
#ifdef SSE41_FOUND
                                   "-msse4.1 "
#endif
#ifdef SSE42_FOUND
                                   "-msse4.2 "
#endif
#ifdef AVX_FOUND
                                   "-mavx "
#endif
#ifdef AVX2_FOUND
                                   "-mavx2 "
#endif
        << PATH_TO_COGADB_SOURCE_CODE
        << "/lib/cogadb/include/query_compilation/minimal_api_c.h -I "
        << PATH_TO_COGADB_SOURCE_CODE
        << "/lib/cogadb/include/  -o minimal_api_c.h.pch" << std::endl;

    ret = system(precompile_header.str().c_str());
    if (ret != 0) {
      std::cout << "Compilation of precompiled header failed!" << std::endl;
      return PipelinePtr();
    } else {
      std::cout << "Compilation of precompiled header successful!" << std::endl;
    }
  }

  SharedCLibPipelineQueryPtr query_ptr;
  boost::shared_ptr<llvm::LLVMContext> context;
  boost::shared_ptr<llvm::ExecutionEngine> engine;
  compile(source_code.str(), query_ptr, engine, context);

  Timestamp end_compile = getTimestamp();

  if (debug_code_generator) {
    std::cout << "Attributes with Hash Tables: " << std::endl;
    for (size_t i = 0; i < param.size(); ++i) {
      std::cout << param[i].getVersionedTableName() << "."
                << param[i].getVersionedAttributeName() << ": "
                << param[i].hasHashTable() << std::endl;
    }
  }

  double compile_time_in_sec =
      double(end_compile - begin_compile) / (1000 * 1000 * 1000);
  return PipelinePtr(new LLVMJitPipeline(query_ptr, param, compile_time_in_sec,
                                         context, engine, pipe_info));
}

}  // end namespace CoGaDB

#pragma GCC diagnostic pop
