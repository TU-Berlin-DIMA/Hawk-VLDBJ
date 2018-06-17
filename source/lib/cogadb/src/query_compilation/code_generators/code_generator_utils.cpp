#include <query_compilation/code_generators/code_generator_utils.hpp>

#include <util/functions.hpp>
#include <util/getname.hpp>

#include <compression/dictionary_compressed_column.hpp>
#include <compression/order_preserving_dictionary_compressed_column.hpp>
#include <core/variable_manager.hpp>

#include <boost/algorithm/string.hpp>
#include <persistence/storage_manager.hpp>

namespace CoGaDB {

void convertToCodeGenerator(const std::string& code_gen_name,
                            CodeGeneratorType& code_gen_type) {
  if (code_gen_name == "cpp") {
    code_gen_type = CPP_CODE_GENERATOR;
  } else if (code_gen_name == "c") {
    code_gen_type = C_CODE_GENERATOR;
  } else if (code_gen_name == "multi_staged") {
    code_gen_type = MULTI_STAGE_CODE_GENERATOR;
  } else if (code_gen_name == "cuda") {
    code_gen_type = CUDA_C_CODE_GENERATOR;
  } else {
    COGADB_FATAL_ERROR("Unknown code generator name " << code_gen_name, "");
  }
}

const std::string toCPPType(const AttributeType& attr) {
  if (attr == VARCHAR) {
    return "string";
  } else if (attr == BOOLEAN) {
    return "bool";
  } else {
    return toCType(attr);
  }
}

const std::string toCType(const AttributeType& attr) {
  switch (attr) {
    case INT:
      return "int32_t";
    case FLOAT:
      return "float";
    case DOUBLE:
      return "double";
    case OID:
      return "uint64_t";
    case VARCHAR:
      return "char*";
    case DATE:
    case UINT32:
      return "uint32_t";
    case CHAR:
    case BOOLEAN:
      return "char";
    default:
      COGADB_FATAL_ERROR("", "");
      return "<INVALID_TYPE>";
  }
}

const std::string getResultType(const AttributeReference& attr,
                                bool ignore_compressed) {
  if (isAttributeDictionaryCompressed(attr) && !ignore_compressed) {
    return "uint32_t";
  } else {
    return toCType(attr.getAttributeType());
  }
}

const std::string toType(const AttributeType& attr) {
  if (targetingCLang()) {
    return toCType(attr);
  } else {
    return toCPPType(attr);
  }
}

const std::string getCTypeFunctionPostFix(const AttributeType& attr) {
  if (attr == VARCHAR) {
    return "cstring";
  } else if (attr == BOOLEAN) {
    return "bool";
  } else {
    return toCType(attr);
  }
}

const std::string getConstant(const boost::any& constant) {
  std::stringstream ss;
  assert(!constant.empty());
  if (constant.type() == typeid(int32_t)) {
    ss << boost::any_cast<int32_t>(constant);
  } else if (constant.type() == typeid(uint32_t)) {
    ss << boost::any_cast<uint32_t>(constant);
  } else if (constant.type() == typeid(int64_t)) {
    ss << boost::any_cast<int64_t>(constant);
  } else if (constant.type() == typeid(uint64_t)) {
    ss << boost::any_cast<uint64_t>(constant);
  } else if (constant.type() == typeid(int16_t)) {
    ss << boost::any_cast<int16_t>(constant);
  } else if (constant.type() == typeid(uint16_t)) {
    ss << boost::any_cast<uint16_t>(constant);
  } else if (constant.type() == typeid(float)) {
    ss << boost::any_cast<float>(constant) << "f";
  } else if (constant.type() == typeid(double)) {
    ss << boost::any_cast<double>(constant);
  } else if (constant.type() == typeid(char)) {
    ss << boost::any_cast<char>(constant);
  } else if (constant.type() == typeid(std::string)) {
    ss << "\"" << boost::any_cast<std::string>(constant) << "\"";
  } else {
    COGADB_FATAL_ERROR("Unknown constant type: " << constant.type().name(), "");
  }
  return ss.str();
}

const std::string getConstant(const std::string& constant) {
  return "\"" + constant + "\"";
}

const std::string getExpression(LogicalOperation log_op) {
  const char* const names[] = {"&&", "||"};
  return std::string(names[log_op]);
}

const std::string getSSEExpression(LogicalOperation log_op) {
  const char* const names[] = {"_mm_and_si128", "_mm_or_si128"};
  return std::string(names[log_op]);
}

const std::string getExpression(ValueComparator x) {
  const char* const names[] = {"<", ">", "==", "<=", ">=", "!="};
  return std::string(names[x]);
}
// only works for int
const std::string getSSEExpression(ValueComparator x) {
  const char* const names[] = {"_mm_cmplt_epi32", "_mm_cmpgt_epi32",
                               "_mm_cmpeq_epi32", "_mm_cmplt_epi32",
                               "_mm_cmpgt_epi32", "_mm_cmpeq_epi32"};
  return std::string(names[x]);
}
// only works for float
const std::string getSSEExpressionFloat(ValueComparator x) {
  const char* const names[] = {"_mm_cmplt_ps", "_mm_cmpgt_ps", "_mm_cmpeq_ps",
                               "_mm_cmple_ps", "_mm_cmpge_ps", "_mm_cmpneq_ps"};
  return std::string(names[x]);
}
// only works for float
const std::string getSSEExpressionDouble(ValueComparator x) {
  const char* const names[] = {"_mm_cmplt_pd", "_mm_cmpgt_pd", "_mm_cmpeq_pd",
                               "_mm_cmple_pd", "_mm_cmpge_pd", "_mm_cmpneq_pd"};
  return std::string(names[x]);
}

const std::string getTupleIDVarName(const AttributeReference& attr_ref) {
  std::stringstream ss;
  if (isPersistent(attr_ref.getTable())) {
    ss << "tuple_id_" << attr_ref.getVersionedTableName();
  } else {
    ss << "tuple_id_" << attr_ref.getUnversionedTableName();
  }
  return ss.str();
}

const std::string getTupleIDVarName(const TablePtr table, uint32_t version) {
  assert(table != NULL);
  std::stringstream ss;
  if (isPersistent(table)) {
    ss << "tuple_id_" << table->getName() << version;
  } else {
    ss << "tuple_id_" << table->getName();
  }
  return ss.str();
}

const std::string getInputArrayVarName(const AttributeReference& attr_ref) {
  std::stringstream ss;
  if (attr_ref.getAttributeReferenceType() == COMPUTED_ATTRIBUTE) {
    ss << "array_" << attr_ref.getVersionedAttributeName();
  } else if (attr_ref.getAttributeReferenceType() == INPUT_ATTRIBUTE) {
    ss << "array_" << attr_ref.getVersionedTableName() << "_"
       << getVariableFromAttributeName(attr_ref.getVersionedAttributeName());
  }
  return ss.str();
}

const std::string getInputColumnVarName(const AttributeReference& attr_ref) {
  std::stringstream ss;
  ss << "col_" << attr_ref.getVersionedTableName() << "_"
     << getVariableFromAttributeName(attr_ref.getVersionedAttributeName());
  return ss.str();
}

const std::string getTableVarName(const AttributeReference& attr_ref) {
  std::stringstream ss;
  if (isPersistent(attr_ref.getTable())) {
    ss << "table_" << attr_ref.getVersionedTableName();
  } else {
    ss << "table_" << attr_ref.getUnversionedTableName();
  }
  return ss.str();
}

const std::string getTableVarName(const TablePtr table, uint32_t version) {
  assert(table != NULL);
  std::stringstream ss;
  if (isPersistent(table)) {
    ss << "table_" << table->getName() << version;
  } else {
    ss << "table_" << table->getName();
  }
  return ss.str();
}

const std::string getResultArrayVarName(const AttributeReference& attr_ref) {
  std::string result_name = attr_ref.getResultAttributeName();
  boost::replace_all(result_name, ".", "_");
  std::stringstream ss;
  ss << "result_array_" << result_name;
  return ss.str();
}

const std::string getHashTableVarName(const AttributeReference& attr_ref) {
  std::stringstream ss;
  ss << "ht_" << attr_ref.getVersionedTableName() << "_"
     << getVariableFromAttributeName(attr_ref.getVersionedAttributeName());
  return ss.str();
}

const std::string getComputedColumnVarName(const AttributeReference& attr_ref) {
  assert(attr_ref.getAttributeReferenceType() == COMPUTED_ATTRIBUTE);
  std::stringstream ss;
  ss << "computed_var_"
     << getVariableFromAttributeName(attr_ref.getVersionedAttributeName());
  return ss.str();
}

const std::string getVarName(const AttributeReference& attr_ref,
                             const bool ignore_compressed) {
  if (attr_ref.getAttributeReferenceType() == INPUT_ATTRIBUTE) {
    std::string var_name = getInputArrayVarName(attr_ref);

    if (isAttributeDictionaryCompressed(attr_ref) && !ignore_compressed) {
      var_name += "_dict_ids";
    }

    return var_name;
  } else if (attr_ref.getAttributeReferenceType() == COMPUTED_ATTRIBUTE) {
    return getComputedColumnVarName(attr_ref);
  } else {
    COGADB_FATAL_ERROR("", "");
  }
}

const std::string getSSEVarName(const uint32_t const_num) {
  std::stringstream ss;
  ss << "SSE_Const_Pred_" << const_num;
  return ss.str();
}

const std::string getElementAccessExpression(const AttributeReference& attr_ref,
                                             std::string tuple_id) {
  if (tuple_id.empty()) {
    tuple_id = getTupleIDVarName(attr_ref);
  }

  if (attr_ref.getAttributeReferenceType() == INPUT_ATTRIBUTE) {
    std::stringstream ss;
    ss << getInputArrayVarName(attr_ref) << "[" << tuple_id << "]";
    return ss.str();
  } else if (attr_ref.getAttributeReferenceType() == COMPUTED_ATTRIBUTE) {
    return getComputedColumnVarName(attr_ref);
  } else {
    COGADB_FATAL_ERROR("", "");
  }
}

const std::string getElementAccessExpressionSIMD(
    const AttributeReference& attr_ref, std::string vector_width,
    std::string tuple_id) {
  if (tuple_id.empty()) {
    tuple_id = getTupleIDVarName(attr_ref);
  }

  if (attr_ref.getAttributeReferenceType() == INPUT_ATTRIBUTE) {
    std::stringstream ss;
    ss << "(" << vector_width << getInputArrayVarName(attr_ref) << ")["
       << tuple_id << "]";
    return ss.str();
  } else if (attr_ref.getAttributeReferenceType() == COMPUTED_ATTRIBUTE) {
    return getComputedColumnVarName(attr_ref);
  } else {
    COGADB_FATAL_ERROR("", "");
  }
}

const std::string getCompressedElementAccessExpression(
    const AttributeReference& attr_ref, std::string tuple_id) {
  /* non input attributes are never compressed */
  if (!isInputAttribute(attr_ref)) {
    return getElementAccessExpression(attr_ref, tuple_id);
  }

  if (tuple_id.empty()) {
    tuple_id = getTupleIDVarName(attr_ref);
  }

  /* check which compression the input attribute has */
  ColumnType col_type = getColumnType(attr_ref);
  if (col_type == PLAIN_MATERIALIZED) {
    return getElementAccessExpression(attr_ref, tuple_id);
  } else if (col_type == DICTIONARY_COMPRESSED) {
    std::stringstream ss;
    ss << getInputArrayVarName(attr_ref) << "_dict_ids"
       << "[" << tuple_id << "]";
    return ss.str();
  } else if (col_type == DICTIONARY_COMPRESSED_ORDER_PRESERVING) {
    std::stringstream ss;
    ss << getInputArrayVarName(attr_ref) << "_dict_ids"
       << "[" << tuple_id << "]";
    return ss.str();
  } else {
    COGADB_FATAL_ERROR(
        "Encountered compression scheme not supported by query compiler: "
            << util::getName(col_type),
        "");
  }
}

const std::string getCompressedElementAccessExpressionSIMD(
    const AttributeReference& attr_ref, std::string vector_width,
    std::string tuple_id) {
  /* non input attributes are never compressed */
  if (!isInputAttribute(attr_ref)) {
    return getElementAccessExpressionSIMD(attr_ref, vector_width, tuple_id);
  }

  if (tuple_id.empty()) {
    tuple_id = getTupleIDVarName(attr_ref);
  }

  /* check which compression the input attribute has */
  ColumnType col_type = getColumnType(attr_ref);
  if (col_type == PLAIN_MATERIALIZED) {
    return getElementAccessExpressionSIMD(attr_ref, vector_width, tuple_id);
  } else if (col_type == DICTIONARY_COMPRESSED) {
    std::stringstream ss;
    ss << "(" << vector_width << getInputArrayVarName(attr_ref) << "_dict_ids"
       << ")[" << tuple_id << "]";
    return ss.str();
  } else if (col_type == DICTIONARY_COMPRESSED_ORDER_PRESERVING) {
    std::stringstream ss;
    ss << "(" << vector_width << getInputArrayVarName(attr_ref) << "_dict_ids"
       << ")[" << tuple_id << "]";
    return ss.str();
  } else {
    COGADB_FATAL_ERROR(
        "Encountered compression scheme not supported by query compiler: "
            << util::getName(col_type),
        "");
  }
}

const std::string getGroupTIDVarName(const AttributeReference& attr_ref) {
  std::stringstream ss;
  ss << "group_tid_"
     << getVariableFromAttributeName(attr_ref.getVersionedAttributeName());
  return ss.str();
}

const AggregationParam getAggregationParam(const GroupbyParam& groupby_param,
                                           const AttributeReference& attr_ref) {
  AggregationFunctions::const_iterator cit;
  for (cit = groupby_param.aggregation_functions.begin();
       cit != groupby_param.aggregation_functions.end(); ++cit) {
    if (cit->first == attr_ref.getVersionedAttributeName()) {
      return cit->second;
    }
  }

  COGADB_FATAL_ERROR("Not found attribute "
                         << attr_ref.getVersionedAttributeName()
                         << " in aggregation functions of groupby parameter!",
                     "");

  return AggregationParam(ProcessorSpecification(hype::PD0), COUNT,
                          HASH_BASED_AGGREGATION, "");
}

const std::string getAggregationPayloadFieldVarName(
    const AttributeReference& attr_ref, const AggregationFunction& agg_func) {
  std::stringstream ss;
  ss << util::getName(agg_func) << "_OF_"
     << getVariableFromAttributeName(attr_ref.getVersionedAttributeName());
  return ss.str();
}

const std::string getAggregationPayloadFieldVarName(
    const AttributeReference& attr_ref, const AggregationParam& param) {
  return getAggregationPayloadFieldVarName(attr_ref, param.agg_func);
}

const std::string getAggregationPayloadFieldCode(
    const AttributeReference& attr_ref, AggregationFunction agg_func) {
  std::stringstream field_code;

  if (agg_func == MIN || agg_func == MAX) {
    field_code << toCType(attr_ref.getAttributeType()) << " "
               << getAggregationPayloadFieldVarName(attr_ref, agg_func) << ";"
               << std::endl;
  } else if (agg_func == COUNT) {
    field_code << getAggregationResultCType(attr_ref, agg_func) << " "
               << getAggregationPayloadFieldVarName(attr_ref, agg_func) << ";"
               << std::endl;
  } else if (agg_func == SUM) {
    field_code << getAggregationResultCType(attr_ref, agg_func) << " "
               << getAggregationPayloadFieldVarName(attr_ref, agg_func) << ";"
               << std::endl;
  } else if (agg_func == AVERAGE) {
    field_code << getAggregationResultCType(attr_ref, COUNT) << " "
               << getAggregationPayloadFieldVarName(attr_ref, COUNT) << ";"
               << std::endl;
    field_code << getAggregationResultCType(attr_ref, AVERAGE) << " "
               << getAggregationPayloadFieldVarName(attr_ref, AVERAGE) << ";"
               << std::endl;
  } else if (agg_func == UDF_AGGREGATION) {
    field_code << toCType(attr_ref.getAttributeType()) << " "
               << getAggregationPayloadFieldVarName(attr_ref, UDF_AGGREGATION)
               << ";" << std::endl;
  } else {
    COGADB_FATAL_ERROR(
        "Unsupported Aggregation Function: " << util::getName(agg_func), "");
  }

  return field_code.str();
}

static thread_local bool intel_hack = false;

void enableIntelAggregationHACK(bool enable) { intel_hack = enable; }

const std::string getAggregationResultCType(const AttributeReference& attr_ref,
                                            AggregationFunction agg_func) {
  std::string type;

  if (agg_func == MIN || agg_func == MAX || agg_func == UDF_AGGREGATION) {
    type = toCType(attr_ref.getAttributeType());
  } else if (agg_func == COUNT) {
    if (intel_hack) {
      type = "uint32_t";
    } else {
      type = "uint64_t";
    }
  } else if (agg_func == SUM || agg_func == AVERAGE) {
    if (intel_hack) {
      type = "float";
    } else {
      type = "double";
    }
  } else {
    COGADB_FATAL_ERROR(
        "Unsupported Aggregation Function: " << util::getName(agg_func), "");
  }

  return type;
}

const std::string getAggregationGroupTIDPayloadFieldCode(
    const AttributeReference& attr_ref) {
  std::stringstream field_code;
  field_code << "TID " << getGroupTIDVarName(attr_ref) << ";" << std::endl;
  return field_code.str();
}

const std::string getComputeGroupIDExpression(
    const GroupingAttributes& grouping_attrs) {
  if (grouping_attrs.empty()) {
    COGADB_FATAL_ERROR(
        "Invalid argument: at least one grouping attribute must exist to "
        "compute the group id!",
        "");
  }

  std::vector<uint64_t> bits_per_column(grouping_attrs.size());
  for (size_t i = 0; i < grouping_attrs.size(); ++i) {
    bits_per_column[i] = getNumberOfRequiredBits(grouping_attrs[i]);
  }

  uint64_t total_number_of_bits =
      std::accumulate(bits_per_column.begin(), bits_per_column.end(), 0u);

  uint32_t max_bits = sizeof(uint64_t) * 8;

  if (intel_hack) {
    max_bits = sizeof(uint32_t) * 8;
  }

  // can we apply our bit packing?
  auto ignore_max_bits = VariableManager::instance().getVariableValueBoolean(
      "code_gen.opt.hack.ignore_bitpacking_max_bits");
  if (total_number_of_bits > max_bits && !ignore_max_bits) {
    COGADB_ERROR(
        "Maximum Number of Bits for optimized groupby exceeded: max value: "
            << max_bits << " Got: " << total_number_of_bits,
        "");

    return "";
  }

  /* special case: if we have only one group by attribute, use value as
   * group key */
  if (grouping_attrs.size() == 1) {
    return getCompressedElementAccessExpression(grouping_attrs.front());
  }

  // we get the number of bits to shift for each column by computing
  // the prefix sum of each columns number of bits
  std::vector<uint64_t> bits_to_shift(grouping_attrs.size() + 1);
  serial_prefixsum(bits_per_column.data(), grouping_attrs.size(),
                   bits_to_shift.data());

  std::stringstream expr;
  for (size_t i = 0; i < grouping_attrs.size(); ++i) {
    expr << "(((TID)" << getCompressedElementAccessExpression(grouping_attrs[i])
         << ") << " << bits_to_shift[i] << ")";

    if (i + 1 < grouping_attrs.size()) {
      expr << " | ";
    }
  }

  return expr.str();
}

bool isBitpackedGroupbyOptimizationApplicable(
    const GroupingAttributes& grouping_attrs) {
  if (grouping_attrs.empty()) {
    return false;
  }

  bool found_floating_point_column = false;
  std::vector<uint64_t> bits_per_column(grouping_attrs.size());
  for (size_t i = 0; i < grouping_attrs.size(); ++i) {
    bits_per_column[i] = getNumberOfRequiredBits(grouping_attrs[i]);
    /* work around bug that computed attribute returns double, even when it
     * isn't! */
    // if(!isComputed(grouping_attrs[i]))
    {
      if (grouping_attrs[i].getAttributeType() == FLOAT ||
          grouping_attrs[i].getAttributeType() == DOUBLE) {
        found_floating_point_column = true;
      }
    }
  }

  /* we can bitpack only if we have integer types, floating point values
     are trouble to say the least */
  if (found_floating_point_column) {
    return false;
  }

  uint64_t total_number_of_bits =
      std::accumulate(bits_per_column.begin(), bits_per_column.end(), 0u);
  if (total_number_of_bits > sizeof(uint64_t) * 8) {
    return false;
  } else {
    return true;
  }
}

const std::string getAggregationCodeGeneric(
    const GroupingAttributes& grouping_columns,
    const AggregateSpecifications& aggregation_specs,
    const std::string access_ht_entry_expression) {
  std::stringstream hash_aggregate;
  for (size_t i = 0; i < aggregation_specs.size(); ++i) {
    hash_aggregate << aggregation_specs[i]->getCodeHashGroupBy(
        access_ht_entry_expression);
    for (size_t i = 0; i < grouping_columns.size(); ++i) {
      AttributeReference attr_ref = grouping_columns[i];
      if (!isComputed(attr_ref)) {
        hash_aggregate << access_ht_entry_expression
                       << getGroupTIDVarName(attr_ref) << " = "
                       << getTupleIDVarName(attr_ref) << ";" << std::endl;
      } else {
        hash_aggregate << access_ht_entry_expression << getVarName(attr_ref)
                       << " = " << getVarName(attr_ref) << ";" << std::endl;
      }
    }
  }
  return hash_aggregate.str();
}

const std::string getAggregationPayloadCodeForGroupingAttributes(
    const GroupingAttributes& grouping_attrs) {
  std::stringstream ss;
  for (const auto& attr_ref : grouping_attrs) {
    if (!isComputed(attr_ref)) {
      ss << getAggregationGroupTIDPayloadFieldCode(attr_ref) << std::endl;
    } else {
      ss << toCType(attr_ref.getAttributeType()) << " " << getVarName(attr_ref)
         << ";" << std::endl;
    }
  }
  return ss.str();
}

const std::string getCodeProjectGroupingColumnsFromHashTable(
    const GroupingAttributes& grouping_attrs,
    const std::string& hash_map_access) {
  std::stringstream ss;
  for (const auto& grouping_attr : grouping_attrs) {
    ss << "    " << getResultArrayVarName(grouping_attr)
       << "[current_result_size] = ";
    if (!isComputed(grouping_attr)) {
      ss << getCompressedElementAccessExpression(
          grouping_attr, hash_map_access + getGroupTIDVarName(grouping_attr));
    } else {
      ss << hash_map_access << getVarName(grouping_attr);
    }
    ss << ";" << std::endl;
  }
  return ss.str();
}

const std::string getCodeDeclareResultMemory(const AttributeReference& ref,
                                             bool uses_c_string) {
  std::stringstream ss;
  AttributeType attr = ref.getAttributeType();

  if (isAttributeDictionaryCompressed(ref)) {
    ss << "uint32_t* " << getResultArrayVarName(ref) << " = NULL;" << std::endl;
  } else if (attr == VARCHAR) {
    if (uses_c_string) {
      ss << "const char** ";
      ss << getResultArrayVarName(ref);
      ss << " = NULL;" << std::endl;
    } else {
      ss << toCPPType(attr) << "* ";
      ss << getResultArrayVarName(ref);
      ss << " = NULL;" << std::endl;
    }
  } else {
    ss << toCPPType(attr) << "* ";
    ss << getResultArrayVarName(ref);
    ss << " = NULL;" << std::endl;
  }

  ss << "uint64_t " << getResultArrayVarName(ref) << "_length = 0;"
     << std::endl;

  return ss.str();
}

const std::string getCodeDeclareResultMemory(const ProjectionParam& param,
                                             bool uses_c_string) {
  std::stringstream ss;

  for (const auto& p : param) {
    ss << getCodeDeclareResultMemory(p, uses_c_string);
  }

  return ss.str();
}

const std::string getCodeMalloc(const std::string& variable,
                                const std::string& type,
                                const std::string& count, const bool realloc) {
  std::stringstream ss;
  ss << variable << " = (" << type << "*)realloc(";

  if (realloc) {
    ss << variable;
  } else {
    ss << "NULL";
  }

  ss << ", " << count << " * sizeof(" << type << "));" << std::endl;
  ss << variable << "_length = allocated_result_elements;" << std::endl;

  return ss.str();
}

const std::string getCodeMallocResultMemory(const AttributeReference& ref,
                                            bool uses_c_string) {
  std::stringstream ss;
  AttributeType attr = ref.getAttributeType();

  if (isAttributeDictionaryCompressed(ref)) {
    return getCodeMalloc(getResultArrayVarName(ref), "uint32_t",
                         "allocated_result_elements", false);
  } else if (attr == VARCHAR) {
    if (uses_c_string) {
      ss << getResultArrayVarName(ref);
      ss << " = (const char**) realloc(NULL, allocated_result_elements * "
            "sizeof(char**));"
         << std::endl;
    } else {
      ss << getResultArrayVarName(ref);
      ss << " = stringMalloc(allocated_result_elements);" << std::endl;
    }
    ss << getResultArrayVarName(ref) << "_length = allocated_result_elements;"
       << std::endl;
  } else {
    return getCodeMalloc(getResultArrayVarName(ref), toCType(attr),
                         "allocated_result_elements", false);
  }

  return ss.str();
}

const std::string getCodeMallocResultMemory(const ProjectionParam& param,
                                            bool uses_c_string) {
  std::stringstream ss;

  for (const auto& p : param) {
    ss << getCodeMallocResultMemory(p, uses_c_string);
  }

  return ss.str();
}

const std::string getCodeReallocResultMemory(const ProjectionParam& param,
                                             bool uses_c_string) {
  std::stringstream ss;
  for (size_t i = 0; i < param.size(); ++i) {
    AttributeType attr = param[i].getAttributeType();

    if (isAttributeDictionaryCompressed(param[i])) {
      ss << getCodeMalloc(getResultArrayVarName(param[i]), "uint32_t",
                          "allocated_result_elements", true);
    } else if (attr == VARCHAR) {
      if (uses_c_string) {
        ss << getResultArrayVarName(param[i]) << " = (const char**) realloc("
           << getResultArrayVarName(param[i])
           << ", allocated_result_elements * sizeof(const char**));"
           << std::endl;
      } else {
        ss << getResultArrayVarName(param[i]);
        ss << " = stringRealloc(" << getResultArrayVarName(param[i])
           << ",  allocated_result_elements);" << std::endl;
      }
    } else {
      ss << getCodeMalloc(getResultArrayVarName(param[i]), toCType(attr),
                          "allocated_result_elements", true);
    }
  }

  return ss.str();
}

const std::string getVariableFromAttributeName(
    const std::string qualified_attribute_name) {
  std::string result = qualified_attribute_name;
  boost::replace_all(result, ".", "_");
  return result;
}

bool targetingCLang() {
  return VariableManager::instance().getVariableValueString(
             "default_code_generator") != "cpp";
}

std::string getMinFunction() { return "C_MIN"; }

std::string getMaxFunction() { return "C_MAX"; }

const std::string generateCCodeWriteResult(ProjectionParam param) {
  std::stringstream ss;
  ss << "if (current_result_size >= allocated_result_elements) {" << std::endl;
  ss << "   allocated_result_elements *= 1.4;" << std::endl;
  ss << getCodeReallocResultMemory(param);
  ss << "}" << std::endl;

  for (size_t i = 0; i < param.size(); ++i) {
    ss << getResultArrayVarName(param[i]) << "[current_result_size] = "
       << getCompressedElementAccessExpression(param[i]) << ";" << std::endl;
  }

  ss << "++current_result_size;" << std::endl;
  return ss.str();
}

const std::string generateCCodeWriteResultFromHashTable(ProjectionParam param) {
  std::stringstream write_result;
  /* write result */

  write_result
      << "if (aggregation_hash_table.size() >= allocated_result_elements) {"
      << std::endl;
  write_result
      << "    allocated_result_elements = aggregation_hash_table.size();"
      << std::endl;
  write_result << "    " << getCodeReallocResultMemory(param);
  write_result << "}" << std::endl;

  write_result << "current_result_size += aggregation_hash_table.size();"
               << std::endl;
  return write_result.str();
}

const std::string generateCCodeCreateResultTable(
    const ProjectionParam& param, const std::string& createResulttableCodeBlock,
    const std::string& cleanupCode, const std::string& result_table_name) {
  std::stringstream ss;
  ss << "C_Column* result_columns[] = { ";

  for (size_t i = 0; i < param.size(); ++i) {
    AttributeType attr = param[i].getAttributeType();

    if (i > 0) {
      ss << ", ";
    }

    if (isAttributeDictionaryCompressed(param[i])) {
      ss << "createResultArray_" << getCTypeFunctionPostFix(attr)
         << "_compressed(" << getInputColumnVarName(param[i]) << ", "
         << getConstant(param[i].getResultAttributeName()) << ", "
         << getResultArrayVarName(param[i]) << ", current_result_size)";
    } else {
      ss << "createResultArray_" << getCTypeFunctionPostFix(attr) << "("
         << getConstant(param[i].getResultAttributeName()) << ", "
         << getResultArrayVarName(param[i]) << ", current_result_size)";
    }
  }
  ss << " };" << std::endl;

  ss << "C_Table* result_table = createTableFromColumns("
     << getConstant(result_table_name) << ", result_columns, " << param.size()
     << ");" << std::endl;

  ss << createResulttableCodeBlock << std::endl;
  ss << cleanupCode << std::endl;
  ss << "return result_table;" << std::endl;

  return ss.str();
}

const std::string generateCCodeAllocateResultTable(
    const ProjectionParam& param) {
  std::stringstream ss;
  ss << getCodeMallocResultMemory(param);
  return ss.str();
}

bool isAttributeDictionaryCompressed(const AttributeReference& attr) {
  ColumnType col_type = getColumnType(attr);

  return col_type == DICTIONARY_COMPRESSED ||
         col_type == DICTIONARY_COMPRESSED_ORDER_PRESERVING;
}

void compareStringWithstrcmp(std::stringstream& expr, const std::string& left,
                             const std::string& right, ValueComparator comp) {
  expr << "strcmp(" << left << ", " << right << ")" << getExpression(comp)
       << "0";
}

void compareStringWithstrcmp(std::stringstream& expr,
                             const AttributeReference& left,
                             const AttributeReference& right,
                             ValueComparator comp) {
  compareStringWithstrcmp(expr, getElementAccessExpression(left),
                          getElementAccessExpression(right), comp);
}

void compareStringWithstrcmp(std::stringstream& expr,
                             const AttributeReference& left,
                             const std::string& right, ValueComparator comp) {
  compareStringWithstrcmp(expr, getElementAccessExpression(left),
                          getConstant(right), comp);
}

void compareStringWithId(std::stringstream& expr, const std::string& left,
                         const std::string& right, ValueComparator comp) {
  expr << left << " " << getExpression(comp) << " " << right;
}

void compareStringWithId(std::stringstream& expr,
                         const AttributeReference& left,
                         const AttributeReference& right,
                         ValueComparator comp) {
  compareStringWithId(expr, getCompressedElementAccessExpression(left),
                      getCompressedElementAccessExpression(right), comp);
}

void compareStringWithId(std::stringstream& expr,
                         const AttributeReference& left,
                         const std::string& right, ValueComparator comp) {
  compareStringWithId(expr, getCompressedElementAccessExpression(left), right,
                      comp);
}

std::string getStringCompareExpression(const AttributeReference& left,
                                       const AttributeReference& right,
                                       ValueComparator comp) {
  std::stringstream expr;
  // if we have dictionary compression we can only exploit the surrogates for
  // equal or unequal compares, but if we got
  // ordered dictionary compression, we can do all compares
  if ((getColumnType(left) == DICTIONARY_COMPRESSED &&
       getColumnType(right) == DICTIONARY_COMPRESSED &&
       (comp == EQUAL || comp == UNEQUAL)) ||
      (getColumnType(left) == DICTIONARY_COMPRESSED_ORDER_PRESERVING &&
       getColumnType(right) == DICTIONARY_COMPRESSED_ORDER_PRESERVING)) {
    compareStringWithId(expr, left, right, comp);
  } else {
    compareStringWithstrcmp(expr, left, right, comp);
  }

  return expr.str();
}

std::string getStringCompareExpression(const AttributeReference& attr,
                                       const std::string& constant,
                                       ValueComparator comp) {
  std::stringstream expr;
  // if we have dictionary compression we can only exploit the surrogates for
  // equal or unequal compares, but if we got
  // ordered dictionary compression, we can do all compares
  if ((getColumnType(attr) == DICTIONARY_COMPRESSED &&
       (comp == EQUAL || comp == UNEQUAL)) ||
      getColumnType(attr) == DICTIONARY_COMPRESSED_ORDER_PRESERVING) {
    ValueComparator rewritten_comp;
    uint32_t filter_id;
    if (getDictionaryIDForPredicate(attr.getColumn(), comp, constant, filter_id,
                                    rewritten_comp)) {
      compareStringWithId(expr, attr, getConstant(filter_id), rewritten_comp);
    } else {
      COGADB_FATAL_ERROR("Failed to retrieve compressed key of value '"
                             << constant << "' from dictionary "
                                            "for attribute: "
                             << CoGaDB::toString(attr),
                         "");
    }
  } else {
    compareStringWithstrcmp(expr, attr, constant, comp);
  }

  return expr.str();
}

bool getDictionaryIDForPredicate(const ColumnPtr col,
                                 const ValueComparator comp,
                                 const std::string& comparison_val,
                                 uint32_t& result_id,
                                 ValueComparator& rewritten_value_comparator) {
  typedef DictionaryCompressedColumn<std::string> StringDictCompressed;
  typedef OrderPreservingDictionaryCompressedColumn<std::string>
      StringOrderDictCompressed;

  boost::shared_ptr<StringDictCompressed> dict_compressed_col;
  boost::shared_ptr<StringOrderDictCompressed> ordered_dict_compressed_col;

  if ((dict_compressed_col =
           boost::dynamic_pointer_cast<StringDictCompressed>(col)) != nullptr) {
    std::pair<bool, uint32_t> ret =
        dict_compressed_col->getDictionaryID(comparison_val);
    if (ret.first) {
      result_id = ret.second;
      rewritten_value_comparator = comp;
      return true;
    } else {
      return false;
    }
  } else if ((ordered_dict_compressed_col =
                  boost::dynamic_pointer_cast<StringOrderDictCompressed>(
                      col)) != nullptr) {
    std::pair<bool, uint32_t> ret =
        ordered_dict_compressed_col->getClosestDictionaryIDForPredicate(
            comparison_val, comp, rewritten_value_comparator);
    if (ret.first) {
      result_id = ret.second;
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

std::string getArrayFromColumnCode(const AttributeReference& attr,
                                   bool ignore_compressed) {
  std::stringstream expr;

  bool is_compressed = isAttributeDictionaryCompressed(attr);

  AttributeType type = attr.getAttributeType();
  std::string pointer = "*";

  if (type == VARCHAR && (ignore_compressed || !is_compressed)) {
    pointer = " const *";
  }

  expr << "const " << getResultType(attr, ignore_compressed) << pointer << " "
       << getVarName(attr, ignore_compressed);

  if (is_compressed && !ignore_compressed) {
    expr << " = getArrayCompressedKeysFromColumn_string";
  } else {
    expr << " = getArrayFromColumn_" << getCTypeFunctionPostFix(type);
  }

  expr << "(" << getInputColumnVarName(attr) << ");" << std::endl;
  expr << "uint64_t " << getVarName(attr, ignore_compressed)
       << "_length = " << getInputColumnVarName(attr) << "_length;"
       << std::endl;

  return expr.str();
}

}  // namespace CoGaDB
