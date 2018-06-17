
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <boost/algorithm/string.hpp>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>

#include <query_processing/definitions.hpp>
#include <query_processing/extension/create_table_operator.hpp>
#include <query_processing/extension/delete_table_operator.hpp>
#include <query_processing/extension/export_into_file.hpp>
#include <query_processing/extension/generic_groupby_operator.hpp>
#include <query_processing/extension/generic_selection_operator.hpp>
#include <query_processing/extension/limit_operator.hpp>
#include <query_processing/extension/map_udf_operator.hpp>
#include <query_processing/extension/rename_table_operator.hpp>
#include <query_processing/extension/sink_operator.hpp>
#include <query_processing/extension/store_table_operator.hpp>
#include <query_processing/query_processor.hpp>

#include <query_compilation/code_generator.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/predicate_specification.hpp>
#include <query_compilation/query_context.hpp>

#include <persistence/storage_manager.hpp>

#include <util/code_generation.hpp>
#include <util/query_processing.hpp>
#include <util/tests.hpp>
#include <util/types.hpp>

#include <parser/json_parser.hpp>

using namespace std;
using namespace rapidjson;

namespace CoGaDB {

typedef hype::queryprocessing::NodePtr NodePtr;

bool convertToAttributeType(const std::string& attr_type_str_,
                            AttributeType& type);
bool convertToValueComparator(const std::string& pred_comp_str,
                              ValueComparator& pred_comp);
bool convertToInternalConstant(const std::string val_str,
                               const AttributeType& type, boost::any& val);
bool convertToJoinType(const std::string& join_type_str, JoinType& join_type);

AttributeReferencePtr parseAttributeReference(
    Document::ConstMemberIterator member);
bool parseAttributeReferences(Document::ConstMemberIterator member,
                              std::vector<AttributeReference>& attrs);

bool convertToAttributeType(const std::string& attr_type_str_,
                            AttributeType& type) {
  std::string attr_type_str = boost::to_upper_copy(attr_type_str_);
  if (attr_type_str == util::getName(INT)) {
    type = INT;
  } else if (attr_type_str == util::getName(FLOAT)) {
    type = FLOAT;
  } else if (attr_type_str == util::getName(VARCHAR)) {
    type = VARCHAR;
  } else if (attr_type_str == util::getName(BOOLEAN)) {
    type = BOOLEAN;
  } else if (attr_type_str == util::getName(UINT32)) {
    type = UINT32;
  } else if (attr_type_str == util::getName(OID)) {
    type = OID;
  } else if (attr_type_str == util::getName(DOUBLE)) {
    type = DOUBLE;
  } else if (attr_type_str == util::getName(CHAR)) {
    type = CHAR;
  } else if (attr_type_str == util::getName(DATE)) {
    type = DATE;
  } else {
    return false;
  }
  return true;
}

bool convertToValueComparator(const std::string& pred_comp_str,
                              ValueComparator& pred_comp) {
  if (pred_comp_str == "LESS_THAN" || pred_comp_str == "<") {
    pred_comp = LESSER;
  } else if (pred_comp_str == "LESS_EQUAL" || pred_comp_str == "<=") {
    pred_comp = LESSER_EQUAL;
  } else if (pred_comp_str == "GREATER_THAN" || pred_comp_str == ">") {
    pred_comp = GREATER;
  } else if (pred_comp_str == "GREATER_EQUAL" || pred_comp_str == ">=") {
    pred_comp = GREATER_EQUAL;
  } else if (pred_comp_str == "EQUAL" || pred_comp_str == "=") {
    pred_comp = EQUAL;
  } else if (pred_comp_str == "UNEQUAL" || pred_comp_str == "<>") {
    pred_comp = UNEQUAL;
  } else {
    return false;
  }
  return true;
}

bool convertToInternalConstant(const std::string val_str,
                               const AttributeType& type, boost::any& val) {
  try {
    if (type == INT) {
      int32_t v = boost::lexical_cast<int32_t>(val_str);
      val = boost::any(v);
    } else if (type == FLOAT) {
      float v = boost::lexical_cast<float>(val_str);
      val = boost::any(v);
    } else if (type == VARCHAR) {
      std::string v = boost::lexical_cast<std::string>(val_str);
      val = boost::any(v);
    } else if (type == BOOLEAN) {
      bool v = boost::lexical_cast<bool>(val_str);
      val = boost::any(v);
    } else if (type == UINT32) {
      uint32_t v = boost::lexical_cast<uint32_t>(val_str);
      val = boost::any(v);
    } else if (type == OID) {
      TID v = boost::lexical_cast<TID>(val_str);
      val = boost::any(v);
    } else if (type == DOUBLE) {
      double v = boost::lexical_cast<double>(val_str);
      val = boost::any(v);
    } else if (type == CHAR) {
      char v = boost::lexical_cast<char>(val_str);
      val = boost::any(v);
    } else if (type == DATE) {
      std::string v = boost::lexical_cast<std::string>(val_str);
      val = boost::any(v);
    } else {
      return false;
    }
  } catch (boost::bad_lexical_cast& e) {
    COGADB_FATAL_ERROR("Exception Occured: " << e.what(), "");
    return false;
  }
  return true;
}

bool convertToJoinType(const std::string& join_type_str, JoinType& join_type) {
  std::string str = boost::to_upper_copy(join_type_str);

  if (str == "INNER_JOIN") {
    join_type = INNER_JOIN;
  } else if (str == "LEFT_SEMI_JOIN") {
    join_type = LEFT_SEMI_JOIN;
  } else if (str == "RIGHT_SEMI_JOIN") {
    join_type = RIGHT_SEMI_JOIN;
  } else if (str == "LEFT_ANTI_SEMI_JOIN") {
    join_type = LEFT_ANTI_SEMI_JOIN;
  } else if (str == "RIGHT_ANTI_SEMI_JOIN") {
    join_type = RIGHT_ANTI_SEMI_JOIN;
  } else if (str == "LEFT_OUTER_JOIN") {
    join_type = LEFT_OUTER_JOIN;
  } else if (str == "RIGHT_OUTER_JOIN") {
    join_type = RIGHT_OUTER_JOIN;
  } else if (str == "FULL_OUTER_JOIN") {
    join_type = FULL_OUTER_JOIN;
  } else if (str == "GATHER_JOIN") {
    join_type = GATHER_JOIN;
  } else {
    return false;
  }
  return true;
}

const std::string readFileContent(const std::string& path_to_file) {
  std::ifstream in(path_to_file.c_str());
  std::stringstream ss_in;
  ss_in << in.rdbuf();
  return ss_in.str();
}

void checkInputMember(Document::ConstMemberIterator member,
                      const std::string& expected_operator_name) {
  Document::ConstMemberIterator operator_name_it =
      member->value.FindMember("OPERATOR_NAME");
  assert(operator_name_it != member->value.MemberEnd());
  if (operator_name_it->value.GetString() !=
      std::string(expected_operator_name)) {
    std::cout << operator_name_it->value.GetString() << std::endl;
  }
  assert(operator_name_it->value.GetString() ==
         std::string(expected_operator_name));
  std::cout << "Parse '" << expected_operator_name << "'..." << std::endl;
}

NodePtr parseSort(Document::ConstMemberIterator member) {
  checkInputMember(member, "SORT BY");

  Document::ConstMemberIterator sort_columns_it =
      member->value.FindMember("SORT_COLUMNS");
  if (sort_columns_it != member->value.MemberEnd()) {
    assert(sort_columns_it->value.IsArray() == true);

    //            Value& sort_columns = member->value["SORT_COLUMNS"];
    //            if()

    SortAttributeList sort_attributes;
    std::string sort_column;
    std::string sort_order = "ASCENDING";
    for (rapidjson::SizeType i = 0; i < sort_columns_it->value.Size();
         i++) {  // rapidjson uses SizeType instead of size_t.
      Document::ConstMemberIterator sort_order_it =
          sort_columns_it->value[i].FindMember("ORDER");
      Document::ConstMemberIterator sort_column_it =
          sort_columns_it->value[i].FindMember("COLUMN_NAME");

      if (sort_column_it != sort_columns_it->value[i].MemberEnd()) {
        assert(sort_column_it->value.IsString());
        sort_column = sort_column_it->value.GetString();
      } else {
        COGADB_FATAL_ERROR(
            "Error! Member 'COLUMN_NAME' not found in JSON object 'SORT BY'!",
            "");
        return NodePtr();
      }

      if (sort_order_it != sort_columns_it->value[i].MemberEnd()) {
        assert(sort_order_it->value.IsString());
        sort_order = sort_order_it->value.GetString();
      }
      std::cout << sort_column << " " << sort_order << std::endl;
      SortOrder order;
      if (sort_order == "ASCENDING") {
        order = ASCENDING;
      } else if (sort_order == "DESCENDING") {
        order = DESCENDING;
      } else {
        COGADB_FATAL_ERROR("Invalid Sort Order: " << sort_order, "");
      }
      SortAttribute attr(sort_column, order);
      sort_attributes.push_back(attr);
    }
    NodePtr result(new CoGaDB::query_processing::logical_operator::Logical_Sort(
        sort_attributes, LOOKUP,
        RuntimeConfiguration::instance().getGlobalDeviceConstraint()));
    return result;
  } else {
    COGADB_FATAL_ERROR("Not found member 'SORT_COLUMNS' in SORT operator!", "");
    return NodePtr();
  }
}

NodePtr parseScan(Document::ConstMemberIterator member) {
  checkInputMember(member, "TABLE_SCAN");

  Document::ConstMemberIterator table_name_it =
      member->value.FindMember("TABLE_NAME");
  if (table_name_it != member->value.MemberEnd()) {
    assert(table_name_it->value.IsString() == true);

    std::string table_name = table_name_it->value.GetString();

    uint32_t version = 1;
    Document::ConstMemberIterator version_it =
        member->value.FindMember("VERSION");
    if (version_it != member->value.MemberEnd()) {
      assert(version_it->value.IsUint() == true);
      version = version_it->value.GetUint();
    }

    NodePtr result(new CoGaDB::query_processing::logical_operator::Logical_Scan(
        table_name, version));
    return result;
  } else {
    COGADB_FATAL_ERROR("Not found member 'TABLE_NAME' in TABLE_SCAN operator!",
                       "");
    return NodePtr();
  }
}

NodePtr parseCreateTable(Document::ConstMemberIterator member) {
  checkInputMember(member, "CREATE_TABLE");

  std::string table_name;
  TableSchema schema;
  CompressionSpecifications compression_specifications;
  std::vector<Tuple> tuples_to_insert;
  std::string path_to_file;
  std::string delimiter;

  /* Parsing mandatory field TABLE_NAME */
  Document::ConstMemberIterator table_name_it =
      member->value.FindMember("TABLE_NAME");
  if (table_name_it != member->value.MemberEnd()) {
    assert(table_name_it->value.IsString() == true);
    table_name = table_name_it->value.GetString();
  } else {
    COGADB_FATAL_ERROR(
        "Not found member 'TABLE_NAME' in CREATE_TABLE operator!", "");
    return NodePtr();
  }
  /* Parsing mandatory field TABLE_SCHEMA */
  Document::ConstMemberIterator table_schema_it =
      member->value.FindMember("TABLE_SCHEMA");
  if (table_schema_it != member->value.MemberEnd()) {
    assert(table_schema_it->value.IsArray());
    for (SizeType i = 0; i < table_schema_it->value.Size();
         i++) {  // Uses SizeType instead of size_t
      Document::ConstMemberIterator attr_type_it;
      Document::ConstMemberIterator attr_name_it;
      Document::ConstMemberIterator attr_compression_it;
      attr_type_it = table_schema_it->value[i].FindMember("ATTRIBUTE_TYPE");
      attr_name_it = table_schema_it->value[i].FindMember("ATTRIBUTE_NAME");
      attr_compression_it =
          table_schema_it->value[i].FindMember("ATTRIBUTE_COMPRESSION");

      if (attr_type_it == table_schema_it->value[i].MemberEnd()) {
        COGADB_FATAL_ERROR(
            "Parse Error: Required Field 'ATTRIBUTE_TYPE' Missing!", "");
      }

      if (attr_name_it == table_schema_it->value[i].MemberEnd()) {
        COGADB_FATAL_ERROR(
            "Parse Error: Required Field 'ATTRIBUTE_NAME' Missing!", "");
      }

      std::string attr_type_str = attr_type_it->value.GetString();
      std::string attr_name_str = attr_name_it->value.GetString();

      boost::to_upper(attr_type_str);
      boost::to_upper(attr_name_str);
      //                INT, FLOAT, VARCHAR, BOOLEAN, UINT32, OID, DOUBLE, CHAR,
      //                DATE
      AttributeType type = INT;
      if (!convertToAttributeType(attr_type_str, type)) {
        COGADB_FATAL_ERROR("Unknown Attribute Type: '" << attr_type_str << "'",
                           "");
      }

      schema.push_back(Attribut(type, attr_name_str));
      /* parsing optional field ATTRIBUTE_COMPRESSION */
      if (attr_compression_it != table_schema_it->value[i].MemberEnd()) {
        std::string compression_method = attr_compression_it->value.GetString();
        boost::to_upper(compression_method);
        ColumnType compr = PLAIN_MATERIALIZED;
        if (compression_method == "NOT_COMPRESSED") {
          compr = PLAIN_MATERIALIZED;
        } else if (compression_method == "DICTIONARY_COMPRESSED") {
          compr = DICTIONARY_COMPRESSED;
        } else if (compression_method == "BITPACKED_DICTIONARY_COMPRESSED") {
          compr = BITPACKED_DICTIONARY_COMPRESSED;
        } else if (compression_method == "RUN_LENGTH_COMPRESSED") {
          compr = RUN_LENGTH_COMPRESSED_PREFIX;
        } else if (compression_method == "RUN_LENGTH_DELTA_ONE_COMPRESSED") {
          compr = RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER_PREFIX;
        } else if (compression_method == "VOID_COMPRESSED") {
          compr = VOID_COMPRESSED_NUMBER;
        } else {
          COGADB_FATAL_ERROR(
              "Unknown Compression Method: '" << compression_method << "'", "");
        }

        CompressionSpecification compr_spec(attr_name_str, compr);
        compression_specifications.insert(compr_spec);
      }
    }
  }
  /* should we insert predefined records? */
  /* \todo implement parsing of tuples */

  /* Should we import data? If so, PATH_TO_DATA_FILE needs to be set (optional
   * field).
   * We can also specify a field FIELD_SEPARATOR, which contains the seperators.
   */
  Document::ConstMemberIterator path_to_file_it =
      member->value.FindMember("PATH_TO_DATA_FILE");
  if (path_to_file_it != member->value.MemberEnd()) {
    assert(path_to_file_it->value.IsString());
    path_to_file = path_to_file_it->value.GetString();
    Document::ConstMemberIterator delimiter_it =
        member->value.FindMember("FIELD_SEPARATOR");
    if (delimiter_it != member->value.MemberEnd()) {
      assert(delimiter_it->value.IsString());
      delimiter = delimiter_it->value.GetString();
    }
  }

  NodePtr result(
      new CoGaDB::query_processing::logical_operator::Logical_CreateTable(
          table_name, schema, compression_specifications, path_to_file,
          delimiter));
  return result;
}

NodePtr parseStoreTable(Document::ConstMemberIterator member) {
  checkInputMember(member, "STORE_TABLE");

  std::string table_name;

  /* Parsing mandatory field TABLE_NAME */
  Document::ConstMemberIterator table_name_it =
      member->value.FindMember("TABLE_NAME");
  if (table_name_it != member->value.MemberEnd()) {
    assert(table_name_it->value.IsString() == true);
    table_name = table_name_it->value.GetString();
  } else {
    COGADB_FATAL_ERROR("Not found member 'TABLE_NAME' in STORE_TABLE operator!",
                       "");
    return NodePtr();
  }

  bool persist_table = false;
  Document::ConstMemberIterator persist_it =
      member->value.FindMember("PERSIST_TABLE_ON_DISK");
  if (persist_it != member->value.MemberEnd()) {
    assert(persist_it->value.IsBool());
    persist_table = persist_it->value.GetBool();
  }

  NodePtr result(
      new CoGaDB::query_processing::logical_operator::Logical_StoreTable(
          table_name, persist_table));
  return result;
}

NodePtr parseDeleteTable(Document::ConstMemberIterator member) {
  checkInputMember(member, "DELETE_TABLE");

  std::string table_name;

  /* Parsing mandatory field TABLE_NAME */
  Document::ConstMemberIterator table_name_it =
      member->value.FindMember("TABLE_NAME");
  if (table_name_it != member->value.MemberEnd()) {
    assert(table_name_it->value.IsString() == true);
    table_name = table_name_it->value.GetString();
  } else {
    COGADB_FATAL_ERROR(
        "Not found member 'TABLE_NAME' in DELETE_TABLE operator!", "");
    return NodePtr();
  }

  NodePtr result(
      new CoGaDB::query_processing::logical_operator::Logical_DeleteTable(
          table_name));
  return result;
}

NodePtr parseLimit(Document::ConstMemberIterator member) {
  checkInputMember(member, "LIMIT");

  /* Parsing mandatory field TABLE_NAME */
  Document::ConstMemberIterator num_rows_it =
      member->value.FindMember("NUMBER_OF_ROWS");

  uint64_t num_rows = 0;
  if (num_rows_it != member->value.MemberEnd()) {
    assert(num_rows_it->value.IsUint64());
    num_rows = num_rows_it->value.GetUint64();
  } else {
    COGADB_FATAL_ERROR("Not found member 'NUMBER_OF_ROWS' in LIMIT operator!",
                       "");
    return NodePtr();
  }
  std::cout << "LIMIT: NUM ROWS: " << num_rows << std::endl;
  NodePtr result(
      new CoGaDB::query_processing::logical_operator::Logical_Limit(num_rows));
  return result;
}

NodePtr parseRenameTable(Document::ConstMemberIterator member) {
  checkInputMember(member, "RENAME_TABLE");

  std::string table_name;
  std::string new_table_name;

  /* Parsing mandatory field TABLE_NAME */
  Document::ConstMemberIterator table_name_it =
      member->value.FindMember("TABLE_NAME");
  if (table_name_it != member->value.MemberEnd()) {
    assert(table_name_it->value.IsString() == true);
    table_name = table_name_it->value.GetString();
  } else {
    COGADB_FATAL_ERROR(
        "Not found member 'TABLE_NAME' in RENAME_TABLE operator!", "");
    return NodePtr();
  }
  /* Parsing mandatory field NEW_TABLE_NAME */
  Document::ConstMemberIterator new_table_name_it =
      member->value.FindMember("NEW_TABLE_NAME");
  if (new_table_name_it != member->value.MemberEnd()) {
    assert(new_table_name_it->value.IsString() == true);
    new_table_name = new_table_name_it->value.GetString();
  } else {
    COGADB_FATAL_ERROR(
        "Not found member 'NEW_TABLE_NAME' in RENAME_TABLE operator!", "");
    return NodePtr();
  }

  NodePtr result(
      new CoGaDB::query_processing::logical_operator::Logical_RenameTable(
          table_name, new_table_name));
  return result;
}

NodePtr parseRename(Document::ConstMemberIterator member) {
  checkInputMember(member, "RENAME");

  std::vector<AttributeReference> attrs;

  /* Parsing mandatory field TABLE_NAME */
  Document::ConstMemberIterator renaming_it =
      member->value.FindMember("RENAMING");
  if (renaming_it != member->value.MemberEnd()) {
    if (!parseAttributeReferences(renaming_it, attrs)) {
      COGADB_FATAL_ERROR("Parsing RENAMINGS failed!", "");
    }
  } else {
    COGADB_FATAL_ERROR("Not found member 'RENAMING' in RENAME operator!", "");
    return NodePtr();
  }

  RenameList renamings;
  for (size_t i = 0; i < attrs.size(); ++i) {
    renamings.push_back(RenameEntry(attrs[i].getUnversionedAttributeName(),
                                    attrs[i].getResultAttributeName()));
  }

  NodePtr result(new CoGaDB::query_processing::logical_operator::Logical_Rename(
      renamings));
  return result;
}

NodePtr parseSink(Document::ConstMemberIterator member) {
  checkInputMember(member, "SINK");
  NodePtr result(
      new CoGaDB::query_processing::logical_operator::Logical_Sink());
  return result;
}

NodePtr parseExportIntoFile(Document::ConstMemberIterator member) {
  checkInputMember(member, "EXPORT_INTO_FILE");

  std::string path_to_file;
  std::string delimiter = "|";
  Document::ConstMemberIterator path_to_file_it =
      member->value.FindMember("PATH_TO_DATA_FILE");
  if (path_to_file_it != member->value.MemberEnd()) {
    assert(path_to_file_it->value.IsString());
    path_to_file = path_to_file_it->value.GetString();
    Document::ConstMemberIterator delimiter_it =
        member->value.FindMember("FIELD_SEPARATOR");
    if (delimiter_it != member->value.MemberEnd()) {
      assert(delimiter_it->value.IsString());
      delimiter = delimiter_it->value.GetString();
    }
  } else {
    COGADB_FATAL_ERROR("Missing field: "
                           << "PATH_TO_DATA_FILE",
                       "");
    return NodePtr();
  }

  NodePtr result(new CoGaDB::query_processing::logical_operator::
                     Logical_ExportTableIntoFile(path_to_file, delimiter));
  return result;
}

AttributeReferencePtr parseAttributeReference(
    Document::ConstMemberIterator member) {
  std::string column_name;
  std::string table_name;
  std::string result_name;
  std::string attr_type_name;
  size_t version = 1;
  Document::ConstMemberIterator column_name_it;
  Document::ConstMemberIterator renaming_it;
  Document::ConstMemberIterator version_it;
  Document::ConstMemberIterator result_name_it;
  Document::ConstMemberIterator attr_type_it;
  column_name_it = member->value.FindMember("COLUMN_NAME");
  renaming_it = member->value.FindMember("TABLE_NAME");
  version_it = member->value.FindMember("VERSION");
  result_name_it = member->value.FindMember("RESULT_NAME");
  attr_type_it = member->value.FindMember("ATTRIBUTE_TYPE");

  if (renaming_it != member->value.MemberEnd()) {
    assert(renaming_it->value.IsString());
    table_name = renaming_it->value.GetString();
  } else {
    COGADB_FATAL_ERROR("", "");
  }
  AttributeType attribute_type = DOUBLE;
  if (table_name == "<COMPUTED>") {
    if (attr_type_it != member->value.MemberEnd()) {
      assert(attr_type_it->value.IsString());
      attr_type_name = attr_type_it->value.GetString();
      if (!convertStringToAttributeType(attr_type_name, attribute_type)) {
        COGADB_FATAL_ERROR("'" << attr_type_name << "' is not "
                               << "a valid attribute type name!",
                           "");
      }
    } else {
      COGADB_FATAL_ERROR("Did not find field 'ATTRIBUTE_TYPE'!", "");
    }
  }

  if (column_name_it != member->value.MemberEnd()) {
    assert(column_name_it->value.IsString());
    column_name = column_name_it->value.GetString();
    if (table_name != "<COMPUTED>") {
      column_name =
          table_name + std::string(".") + column_name_it->value.GetString();
    }
  } else {
    COGADB_FATAL_ERROR("", "");
  }

  /* if result name is not explicitely specified, we assume the same name as the
   * column name */
  if (result_name_it != member->value.MemberEnd()) {
    assert(result_name_it->value.IsString());
    result_name = result_name_it->value.GetString();
  } else {
    result_name = column_name;
  }

  /* if version is not explicitely specified, we assume version number 1 */
  if (version_it != member->value.MemberEnd()) {
    assert(version_it->value.IsUint64());
    version = version_it->value.GetUint64();
  } else {
    version = 1;
  }
  //
  TablePtr table = getTablebyName(table_name);
  if (table_name != "<COMPUTED>") {
    if (!table) {
      COGADB_FATAL_ERROR("Table '" << table_name << "' not found!", "");
    }
    if (!table->hasColumn(column_name)) {
      COGADB_FATAL_ERROR("Column '" << column_name << "' not found in Table '"
                                    << table_name << "'!",
                         "");
    }
  }

  if (table_name == "<COMPUTED>") {
    return boost::make_shared<AttributeReference>(column_name, attribute_type,
                                                  result_name, version);
  } else {
    return boost::make_shared<AttributeReference>(table, column_name,
                                                  result_name, version);
  }
}

bool parseAttributeReferences(Document::ConstMemberIterator member,
                              std::vector<AttributeReference>& attrs) {
  assert(member->value.IsArray());
  Value::ConstValueIterator attr_it;
  for (attr_it = member->value.Begin(); attr_it != member->value.End();
       ++attr_it) {
    //            for (SizeType i = 0; i < grouping_cols_it->value.Size(); i++){
    //            // Uses SizeType instead of size_t
    assert(attr_it->IsObject());
    Document::ConstMemberIterator attr_ref_it =
        attr_it->FindMember("ATTRIBUTE_REFERENCE");
    assert(attr_ref_it != attr_it->MemberEnd());
    AttributeReferencePtr attr;
    attr = parseAttributeReference(attr_ref_it);
    if (!attr) {
      return false;
    }
    attrs.push_back(*attr);
  }

  return true;
}

bool parseGroupingAttributes(Document::ConstMemberIterator member,
                             GroupingAttributes& grouping_attrs) {
  Document::ConstMemberIterator grouping_cols_it =
      member->value.FindMember("GROUPING_COLUMNS");
  if (grouping_cols_it != member->value.MemberEnd()) {
    if (!parseAttributeReferences(grouping_cols_it, grouping_attrs)) {
      COGADB_FATAL_ERROR("Parsing Grouping Attributes failed!", "");
    }
  }
  return true;
}

OutputAttribute::OutputAttribute(const AttributeType& _field_type,
                                 const std::string& _field_name,
                                 const std::string& _attribute_name)
    : field_name(_field_name),
      field_type(_field_type),
      attribute_name(_attribute_name) {}

const std::vector<StructFieldPtr> parseReduceUDFPayloadAttributes(
    Document::ConstMemberIterator member) {
  std::vector<StructFieldPtr> result;
  for (SizeType i = 0; i < member->value.Size(); i++) {
    Document::ConstMemberIterator attr_type_it =
        member->value[i].FindMember("ATTRIBUTE_TYPE");
    Document::ConstMemberIterator attr_name_it =
        member->value[i].FindMember("ATTRIBUTE_NAME");
    Document::ConstMemberIterator attr_init_val_it =
        member->value[i].FindMember("ATTRIBUTE_INIT_VALUE");
    assert(attr_type_it != member->value[i].MemberEnd());
    assert(attr_name_it != member->value[i].MemberEnd());
    assert(attr_init_val_it != member->value[i].MemberEnd());

    std::string attr_type = attr_type_it->value.GetString();
    std::string attr_name = attr_name_it->value.GetString();
    std::string attr_init_val = attr_init_val_it->value.GetString();

    boost::to_upper(attr_type);
    boost::to_upper(attr_name);

    AttributeType attribute_type;
    if (!convertStringToAttributeType(attr_type, attribute_type)) {
      COGADB_FATAL_ERROR("'" << attr_type << "' is not "
                             << "a valid attribute type name!",
                         "");
    }

    boost::to_upper(attr_name);

    StructFieldPtr field;
    field = boost::make_shared<StructField>(attribute_type, attr_name,
                                            attr_init_val);
    result.push_back(field);
  }

  return result;
}

const std::vector<OutputAttributePtr> parseReduceUDFOutputAttributes(
    Document::ConstMemberIterator member) {
  std::vector<OutputAttributePtr> result;

  for (SizeType i = 0; i < member->value.Size(); i++) {
    Document::ConstMemberIterator attr_type_it =
        member->value[i].FindMember("ATTRIBUTE_TYPE");
    Document::ConstMemberIterator attr_name_it =
        member->value[i].FindMember("ATTRIBUTE_NAME");
    //    Document::ConstMemberIterator var_name_it =
    //        member->value[i].FindMember("INTERNAL_VARIABLE_NAME");
    assert(attr_type_it != member->value[i].MemberEnd());
    assert(attr_name_it != member->value[i].MemberEnd());
    //    assert(var_name_it != member->value[i].MemberEnd());

    std::string attr_type = attr_type_it->value.GetString();
    std::string attr_name = attr_name_it->value.GetString();
    //    std::string var_name = var_name_it->value.GetString();
    /* workaround issue with computed attributes by a UDF that
     * are referenced via an attribute reference */
    std::string var_name = attr_name;

    boost::to_upper(attr_type);
    boost::to_upper(attr_name);

    AttributeType attribute_type;
    if (!convertStringToAttributeType(attr_type, attribute_type)) {
      COGADB_FATAL_ERROR("'" << attribute_type << "' is not "
                             << "a valid attribute type name!",
                         "");
    }

    OutputAttributePtr field;
    field = boost::make_shared<OutputAttribute>(attribute_type, var_name,
                                                attr_name);
    result.push_back(field);
  }
  return result;
}

// const UDF_CodePtr parseUDFCode(const std::vector<std::string>& code_lines,
//                               const std::vector<OutputAttributePtr>&
//                               output_attributes,
//                               UDF_Type udf_type){
//  std::stringstream code;
//  std::vector<AttributeReferencePtr> scanned_attributes;
//  std::vector<AttributeReferencePtr> result_attributes;
//  for (size_t i = 0; i < code_lines.size(); i++) {
//    std::string code_line = code_lines[i]; // member->value[i].GetString();

//    std::vector<std::string> tokens;
//    boost::split(tokens, code_line, boost::is_any_of("#"));
//    /* n tokens == n-1 '#' characters -> if number of '#' characters should be
//     even, we need an uneven number of tokens! */
//    if (tokens.size() % 2 == 0) {
//      COGADB_FATAL_ERROR(
//          "Parse Error in UDF: uneven number of '#' characters in line: "
//              << "'" << code_line << "'",
//          "");
//    }

//    std::stringstream transformed_code_line;
//    for (size_t j = 0; j < tokens.size(); ++j) {
//      if (j % 2 == 1) {
//        std::cout << "Attribute Reference: " << tokens[j] << std::endl;

//        std::string attribute_type_identifier;
//        std::string attribute_name;
//        uint32_t version = 1;
//        if (!parseColumnIndentifierName(tokens[j], attribute_type_identifier,
//                                        attribute_name, version)) {
//          COGADB_FATAL_ERROR("Invalid attribute reference in UDF Code: '"
//                                 << tokens[j] << "'",
//                             "");
//        }
//        std::cout << "Attribute Type Identifier: " <<
//        attribute_type_identifier
//                  << std::endl;
//        std::cout << "Attribute Name: " << attribute_name << std::endl;
//        std::cout << "Attribute Version: " << version << std::endl;

//        if (attribute_type_identifier == "<HASH_ENTRY>") {
//          // do nothing
//          transformed_code_line << boost::to_upper_copy(tokens[j]);
//        } else if (attribute_type_identifier == "<OUT>") {
//          OutputAttributePtr output_attr;
//          for (size_t k = 0; k < output_attributes.size(); ++k) {
//            if (output_attributes[i]->attribute_name == attribute_name) {
//              output_attr = output_attributes[i];
//            }
//          }
//          assert(output_attr != NULL);
//          AttributeReferencePtr attr;
//          attr = boost::make_shared<AttributeReference>(
//              output_attr->field_name,  // output_attr->attribute_name,
//              output_attr->field_type,
//              output_attr->attribute_name);
//          result_attributes.push_back(attr);

//          if (udf_type == MAP_UDF) {
//            transformed_code_line << getElementAccessExpression(*attr);
//          } else if (udf_type == REDUCE_UDF) {
//            transformed_code_line << getResultArrayVarName(*attr)
//                                  << "[current_result_size]";
//          } else {
//            COGADB_FATAL_ERROR("", "");
//          }

//        } else if (attribute_type_identifier == "<COMPUTED>") {
//          COGADB_FATAL_ERROR("Unhandled case!", "");
//        } else {
//          attribute_name =
//              attribute_type_identifier + std::string(".") + attribute_name;
//          TablePtr table = getTablebyName(attribute_type_identifier);
//          assert(table != NULL);
//          assert(table->hasColumn(attribute_name));
//          AttributeReferencePtr attr;
//          attr = boost::make_shared<AttributeReference>(
//              table, attribute_name, attribute_name, version);
//          scanned_attributes.push_back(attr);
//          transformed_code_line << getElementAccessExpression(*attr);
//        }
//      } else {
//        transformed_code_line << tokens[j];
//      }
//    }
//    code << transformed_code_line.str();
//    if (i + 1 < code_lines.size()) code << std::endl;
//  }
//  return boost::make_shared<UDF_Code>(code.str(), scanned_attributes,
//                                      result_attributes);
//}

const std::vector<std::string> getUDFCode(
    Document::ConstMemberIterator member) {
  std::vector<std::string> code_lines;
  for (SizeType i = 0; i < member->value.Size(); i++) {
    assert(member->value[i].IsString());
    std::string code_line = member->value[i].GetString();
    code_lines.push_back(code_line);
  }
  return code_lines;
}

const AggregateSpecificationPtr parseReduceUDF(
    Document::ConstMemberIterator member) {
  Document::ConstMemberIterator payload_attributes_it;
  Document::ConstMemberIterator payload_output_it;
  Document::ConstMemberIterator udf_code_it;
  Document::ConstMemberIterator final_udf_code_it;

  payload_attributes_it =
      member->value.FindMember("REDUCE_UDF_PAYLOAD_ATTRIBUTES");
  payload_output_it = member->value.FindMember("REDUCE_UDF_OUTPUT_ATTRIBUTES");
  udf_code_it = member->value.FindMember("REDUCE_UDF_CODE");
  final_udf_code_it = member->value.FindMember("REDUCE_UDF_FINAL_CODE");

  //\todo read this from JSON!
  AggregationFunctionType agg_func_type = ALGEBRAIC;
  std::vector<StructFieldPtr> struct_fields =
      parseReduceUDFPayloadAttributes(payload_attributes_it);
  std::vector<OutputAttributePtr> output_attrs =
      parseReduceUDFOutputAttributes(payload_output_it);

  std::vector<std::string> udf_code_lines = getUDFCode(udf_code_it);
  UDF_CodePtr udf_code = parseUDFCode(udf_code_lines, output_attrs, REDUCE_UDF);
  std::vector<std::string> final_udf_code_lines = getUDFCode(final_udf_code_it);
  UDF_CodePtr final_udf_code =
      parseUDFCode(final_udf_code_lines, output_attrs, REDUCE_UDF);

  for (size_t i = 0; i < struct_fields.size(); ++i) {
    std::cout << "StructField: " << util::getName(struct_fields[i]->field_type)
              << " " << struct_fields[i]->field_name;
    if (!struct_fields[i]->field_init_val.empty()) {
      std::cout << " = " << struct_fields[i]->field_init_val;
    }
    std::cout << ";" << std::endl;
  }
  for (size_t i = 0; i < output_attrs.size(); ++i) {
    std::cout << "OutputAttribute: "
              << util::getName(output_attrs[i]->field_type) << " "
              << output_attrs[i]->field_name << " ("
              << output_attrs[i]->attribute_name << ")" << std::endl;
  }
  std::cout << "UDF_CODE: " << std::endl << udf_code->getCode() << std::endl;
  std::cout << "FINAL_UDF_CODE: " << std::endl
            << final_udf_code->getCode() << std::endl;

  return createAggregateSpecificationUDF(agg_func_type, struct_fields, udf_code,
                                         final_udf_code);
}

NodePtr parseMapUDF(Document::ConstMemberIterator member) {
  checkInputMember(member, "MAP_UDF");

  Document::ConstMemberIterator output_attributes_it;
  Document::ConstMemberIterator udf_code_it;

  output_attributes_it = member->value.FindMember("MAP_UDF_OUTPUT_ATTRIBUTES");
  udf_code_it = member->value.FindMember("MAP_UDF_CODE");

  std::vector<OutputAttributePtr> output_attrs =
      parseReduceUDFOutputAttributes(output_attributes_it);

  std::vector<std::string> code_lines = getUDFCode(udf_code_it);
  UDF_CodePtr udf_code = parseUDFCode(code_lines, output_attrs, MAP_UDF);

  for (size_t i = 0; i < output_attrs.size(); ++i) {
    std::cout << "OutputAttribute: "
              << util::getName(output_attrs[i]->field_type) << " "
              << output_attrs[i]->field_name << " ("
              << output_attrs[i]->attribute_name << ")" << std::endl;
  }
  std::cout << "UDF_CODE: " << std::endl << udf_code->getCode() << std::endl;

  std::vector<StructFieldPtr> declared_variables;
  for (size_t i = 0; i < output_attrs.size(); ++i) {
    StructFieldPtr declared_var(new StructField(
        output_attrs[i]->field_type, output_attrs[i]->field_name, "0"));
    declared_variables.push_back(declared_var);
  }

  Map_UDFPtr map_udf(new Generic_Map_UDF(udf_code, declared_variables));
  Map_UDF_ParamPtr param(new Map_UDF_Param(map_udf));
  return boost::make_shared<
      CoGaDB::query_processing::logical_operator::Logical_MapUDF>(param);
}

const AggregateSpecificationPtr parseAggregateSpecification(
    Value::ConstValueIterator aggr_spec_it) {
  AggregateSpecificationPtr aggr_spec;
  std::string result_name;
  Document::ConstMemberIterator attr_ref_it;
  Document::ConstMemberIterator agg_func_it;

  attr_ref_it = aggr_spec_it->FindMember("ATTRIBUTE_REFERENCE");
  agg_func_it = aggr_spec_it->FindMember("AGGREGATION_FUNCTION");

  std::string agg_func;
  if (agg_func_it != aggr_spec_it->MemberEnd()) {
    assert(agg_func_it->value.IsString());
    agg_func = agg_func_it->value.GetString();
  } else {
    COGADB_FATAL_ERROR("", "");
  }
  if (agg_func == "UDF") {
    Document::ConstMemberIterator reduce_udf_it;
    reduce_udf_it = aggr_spec_it->FindMember("REDUCE_UDF");
    assert(reduce_udf_it != aggr_spec_it->MemberEnd());
    aggr_spec = parseReduceUDF(reduce_udf_it);
  } else {
    if (attr_ref_it != aggr_spec_it->MemberEnd()) {
      AttributeReferencePtr attr = parseAttributeReference(attr_ref_it);
      Document::ConstMemberIterator result_name_it;
      result_name_it = aggr_spec_it->FindMember("RESULT_NAME");
      if (result_name_it != aggr_spec_it->MemberEnd()) {
        assert(result_name_it->value.IsString());
        result_name = result_name_it->value.GetString();
      } else {
        result_name = agg_func;
        result_name += "_";
        result_name += attr->getVersionedAttributeName();
      }
      AggregationFunction agg = COUNT;
      if (agg_func == "COUNT") {
        agg = COUNT;
      } else if (agg_func == "SUM") {
        agg = SUM;
      } else if (agg_func == "MIN") {
        agg = MIN;
      } else if (agg_func == "MAX") {
        agg = MAX;
      } else if (agg_func == "AVG") {
        agg = AVERAGE;
      } else if (agg_func == "GENOTYPE") {
        agg = AGG_GENOTYPE;
      } else if (agg_func == "CONCAT_BASES") {
        agg = AGG_CONCAT_BASES;
      } else if (agg_func == "IS_HOMOPOLYMER") {
        agg = AGG_IS_HOMOPOLYMER;
      } else if (agg_func == "AGG_GENOTYPE_STATISTICS") {
        agg = AGG_GENOTYPE_STATISTICS;
      } else {
        COGADB_FATAL_ERROR("Unknown Aggregation function: '" << agg_func << "'",
                           "");
      }
      aggr_spec = createAggregateSpecification(*attr, agg, result_name);
    } else {
      COGADB_FATAL_ERROR("", "");
    }
  }
  return aggr_spec;
}

bool parseAggregateSpecifications(Document::ConstMemberIterator member,
                                  AggregateSpecifications& aggregation_specs) {
  Document::ConstMemberIterator aggr_specs_it =
      member->value.FindMember("AGGREGATION_SPECIFICATION");
  if (aggr_specs_it != member->value.MemberEnd()) {
    assert(aggr_specs_it->value.IsArray());

    Value::ConstValueIterator aggr_spec_it;
    for (aggr_spec_it = aggr_specs_it->value.Begin();
         aggr_spec_it != aggr_specs_it->value.End(); ++aggr_spec_it) {
      assert(aggr_spec_it->IsObject());
      assert(aggr_spec_it->FindMember("AGGREGATION_FUNCTION") !=
             aggr_spec_it->MemberEnd());
      AggregateSpecificationPtr aggr_spec =
          parseAggregateSpecification(aggr_spec_it);
      //                assert(aggr_spec!=NULL);
      if (aggr_spec) aggregation_specs.push_back(aggr_spec);
    }
  }
  return true;
}

NodePtr parseGenericGroupby(Document::ConstMemberIterator member) {
  checkInputMember(member, "GENERIC_GROUPBY");

  GroupingAttributes grouping_attrs;
  AggregateSpecifications aggregation_specs;
  if (!parseGroupingAttributes(member, grouping_attrs)) {
    COGADB_FATAL_ERROR("", "");
  }
  if (!parseAggregateSpecifications(member, aggregation_specs)) {
    COGADB_FATAL_ERROR("", "");
  }
  ProcessorSpecification proc_spec(hype::PD0);
  GroupByAggregateParam groupby_param(proc_spec, grouping_attrs,
                                      aggregation_specs);
  NodePtr result(
      new CoGaDB::query_processing::logical_operator::Logical_GenericGroupby(
          groupby_param));
  return result;
}

PredicateExpressionPtr parsePredicateExpression(
    Document::ConstMemberIterator member);

PredicateExpressionPtr parseColumnConstantPredicateExpression(
    Document::ConstMemberIterator member) {
  Document::ConstMemberIterator pred_type_it =
      member->value.FindMember("PREDICATE_TYPE");
  Document::ConstMemberIterator attr_ref_it =
      member->value.FindMember("ATTRIBUTE_REFERENCE");
  Document::ConstMemberIterator pred_comp_it =
      member->value.FindMember("PREDICATE_COMPARATOR");
  Document::ConstMemberIterator constant_it =
      member->value.FindMember("CONSTANT");

  assert(pred_type_it != member->value.MemberEnd());
  assert(attr_ref_it != member->value.MemberEnd());
  assert(pred_comp_it != member->value.MemberEnd());
  assert(constant_it != member->value.MemberEnd());

  assert(pred_type_it->value.IsString());
  std::string predicate_type = pred_type_it->value.GetString();
  boost::to_upper(predicate_type);
  assert(predicate_type == "COLUMN_CONSTANT_PREDICATE");

  AttributeReferencePtr attr = parseAttributeReference(attr_ref_it);
  assert(attr != NULL);

  assert(pred_comp_it->value.IsString());
  std::string pred_comp_str = pred_comp_it->value.GetString();
  boost::to_upper(pred_comp_str);
  ValueComparator pred_comp;
  if (!convertToValueComparator(pred_comp_str, pred_comp)) {
    COGADB_FATAL_ERROR("Unknown Value Comparator: '" << pred_comp_str << "'",
                       "");
  }

  Document::ConstMemberIterator constant_value_it =
      constant_it->value.FindMember("CONSTANT_VALUE");
  Document::ConstMemberIterator constant_type_it =
      constant_it->value.FindMember("CONSTANT_TYPE");
  assert(constant_value_it != constant_it->value.MemberEnd());
  assert(constant_type_it != constant_it->value.MemberEnd());

  std::string constant_value_str = constant_value_it->value.GetString();
  std::string constant_type_str = constant_type_it->value.GetString();

  AttributeType constant_type;
  boost::any constant_value;

  if (!convertToAttributeType(constant_type_str, constant_type)) {
    COGADB_FATAL_ERROR("Unknown Attribute Type: '" << constant_type_str << "'",
                       "");
  }

  if (!convertToInternalConstant(constant_value_str, constant_type,
                                 constant_value)) {
    COGADB_FATAL_ERROR("Cannot Parse Value '" << constant_value_str
                                              << "' as Attribute Type: '"
                                              << constant_type_str << "'",
                       "");
  }

  PredicateExpressionPtr result =
      createColumnConstantComparisonPredicateExpression(attr, constant_value,
                                                        pred_comp);
  assert(result != NULL);
  return result;
}

PredicateExpressionPtr parseColumnColumnPredicateExpression(
    Document::ConstMemberIterator member) {
  Document::ConstMemberIterator pred_type_it =
      member->value.FindMember("PREDICATE_TYPE");
  Document::ConstMemberIterator lhs_attr_ref_it =
      member->value.FindMember("LEFT_HAND_SIDE_ATTRIBUTE_REFERENCE");
  Document::ConstMemberIterator pred_comp_it =
      member->value.FindMember("PREDICATE_COMPARATOR");
  Document::ConstMemberIterator rhs_attr_ref_it =
      member->value.FindMember("RIGHT_HAND_SIDE_ATTRIBUTE_REFERENCE");

  assert(pred_type_it != member->value.MemberEnd());
  assert(lhs_attr_ref_it != member->value.MemberEnd());
  assert(pred_comp_it != member->value.MemberEnd());
  assert(rhs_attr_ref_it != member->value.MemberEnd());

  assert(pred_type_it->value.IsString());
  std::string predicate_type = pred_type_it->value.GetString();
  boost::to_upper(predicate_type);
  assert(predicate_type == "COLUMN_COLUMN_PREDICATE");

  AttributeReferencePtr lhs_attr = parseAttributeReference(lhs_attr_ref_it);
  AttributeReferencePtr rhs_attr = parseAttributeReference(rhs_attr_ref_it);

  assert(lhs_attr != NULL);
  assert(rhs_attr != NULL);

  assert(pred_comp_it->value.IsString());
  std::string pred_comp_str = pred_comp_it->value.GetString();
  boost::to_upper(pred_comp_str);
  ValueComparator pred_comp;
  if (!convertToValueComparator(pred_comp_str, pred_comp)) {
    COGADB_FATAL_ERROR("Unknown Value Comparator: '" << pred_comp_str << "'",
                       "");
  }

  PredicateExpressionPtr result =
      createColumnColumnComparisonPredicateExpression(lhs_attr, rhs_attr,
                                                      pred_comp);
  return result;
}

PredicateExpressionPtr parseCombinatorPredicateExpression(
    Document::ConstMemberIterator member, const std::string& predicate_type) {
  std::vector<PredicateExpressionPtr> predicates;
  Document::ConstMemberIterator predicates_it =
      member->value.FindMember("PREDICATES");
  for (SizeType i = 0; i < predicates_it->value.Size(); i++) {
    Document::ConstMemberIterator predicate_it =
        predicates_it->value[i].FindMember("PREDICATE");
    assert(predicate_it != predicates_it->value[i].MemberEnd());
    PredicateExpressionPtr predicate = parsePredicateExpression(predicate_it);
    assert(predicate != NULL);
    predicates.push_back(predicate);
  }

  LogicalOperation op;
  if (predicate_type == "AND_PREDICATE") {
    op = LOGICAL_AND;
  } else if (predicate_type == "OR_PREDICATE") {
    op = LOGICAL_OR;
  } else {
    COGADB_FATAL_ERROR("Unknown predicate type: '" << predicate_type << "'",
                       "");
  }

  return createPredicateExpression(predicates, op);
}

PredicateExpressionPtr parsePredicateExpression(
    Document::ConstMemberIterator member) {
  Document::ConstMemberIterator pred_type_it =
      member->value.FindMember("PREDICATE_TYPE");
  std::string predicate_type;
  if (pred_type_it != member->value.MemberEnd()) {
    assert(pred_type_it->value.IsString());
    predicate_type = pred_type_it->value.GetString();
  } else {
    COGADB_FATAL_ERROR(
        "Did not find field 'PREDICATE_TYPE' in PREDICATE object!", "");
  }

  if (predicate_type == "COLUMN_CONSTANT_PREDICATE") {
    return parseColumnConstantPredicateExpression(member);
  } else if (predicate_type == "COLUMN_COLUMN_PREDICATE") {
    return parseColumnColumnPredicateExpression(member);
  } else if (predicate_type == "AND_PREDICATE" ||
             predicate_type == "OR_PREDICATE") {
    return parseCombinatorPredicateExpression(member, predicate_type);
  } else {
    COGADB_FATAL_ERROR("Unkown predicate type: '" << predicate_type << "'", "");
  }

  return PredicateExpressionPtr();
}

NodePtr parseGenericSelection(Document::ConstMemberIterator member) {
  checkInputMember(member, "GENERIC_SELECTION");

  Document::ConstMemberIterator predicate_it =
      member->value.FindMember("PREDICATE");
  assert(predicate_it != member->value.MemberEnd());

  PredicateExpressionPtr pred_expr = parsePredicateExpression(predicate_it);
  assert(pred_expr != NULL);
  std::cout << pred_expr->getCPPExpression() << std::endl;

  //        ProcessorSpecification proc_spec(hype::PD0);
  //        GroupByAggregateParam groupby_param(proc_spec, grouping_attrs,
  //        aggregation_specs);
  NodePtr result;
  result = NodePtr(
      new CoGaDB::query_processing::logical_operator::Logical_GenericSelection(
          pred_expr));
  //        (new
  //        CoGaDB::query_processing::logical_operator::Logical_GenericGroupby(groupby_param));
  return result;
}

NodePtr parseGenericJoin(Document::ConstMemberIterator member) {
  checkInputMember(member, "GENERIC_JOIN");

  Document::ConstMemberIterator join_type_it =
      member->value.FindMember("JOIN_TYPE");
  assert(join_type_it != member->value.MemberEnd());
  std::string join_type_str = join_type_it->value.GetString();

  JoinType join_type;
  if (!convertToJoinType(join_type_str, join_type)) {
    COGADB_FATAL_ERROR("Unknown Join Type: '" << join_type_str << "'", "");
  }

  Document::ConstMemberIterator predicate_it =
      member->value.FindMember("PREDICATE");
  assert(predicate_it != member->value.MemberEnd());

  PredicateExpressionPtr pred_expr = parsePredicateExpression(predicate_it);
  assert(pred_expr != NULL);
  std::cout << pred_expr->getCPPExpression() << std::endl;

  /* We currently support equi join operator only. Therefore, we try to
   * find an equality join predicate (ValueValuePredicate). The remaining
   * predicates are put in a selection operator. Note that this should have
   * no effect on performance of join processing, as the code ends up inside the
   * same pipeline. */

  boost::shared_ptr<PredicateSpecification> equality_join_predicate;
  std::vector<PredicateExpressionPtr> pred_exprs = getPredicates(pred_expr);
  for (size_t i = 0; i < pred_exprs.size(); ++i) {
    boost::shared_ptr<PredicateSpecification> ptr;
    ptr = boost::dynamic_pointer_cast<PredicateSpecification>(pred_exprs[i]);
    if (ptr) {
      if (ptr->getPredicateSpecificationType() == ValueValuePredicateSpec &&
          ptr->getValueComparator() == EQUAL) {
        //                    /* let us asume for now that attributes come from
        //                    different tables */
        //                    assert(ptr->getLeftAttribute()->getTable()!=ptr->getRightAttribute()->getTable());

        if (equality_join_predicate) {
          COGADB_FATAL_ERROR(
              "Found multiple join precdicates in same join operator, which is "
              "currently not supported!",
              "");
        }

        equality_join_predicate = ptr;
      } else if (ptr->getPredicateSpecificationType() ==
                 ValueConstantPredicateSpec) {
        COGADB_FATAL_ERROR(
            "No Column Constant comparison predicates allowed in join "
            "operator!",
            "");
      } else if (ptr->getPredicateSpecificationType() ==
                 ValueRegularExpressionPredicateSpec) {
        COGADB_FATAL_ERROR(
            "No Column regular expression comparison predicates allowed in "
            "join operator!",
            "");
      }
    }
  }
  if (!equality_join_predicate) {
    COGADB_FATAL_ERROR(
        "Could not create operator for generic join because of too complex "
        "predicate expression!"
            << " Unimplemented Feature: Arbitrary Join Predicates",
        "");
  }

  NodePtr result;
  //        result = NodePtr(new
  //        CoGaDB::query_processing::logical_operator::Logical_Join(
  //                equality_join_predicate->getLeftAttribute()->getUnversionedAttributeName(),
  //                equality_join_predicate->getRightAttribute()->getUnversionedAttributeName(),
  //                join_type));
  result = NodePtr(new CoGaDB::query_processing::logical_operator::Logical_Join(
      createFullyQualifiedColumnIdentifier(
          equality_join_predicate->getLeftAttribute()),
      createFullyQualifiedColumnIdentifier(
          equality_join_predicate->getRightAttribute()),
      join_type));

  return result;
}

NodePtr parseProjection(Document::ConstMemberIterator member) {
  checkInputMember(member, "PROJECTION");
  Document::ConstMemberIterator proj_cols_it =
      member->value.FindMember("ATTRIBUTES");
  if (proj_cols_it != member->value.MemberEnd()) {
    std::vector<AttributeReference> attrs;
    if (!parseAttributeReferences(proj_cols_it, attrs)) {
      COGADB_FATAL_ERROR("Parsing Grouping Attributes failed!", "");
    }
    //            std::list<std::string> columns_to_select;
    //            for(size_t i=0;i<attrs.size();++i){
    //                columns_to_select.push_back(attrs[i].getUnversionedAttributeName());
    //            }

    std::vector<AttributeReferencePtr> attr_refs;
    for (size_t i = 0; i < attrs.size(); ++i) {
      attr_refs.push_back(boost::make_shared<AttributeReference>(attrs[i]));
    }

    return NodePtr(
        new CoGaDB::query_processing::logical_operator::Logical_Projection(
            attr_refs));
  }
  return NodePtr();
}

NodePtr parseCrossJoin(Document::ConstMemberIterator member) {
  checkInputMember(member, "CROSS_JOIN");

  return NodePtr(
      new CoGaDB::query_processing::logical_operator::Logical_CrossJoin());
}

NodePtr parseMember(Document::ConstMemberIterator member) {
  NodePtr node;
  NodePtr left_child;
  NodePtr right_child;

  if (member->value.IsObject()) {
    Document::ConstMemberIterator operator_name_it =
        member->value.FindMember("OPERATOR_NAME");
    if (operator_name_it != member->value.MemberEnd()) {
      const std::string operator_name = operator_name_it->value.GetString();
      std::cout << "Operator: '" << operator_name << "'" << std::endl;

      if (operator_name == "SORT BY") {
        node = parseSort(member);
      } else if (operator_name == "TABLE_SCAN") {
        node = parseScan(member);
      } else if (operator_name == "CREATE_TABLE") {
        node = parseCreateTable(member);
      } else if (operator_name == "STORE_TABLE") {
        node = parseStoreTable(member);
      } else if (operator_name == "EXPORT_INTO_FILE") {
        node = parseExportIntoFile(member);
      } else if (operator_name == "GENERIC_GROUPBY") {
        node = parseGenericGroupby(member);
      } else if (operator_name == "MAP_UDF") {
        node = parseMapUDF(member);
      } else if (operator_name == "GENERIC_SELECTION") {
        node = parseGenericSelection(member);
      } else if (operator_name == "PROJECTION") {
        node = parseProjection(member);
      } else if (operator_name == "CROSS_JOIN") {
        node = parseCrossJoin(member);
      } else if (operator_name == "GENERIC_JOIN") {
        node = parseGenericJoin(member);
      } else if (operator_name == "DELETE_TABLE") {
        node = parseDeleteTable(member);
      } else if (operator_name == "LIMIT") {
        node = parseLimit(member);
      } else if (operator_name == "RENAME_TABLE") {
        node = parseRenameTable(member);
      } else if (operator_name == "RENAME") {
        node = parseRename(member);
      } else if (operator_name == "SINK") {
        node = parseSink(member);
      } else {
        COGADB_FATAL_ERROR("Unknown Operator Name: " << operator_name, "");
      }

      if (!node) {
        return NodePtr();
      }

      Document::ConstMemberIterator left_child_it =
          member->value.FindMember("LEFT_CHILD");
      if (left_child_it != member->value.MemberEnd()) {
        left_child = parseMember(left_child_it);
      }

      Document::ConstMemberIterator right_child_it =
          member->value.FindMember("RIGHT_CHILD");
      if (right_child_it != member->value.MemberEnd()) {
        right_child = parseMember(right_child_it);
      }

      if (!left_child && operator_name != "TABLE_SCAN" &&
          operator_name != "CREATE_TABLE" && operator_name != "DELETE_TABLE" &&
          operator_name != "RENAME_TABLE") {
        COGADB_FATAL_ERROR("No valid left child!", "");
        return NodePtr();
      }

      node->setLeft(left_child);
      node->setRight(right_child);

    } else {
      COGADB_FATAL_ERROR("Error: found object without member 'OPERATOR_NAME'",
                         "");
      return NodePtr();
    }
  }
  return node;
}

#ifdef JSON_VALIDATION_ENABLED
::rapidjson::SchemaDocument load_schema_from_file(
    const std::string& path_to_file) {
  ::rapidjson::Document const& document = load_json_from_file(path_to_file);
  // TODO: What happens if the schema document itself is invalid?
  return ::rapidjson::SchemaDocument(document);
}

void validate_jsondocument(const ::rapidjson::Document& document,
                           const ::rapidjson::SchemaDocument& schema) {
  SchemaValidator validator(schema);
  const bool valid = document.Accept(validator);
  validator.Reset();
  if (!valid) {
    StringBuffer schema_pointer;
    StringBuffer document_pointer;
    validator.GetInvalidSchemaPointer().StringifyUriFragment(schema_pointer);
    validator.GetInvalidDocumentPointer().StringifyUriFragment(
        document_pointer);
    std::cout << "Schema: " << schema_pointer.GetString() << std::endl;
    std::cout << "Pointer: " << document_pointer.GetString() << std::endl;
    std::cout << "Keyword: " << validator.GetInvalidSchemaKeyword()
              << std::endl;
    throw json_invalid_plan_document_exception(
        "The given JSON "
        "query plan does not conform to the schema");
  }
}
#endif  // JSON_VALIDATION_ENABLED

::rapidjson::Document load_json_from_file(const std::string& path_to_file) {
  if (!boost::filesystem::exists(path_to_file)) {
    std::stringstream error_msg;
    error_msg << "Error: File '" << path_to_file << "' not found!";
    throw json_file_not_found_exception(error_msg.str());
  }

  std::string json = readFileContent(path_to_file);

  // TODO: Dump to client->getOutputStream() instead of std::cout?
  std::cout << "'" << json << "'" << std::endl;

  ::rapidjson::Document document;
  // "normal" parsing, decode strings to new buffers.
  // Can use other input stream via ParseStream().
  if (document.Parse(json.c_str()).HasParseError()) {
    std::stringstream error_msg;
    error_msg << "Error during parsing '" << path_to_file << "'" << std::endl;

    error_msg << "Error Code: " << (int)document.GetParseError();

    throw json_parsing_exception(error_msg.str());
  }

  return document;
}

const query_processing::LogicalQueryPlanPtr import_query_from_jsondocument(
    ::rapidjson::Document const& document) {
  Document::ConstMemberIterator plan = document.FindMember("QUERY_PLAN");
  if (plan == document.MemberEnd()) {
    throw json_plan_import_exception(
        "Error: found root object without member 'QUERY_PLAN'");
  }
  NodePtr root = parseMember(plan);
  query_processing::TypedNodePtr typed_root =
      boost::dynamic_pointer_cast<query_processing::TypedNodePtr::element_type>(
          root);
  query_processing::LogicalQueryPlanPtr log_plan(
      new query_processing::LogicalQueryPlan(typed_root));
  return log_plan;
}

const query_processing::LogicalQueryPlanPtr import_query_from_json(
    const std::string& path_to_file, ClientPtr client) {
  std::ostream& out = client->getOutputStream();

#ifdef JSON_VALIDATION_ENABLED
  std::string schema_file =
      std::string(PATH_TO_COGADB_EXECUTABLE)
          .append("/share/cogadb/parser/json/query_plan_schema.json");
#endif

  try {
    Document const& document = load_json_from_file(path_to_file);
#ifdef JSON_VALIDATION_ENABLED
    SchemaDocument const& schema = load_schema_from_file(schema_file);
//  validate_jsondocument(document, schema);
#endif
    return import_query_from_jsondocument(document);
  } catch (json_file_not_found_exception const& nfe) {
    out << nfe.what() << std::endl;
    return query_processing::LogicalQueryPlanPtr();
  } catch (json_parsing_exception const& jpe) {
    out << jpe.what() << std::endl;
    return query_processing::LogicalQueryPlanPtr();
  } catch (json_plan_import_exception const& jpe) {
    out << jpe.what() << std::endl;
    return query_processing::LogicalQueryPlanPtr();
  } catch (std::exception const& e) {
    out << "Error: Loading and parsing of JSON file failed.";
    return query_processing::LogicalQueryPlanPtr();
  }
}

const std::pair<bool, TablePtr> load_and_execute_query_from_json(
    const std::string& path_to_file, ClientPtr client) {
  query_processing::LogicalQueryPlanPtr log_plan =
      import_query_from_json(path_to_file, client);
  if (!log_plan || !log_plan->getRoot()) {
    return std::make_pair(false, TablePtr());
  }
  if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
    log_plan->print();
  }
  Timestamp begin = getTimestamp();
  TablePtr result = executeQueryPlan(log_plan, client);
  Timestamp end = getTimestamp();
  assert(end >= begin);
  double exec_time_in_milliseconds = double(end - begin) / (1000 * 1000);
  if (log_plan->getRoot()->getOperationName() != "SINK") {
    printResult(result, client, exec_time_in_milliseconds);
  }
  return std::make_pair(true, result);
}

}  // end namespace CoGaDB
