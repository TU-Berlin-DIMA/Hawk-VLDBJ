

#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/create_table_operator.hpp>

namespace CoGaDB {
namespace query_processing {

Physical_Operator_Map_Ptr map_init_function_create_table_operator() {
  return Physical_Operator_Map_Ptr();
}

namespace logical_operator {

Logical_CreateTable::Logical_CreateTable(
    const std::string& table_name, const TableSchema& schema,
    const CompressionSpecifications& compression_specifications,
    const std::vector<Tuple>& tuples_to_insert)
    : TypedNode_Impl<TablePtr, map_init_function_create_table_operator>(
          false, hype::ANY_DEVICE),
      table_name_(table_name),
      schema_(schema),
      compression_specifications_(compression_specifications),
      tuples_to_insert_(tuples_to_insert),
      path_to_file_(),
      delimiter_() {}

Logical_CreateTable::Logical_CreateTable(
    const std::string& table_name, const TableSchema& schema,
    const CompressionSpecifications& compression_specifications,
    const std::string& path_to_file, const std::string& delimiter)
    : TypedNode_Impl<TablePtr, map_init_function_create_table_operator>(
          false, hype::ANY_DEVICE),
      table_name_(table_name),
      schema_(schema),
      compression_specifications_(compression_specifications),
      tuples_to_insert_(),
      path_to_file_(path_to_file),
      delimiter_(delimiter) {}

Logical_CreateTable::Logical_CreateTable(
    const std::string& table_name, const TableSchema& schema,
    const CompressionSpecifications& compression_specifications,
    const std::vector<Tuple>& tuples_to_insert, const std::string& path_to_file,
    const std::string& delimiter)
    : TypedNode_Impl<TablePtr, map_init_function_create_table_operator>(
          false, hype::ANY_DEVICE),
      table_name_(table_name),
      schema_(schema),
      compression_specifications_(compression_specifications),
      tuples_to_insert_(tuples_to_insert),
      path_to_file_(path_to_file),
      delimiter_(delimiter) {}

unsigned int Logical_CreateTable::getOutputResultSize() const { return 0; }

double Logical_CreateTable::getCalculatedSelectivity() const { return 1; }

const hype::Tuple Logical_CreateTable::getFeatureVector() const {
  hype::Tuple t;
  t.push_back(getOutputResultSize());
  return t;
}

void Logical_CreateTable::produce_impl(CodeGeneratorPtr code_gen,
                                       QueryContextPtr context) {
  TablePtr table(new Table(table_name_, schema_, compression_specifications_));

  for (size_t i = 0; i < tuples_to_insert_.size(); ++i) {
    table->insert(tuples_to_insert_[i]);
  }

  if (path_to_file_ != "") {
    /* \todo make delimiter configurable! */
    assert(this->delimiter_ == "|");
    table->loadDatafromFile(path_to_file_, true);
  }

  retrieveScannedAndProjectedAttributesFromScannedTable(code_gen, context,
                                                        table, 1);

  // create for loop for code_gen
  if (!code_gen->createForLoop(table, 1)) {
    COGADB_FATAL_ERROR("Creating of for loop failed!", "");
  }
  if (parent_) {
    parent_->consume(code_gen, context);
  }
}

void Logical_CreateTable::consume_impl(CodeGeneratorPtr code_gen,
                                       QueryContextPtr context) {
  /* this is a bulk operator, so no consume here! */
}

std::string Logical_CreateTable::getOperationName() const {
  return "CREATE_TABLE";
}

std::string Logical_CreateTable::toString(bool verbose) const {
  std::string result = this->getOperationName();
  if (verbose) {
    result += " (";
    TableSchema::const_iterator it;
    for (it = schema_.begin(); it != schema_.end(); ++it) {
      if (it != schema_.begin()) result += ", ";
      result += util::getName(it->first);
      result += ": ";
      result += it->second;
    }
    if (!path_to_file_.empty()) {
      result += "; ";
      result += "IMPORT_FILE='";
      result += path_to_file_;
      result += "', SEPARATOR='";
      result += delimiter_;
      result += "'";
    }
    result += ")";
  }
  return result;
}
}  // end namespace logical_operator

}  // end namespace query_processing
}  // end namespace CogaDB
