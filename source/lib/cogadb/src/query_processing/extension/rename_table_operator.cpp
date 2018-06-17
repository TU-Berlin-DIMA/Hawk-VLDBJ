

#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/rename_table_operator.hpp>

namespace CoGaDB {
namespace query_processing {

namespace logical_operator {

Logical_RenameTable::Logical_RenameTable(const std::string& table_name,
                                         const std::string& new_table_name)
    : Logical_BulkOperator(),
      table_name_(table_name),
      new_table_name_(new_table_name) {}

unsigned int Logical_RenameTable::getOutputResultSize() const { return 0; }

double Logical_RenameTable::getCalculatedSelectivity() const { return 1; }

std::string Logical_RenameTable::getOperationName() const {
  return "RENAME_TABLE";
}

std::string Logical_RenameTable::toString(bool verbose) const {
  std::string result = this->getOperationName();
  if (verbose) {
    result += " (";
    result += table_name_;
    result += ")";
  }
  return result;
}

const TablePtr Logical_RenameTable::executeBulkOperator(TablePtr table) const {
  assert(table == NULL);

  bool return_value = renameTable(table_name_, new_table_name_);

  TableSchema schema;
  schema.push_back(std::make_pair(INT, std::string("RETURN_VALUE")));
  TablePtr result(new Table("RENAMED_TABLE", schema));
  Tuple t;
  t.push_back(int32_t(return_value));
  result->insert(t);
  return result;
}

const std::list<std::string> Logical_RenameTable::getNamesOfReferencedColumns()
    const {
  std::list<std::string> cols;

  return cols;
}

}  // end namespace logical_operator

}  // end namespace query_processing
}  // end namespace CogaDB
