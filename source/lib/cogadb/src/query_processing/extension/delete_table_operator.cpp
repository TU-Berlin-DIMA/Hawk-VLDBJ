

#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/delete_table_operator.hpp>

namespace CoGaDB {
namespace query_processing {

namespace logical_operator {

Logical_DeleteTable::Logical_DeleteTable(const std::string& table_name)
    : Logical_BulkOperator(), table_name_(table_name) {}

unsigned int Logical_DeleteTable::getOutputResultSize() const { return 0; }

double Logical_DeleteTable::getCalculatedSelectivity() const { return 1; }

std::string Logical_DeleteTable::getOperationName() const {
  return "DELETE_TABLE";
}

std::string Logical_DeleteTable::toString(bool verbose) const {
  std::string result = this->getOperationName();
  if (verbose) {
    result += " (";
    result += table_name_;
    result += ")";
  }
  return result;
}

const TablePtr Logical_DeleteTable::executeBulkOperator(TablePtr table) const {
  assert(table == NULL);
  TablePtr tab = getTablebyName(table_name_);
  bool return_value = true;
  if (tab) {
    if (!dropTable(table_name_)) {
      *(this->out) << "Deletion of Table '" << table_name_ << "' failed!"
                   << std::endl;
      return_value = false;
    }
  } else {
    *(this->out) << "Table '" << table_name_ << "' not found!" << std::endl;
    return_value = false;
  }
  TableSchema schema;
  schema.push_back(std::make_pair(INT, std::string("RETURN_VALUE")));
  TablePtr result(new Table("DELETED_TABLE", schema));
  Tuple t;
  t.push_back(int32_t(return_value));
  result->insert(t);
  return result;
}

const std::list<std::string> Logical_DeleteTable::getNamesOfReferencedColumns()
    const {
  std::list<std::string> cols;

  return cols;
}

}  // end namespace logical_operator

}  // end namespace query_processing
}  // end namespace CogaDB
