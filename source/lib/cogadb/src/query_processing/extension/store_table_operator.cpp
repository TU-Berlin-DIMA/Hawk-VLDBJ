

#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/store_table_operator.hpp>

namespace CoGaDB {
namespace query_processing {

namespace logical_operator {

Logical_StoreTable::Logical_StoreTable(const std::string& table_name,
                                       const bool persist_to_disk)
    : Logical_BulkOperator(),
      table_name_(table_name),
      persist_to_disk_(persist_to_disk) {}

unsigned int Logical_StoreTable::getOutputResultSize() const { return 0; }

double Logical_StoreTable::getCalculatedSelectivity() const { return 1; }

std::string Logical_StoreTable::getOperationName() const {
  return "STORE_TABLE";
}

std::string Logical_StoreTable::toString(bool verbose) const {
  std::string result = this->getOperationName();
  if (verbose) {
    result += " (";
    result += table_name_;
    result += ")";
  }
  return result;
}

const TablePtr Logical_StoreTable::executeBulkOperator(TablePtr table) const {
  /* handle case where table is intermediate table */
  assert(table != NULL);
  if (!table->isMaterialized()) {
    table = table->materialize();
    assert(table != NULL);
  }

  std::cout << "Table Name: " << table->getName() << std::endl;
  std::cout << "Stored Table Name: " << this->table_name_ << std::endl;

  if (getTablebyName(table_name_)) {
    COGADB_FATAL_ERROR("Table '" << table_name_ << "' already exists!", "");
  }

  if (table == getTablebyName(table->getName())) {
    /* In case we want to copy a table in a database and use this store
     * operator,
     * we need to copy the table. */
    table = table->materialize();
    assert(table != NULL);
  }
  table->setName(table_name_);
  /* remove old qualified attribute names */
  renameFullyQualifiedNamesToUnqualifiedNames(table);
  /* add new qualified attribute names based on new table name */
  expandUnqualifiedColumnNamesToQualifiedColumnNames(table);
  if (!addToGlobalTableList(table)) {
    COGADB_FATAL_ERROR("Failed to add table: '"
                           << table->getName() << "' to global table list! "
                           << "A table with the same name already exists!",
                       "");
  }
  if (persist_to_disk_) {
    if (!table->store(RuntimeConfiguration::instance().getPathToDatabase())) {
      COGADB_FATAL_ERROR("Failed to store table: '" << table->getName() << "'",
                         "");
    }
  }
  //  table->setName(table_name_);
  //  if(!storeTable(table, persist_to_disk_)){
  //    std::string table_name;
  //    if(table)
  //      table_name=table->getName();
  //    COGADB_FATAL_ERROR("Failed to stored table '" << table_name << "'","");
  //  }
  return table;
}

const std::list<std::string> Logical_StoreTable::getNamesOfReferencedColumns()
    const {
  std::list<std::string> cols;

  return cols;
}

}  // end namespace logical_operator

}  // end namespace query_processing
}  // end namespace CogaDB
