

#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/export_into_file.hpp>

namespace CoGaDB {
namespace query_processing {

namespace logical_operator {

Logical_ExportTableIntoFile::Logical_ExportTableIntoFile(
    const std::string& path_to_file, const std::string& delimiter)
    : Logical_BulkOperator(),
      path_to_file_(path_to_file),
      delimiter_(delimiter) {}

unsigned int Logical_ExportTableIntoFile::getOutputResultSize() const {
  return 0;
}

double Logical_ExportTableIntoFile::getCalculatedSelectivity() const {
  return 1;
}

std::string Logical_ExportTableIntoFile::getOperationName() const {
  return "EXPORT_INTO_FILE";
}

std::string Logical_ExportTableIntoFile::toString(bool verbose) const {
  std::string result = this->getOperationName();
  if (verbose) {
    result += " (";
    //                    result += table_name_;
    //                    TableSchema::const_iterator it;
    if (!path_to_file_.empty()) {
      result += "; ";
      result += "TARGET_FILE='";
      result += path_to_file_;
      result += "', SEPARATOR='";
      result += delimiter_;
      result += "'";
    }
    result += ")";
  }
  return result;
}

const TablePtr Logical_ExportTableIntoFile::executeBulkOperator(
    TablePtr table) const {
  /* handle case where table is intermediate table */
  assert(table != NULL);
  std::ofstream file;
  file.open(path_to_file_.c_str());
  assert(delimiter_ == "|");
  if (file.is_open()) {
    file << table->toString("csv", true);
  } else {
    COGADB_FATAL_ERROR("Error exporting File!", "");
  }
  file.close();

  return table;
}

const std::list<std::string>
Logical_ExportTableIntoFile::getNamesOfReferencedColumns() const {
  std::list<std::string> cols;

  return cols;
}

}  // end namespace logical_operator

}  // end namespace query_processing
}  // end namespace CogaDB
