

#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/sink_operator.hpp>

namespace CoGaDB {
namespace query_processing {

namespace logical_operator {

Logical_Sink::Logical_Sink() : Logical_BulkOperator() {}

unsigned int Logical_Sink::getOutputResultSize() const { return 0; }

double Logical_Sink::getCalculatedSelectivity() const { return 1; }

std::string Logical_Sink::getOperationName() const { return "SINK"; }

std::string Logical_Sink::toString(bool verbose) const {
  std::string result = this->getOperationName();
  return result;
}

const TablePtr Logical_Sink::executeBulkOperator(TablePtr table) const {
  return table;
}

const std::list<std::string> Logical_Sink::getNamesOfReferencedColumns() const {
  std::list<std::string> cols;

  return cols;
}

}  // end namespace logical_operator

}  // end namespace query_processing
}  // end namespace CogaDB
